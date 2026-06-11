"""Repository helpers — narrow, intention-revealing CRUD."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select

from betbot.logging import get_logger
from betbot.storage.db import session_scope
from betbot.storage.models import (
    ArbExecution,
    ArbScanResult,
    Deposit,
    GasTopup,
    GlickoRating,
    KillSwitch,
    ModelPrediction,
    PaperBet,
    PredictionRow,
    TreasuryBridge,
    User,
)
from betbot.strategy.engine import BetDecision, Outcome, Prediction
from betbot.strategy.glicko import Glicko2Rating, update_rating

log = get_logger(__name__)


def upsert_prediction(
    prediction: Prediction, *, kickoff: datetime
) -> int:
    """Insert a prediction row (or update if (fixture_id, run_date) exists).

    Returns the row id.
    """
    run_date = date.today().isoformat()
    with session_scope() as s:
        existing = s.execute(
            select(PredictionRow)
            .where(PredictionRow.fixture_id == prediction.fixture_id)
            .where(PredictionRow.run_date == run_date)
        ).scalar_one_or_none()
        if existing is not None:
            existing.p_home = prediction.p_home
            existing.p_draw = prediction.p_draw
            existing.p_away = prediction.p_away
            existing.home_score = prediction.home_score
            existing.away_score = prediction.away_score
            existing.draw_score = prediction.draw_score
            s.flush()
            return existing.id
        row = PredictionRow(
            fixture_id=prediction.fixture_id,
            competition_code=prediction.competition_code,
            kickoff=kickoff,
            run_date=run_date,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            p_home=prediction.p_home,
            p_draw=prediction.p_draw,
            p_away=prediction.p_away,
            home_score=prediction.home_score,
            away_score=prediction.away_score,
            draw_score=prediction.draw_score,
        )
        s.add(row)
        s.flush()
        return row.id


def insert_paper_bet(decision: BetDecision, prediction_id: int) -> bool:
    """Insert an edge-filtered paper bet. Returns False if a duplicate exists."""
    with session_scope() as s:
        existing = s.execute(
            select(PaperBet)
            .where(PaperBet.fixture_id == decision.fixture_id)
            .where(PaperBet.outcome == decision.outcome.value)
        ).scalar_one_or_none()
        if existing is not None:
            return False
        s.add(
            PaperBet(
                prediction_id=prediction_id,
                fixture_id=decision.fixture_id,
                outcome=decision.outcome.value,
                our_probability=decision.our_probability,
                market_price=decision.market_price,
                edge=decision.edge,
                stake_usd=decision.stake_usd,
                rationale=decision.rationale,
            )
        )
        return True


def insert_paper_bet_no_market(
    prediction: Prediction,
    prediction_id: int,
    outcome: Outcome,
    *,
    stake_usd: float,
    rationale: str,
) -> bool:
    """Insert a Phase-1-style favourite-only paper bet (no market price)."""
    our_p = {
        Outcome.HOME: prediction.p_home,
        Outcome.DRAW: prediction.p_draw,
        Outcome.AWAY: prediction.p_away,
    }[outcome]
    with session_scope() as s:
        existing = s.execute(
            select(PaperBet)
            .where(PaperBet.fixture_id == prediction.fixture_id)
            .where(PaperBet.outcome == outcome.value)
        ).scalar_one_or_none()
        if existing is not None:
            return False
        s.add(
            PaperBet(
                prediction_id=prediction_id,
                fixture_id=prediction.fixture_id,
                outcome=outcome.value,
                our_probability=our_p,
                market_price=None,
                edge=None,
                stake_usd=stake_usd,
                rationale=rationale,
            )
        )
        return True


def list_recent_paper_bets(days: int = 7) -> list[PaperBet]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet)
                .where(PaperBet.created_at >= cutoff)
                .order_by(PaperBet.created_at.desc())
            ).scalars()
        )
        # Detach so attribute access works after the session closes.
        s.expunge_all()
        return rows


def daily_paper_exposure_usd() -> float:
    """Sum of stakes placed today (used by the risk control)."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    with session_scope() as s:
        rows = s.execute(
            select(PaperBet.stake_usd).where(PaperBet.created_at >= today_start)
        ).scalars()
        return float(sum(rows))


# ----------------------------------------------------------------------
# Settlement (Phase 4)
# ----------------------------------------------------------------------
def list_unsettled_bets_due(now: datetime, grace_minutes: int) -> list[PaperBet]:
    """Unsettled bets whose kickoff + grace has passed — ready to settle."""
    cutoff = now - timedelta(minutes=grace_minutes)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet)
                .join(PredictionRow, PaperBet.prediction_id == PredictionRow.id)
                .where(PaperBet.settled_at.is_(None))
                .where(PredictionRow.kickoff <= cutoff)
                .order_by(PaperBet.fixture_id)
            ).scalars()
        )
        # Detach so attribute access works after the session closes.
        s.expunge_all()
        return rows


def record_settlement(
    bet_id: int, settled_outcome: str, pnl_usd: float, settled_at: datetime
) -> None:
    with session_scope() as s:
        bet = s.get(PaperBet, bet_id)
        if bet is None:
            return
        bet.settled_at = settled_at
        bet.settled_outcome = settled_outcome
        bet.pnl_usd = pnl_usd


def settled_pnl_window(days: int) -> tuple[float, float]:
    """``(realized_pnl, staked)`` over settled MARKET bets in the trailing window.

    No-market (favourite-only) bets are excluded — they have no real-money
    equivalent and must not pollute the kill-switch / gate signal.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet.pnl_usd, PaperBet.stake_usd)
                .where(PaperBet.settled_at.is_not(None))
                .where(PaperBet.settled_at >= cutoff)
                .where(PaperBet.market_price.is_not(None))
            )
        )
    pnl = float(sum((r[0] or 0.0) for r in rows))
    staked = float(sum((r[1] or 0.0) for r in rows))
    return pnl, staked


def list_recent_predictions(days: int = 7) -> list[PredictionRow]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PredictionRow)
                .where(PredictionRow.created_at >= cutoff)
                .order_by(PredictionRow.kickoff.asc())
            ).scalars()
        )
        s.expunge_all()
        return rows


def list_settled_market_bets(window_days: int | None = None) -> list[PaperBet]:
    """Settled bets that carried a market price (favourite-only bets excluded).

    Ordered oldest-first by settlement time so callers can measure the window
    span. Used by the backtest + gate.
    """
    with session_scope() as s:
        stmt = (
            select(PaperBet)
            .where(PaperBet.settled_at.is_not(None))
            .where(PaperBet.market_price.is_not(None))
            .order_by(PaperBet.settled_at.asc())
        )
        if window_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
            stmt = stmt.where(PaperBet.settled_at >= cutoff)
        rows = list(s.execute(stmt).scalars())
        s.expunge_all()
        return rows


# ----------------------------------------------------------------------
# Daily report queries (scheduled Telegram jobs)
# ----------------------------------------------------------------------
def list_bets_created_between(
    start: datetime, end: datetime
) -> list[tuple[PaperBet, str, str]]:
    """``(bet, home_team, away_team)`` for bets logged in ``[start, end)``.

    Team names are joined in here (rather than via the lazy relationship)
    because rows are detached before returning — relationship access on a
    detached row raises DetachedInstanceError.
    """
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet, PredictionRow.home_team, PredictionRow.away_team)
                .join(PredictionRow, PaperBet.prediction_id == PredictionRow.id)
                .where(PaperBet.created_at >= start)
                .where(PaperBet.created_at < end)
                .order_by(PaperBet.created_at.asc())
            )
        )
        s.expunge_all()
        return [(r[0], r[1], r[2]) for r in rows]


def list_bets_settled_between(
    start: datetime, end: datetime
) -> list[tuple[PaperBet, str, str]]:
    """``(bet, home_team, away_team)`` for bets settled in ``[start, end)``."""
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet, PredictionRow.home_team, PredictionRow.away_team)
                .join(PredictionRow, PaperBet.prediction_id == PredictionRow.id)
                .where(PaperBet.settled_at.is_not(None))
                .where(PaperBet.settled_at >= start)
                .where(PaperBet.settled_at < end)
                .order_by(PaperBet.settled_at.asc())
            )
        )
        s.expunge_all()
        return [(r[0], r[1], r[2]) for r in rows]


def cumulative_realized_pnl_usd() -> float:
    """All-time realised P&L over settled bets.

    No-market (favourite-only) bets settle at 0 by design, so including them
    changes nothing — this is the honest "since inception" number.
    """
    with session_scope() as s:
        rows = s.execute(
            select(PaperBet.pnl_usd).where(PaperBet.settled_at.is_not(None))
        ).scalars()
        return float(sum(r or 0.0 for r in rows))


def record_arb_opportunity(
    *,
    home_team: str,
    away_team: str,
    venues: str,
    margin: float,
    price_sum: float,
    scanned_at: datetime | None = None,
) -> None:
    """Persist one arb-scan hit so the daily report can count today's total."""
    with session_scope() as s:
        s.add(
            ArbScanResult(
                scanned_at=scanned_at or datetime.now(timezone.utc),
                home_team=home_team[:80],
                away_team=away_team[:80],
                venues=venues[:120],
                margin=margin,
                price_sum=price_sum,
            )
        )


def count_arb_opportunities_between(start: datetime, end: datetime) -> int:
    """How many arb opportunities all scans recorded in ``[start, end)``."""
    with session_scope() as s:
        n = s.execute(
            select(func.count())
            .select_from(ArbScanResult)
            .where(ArbScanResult.scanned_at >= start)
            .where(ArbScanResult.scanned_at < end)
        ).scalar_one()
        return int(n)


# ----------------------------------------------------------------------
# Kill switch (Phase 4)
# ----------------------------------------------------------------------
def get_kill_switch() -> KillSwitch:
    """Return the single kill-switch row, creating it (untripped) if absent."""
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        if ks is None:
            ks = KillSwitch(id=1)
            s.add(ks)
            s.flush()
        s.expunge_all()
        return ks


def is_kill_switch_tripped() -> bool:
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        return ks is not None and ks.tripped_at is not None


def trip_kill_switch(
    reason: str, realized_pnl_usd: float, staked_usd: float
) -> None:
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        if ks is None:
            ks = KillSwitch(id=1)
            s.add(ks)
        ks.tripped_at = datetime.now(timezone.utc)
        ks.reason = reason[:300]
        ks.realized_pnl_usd = realized_pnl_usd
        ks.staked_usd = staked_usd


def reset_kill_switch() -> None:
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        if ks is None:
            return
        ks.tripped_at = None
        ks.reason = None
        ks.realized_pnl_usd = None
        ks.staked_usd = None


# ----------------------------------------------------------------------
# Glicko ratings (Phase 5.5)
# ----------------------------------------------------------------------
def get_rating(
    team_name: str,
    *,
    default_rating: float = 1500.0,
    default_rd: float = 200.0,
    default_vol: float = 0.06,
) -> Glicko2Rating:
    with session_scope() as s:
        row = s.execute(
            select(GlickoRating).where(GlickoRating.team_name == team_name)
        ).scalar_one_or_none()
        if row is None:
            return Glicko2Rating(default_rating, default_rd, default_vol)
        return Glicko2Rating(row.rating, row.rd, row.volatility, row.last_period)


def upsert_rating(team_name: str, rating: Glicko2Rating, *, team_id: int | None = None) -> None:
    with session_scope() as s:
        row = s.execute(
            select(GlickoRating).where(GlickoRating.team_name == team_name)
        ).scalar_one_or_none()
        if row is None:
            s.add(GlickoRating(
                team_name=team_name, team_id=team_id, rating=rating.rating,
                rd=rating.rd, volatility=rating.volatility, last_period=rating.last_period,
            ))
        else:
            row.rating, row.rd, row.volatility = rating.rating, rating.rd, rating.volatility
            row.last_period = rating.last_period
            if team_id is not None:
                row.team_id = team_id


def all_ratings() -> list[tuple[str, Glicko2Rating]]:
    with session_scope() as s:
        rows = list(
            s.execute(select(GlickoRating).order_by(GlickoRating.rating.desc())).scalars()
        )
        return [(r.team_name, Glicko2Rating(r.rating, r.rd, r.volatility, r.last_period))
                for r in rows]


def apply_rating_period(
    matches: list[tuple[str, str, str]],
    period: str,
    *,
    tau: float = 0.5,
    default_rating: float = 1500.0,
    default_rd: float = 200.0,
    default_vol: float = 0.06,
) -> int:
    """Apply one Glicko-2 rating period from ``(home, away, outcome)`` results.

    ``outcome`` is "HOME"/"AWAY"/"DRAW". All updates use opponents' PRE-period
    ratings (correct Glicko-2 semantics), then persist. Returns teams updated.
    """
    teams: set[str] = set()
    for home, away, _ in matches:
        teams.add(home)
        teams.add(away)
    current = {
        t: get_rating(t, default_rating=default_rating, default_rd=default_rd,
                      default_vol=default_vol)
        for t in teams
    }
    per_team: dict[str, list[tuple[float, float, float]]] = {t: [] for t in teams}
    for home, away, outcome in matches:
        sh = 1.0 if outcome == "HOME" else (0.5 if outcome == "DRAW" else 0.0)
        sa = 1.0 if outcome == "AWAY" else (0.5 if outcome == "DRAW" else 0.0)
        per_team[home].append((current[away].rating, current[away].rd, sh))
        per_team[away].append((current[home].rating, current[home].rd, sa))
    for t in teams:
        upsert_rating(t, update_rating(current[t], per_team[t], tau=tau, period=period))
    return len(teams)


# ----------------------------------------------------------------------
# Deposit pipeline (CCTP bridging — betbot/bridge.py)
# ----------------------------------------------------------------------
# Terminal status: a leg whose funds have been delivered AND venue approvals
# have run. Everything else is "active" and (a) blocks re-detection on its
# source chain and (b) gets resumed by every scan tick.
DEPOSIT_DONE = "done"


def create_deposit(
    *,
    user_id: int,
    wallet_address: str,
    source_chain: str,
    dest_chain: str,
    amount_usdc: float,
    balance_snapshot: float,
    status: str,
) -> int:
    with session_scope() as s:
        row = Deposit(
            user_id=user_id,
            wallet_address=wallet_address,
            source_chain=source_chain,
            dest_chain=dest_chain,
            amount_usdc=amount_usdc,
            balance_snapshot=balance_snapshot,
            status=status,
        )
        s.add(row)
        s.flush()
        return row.id


def has_active_source_deposit(wallet_address: str, source_chain: str) -> bool:
    """True while ANY leg sourced from this (wallet, chain) is unfinished.

    This is the first idempotency guard: a balance seen twice while its
    pipeline is in flight must not create a second deposit record.
    """
    with session_scope() as s:
        row = s.execute(
            select(Deposit.id)
            .where(Deposit.wallet_address == wallet_address)
            .where(Deposit.source_chain == source_chain)
            .where(Deposit.status != DEPOSIT_DONE)
            .limit(1)
        ).scalar_one_or_none()
        return row is not None


def has_active_dest_deposit(wallet_address: str, dest_chain: str) -> bool:
    """True while ANY unfinished leg is bridging TOWARD this (wallet, chain).

    Detection guard: an in-flight leg's mint can land on the destination
    chain before its status persists; detecting balances there while the leg
    is active would record the bridged funds as a phantom new deposit and
    double-count the delivered baseline.
    """
    with session_scope() as s:
        row = s.execute(
            select(Deposit.id)
            .where(Deposit.wallet_address == wallet_address)
            .where(Deposit.dest_chain == dest_chain)
            .where(Deposit.status != DEPOSIT_DONE)
            .limit(1)
        ).scalar_one_or_none()
        return row is not None


def list_active_deposits() -> list[Deposit]:
    """Unfinished legs, oldest first — the scan tick resumes each of these."""
    with session_scope() as s:
        rows = list(
            s.execute(
                select(Deposit)
                .where(Deposit.status != DEPOSIT_DONE)
                .order_by(Deposit.created_at.asc())
            ).scalars()
        )
        s.expunge_all()
        return rows


def delivered_to_chain_usdc(wallet_address: str, dest_chain: str) -> float:
    """USDC already delivered to ``dest_chain`` for this wallet by past legs.

    Used as the detection baseline on TRADING chains (where delivered funds
    stay in the wallet): a new deposit is only the balance ABOVE this number.
    Local legs (source == dest) count from creation — the funds never left;
    bridged legs count once minted. Trading spend pushes the real balance
    below this baseline, which only makes detection more conservative (we
    under-detect rather than ever double-bridge).
    """
    with session_scope() as s:
        rows = list(
            s.execute(
                select(Deposit.amount_usdc, Deposit.source_chain, Deposit.status)
                .where(Deposit.wallet_address == wallet_address)
                .where(Deposit.dest_chain == dest_chain)
            )
        )
    total = 0.0
    for amount, source_chain, status in rows:
        if source_chain == dest_chain or status in ("minted", DEPOSIT_DONE):
            total += amount or 0.0
    return total


def update_deposit(
    deposit_id: int,
    *,
    status: str | None = None,
    burn_tx: str | None = None,
    mint_tx: str | None = None,
    error: str | None = None,
) -> None:
    with session_scope() as s:
        row = s.get(Deposit, deposit_id)
        if row is None:
            return
        if status is not None:
            row.status = status
            row.error = None  # a successful step clears the previous error
        if burn_tx is not None:
            row.burn_tx = burn_tx
        if mint_tx is not None:
            row.mint_tx = mint_tx
        if error is not None:
            row.error = error[:300]


# ----------------------------------------------------------------------
# Treasury rebalancer (agent-owned float; betbot/bridge.py TreasuryRebalancer)
# ----------------------------------------------------------------------
def create_treasury_bridge(
    *, source_chain: str, dest_chain: str, amount_usdc: float, status: str
) -> int:
    with session_scope() as s:
        row = TreasuryBridge(
            source_chain=source_chain,
            dest_chain=dest_chain,
            amount_usdc=amount_usdc,
            status=status,
        )
        s.add(row)
        s.flush()
        return row.id


def active_treasury_bridge() -> TreasuryBridge | None:
    """The single in-flight treasury leg, if any (oldest non-``done`` row).

    This IS the rebalancer's idempotency lock: a non-None result means a
    bridge is already mid-flight, so no new one may start. Detached for use
    after the session closes.
    """
    with session_scope() as s:
        row = s.execute(
            select(TreasuryBridge)
            .where(TreasuryBridge.status != DEPOSIT_DONE)
            .order_by(TreasuryBridge.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()
        if row is not None:
            s.expunge_all()
        return row


def update_treasury_bridge(
    bridge_id: int,
    *,
    status: str | None = None,
    burn_tx: str | None = None,
    mint_tx: str | None = None,
    error: str | None = None,
) -> None:
    with session_scope() as s:
        row = s.get(TreasuryBridge, bridge_id)
        if row is None:
            return
        if status is not None:
            row.status = status
            row.error = None  # a successful step clears the previous error
        if burn_tx is not None:
            row.burn_tx = burn_tx
        if mint_tx is not None:
            row.mint_tx = mint_tx
        if error is not None:
            row.error = error[:300]


# ----------------------------------------------------------------------
# Agent gas spend audit (deposit pipeline abuse guard)
# ----------------------------------------------------------------------
def record_gas_topup(
    *, wallet_address: str, chain: str, amount: float, tx: str | None = None
) -> None:
    """Persist one agent-funded native-gas top-up (see bridge._ensure_gas)."""
    with session_scope() as s:
        s.add(
            GasTopup(
                wallet_address=wallet_address, chain=chain, amount=amount, tx=tx
            )
        )


def count_gas_topups_since(wallet_address: str, since: datetime) -> int:
    """Top-ups this wallet received since ``since`` — feeds the daily cap.

    Persisted (not in-memory) ON PURPOSE: a daemon restart must not reset
    the cap, or an attacker could trigger unlimited agent gas spend by
    timing deposits around restarts.
    """
    with session_scope() as s:
        n = s.execute(
            select(func.count())
            .select_from(GasTopup)
            .where(GasTopup.wallet_address == wallet_address)
            .where(GasTopup.created_at >= since)
        ).scalar_one()
        return int(n)


# Arbitrage executions
# ----------------------------------------------------------------------
def insert_arb_execution(
    *,
    home_team: str,
    away_team: str,
    margin: float,
    price_sum: float,
    stake_usd: float,
    status: str,
    legs_json: str,
    net_expected_usd: float | None = None,
    error: str | None = None,
) -> int:
    """Persist one arb execution attempt; returns the row id."""
    with session_scope() as s:
        row = ArbExecution(
            home_team=home_team[:80],
            away_team=away_team[:80],
            margin=margin,
            price_sum=price_sum,
            stake_usd=stake_usd,
            status=status,
            legs_json=legs_json,
            net_expected_usd=net_expected_usd,
            error=error[:500] if error else None,
        )
        s.add(row)
        s.flush()
        return row.id


def update_arb_execution(
    exec_id: int,
    *,
    status: str | None = None,
    legs_json: str | None = None,
    net_expected_usd: float | None = None,
    error: str | None = None,
) -> None:
    with session_scope() as s:
        row = s.get(ArbExecution, exec_id)
        if row is None:
            return
        if status is not None:
            row.status = status
        if legs_json is not None:
            row.legs_json = legs_json
        if net_expected_usd is not None:
            row.net_expected_usd = net_expected_usd
        if error is not None:
            row.error = error[:500]


def arb_staked_today_usd() -> float:
    """Today's total arb stake over rows where money moved (or may have).

    ``rejected_*`` rows were gated before any order and don't count; everything
    else (executing / aborted / partial / filled) counts conservatively toward
    the BETBOT_ARB_DAILY_CAP_USD daily cap.
    """
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    with session_scope() as s:
        rows = s.execute(
            select(ArbExecution.stake_usd)
            .where(ArbExecution.created_at >= today_start)
            .where(ArbExecution.status.not_like("rejected%"))
        ).scalars()
        return float(sum(rows))


def list_recent_arb_executions(days: int = 7) -> list[ArbExecution]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(ArbExecution)
                .where(ArbExecution.created_at >= cutoff)
                .order_by(ArbExecution.created_at.desc())
            ).scalars()
        )
        s.expunge_all()
        return rows


# ----------------------------------------------------------------------
# Multi-user accounts (each user has their OWN isolated wallet)
# ----------------------------------------------------------------------
def get_or_create_user(
    telegram_user_id: int, name: str, *, secrets_dir: str, keyfile: str | None = None
) -> User:
    """Return the user, generating a fresh per-user wallet on first contact.

    The wallet key lives at ``<secrets_dir>/users/<telegram_id>.key`` (0600,
    gitignored) unless ``keyfile`` overrides it — used to map the operator onto
    the existing agent wallet so their prior deposit isn't stranded. Funds stay
    in each user's own wallet — never pooled.
    """
    from pathlib import Path

    from betbot.wallet import get_or_create_address

    keyfile = Path(keyfile) if keyfile else Path(secrets_dir) / "users" / f"{telegram_user_id}.key"
    with session_scope() as s:
        u = s.execute(
            select(User).where(User.telegram_user_id == telegram_user_id)
        ).scalar_one_or_none()
        if u is None:
            address = get_or_create_address(keyfile)
            u = User(
                telegram_user_id=telegram_user_id, name=name[:80],
                wallet_address=address, wallet_keyfile=str(keyfile), active=True,
            )
            s.add(u)
            s.flush()
        s.expunge_all()
        return u


def get_user(telegram_user_id: int) -> User | None:
    with session_scope() as s:
        u = s.execute(
            select(User).where(User.telegram_user_id == telegram_user_id)
        ).scalar_one_or_none()
        if u is not None:
            s.expunge_all()
        return u


def list_users() -> list[User]:
    with session_scope() as s:
        rows = list(
            s.execute(select(User).where(User.active.is_(True))
                      .order_by(User.created_at.asc())).scalars()
        )
        s.expunge_all()
        return rows


# ---- Online model selection (Hedge) — see strategy/model_select.py ------

_OUTCOME_INDEX = {"HOME": 0, "DRAW": 1, "AWAY": 2}


def upsert_model_prediction(
    fixture_id: int,
    home_team: str,
    away_team: str,
    glicko: tuple[float, float, float],
    ensemble: tuple[float, float, float],
    weights: tuple[float, float],
) -> None:
    """Record both models' pre-match view. Re-predicting before kickoff
    overwrites; once an outcome is scored the row is frozen."""
    with session_scope() as s:
        row = s.execute(
            select(ModelPrediction).where(ModelPrediction.fixture_id == fixture_id)
        ).scalar_one_or_none()
        if row is not None and row.outcome is not None:
            return  # already settled — never rewrite history
        if row is None:
            row = ModelPrediction(fixture_id=fixture_id,
                                  home_team=home_team, away_team=away_team,
                                  g_home=0, g_draw=0, g_away=0,
                                  e_home=0, e_draw=0, e_away=0,
                                  w_glicko=0, w_ensemble=0)
            s.add(row)
        row.home_team, row.away_team = home_team, away_team
        row.g_home, row.g_draw, row.g_away = glicko
        row.e_home, row.e_draw, row.e_away = ensemble
        row.w_glicko, row.w_ensemble = weights


def score_model_prediction(fixture_id: int, outcome: str) -> bool:
    """Fill RPS for both models once the result is known. Idempotent —
    returns False if there's no row or it was already scored."""
    from betbot.strategy.ensemble import ranked_probability_score

    oi = _OUTCOME_INDEX.get(outcome)
    if oi is None:
        return False
    with session_scope() as s:
        row = s.execute(
            select(ModelPrediction).where(ModelPrediction.fixture_id == fixture_id)
        ).scalar_one_or_none()
        if row is None or row.outcome is not None:
            return False
        row.outcome = outcome
        row.rps_glicko = ranked_probability_score(
            (row.g_home, row.g_draw, row.g_away), oi
        )
        row.rps_ensemble = ranked_probability_score(
            (row.e_home, row.e_draw, row.e_away), oi
        )
        return True


def model_select_losses() -> tuple[float, float, int]:
    """(cumulative RPS glicko, cumulative RPS ensemble, n scored matches)."""
    with session_scope() as s:
        rows = list(
            s.execute(
                select(ModelPrediction).where(ModelPrediction.outcome.is_not(None))
            ).scalars()
        )
        return (
            sum(r.rps_glicko or 0.0 for r in rows),
            sum(r.rps_ensemble or 0.0 for r in rows),
            len(rows),
        )


def model_select_summary(eta: float = 2.0) -> dict:
    """Operator-facing snapshot: who's winning the live model race."""
    from betbot.strategy.model_select import hedge_weights

    lg, le, n = model_select_losses()
    w_g, w_e = hedge_weights(lg, le, eta=eta)
    return {
        "scored_matches": n,
        "cum_rps_glicko": round(lg, 4),
        "cum_rps_ensemble": round(le, 4),
        "avg_rps_glicko": round(lg / n, 4) if n else None,
        "avg_rps_ensemble": round(le / n, 4) if n else None,
        "weight_glicko": round(w_g, 3),
        "weight_ensemble": round(w_e, 3),
        "leader": ("glicko" if lg < le else "ensemble") if n else "no data yet",
    }
