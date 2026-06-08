"""Repository helpers — narrow, intention-revealing CRUD."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from betbot.logging import get_logger
from betbot.storage.db import session_scope
from betbot.storage.models import (
    GlickoRating,
    KillSwitch,
    PaperBet,
    PredictionRow,
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
# Multi-user accounts (each user has their OWN isolated wallet)
# ----------------------------------------------------------------------
def get_or_create_user(telegram_user_id: int, name: str, *, secrets_dir: str) -> User:
    """Return the user, generating a fresh per-user wallet on first contact.

    The wallet key lives at ``<secrets_dir>/users/<telegram_id>.key`` (0600,
    gitignored). Funds stay in this user's own wallet — never pooled.
    """
    from pathlib import Path

    from betbot.wallet import get_or_create_address

    keyfile = Path(secrets_dir) / "users" / f"{telegram_user_id}.key"
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
