"""Repository helpers — narrow, intention-revealing CRUD."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from betbot.logging import get_logger
from betbot.storage.db import session_scope
from betbot.storage.models import KillSwitch, PaperBet, PredictionRow
from betbot.strategy.engine import BetDecision, Outcome, Prediction

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
