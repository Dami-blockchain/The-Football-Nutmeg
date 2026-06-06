"""Repository helpers — narrow, intention-revealing CRUD."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from betbot.logging import get_logger
from betbot.storage.db import session_scope
from betbot.storage.models import PaperBet, PredictionRow
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
