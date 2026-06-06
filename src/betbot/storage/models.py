"""ORM models. Kept narrow — only the tables we actually use."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    """One row per (fixture_id, run_date) — what the model thought today."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("fixture_id", "run_date", name="uq_predictions_fix_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, index=True)
    competition_code: Mapped[str] = mapped_column(String(8))
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    run_date: Mapped[str] = mapped_column(String(10))  # ISO date string

    home_team: Mapped[str] = mapped_column(String(80))
    away_team: Mapped[str] = mapped_column(String(80))

    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)

    home_score: Mapped[float] = mapped_column(Float)
    away_score: Mapped[float] = mapped_column(Float)
    draw_score: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    paper_bets: Mapped[list["PaperBet"]] = relationship(back_populates="prediction")


class PaperBet(Base):
    """A logged bet. ``market_price`` is None for Phase-1 favourite-only bets."""

    __tablename__ = "paper_bets"
    __table_args__ = (
        UniqueConstraint("fixture_id", "outcome", name="uq_paper_bets_fix_out"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id"), index=True
    )
    fixture_id: Mapped[int] = mapped_column(Integer, index=True)

    outcome: Mapped[str] = mapped_column(String(4))  # HOME / DRAW / AWAY
    our_probability: Mapped[float] = mapped_column(Float)
    market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge: Mapped[float | None] = mapped_column(Float, nullable=True)
    stake_usd: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Settlement fields — filled in by SettlementWatcher in Phase 4.
    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    settled_outcome: Mapped[str | None] = mapped_column(String(4), nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    prediction: Mapped["PredictionRow"] = relationship(back_populates="paper_bets")
