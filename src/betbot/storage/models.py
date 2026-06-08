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


class KillSwitch(Base):
    """Single-row (id=1) drawdown kill switch. Last-write-wins.

    When ``tripped_at`` is set, the scoring loop refuses to log new bets until
    an operator runs ``tfsm kill-switch reset``. ``realized_pnl_usd`` /
    ``staked_usd`` record the trailing-window numbers that tripped it.
    """

    __tablename__ = "kill_switch"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # always 1
    tripped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reason: Mapped[str | None] = mapped_column(String(300), nullable=True)
    realized_pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    staked_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    @property
    def is_tripped(self) -> bool:
        return self.tripped_at is not None


class GlickoRating(Base):
    """Current Glicko-2 rating per national team (Phase 5.5). Upsert on name."""

    __tablename__ = "glicko_ratings"
    __table_args__ = (
        UniqueConstraint("team_name", name="uq_glicko_team_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_name: Mapped[str] = mapped_column(String(80), index=True)
    team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating: Mapped[float] = mapped_column(Float)
    rd: Mapped[float] = mapped_column(Float)
    volatility: Mapped[float] = mapped_column(Float)
    last_period: Mapped[str | None] = mapped_column(String(10), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
