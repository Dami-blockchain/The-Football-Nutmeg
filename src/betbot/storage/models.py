"""ORM models. Kept narrow — only the tables we actually use."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
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

    # Expected-goals readout (display-only). Nullable: only populated when the
    # Dixon-Coles component is available for both teams; None on fallback paths.
    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)

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


class ArbScanResult(Base):
    """One row per arb opportunity surfaced by any scan (the periodic watcher
    or the 09:00 digest).

    Scans were previously fire-and-forget Telegram messages; persisting each
    hit is what lets the 21:00 daily report answer "how many arb opportunities
    did we see today?" without re-scanning.
    """

    __tablename__ = "arb_scan_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scanned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    home_team: Mapped[str] = mapped_column(String(80))
    away_team: Mapped[str] = mapped_column(String(80))
    venues: Mapped[str] = mapped_column(String(120))  # e.g. "LIMITLESS+POLYMARKET"
    margin: Mapped[float] = mapped_column(Float)
    price_sum: Mapped[float] = mapped_column(Float)


class Deposit(Base):
    """One leg of a processed user deposit (deposit pipeline, betbot/bridge.py).

    A detected deposit is split into one row per DESTINATION chain (the
    Polygon/Base allocation), each advancing through an explicit per-step
    status (``detected → gas_topped_up → burn_submitted → burned → minted →
    done``; local legs where source == dest start at ``minted``). This table
    is the pipeline's idempotency record: ``burn_submitted`` persists the
    signed burn's tx hash BEFORE it is broadcast, so a crash or receipt
    timeout in the broadcast window can never lose the hash — recovery
    verifies that tx on-chain instead of re-burning, and a half-finished
    pipeline resumes from the last completed step.

    No unique constraint on the balance snapshot ON PURPOSE: two deposits of
    the same amount on the same chain are legitimate over time. Double
    processing is prevented by (a) refusing to detect on a chain while any
    non-terminal leg for that (wallet, source_chain) exists and (b) the
    delivered-balance baseline in ``delivered_to_chain_usdc``.
    """

    __tablename__ = "deposits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    wallet_address: Mapped[str] = mapped_column(String(64), index=True)
    source_chain: Mapped[str] = mapped_column(String(16), index=True)
    dest_chain: Mapped[str] = mapped_column(String(16))
    amount_usdc: Mapped[float] = mapped_column(Float)
    # USDC balance observed on source_chain at detection time (audit trail).
    balance_snapshot: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24), index=True)
    burn_tx: Mapped[str | None] = mapped_column(String(80), nullable=True)
    mint_tx: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class TreasuryBridge(Base):
    """One in-flight AGENT-treasury rebalance leg (betbot/bridge.py).

    Separate from ``deposits`` ON PURPOSE: deposit legs belong to a registered
    user (NOT-NULL ``user_id`` FK) and end in venue approvals; a treasury leg
    is the operator's own seed float repositioned between trading chains by the
    agent wallet itself (both burner AND mint-relayer), with no user and no
    venue setup. Overloading the deposits table would mean a sentinel user and
    special-casing every deposit guard. A small dedicated table keeps the
    user-deposit invariants untouched.

    Crash-safety mirrors the deposit pipeline exactly: the signed burn's tx
    hash is persisted (status ``burn_submitted``) BEFORE the broadcast, so a
    receipt timeout or daemon death can never lose the hash and re-burn —
    recovery verifies that tx on-chain. At most ONE non-``done`` row exists at
    a time (the single-in-flight invariant), so a query for an active row is
    the rebalancer's idempotency lock.
    """

    __tablename__ = "treasury_bridges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_chain: Mapped[str] = mapped_column(String(16), index=True)
    dest_chain: Mapped[str] = mapped_column(String(16))
    amount_usdc: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24), index=True)
    burn_tx: Mapped[str | None] = mapped_column(String(80), nullable=True)
    mint_tx: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class GasTopup(Base):
    """One agent-funded native-gas top-up sent to a user wallet.

    The deposit pipeline's abuse guard: registration is open to the public,
    so agent-wallet gas spend triggered by third-party deposits must be
    bounded. ``bridge._ensure_gas`` counts rows here to enforce the
    per-wallet per-UTC-day cap (``BETBOT_GAS_TOPUP_DAILY_CAP``) — persisted
    rather than in-memory so a daemon restart never resets the budget.
    """

    __tablename__ = "gas_topups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    wallet_address: Mapped[str] = mapped_column(String(64), index=True)
    chain: Mapped[str] = mapped_column(String(16))
    amount: Mapped[float] = mapped_column(Float)  # native units (POL / ETH)
    tx: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


class User(Base):
    """A tenant of the multi-user bot. Each user has their OWN isolated wallet
    and funds — nothing is pooled. The bot trades each user's wallet
    independently; one user's outcome never touches another's."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("telegram_user_id", name="uq_users_tg"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String(80))
    wallet_address: Mapped[str] = mapped_column(String(64))
    wallet_keyfile: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(default=True)
    # Set True once the user taps the cross-venue arbitrage "tell me more"
    # button on /start. Legacy column, retained for DB compatibility only.
    arb_interest: Mapped[bool] = mapped_column(default=False)
    # Paid predictions this user has consumed (post-trial). Each 1 USDC held
    # buys one reveal; ``credits_remaining = floor(usdc) - predictions_consumed``.
    # ``created_at`` is the trial start. See :mod:`betbot.entitlement`.
    predictions_consumed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class PredictionReveal(Base):
    """Per-(user, fixture) reveal ledger — the money-idempotency record.

    A single fixture prediction reaches a user through THREE paths (the
    matchday-morning alert, the ~kickoff-60m alert, and every repeat of
    ``/predictions``). Without this ledger each path — and each repeat —
    re-charges a paying user for the SAME fixture. One row here per fixture a
    user has been shown means each fixture is charged AT MOST ONCE and re-shown
    FREE forever after. ``charged`` records whether the reveal actually cost a
    credit (True only for a paid reveal; operator/trial reveals are recorded
    ``charged=False`` so they stay free once the trial ends).

    The unique constraint is the idempotency lock: an INSERT that collides is a
    no-op (see :func:`betbot.storage.repos.record_reveal`), so committing a
    reveal twice (e.g. a retried send) never double-charges.
    """

    __tablename__ = "prediction_reveals"
    __table_args__ = (
        UniqueConstraint(
            "telegram_user_id", "fixture_id", name="uq_reveal_user_fixture"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, index=True)
    fixture_id: Mapped[int] = mapped_column(Integer, index=True)
    charged: Mapped[bool] = mapped_column(default=False)
    revealed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class ModelPrediction(Base):
    """Dual-logged model comparison for online selection (Hedge).

    One row per WC fixture: what pure Glicko and the ensemble each predicted
    pre-match, the Hedge weights actually used, and — once settled — each
    model's RPS. The cumulative RPS sums drive the live weighting, so the bot
    converges onto whichever model is winning THIS tournament.
    """

    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint("fixture_id", name="uq_model_pred_fixture"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, index=True)
    home_team: Mapped[str] = mapped_column(String(80))
    away_team: Mapped[str] = mapped_column(String(80))

    g_home: Mapped[float] = mapped_column(Float)   # pure-Glicko triple
    g_draw: Mapped[float] = mapped_column(Float)
    g_away: Mapped[float] = mapped_column(Float)
    e_home: Mapped[float] = mapped_column(Float)   # ensemble triple
    e_draw: Mapped[float] = mapped_column(Float)
    e_away: Mapped[float] = mapped_column(Float)
    w_glicko: Mapped[float] = mapped_column(Float)     # Hedge weights used
    w_ensemble: Mapped[float] = mapped_column(Float)

    # Dispersion challenger triple (flag-gated experiment, dual-logged always).
    c_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_away: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Margin-of-victory challenger triple (flag-gated experiment, dual-logged).
    m_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    m_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    m_away: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Filled at settlement.
    outcome: Mapped[str | None] = mapped_column(String(4), nullable=True)
    rps_glicko: Mapped[float | None] = mapped_column(Float, nullable=True)
    rps_ensemble: Mapped[float | None] = mapped_column(Float, nullable=True)
    rps_challenger: Mapped[float | None] = mapped_column(Float, nullable=True)
    rps_mov: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ArbExecution(Base):
    """One cross-venue arbitrage execution attempt (armed executor only).

    ``legs_json`` is the opportunity snapshot plus per-leg execution state
    (venue, quoted price, planned stake, status, fill price, order id).
    ``status``: ``rejected_*`` (gated before any order — no money moved),
    ``executing`` (orders in flight), ``aborted_no_exposure`` (first leg failed,
    nothing filled), ``partial_exposure_open`` (a later leg failed AFTER fills —
    operator must close manually), ``filled`` (all legs filled).
    """

    __tablename__ = "arb_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    home_team: Mapped[str] = mapped_column(String(80))
    away_team: Mapped[str] = mapped_column(String(80))
    margin: Mapped[float] = mapped_column(Float)
    price_sum: Mapped[float] = mapped_column(Float)
    stake_usd: Mapped[float] = mapped_column(Float)  # total planned across legs
    status: Mapped[str] = mapped_column(String(32), index=True)
    legs_json: Mapped[str] = mapped_column(Text, default="{}")
    net_expected_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
