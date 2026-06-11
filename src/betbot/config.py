"""Centralised configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Top-5 European leagues + UCL. Codes are football-data.org competition IDs.
LEAGUE_CODES: tuple[str, ...] = ("PL", "PD", "BL1", "SA", "FL1", "CL", "WC")
INTERNATIONAL_COMPETITIONS: frozenset[str] = frozenset({"WC"})

class Settings(BaseSettings):
    """All runtime knobs. Loaded from .env at process start.

    Mutable by tests (we don't ``frozen=True``), but the production daemon
    only reads it once at startup via :func:`get_settings`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Mode ----------------------------------------------------------
    mode: Literal["paper", "live"] = Field(default="paper", alias="BETBOT_MODE")

    # ---- Football-data.org --------------------------------------------
    football_data_api_key: str = Field(default="", alias="FOOTBALL_DATA_API_KEY")
    football_data_base_url: str = Field(
        default="https://api.football-data.org/v4",
        alias="FOOTBALL_DATA_BASE_URL",
    )
    football_data_rate_limit_per_min: int = Field(
        default=10, alias="FOOTBALL_DATA_RATE_LIMIT_PER_MIN"
    )

    # League scope (immutable for v1).
    leagues: tuple[str, ...] = LEAGUE_CODES

    # ---- Strategy knobs -----------------------------------------------
    home_advantage: float = Field(default=0.3, alias="BETBOT_HOME_ADVANTAGE")
    draw_score: float = Field(default=2.4, alias="BETBOT_DRAW_SCORE")
    softmax_temp: float = Field(default=1.0, alias="BETBOT_SOFTMAX_TEMP")
    opp_strength_weight: float = Field(
        default=0.5, alias="BETBOT_OPP_STRENGTH_WEIGHT"
    )

    # ---- Risk controls ------------------------------------------------
    fixed_stake_usd: float = Field(default=10.0, alias="BETBOT_FIXED_STAKE_USD")
    max_bet_usd: float = Field(default=50.0, alias="BETBOT_MAX_BET_USD")
    daily_exposure_cap_usd: float = Field(
        default=200.0, alias="BETBOT_DAILY_EXPOSURE_CAP_USD"
    )
    edge_threshold: float = Field(default=0.05, alias="BETBOT_EDGE_THRESHOLD")

    # ---- Settlement + drawdown kill switch (Phase 4) ------------------
    settle_grace_minutes: int = Field(
        default=150, alias="BETBOT_SETTLE_GRACE_MINUTES"
    )
    drawdown_kill_pct: float = Field(
        default=0.20, alias="BETBOT_DRAWDOWN_KILL_PCT"
    )
    drawdown_window_days: int = Field(
        default=7, alias="BETBOT_DRAWDOWN_WINDOW_DAYS"
    )
    drawdown_min_staked_usd: float = Field(
        default=100.0, alias="BETBOT_DRAWDOWN_MIN_STAKED_USD"
    )

    # ---- Live-readiness gate (Phase 5) --------------------------------
    gate_min_bets: int = Field(default=20, alias="BETBOT_GATE_MIN_BETS")
    gate_min_window_days: float = Field(
        default=14.0, alias="BETBOT_GATE_MIN_WINDOW_DAYS"
    )
    gate_min_hit_rate: float = Field(default=0.30, alias="BETBOT_GATE_MIN_HIT_RATE")
    gate_min_roi: float = Field(default=0.0, alias="BETBOT_GATE_MIN_ROI")

    # ---- API / wallet / Telegram (backend + TG bot) ------------------
    api_token: str = Field(default="", alias="TFSM_API_TOKEN")
    wallet_keyfile: Path = Field(
        default=Path("./.secrets/agent_wallet.key"), alias="BETBOT_WALLET_KEYFILE"
    )
    polygon_rpc_url: str = Field(
        default="https://polygon.drpc.org", alias="POLYGON_RPC_URL"
    )
    base_rpc_url: str = Field(
        default="https://mainnet.base.org", alias="BASE_RPC_URL"
    )
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_user_id: int = Field(
        default=0, alias="TELEGRAM_ALLOWED_USER_ID"
    )
    # Multi-user: comma-separated extra Telegram user ids allowed to register.
    # If telegram_open_registration is true, anyone who messages the bot can
    # register (the operator controls access by who they share the bot with).
    telegram_allowed_user_ids: str = Field(
        default="", alias="TELEGRAM_ALLOWED_USER_IDS"
    )
    telegram_open_registration: bool = Field(
        default=False, alias="TELEGRAM_OPEN_REGISTRATION"
    )

    @property
    def allowed_telegram_ids(self) -> set[int]:
        ids: set[int] = set()
        if self.telegram_allowed_user_id:
            ids.add(self.telegram_allowed_user_id)
        for part in self.telegram_allowed_user_ids.split(","):
            part = part.strip()
            if part:
                try:
                    ids.add(int(part))
                except ValueError:
                    pass
        return ids

    @property
    def secrets_dir(self) -> str:
        from pathlib import Path

        return str(Path(self.wallet_keyfile).resolve().parent)

    # ---- Live-trading secrets (Phase 5; only used in live mode) -------
    polymarket_private_key: str = Field(default="", alias="POLYMARKET_PRIVATE_KEY")
    polymarket_funder: str = Field(default="", alias="POLYMARKET_FUNDER")
    limitless_private_key: str = Field(default="", alias="LIMITLESS_PRIVATE_KEY")
    # Limitless API auth (create in the Limitless app: connect the agent wallet,
    # then derive a scoped API token). Without these, live orders can't post.
    limitless_api_key: str = Field(default="", alias="LIMITLESS_API_KEY")
    limitless_api_secret: str = Field(default="", alias="LIMITLESS_API_SECRET")
    # Limitless feeRateBps must fall in the exchange's per-user band (a 0 fee is
    # rejected). Set the value from the Limitless docs/support before live orders.
    limitless_fee_rate_bps: int = Field(default=0, alias="LIMITLESS_FEE_RATE_BPS")
    # Multi-tenant: minimum per-user stake. A user whose wallet balance is below
    # this is skipped for that bet (rather than placing a dust order or failing).
    min_user_stake_usd: float = Field(default=1.0, alias="BETBOT_MIN_USER_STAKE_USD")
    # Max slippage added to the quoted price when sending a market buy.
    order_slippage: float = Field(default=0.02, alias="BETBOT_ORDER_SLIPPAGE")
    # Allow live orders on INTERNATIONAL_COMPETITIONS (World Cup). Default OFF:
    # the Glicko model is weak vs efficient WC markets and Appendix A makes WC
    # paper-only by default. Set true to opt in (still gated by live mode + gate).
    allow_international_live: bool = Field(
        default=False, alias="BETBOT_ALLOW_INTERNATIONAL_LIVE"
    )
    # Bet EVERY international (WC) match that has a market, bypassing the edge
    # filter. This is -EV against efficient markets (see the Qatar 2022 backtest)
    # but is what the operator wants for the World Cup. The kill switch + daily
    # exposure cap still apply as guardrails.
    international_bet_every_match: bool = Field(
        default=False, alias="BETBOT_INTERNATIONAL_BET_EVERY_MATCH"
    )
    # Require the live-readiness gate before placing live orders. Set false to
    # go live without a paper-trading record (NOT recommended — you lose the
    # "earned the right to trade" safety check).
    require_gate: bool = Field(default=True, alias="BETBOT_REQUIRE_GATE")

    # ---- Arbitrage watch (Telegram alerts) ---------------------------
    arb_notify_min_margin: float = Field(
        default=0.01, alias="BETBOT_ARB_NOTIFY_MIN_MARGIN"
    )
    arb_scan_interval_min: int = Field(
        default=10, alias="BETBOT_ARB_SCAN_INTERVAL_MIN"
    )
    arb_scan_limit: int = Field(default=80, alias="BETBOT_ARB_SCAN_LIMIT")

    # ---- Glicko-2 (international / World Cup, Phase 5.5) --------------
    glicko_tau: float = Field(default=0.5, alias="BETBOT_GLICKO_TAU")
    glicko_default_rating: float = Field(default=1500.0, alias="BETBOT_GLICKO_DEFAULT_RATING")
    glicko_default_rd: float = Field(default=200.0, alias="BETBOT_GLICKO_DEFAULT_RD")
    glicko_default_vol: float = Field(default=0.06, alias="BETBOT_GLICKO_DEFAULT_VOL")
    glicko_draw_rho: float = Field(default=0.28, alias="BETBOT_GLICKO_DRAW_RHO")
    glicko_host_home_mu: float = Field(default=0.2, alias="BETBOT_GLICKO_HOST_HOME_MU")
    glicko_results_csv: str = Field(default="", alias="BETBOT_GLICKO_RESULTS_CSV")

    # ---- Ensemble (Klement fundamentals + Dixon-Coles + market) -------
    # Artifacts are regenerable: scripts/fit_dixon_coles.py writes the DC
    # params; scripts/backtest_ensemble.py writes the calibration. Missing
    # files are fine — the engine falls back to pure Glicko.
    dc_params_path: Path = Field(
        default=Path("./data/dc_params.json"), alias="BETBOT_DC_PARAMS_PATH"
    )
    ensemble_calibration_path: Path = Field(
        default=Path("./data/ensemble_calibration.json"),
        alias="BETBOT_ENSEMBLE_CALIBRATION_PATH",
    )
    # Relative log-pool weights. Market-leaning by design: the closing line
    # is the strongest single forecaster, so the models act as a
    # disagreement detector rather than trying to out-predict it.
    ensemble_weight_glicko: float = Field(
        default=1.0, alias="BETBOT_ENSEMBLE_W_GLICKO"
    )
    ensemble_weight_dc: float = Field(default=1.5, alias="BETBOT_ENSEMBLE_W_DC")
    ensemble_weight_market: float = Field(
        default=2.5, alias="BETBOT_ENSEMBLE_W_MARKET"
    )

    # ---- Storage ------------------------------------------------------
    db_path: Path = Field(
        default=Path("./data/betbot.sqlite"), alias="BETBOT_DB_PATH"
    )

    # ---- Logging ------------------------------------------------------
    log_level: str = Field(default="INFO", alias="BETBOT_LOG_LEVEL")

    # ---- Scheduler ----------------------------------------------------
    daemon_cron: str = Field(default="0 8 * * *", alias="BETBOT_DAEMON_CRON")

    # ---- Convenience properties ---------------------------------------
    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Re-read by clearing the cache."""
    return Settings()
