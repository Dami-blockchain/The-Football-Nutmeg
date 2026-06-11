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

    # ---- Market-match sanity guard (matcher hardening) ----------------
    # A 1X2 outcome quoted outside this band is implausible (a 0.014 "edge"
    # almost always means the fixture was paired to the WRONG market — a
    # longshot prop or a different fixture). Reject such matches at the router
    # so neither the favourite-edge path nor the market route logs a phantom
    # edge. Bounds are inclusive.
    min_plausible_price: float = Field(
        default=0.02, alias="BETBOT_MIN_PLAUSIBLE_PRICE"
    )
    max_plausible_price: float = Field(
        default=0.98, alias="BETBOT_MAX_PLAUSIBLE_PRICE"
    )

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
    # Open to the public by default: anyone who messages the bot can register
    # and gets their own isolated wallet. Funds are never pooled, and live
    # trading stays behind the mode=paper default + gate, so opening
    # registration does not weaken any money-moving safeguard.
    telegram_open_registration: bool = Field(
        default=True, alias="TELEGRAM_OPEN_REGISTRATION"
    )

    # ---- LLM assistant (free-text Telegram Q&A via Anthropic) ---------
    # No SDK dependency by design — llm_agent.py calls the Messages API with
    # httpx directly. Empty key = graceful fallback (commands still work).
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(
        default="claude-haiku-4-5-20251001", alias="BETBOT_LLM_MODEL"
    )
    llm_max_tokens: int = Field(default=500, alias="BETBOT_LLM_MAX_TOKENS")
    # Per-user daily question cap: the bot is public, so this bounds API spend
    # per Telegram user per UTC day.
    llm_daily_limit_per_user: int = Field(
        default=20, alias="BETBOT_LLM_DAILY_LIMIT_PER_USER"
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

    # ---- Deposit pipeline (CCTP bridging; betbot/bridge.py) -----------
    # Master gate for EVERY on-chain action the pipeline takes (gas top-ups,
    # CCTP burns/mints, venue approvals). False = detect + log only.
    auto_bridge: bool = Field(default=True, alias="BETBOT_AUTO_BRIDGE")
    # Deposits below this are ignored (logged): bridging dust costs more in
    # gas + operational noise than the deposit is worth.
    min_deposit_usdc: float = Field(default=10.0, alias="BETBOT_MIN_DEPOSIT_USDC")
    # Fraction of each bridged deposit reserved for Base / Limitless. Default
    # 0.0: Polymarket (Polygon) is the only live venue today — Limitless live
    # orders are parked — so 100% of a deposit goes to Polygon.
    bridge_split_base: float = Field(default=0.0, alias="BETBOT_BRIDGE_SPLIT_BASE")
    # Native-gas top-ups sent from the agent wallet: user wallets are created
    # empty and can't pay for their own approvals/burns. Small by design —
    # enough for a handful of simple transactions, not a balance to manage.
    gas_topup_pol: float = Field(default=0.5, alias="BETBOT_GAS_TOPUP_POL")
    gas_topup_eth: float = Field(default=0.0005, alias="BETBOT_GAS_TOPUP_ETH")
    # Abuse guard for the public bot: max agent-funded gas top-ups per user
    # wallet per UTC day (persisted in the gas_topups table, so restarts
    # don't reset it). A normal deposit needs at most 2-3 top-ups (source
    # chain + each trading chain); 4 leaves headroom for a retry. 0 = no cap.
    gas_topup_daily_cap: int = Field(
        default=4, alias="BETBOT_GAS_TOPUP_DAILY_CAP"
    )
    deposit_scan_minutes: int = Field(
        default=10, alias="BETBOT_DEPOSIT_SCAN_MINUTES"
    )
    # Extra source-chain RPCs (Polygon/Base RPCs are defined above).
    ethereum_rpc_url: str = Field(
        default="https://eth.drpc.org", alias="ETHEREUM_RPC_URL"
    )
    arbitrum_rpc_url: str = Field(
        default="https://arbitrum.drpc.org", alias="ARBITRUM_RPC_URL"
    )
    # Circle's attestation service for CCTP (poll after depositForBurn).
    iris_api_url: str = Field(
        default="https://iris-api.circle.com", alias="BETBOT_IRIS_API_URL"
    )
    # Limitless CTF Exchange address on Base (market.venue.exchange). Needed
    # only when bridge_split_base > 0; empty = skip the Limitless approval
    # with a logged warning (see scripts/limitless_approve.py).
    limitless_exchange: str = Field(default="", alias="LIMITLESS_EXCHANGE")

    # ---- Arbitrage watch (Telegram alerts) ---------------------------
    arb_notify_min_margin: float = Field(
        default=0.01, alias="BETBOT_ARB_NOTIFY_MIN_MARGIN"
    )
    arb_scan_interval_min: int = Field(
        default=10, alias="BETBOT_ARB_SCAN_INTERVAL_MIN"
    )
    arb_scan_limit: int = Field(default=80, alias="BETBOT_ARB_SCAN_LIMIT")

    # ---- Arbitrage EXECUTION (heavily gated; OFF by default) ----------
    # Master switch. Even when true, execution also requires mode=live, a clear
    # kill switch, margin >= arb_execute_min_margin, and funded balances on
    # BOTH venues. The executor self-gates on every opportunity.
    arb_execute: bool = Field(default=False, alias="BETBOT_ARB_EXECUTE")
    # Execution threshold — deliberately stricter than the 0.01 alert threshold.
    arb_execute_min_margin: float = Field(
        default=0.02, alias="BETBOT_ARB_EXECUTE_MIN_MARGIN"
    )
    # Total stake per opportunity, summed across all legs.
    arb_max_stake_usd: float = Field(default=25.0, alias="BETBOT_ARB_MAX_STAKE_USD")
    # Per-day cap summed over arb_executions rows that (may have) moved money.
    arb_daily_cap_usd: float = Field(default=100.0, alias="BETBOT_ARB_DAILY_CAP_USD")

    # ---- Daily Telegram jobs (Nairobi wall-clock schedule) ------------
    # The cron triggers pin timezone="Africa/Nairobi" (NOT a UTC offset), so
    # "09:00" and "exactly 9pm" stay true to the operator's wall clock even
    # if the zone's rules ever change.
    arb_digest_enabled: bool = Field(
        default=True, alias="BETBOT_ARB_DIGEST_ENABLED"
    )
    arb_digest_hour: int = Field(default=9, alias="BETBOT_ARB_DIGEST_HOUR")
    daily_report_enabled: bool = Field(
        default=True, alias="BETBOT_DAILY_REPORT_ENABLED"
    )
    daily_report_hour: int = Field(default=21, alias="BETBOT_DAILY_REPORT_HOUR")

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
    # Qatar-2022 walk-forward: DC monotonically improves RPS on 1k+ pre-WC
    # competitive matches, but on the 64 WC-finals matches the ensemble was
    # NOT distinguishable from pure Glicko (+0.021 ± 0.024 RPS). Default is
    # therefore moderate, not the in-sample optimum (3.0). See
    # scripts/backtest_ensemble.py.
    ensemble_weight_dc: float = Field(default=1.0, alias="BETBOT_ENSEMBLE_W_DC")
    ensemble_weight_market: float = Field(
        default=2.5, alias="BETBOT_ENSEMBLE_W_MARKET"
    )
    # Online model selection (Hedge): dual-log pure-Glicko vs ensemble per
    # WC fixture, score both at settlement, and weight the live prediction
    # toward whichever is winning this tournament. Eta = learning rate
    # (0 = ignore evidence; higher = converge faster).
    model_select_enabled: bool = Field(default=True, alias="BETBOT_MODEL_SELECT")
    model_select_eta: float = Field(default=2.0, alias="BETBOT_MODEL_SELECT_ETA")

    # ---- Tournament simulator (outright/futures, scripts/simulate_wc.py)
    # How random extra-time/penalties are when a knockout match draws:
    # 1.0 = pure coin flip, 0.0 = fully decided by relative DC strength.
    sim_penalty_randomness: float = Field(
        default=0.5, alias="BETBOT_SIM_PENALTY_RANDOMNESS"
    )
    sim_runs: int = Field(default=20_000, alias="BETBOT_SIM_RUNS")

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
