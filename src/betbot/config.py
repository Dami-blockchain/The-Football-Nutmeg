"""Centralised configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Top-5 European leagues + UCL. Codes are football-data.org competition IDs.
LEAGUE_CODES: tuple[str, ...] = ("PL", "PD", "BL1", "SA", "FL1", "CL")


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

    # ---- Football-data.org --------------------------------------------
    football_data_api_key: str = Field(default="", alias="FOOTBALL_DATA_API_KEY")
    football_data_base_url: str = Field(
        default="https://api.football-data.org/v4",
        alias="FOOTBALL_DATA_BASE_URL",
    )
    football_data_rate_limit_per_min: int = Field(
        default=10, alias="FOOTBALL_DATA_RATE_LIMIT_PER_MIN"
    )

    # ---- api-football (api-sports.io) FREE tier -----------------------
    # FREE-tier resource for confirmed starting lineups, injuries and
    # season player-minutes (R4a). 100 requests/day, ~10/min. Season is the
    # START year (2025/26 -> 2025). Key already lives in .env.
    api_football_key: str = Field(default="", alias="API_FOOTBALL_KEY")
    api_football_base_url: str = Field(
        default="https://v3.football.api-sports.io",
        alias="API_FOOTBALL_BASE_URL",
    )
    api_football_rate_limit_per_min: int = Field(
        default=10, alias="API_FOOTBALL_RATE_LIMIT_PER_MIN"
    )
    api_football_season: int = Field(default=2026, alias="BETBOT_AF_SEASON")

    # ---- Lineup-adjusted scoring (R4a) --------------------------------
    # Max Glicko-point penalty when a team's entire expected first XI is
    # absent from the confirmed starting XI. Scales linearly with the
    # minutes-weighted share of missing regulars.
    lineup_max_penalty: float = Field(
        default=120.0, alias="BETBOT_LINEUP_MAX_PENALTY"
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
    # Open to the public by default: anyone who messages the bot can register.
    # This is a predictions-only service — the bot places no orders and moves
    # no funds — so opening registration exposes no money-moving surface.
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

    # ---- Tipster Telegram alerts (Nairobi wall-clock schedule) --------
    # The cron triggers pin timezone="Africa/Nairobi" (NOT a UTC offset), so
    # the alert hour stays true to the operator's wall clock even if the zone's
    # rules ever change. The matchday-morning alert fires at this hour; the
    # per-fixture kickoff reminder fires kickoff_alert_lead_minutes before KO.
    matchday_alert_hour: int = Field(
        default=8, alias="BETBOT_MATCHDAY_ALERT_HOUR"
    )
    # Per-competition lead for the pre-match lineup-adjusted alert. Confirmed
    # XIs post ~1h before KO; PL posts earliest (~75m) so we fire at KO-70,
    # every other competition at KO-55. The morning heads-up quotes the SAME
    # lead so its stated "prediction at HH:MM" == the actual firing time.
    pl_lineup_alert_lead_minutes: int = Field(
        default=70, alias="BETBOT_PL_LINEUP_ALERT_LEAD_MIN"
    )
    lineup_alert_lead_minutes_default: int = Field(
        default=55, alias="BETBOT_LINEUP_ALERT_LEAD_MIN"
    )

    def lineup_alert_lead_minutes(self, competition_code: str | None) -> int:
        """Minutes before kickoff to fire the pre-match lineup alert.

        Premier League (``PL``) -> 70; every other competition -> 55. Used by
        BOTH the morning heads-up (to state "prediction at HH:MM") and the
        scheduler (to fire the alert), so the two can never drift apart.
        """
        if (competition_code or "").upper() == "PL":
            return self.pl_lineup_alert_lead_minutes
        return self.lineup_alert_lead_minutes_default

    # ---- Glicko-2 defaults (shared by the club rating machinery) ------
    glicko_tau: float = Field(default=0.5, alias="BETBOT_GLICKO_TAU")
    glicko_default_rating: float = Field(default=1500.0, alias="BETBOT_GLICKO_DEFAULT_RATING")
    glicko_default_rd: float = Field(default=200.0, alias="BETBOT_GLICKO_DEFAULT_RD")
    glicko_default_vol: float = Field(default=0.06, alias="BETBOT_GLICKO_DEFAULT_VOL")

    # ---- Club ensemble engine (domestic leagues: PL/PD/BL1/SA/FL1) -----
    # The same Glicko+Dixon-Coles machinery the WC engine uses, wired to club
    # fixtures. Ratings/params are seeded by scripts/seed_glicko_club.py +
    # scripts/fit_dixon_coles_club.py. Disable to fall back to the naive
    # form-only StrategyEngine for clubs. Scoped to domestic leagues only —
    # cross-league ratings (CL) aren't calibrated, so CL keeps its old path.
    club_ensemble_enabled: bool = Field(default=True, alias="BETBOT_CLUB_ENSEMBLE")
    dc_params_club_path: Path = Field(
        default=Path("./data/dc_params_club.json"), alias="BETBOT_DC_PARAMS_CLUB_PATH"
    )
    club_name_map_path: Path = Field(
        default=Path("./data/club_name_map.json"), alias="BETBOT_CLUB_NAME_MAP_PATH"
    )
    ensemble_calibration_club_path: Path = Field(
        default=Path("./data/ensemble_calibration_club.json"),
        alias="BETBOT_CLUB_CALIBRATION_PATH",
    )
    # Clubs have a real, always-on home advantage (unlike neutral-venue WC).
    # home_mu is the Glicko logit boost for the home side; draw_rho targets the
    # ~25% domestic draw rate.
    glicko_club_home_mu: float = Field(default=0.30, alias="BETBOT_GLICKO_CLUB_HOME_MU")
    glicko_club_draw_rho: float = Field(default=0.28, alias="BETBOT_GLICKO_CLUB_DRAW_RHO")
    # Log-pool weights for the club ensemble components + market anchoring.
    club_weight_glicko: float = Field(default=1.0, alias="BETBOT_CLUB_W_GLICKO")
    club_weight_dc: float = Field(default=1.0, alias="BETBOT_CLUB_W_DC")
    club_weight_form: float = Field(default=0.5, alias="BETBOT_CLUB_W_FORM")
    club_weight_market: float = Field(default=1.0, alias="BETBOT_CLUB_W_MARKET")

    # ---- Cross-league Elo engine (Champions League only, R2) -----------
    # The club Glicko/DC ratings are calibrated WITHIN a domestic league, so
    # they can't price a CL tie that mixes leagues. ClubElo is a single
    # Europe-wide Elo scale, so elo_home - elo_away is comparable across
    # leagues — that is what unlocks CL predictions. Disable to fall back to
    # the naive form engine for the CL. HA/rho are the values tuned by
    # scripts/backtest_cl.py on seasons 2023+2024 (walk-forward, gate CI>0:
    # +0.056 RPS/match vs naive on the held-out 2025 season). The DC blend
    # beat pure Elo on train, so cl_weight_dc ships non-zero.
    cl_elo_enabled: bool = Field(default=True, alias="BETBOT_CL_ELO")
    clubelo_latest_path: Path = Field(
        default=Path("./data/clubelo_latest.csv"), alias="BETBOT_CLUBELO_LATEST_PATH"
    )
    cl_elo_home_adv: float = Field(default=65.0, alias="BETBOT_CL_ELO_HOME_ADV")
    cl_elo_draw_rho: float = Field(default=0.26, alias="BETBOT_CL_ELO_DRAW_RHO")
    # Log-pool weights for the CL Elo ensemble components + market anchoring.
    cl_weight_elo: float = Field(default=1.0, alias="BETBOT_CL_W_ELO")
    cl_weight_dc: float = Field(default=1.0, alias="BETBOT_CL_W_DC")
    cl_weight_market: float = Field(default=1.0, alias="BETBOT_CL_W_MARKET")

    # ---- Storage ------------------------------------------------------
    db_path: Path = Field(
        default=Path("./data/betbot.sqlite"), alias="BETBOT_DB_PATH"
    )

    # ---- Logging ------------------------------------------------------
    log_level: str = Field(default="INFO", alias="BETBOT_LOG_LEVEL")

    # ---- Scheduler ----------------------------------------------------
    daemon_cron: str = Field(default="0 8 * * *", alias="BETBOT_DAEMON_CRON")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Re-read by clearing the cache."""
    return Settings()
