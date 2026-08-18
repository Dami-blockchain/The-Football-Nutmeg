"""Centralised configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Obvious ``.env`` placeholder values that must be treated as "unset".
_PLACEHOLDER_KEYS: frozenset[str] = frozenset({"YOURKEY", "YOUR_KEY", "CHANGEME", ""})


def _first_real_env_value(env_var: str, env_file: str = ".env") -> str:
    """First non-placeholder value for ``env_var`` in ``env_file`` ('' if none).

    python-dotenv (used by pydantic-settings) resolves a DUPLICATE key to the
    LAST occurrence. The production ``.env`` carries a real HIGHLIGHTLY_API_KEY
    followed by a leftover ``=YOURKEY`` placeholder line, so the loaded value is
    the placeholder. Rather than edit ``.env`` (off-limits), we scan for the
    FIRST real value. Missing file / no real value -> ''.
    """
    try:
        lines = Path(env_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    prefix = f"{env_var}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith(prefix):
            continue
        value = stripped[len(prefix):].strip().strip("'\"")
        if value and value.upper() not in _PLACEHOLDER_KEYS:
            return value
    return ""

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

    # ---- Highlightly Soccer FREE tier (confirmed lineups, CURRENT season) ---
    # api-football's FREE tier blocks the current season, so the confirmed-XI
    # source is Highlightly (serves the current season). Behind Cloudflare — the
    # client always sends a browser User-Agent + the key (x-rapidapi-key).
    highlightly_api_key: str = Field(default="", alias="HIGHLIGHTLY_API_KEY")
    highlightly_base_url: str = Field(
        default="https://soccer.highlightly.net",
        alias="HIGHLIGHTLY_BASE_URL",
    )

    # ---- Free pre-match odds anchoring (flag-gated, DEFAULT OFF) -------
    # anchor_to_market previously fired ONLY when Polymarket happened to list
    # the fixture, so most big-5 matches shipped raw, unanchored model output.
    # This flag turns on a free pre-match odds feed (football-data.co.uk —
    # no key, no signup, no quota, no cost) so EVERY scored club fixture can
    # be anchored to a de-vigged bookmaker line.
    #
    # HONESTY: anchoring moves the model TOWARD market-level accuracy. It
    # cannot beat the market and it is not an edge. Default OFF until the
    # pre-registered live gate passes (see RONALDO_PLAN.md).
    odds_anchor_enabled: bool = Field(default=False, alias="BETBOT_ODDS_ANCHOR")
    # Weight given to the de-vigged odds line, relative to the engine's own
    # summed component weights, in the logit-space anchor.
    odds_anchor_market_weight: float = Field(
        default=1.0, alias="BETBOT_ODDS_ANCHOR_W"
    )
    # Shared-cache TTL: one HTTP GET covers every division's whole card, so
    # 6h means a 20-fixture Saturday costs ONE request, not twenty.
    odds_cache_ttl_seconds: float = Field(
        default=21600.0, alias="BETBOT_ODDS_TTL_SEC"
    )
    odds_min_request_interval_seconds: float = Field(
        default=60.0, alias="BETBOT_ODDS_MIN_INTERVAL_SEC"
    )
    odds_http_timeout_seconds: float = Field(
        default=30.0, alias="BETBOT_ODDS_HTTP_TIMEOUT_SEC"
    )
    # Feed dates are local match dates; kickoff dates are UTC. A small slack
    # absorbs the offset without letting a fixture match the reverse leg.
    odds_max_date_slack_days: int = Field(
        default=3, alias="BETBOT_ODDS_DATE_SLACK_DAYS"
    )
    odds_team_alias_path: Path = Field(
        default=Path("./config/odds_team_aliases.yaml"),
        alias="BETBOT_ODDS_ALIAS_PATH",
    )

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
    # Sample-size shrinkage for the naive engine's per-game form. Each side's
    # per-game score is pulled toward the neutral prior (0.0) with data weight
    # n / (n + form_shrinkage_k): n=1 => 20% data, n=5 => 56%, large n => ~raw.
    # Bigger K = more conservative at season start. K=0 disables shrinkage.
    form_shrinkage_k: float = Field(default=4.0, alias="BETBOT_FORM_SHRINKAGE_K")
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

    # ---- Interactive chat assistant (free-text Telegram, via Groq) ----
    # The chat runs on the FREE Groq API (OpenAI-compatible). No SDK by design —
    # llm_agent.py calls /openai/v1/chat/completions with httpx directly. Empty
    # key = graceful fallback (commands still work). The legacy anthropic_api_key
    # field is retained (harmless) but no longer drives the chat path.
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="openai/gpt-oss-120b", alias="BETBOT_GROQ_MODEL"
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1", alias="BETBOT_GROQ_BASE_URL"
    )
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(
        default="claude-haiku-4-5-20251001", alias="BETBOT_LLM_MODEL"
    )
    llm_max_tokens: int = Field(default=1024, alias="BETBOT_LLM_MAX_TOKENS")
    # Per-user daily question cap: the bot is public, so this bounds API spend
    # per Telegram user per UTC day.
    llm_daily_limit_per_user: int = Field(
        default=20, alias="BETBOT_LLM_DAILY_LIMIT_PER_USER"
    )

    @model_validator(mode="after")
    def _repair_placeholder_highlightly_key(self) -> "Settings":
        """Recover the real HIGHLIGHTLY key when a duplicate ``.env`` line wins.

        See :func:`_first_real_env_value`: a leftover ``HIGHLIGHTLY_API_KEY=YOURKEY``
        placeholder line shadows the real key under python-dotenv's last-wins rule.
        If the loaded value is empty or a known placeholder, fall back to the FIRST
        real value in the env file so the confirmed-lineup path works live.
        """
        current = (self.highlightly_api_key or "").strip()
        if not current or current.upper() in _PLACEHOLDER_KEYS:
            env_file = (self.model_config.get("env_file") or ".env")
            recovered = _first_real_env_value("HIGHLIGHTLY_API_KEY", str(env_file))
            if recovered:
                object.__setattr__(self, "highlightly_api_key", recovered)
        return self

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
    # per-fixture kickoff reminders fire before KO (see the two-alert model).
    matchday_alert_hour: int = Field(
        default=8, alias="BETBOT_MATCHDAY_ALERT_HOUR"
    )

    # ---- TWO-alert pre-match model ------------------------------------
    # Highlightly's FREE tier posts confirmed XIs only ~6-15 min before KO
    # (verified: a La Liga XI absent at KO-17 was present at KO-6), NOT the
    # official ~60 min. So a single ~55-min alert never carries the lineup.
    # Instead we fire TWO one-off alerts per fixture:
    #   (1) EARLY model-prediction alert — lead time, XI not yet posted;
    #   (2) LATE confirmed-XI + lineup-adjusted alert — once the XI drops.
    #
    # EARLY (model) lead: per-competition. Confirmed XIs post ~1h before KO;
    # PL posts earliest so we fire at KO-70, every other competition at KO-55.
    # The morning heads-up quotes the SAME early lead so its stated "prediction
    # at HH:MM" == the actual early firing time.
    pl_lineup_alert_lead_minutes: int = Field(
        default=70, alias="BETBOT_PL_LINEUP_ALERT_LEAD_MIN"
    )
    lineup_alert_lead_minutes_default: int = Field(
        default=55, alias="BETBOT_LINEUP_ALERT_LEAD_MIN"
    )
    # LATE (lineup-confirm) lead: same for every league, KO-minus this many
    # minutes. 10 is a STARTING value — Highlightly free posts the XI ~6-15 min
    # pre-KO, so this may need tuning as more matches are observed. Kept easily
    # tunable via BETBOT_LINEUP_CONFIRM_LEAD_MIN.
    lineup_confirm_lead_minutes_value: int = Field(
        default=10, alias="BETBOT_LINEUP_CONFIRM_LEAD_MIN"
    )

    def early_alert_lead_minutes(self, competition_code: str | None) -> int:
        """Minutes before kickoff to fire the EARLY (model-prediction) alert.

        Premier League (``PL``) -> 70; every other competition -> 55. Used by
        BOTH the morning heads-up (to state "prediction at HH:MM") and the
        scheduler (to fire the early alert), so the two can never drift apart.
        """
        if (competition_code or "").upper() == "PL":
            return self.pl_lineup_alert_lead_minutes
        return self.lineup_alert_lead_minutes_default

    # Back-compat alias: existing callers/tests still reference this name for
    # the early (model) lead. Kept pointing at early_alert_lead_minutes so the
    # two never diverge.
    def lineup_alert_lead_minutes(self, competition_code: str | None) -> int:
        """Alias for :meth:`early_alert_lead_minutes` (the EARLY model lead)."""
        return self.early_alert_lead_minutes(competition_code)

    def lineup_confirm_lead_minutes(self) -> int:
        """Minutes before kickoff to fire the LATE confirmed-XI alert.

        Same for every league. Default 10 (tunable via
        BETBOT_LINEUP_CONFIRM_LEAD_MIN) — Highlightly free posts the XI ~6-15
        min pre-KO, so this may need tuning as more matches are observed.
        """
        return self.lineup_confirm_lead_minutes_value

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
    # Form component disabled — the live FormService scale degrades the
    # log-pooled draw; re-enable only after a proper per-game rescale + club
    # backtest re-validates it. (The naive engine's own per-game fix, used for
    # the unrated-team fallback, is separate and still active.)
    club_weight_form: float = Field(default=0.0, alias="BETBOT_CLUB_W_FORM")
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

    # ---- Confidence filter on the BET / NO BET call (flag-gated, OFF) ---
    # A PRE-REGISTERED selection rule, not a model change: it never alters a
    # probability, it only decides whether the argmax is put forward as
    # a BET or falls back to the standing NO BET default. See
    # betbot/strategy/confidence.py. Default OFF, same discipline as the
    # dispersion/MOV challengers — turn on only after the live gate below.
    #
    # Measured on data/club_results.csv (n=10,734, Pinnacle CLOSING odds — so
    # optimistic vs the T-24h prices we would have live): favourite hit rate
    # p>=0.55 -> 68.1% (38% of fixtures), p>=0.60 -> 72.2% (27%), p>=0.65 ->
    # 75.0% (19%).
    #
    # HONESTY: that hit rate is an ACCURACY KPI. It is NOT edge and NOT +EV —
    # backing favourites at a fair market price is ~0 EV by construction. It
    # must never be presented as beating the market.
    #
    # Applies to whatever FINAL blended probability the caller passes, so it
    # keeps working unchanged once market anchoring lands on all fixtures.
    club_confidence_filter: bool = Field(
        default=False, alias="BETBOT_CONFIDENCE_FILTER"
    )
    # Minimum favourite probability for a BET call.
    club_confidence_threshold: float = Field(
        default=0.60, alias="BETBOT_CONFIDENCE_THRESHOLD"
    )
    # Force NO BET when p_draw is within this of the favourite. Draws are ~25%
    # of outcomes and effectively unpickable — this drops the worst-calibrated
    # call category rather than trying to price it.
    club_confidence_draw_margin: float = Field(
        default=0.05, alias="BETBOT_CONFIDENCE_DRAW_MARGIN"
    )
    # Accuracy-ledger epoch: outcomes settled BEFORE this date are excluded
    # from every accuracy read. Predictions made before 2026-08-17 are poisoned
    # by the degenerate 0/0/100-AWAY rating bug, so quoting them to a user
    # would be dishonest. Set empty to disable the cutoff.
    accuracy_ledger_epoch: str = Field(
        default="2026-08-17", alias="BETBOT_ACCURACY_LEDGER_EPOCH"
    )

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
