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
