"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from betbot.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Default Settings instance with deterministic knobs for tests."""
    return Settings(
        BETBOT_MODE="paper",
        FOOTBALL_DATA_API_KEY="fake-test-key",
        BETBOT_FIXED_STAKE_USD=10,
        BETBOT_MAX_BET_USD=50,
        BETBOT_DAILY_EXPOSURE_CAP_USD=200,
        BETBOT_EDGE_THRESHOLD=0.05,
        BETBOT_HOME_ADVANTAGE=0.3,
        BETBOT_DRAW_SCORE=2.4,
        BETBOT_SOFTMAX_TEMP=1.0,
        BETBOT_OPP_STRENGTH_WEIGHT=0.5,
    )
