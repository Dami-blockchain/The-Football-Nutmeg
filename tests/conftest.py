"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from betbot.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Default Settings instance with deterministic knobs for tests.

    ``_env_file=None`` makes the fixture hermetic: it must NOT inherit the
    deployment ``.env`` (on the droplet that carries live-trading flags like
    BETBOT_ALLOW_INTERNATIONAL_LIVE=true), or tests asserting default
    behaviour — e.g. the WC live-order guard — would flip with operator config.
    """
    return Settings(
        _env_file=None,
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
        # Point ensemble artifacts at nonexistent paths so tests stay
        # deterministic regardless of what's in the working tree's data/.
        BETBOT_DC_PARAMS_PATH="./tests/_no_such_dc_params.json",
        BETBOT_ENSEMBLE_CALIBRATION_PATH="./tests/_no_such_calibration.json",
        BETBOT_GLICKO_MOV_PATH="./tests/_no_such_mov.json",
        # Pin live-trading flags to their safe defaults EXPLICITLY (init kwargs
        # outrank both .env and exported env vars), so a deployment box with
        # live config set can't flip tests that assert default-OFF behaviour.
        BETBOT_ALLOW_INTERNATIONAL_LIVE="false",
        BETBOT_INTERNATIONAL_BET_EVERY_MATCH="false",
        BETBOT_REQUIRE_GATE="true",
        BETBOT_ARB_EXECUTE="false",
    )


@pytest.fixture(autouse=True)
def _no_ledger_epoch(monkeypatch):
    """Disable the accuracy-ledger epoch cutoff for the suite.

    Production defaults BETBOT_ACCURACY_LEDGER_EPOCH=2026-08-17 so the
    poisoned pre-fix 0/0/100-AWAY outcomes never reach a user-facing accuracy
    figure. Several settlement tests pin a frozen "now" in the past, and a
    wall-clock-anchored cutoff in the code under test would silently filter
    their fixtures out — the same staleness trap that bit the kill-switch
    drawdown test. So the cutoff is off by default here and switched ON
    explicitly by the tests that actually exercise it
    (tests/test_confidence_ledger.py).
    """
    from betbot.config import get_settings

    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_season_scope(monkeypatch):
    """Disable the club-season scope on the accuracy ledger for the suite.

    Same trap, same reason as :func:`_no_ledger_epoch`. Production defaults
    BETBOT_SEASON_START=2026-08-01 so /record covers this season's club
    football only; most settlement tests pin a frozen "now" in June 2026, and
    a wall-clock-anchored season boundary would silently filter every one of
    their fixtures out of the ledger they are asserting on. Off by default
    here, switched ON explicitly by the tests that exercise it
    (tests/test_season_record.py).

    Note this disables only the DATE half of the scope. The club-competition
    allowlist is not date-dependent and stays on everywhere, so a test that
    scores a "WC" fixture is excluded from the ledger in the suite exactly as
    it would be in production.
    """
    from betbot.config import get_settings

    monkeypatch.setenv("BETBOT_SEASON_START", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_operator_notify_cooldowns():
    """Operator-notification cooldowns are process-global state.

    Without this, a test that sends a notification of some kind would silently
    suppress the next test that sends the same kind — and the second test would
    fail for a reason that has nothing to do with what it is asserting.
    """
    from betbot.notify import reset_operator_notify_cooldowns

    reset_operator_notify_cooldowns()
    yield
    reset_operator_notify_cooldowns()
