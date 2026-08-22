"""Smoke coverage for the FastAPI backend's read-only endpoints.

WHY THIS FILE EXISTS: ``/api/health`` returned HTTP 500 in production for
weeks. The predictions-only pivot deleted ``Settings.mode`` but left three
endpoints reading ``settings.mode``, and *the backend had no tests at all*, so
nothing failed. The point of these tests is not the health payload's shape —
it is that every endpoint the operator and the frontend call is exercised at
least once, so an attribute that stops existing fails the suite instead of the
probe.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from betbot.config import Settings, get_settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """App wired to a throwaway sqlite file and no auth token."""
    monkeypatch.setenv("BETBOT_DB_PATH", str(tmp_path / "t.sqlite"))
    monkeypatch.setenv("TFSM_API_TOKEN", "")
    get_settings.cache_clear()
    from backend.tfsm_api.app import create_app

    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_health_returns_200_not_500(client):
    """The regression itself: this asserted nothing before and 500'd live."""
    r = client.get("/api/health")
    assert r.status_code == 200, r.text


def test_health_reports_real_state_not_just_a_literal(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    # DB reachability is PROVEN by a query, not assumed from startup.
    assert body["checks"]["db"]["ok"] is True
    assert body["checks"]["db"]["error"] is None
    # Kill-switch state is visible to the monitor.
    assert body["kill_switch"]["tripped"] is False
    # The booted commit is what makes a stale deploy detectable.
    assert body["build"]["commit"]
    assert body["uptime_seconds"] >= 0


def test_health_reports_paper_because_no_live_order_path_exists(client):
    """Mode must track the BUILD, not a .env string."""
    from betbot.config import LIVE_ORDER_PATH_AVAILABLE

    assert LIVE_ORDER_PATH_AVAILABLE is False
    assert client.get("/api/health").json()["mode"] == "paper"


def test_betbot_mode_in_env_cannot_claim_live_trading(monkeypatch):
    """A leftover BETBOT_MODE=live must not make the API report live."""
    monkeypatch.setenv("BETBOT_MODE", "live")
    get_settings.cache_clear()
    try:
        assert Settings(_env_file=None).mode == "paper"
    finally:
        get_settings.cache_clear()


def test_settings_exposes_mode_for_every_endpoint_that_reads_it():
    """`/api/health`, `/api/status` and `/api/settings` all read settings.mode.

    The live AttributeError was 'Settings' object has no attribute 'mode'.
    """
    assert Settings(_env_file=None).mode in {"paper", "live"}


@pytest.fixture()
def unreachable_db(monkeypatch):
    """Point the process-global session factory at a database that cannot open.

    A GENUINE outage, not a stubbed ``_db_ok``. Stubbing that one function
    faked the symptom while leaving a perfectly healthy test DB underneath, so
    every OTHER database read in the handler still succeeded — which is exactly
    how an unconditional ``is_kill_switch_tripped()`` call sat on line 214
    unnoticed and turned a real outage into a generic HTTP 500. Here every
    query fails, the way it does when the sqlite file is deleted, unmounted or
    permission-changed under a running process.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import betbot.storage.db as db_mod

    # A directory that does not exist: sqlite raises "unable to open database
    # file" on CONNECT, so nothing in the app can reach the DB at all.
    dead = create_engine("sqlite:////nonexistent-dir-4f2a/dead.sqlite", future=True)
    monkeypatch.setattr(db_mod, "_engine", dead)
    monkeypatch.setattr(
        db_mod,
        "_SessionLocal",
        sessionmaker(bind=dead, expire_on_commit=False, future=True),
    )
    return dead


def test_health_degrades_to_503_when_the_db_is_unreachable(client, unreachable_db):
    """A probe that stays 200 through a dead DB is worse than no probe."""
    r = client.get("/api/health")
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["db"]["ok"] is False
    # The diagnostic is the POINT of the 503 payload — without it the operator
    # learns only "something is wrong".
    assert body["checks"]["db"]["error"]


def test_a_dead_db_does_not_turn_the_probe_into_a_500(client, unreachable_db):
    """The regression: reading the kill switch re-entered the dead DB.

    ``is_kill_switch_tripped()`` opens its own session, so building the payload
    raised before the 503 could be returned. The handler answered 500 with a
    generic body, the ``db.error`` diagnostic was lost, and the
    ``health_degraded`` log line never fired.
    """
    r = client.get("/api/health")
    assert r.status_code != 500, r.text
    # Unknown must be reported as UNKNOWN. Claiming ``False`` here would tell a
    # monitor the trading kill switch is confirmed un-tripped on the strength
    # of a read that never happened.
    assert r.json()["kill_switch"]["tripped"] is None


def test_health_logs_the_degradation_when_the_db_is_dead(client, unreachable_db, monkeypatch):
    """`log.error("health_degraded")` is how the outage reaches the logs."""
    import backend.tfsm_api.app as app_mod

    seen: list[tuple] = []
    monkeypatch.setattr(
        app_mod.log, "error", lambda event, **kw: seen.append((event, kw))
    )
    client.get("/api/health")
    assert [e for e, _ in seen] == ["health_degraded"]


@pytest.mark.parametrize(
    "path",
    ["/api/status", "/api/settings", "/api/gate", "/api/kill-switch", "/api/bets",
     "/api/predictions"],
)
def test_read_only_endpoints_do_not_500(client, path):
    """Blanket guard: the pivot broke /api/status and /api/settings too."""
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text}"


def test_betbot_mode_is_no_longer_an_editable_setting(client):
    """Writing a setting the code ignores must fail loudly, not report 200."""
    from backend.tfsm_api.app import EDITABLE_SETTINGS

    assert "BETBOT_MODE" not in EDITABLE_SETTINGS
    r = client.post("/api/settings", json={"key": "BETBOT_MODE", "value": "live"})
    assert r.status_code == 400


def test_every_editable_setting_reports_restart_required(client, monkeypatch):
    """The daemon caches settings at startup, so no knob takes effect live.

    RESTART_REQUIRED was {"BETBOT_MODE"}, which implied the risk controls
    (max bet, exposure cap, drawdown kill) applied immediately. They do not —
    the API clears its OWN settings cache while the daemon keeps the old
    values until restarted.
    """
    import backend.tfsm_api.app as app_mod
    from backend.tfsm_api.app import EDITABLE_SETTINGS, RESTART_REQUIRED

    assert RESTART_REQUIRED == EDITABLE_SETTINGS

    # The real writer edits _REPO_ROOT/".env". Run on the droplet checkout that
    # is the LIVE .env, so this test must never invoke it.
    written: list[tuple] = []
    monkeypatch.setattr(
        app_mod, "_update_env_file", lambda path, k, v: written.append((k, v))
    )
    r = client.post("/api/settings", json={"key": "BETBOT_MAX_BET_USD", "value": "50"})
    assert r.status_code == 200, r.text
    assert r.json()["restart_required"] is True
    assert written == [("BETBOT_MAX_BET_USD", "50")]
