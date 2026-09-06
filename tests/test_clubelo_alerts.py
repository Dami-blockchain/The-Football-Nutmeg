"""Unit tests for the ClubElo stale-snapshot operator alerting.

Covers the cadence state machine (first alert, once-per-day reminder cap,
one-time recovery) and the async glue that routes a decision to the operator
notifier without ever raising into the daemon tick.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from betbot.clubelo_alerts import (
    AlertAction,
    ClubEloAlerter,
    format_recovered_message,
    format_stale_message,
    run_clubelo_alert,
)
from betbot.data.clubelo import SnapshotStatus

DAY = 86400.0
PATH = Path("data/clubelo_latest.csv")


def _stale(age: float = 6.0) -> SnapshotStatus:
    return SnapshotStatus(
        path=PATH,
        exists=True,
        age_days=age,
        stale=True,
        reason=f"stale_{age:.1f}d",
        snapshot_date=date(2026, 8, 31),
        clubs=600,
    )


def _fresh(age: float = 0.0) -> SnapshotStatus:
    return SnapshotStatus(
        path=PATH,
        exists=True,
        age_days=age,
        stale=False,
        reason="fresh",
        snapshot_date=date(2026, 9, 6),
        clubs=600,
    )


# --------------------------------------------------------------------------
# decision cadence
# --------------------------------------------------------------------------
def test_first_stale_triggers_alert():
    a = ClubEloAlerter()
    assert a.decide(_stale(), now=0.0) is AlertAction.STALE


def test_daily_reminder_cap_suppresses_within_24h():
    a = ClubEloAlerter()
    assert a.decide(_stale(), now=0.0) is AlertAction.STALE
    # A few hours later, still stale -> no second page.
    assert a.decide(_stale(), now=6 * 3600.0) is AlertAction.NONE
    assert a.decide(_stale(), now=DAY - 1) is AlertAction.NONE


def test_reminder_fires_after_24h_still_stale():
    a = ClubEloAlerter()
    assert a.decide(_stale(), now=0.0) is AlertAction.STALE
    assert a.decide(_stale(), now=DAY) is AlertAction.STALE  # daily reminder
    assert a.decide(_stale(), now=DAY + 3600.0) is AlertAction.NONE
    assert a.decide(_stale(), now=2 * DAY) is AlertAction.STALE


def test_recovery_sent_once_after_stale():
    a = ClubEloAlerter()
    a.decide(_stale(), now=0.0)
    assert a.decide(_fresh(), now=3600.0) is AlertAction.RECOVERED
    # No repeated recovery spam while it stays fresh.
    assert a.decide(_fresh(), now=7200.0) is AlertAction.NONE


def test_fresh_from_the_start_is_silent():
    a = ClubEloAlerter()
    assert a.decide(_fresh(), now=0.0) is AlertAction.NONE
    assert a.decide(_fresh(), now=DAY) is AlertAction.NONE


def test_new_incident_after_recovery_alerts_again():
    a = ClubEloAlerter()
    a.decide(_stale(), now=0.0)              # incident 1
    a.decide(_fresh(), now=1000.0)           # recovered
    # Goes stale again shortly after -> must alert (not suppressed by the old cap).
    assert a.decide(_stale(), now=2000.0) is AlertAction.STALE


# --------------------------------------------------------------------------
# async glue -> notifier
# --------------------------------------------------------------------------
class _Recorder:
    def __init__(self, ok: bool = True, boom: bool = False):
        self.calls: list[dict] = []
        self._ok = ok
        self._boom = boom

    async def __call__(self, settings, text, *, kind=None, cooldown_seconds=None):
        if self._boom:
            raise RuntimeError("telegram down")
        self.calls.append(
            {"text": text, "kind": kind, "cooldown_seconds": cooldown_seconds}
        )
        return self._ok


@pytest.mark.asyncio
async def test_run_alert_sends_stale_with_expected_kind():
    a = ClubEloAlerter()
    rec = _Recorder()
    action = await run_clubelo_alert(
        object(), _stale(), a, notify=rec, now=0.0,
        wall_now=datetime(2026, 9, 6, 5, 0, tzinfo=timezone.utc),
    )
    assert action is AlertAction.STALE
    assert len(rec.calls) == 1
    assert rec.calls[0]["kind"] == "clubelo_stale"
    # Alerter owns cadence, so notify must not add its own cooldown.
    assert rec.calls[0]["cooldown_seconds"] == 0
    assert "stale" in rec.calls[0]["text"].lower()
    assert "EAT" in rec.calls[0]["text"]


@pytest.mark.asyncio
async def test_run_alert_recovery_kind():
    a = ClubEloAlerter()
    a.decide(_stale(), now=0.0)
    rec = _Recorder()
    action = await run_clubelo_alert(object(), _fresh(), a, notify=rec, now=100.0)
    assert action is AlertAction.RECOVERED
    assert rec.calls[0]["kind"] == "clubelo_recovered"


@pytest.mark.asyncio
async def test_run_alert_none_sends_nothing():
    a = ClubEloAlerter()
    rec = _Recorder()
    action = await run_clubelo_alert(object(), _fresh(), a, notify=rec, now=0.0)
    assert action is AlertAction.NONE
    assert rec.calls == []


@pytest.mark.asyncio
async def test_run_alert_never_raises_on_notifier_failure():
    a = ClubEloAlerter()
    rec = _Recorder(boom=True)
    # Must not propagate the notifier exception into the daemon tick.
    action = await run_clubelo_alert(object(), _stale(), a, notify=rec, now=0.0)
    assert action is AlertAction.STALE


# --------------------------------------------------------------------------
# message formatting
# --------------------------------------------------------------------------
def test_stale_message_has_key_fields():
    wall = datetime(2026, 9, 6, 5, 0, tzinfo=timezone.utc)
    msg = format_stale_message(_stale(6.2), wall_now=wall)
    assert "clubelo_latest.csv" in msg
    assert "2026-08-31" in msg      # snapshot date
    assert "6.2 d" in msg           # age
    assert "EAT" in msg


def test_recovered_message_has_key_fields():
    wall = datetime(2026, 9, 6, 8, 0, tzinfo=timezone.utc)
    msg = format_recovered_message(_fresh(), wall_now=wall)
    assert "recovered" in msg.lower()
    assert "EAT" in msg
