"""Operator alerting for a stale ClubElo snapshot.

``betbot.data.clubelo`` deliberately knows nothing about Telegram: it only makes
the degraded state loud and machine-readable via :func:`snapshot_status`. This
module is the seam that turns that status into an operator message, with cadence
control so a multi-day outage pages **once**, then reminds **at most once a day**
while it stays stale, and sends a **single** recovery note when the feed comes
back.

The decision layer (:meth:`ClubEloAlerter.decide`) does no I/O and owns all
cadence, so it is trivially unit-testable. The async :func:`run_clubelo_alert`
wires a decision to :func:`betbot.notify.notify_operator` (or any injected
sender), formatting the message with EAT timestamps per the standing rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

from betbot.data.clubelo import STALE_AFTER_DAYS, SnapshotStatus
from betbot.logging import get_logger
from betbot.timefmt import eat_datetime

log = get_logger(__name__)

#: Remind at most once per day while the snapshot stays stale.
DEFAULT_REMINDER_INTERVAL_S = 86400.0


class AlertAction(Enum):
    """What :meth:`ClubEloAlerter.decide` decided to do this evaluation."""

    NONE = "none"
    #: First stale alert, or a once-per-day reminder while still stale.
    STALE = "stale"
    #: Feed came back after we had alerted at least once.
    RECOVERED = "recovered"


@dataclass
class ClubEloAlerter:
    """Cadence state machine for stale-snapshot operator alerts.

    Long-lived: one instance per daemon process. :meth:`decide` is pure with
    respect to I/O — given a freshness status and the current clock it returns
    the action to take and mutates only its own small state. It owns the daily
    reminder cap and the one-time recovery, so the caller can pass
    ``cooldown_seconds=0`` to ``notify_operator`` and not double-suppress.
    """

    reminder_interval_s: float = DEFAULT_REMINDER_INTERVAL_S
    _stale_active: bool = False
    _last_alert_ts: float | None = None

    def decide(self, status: SnapshotStatus, *, now: float) -> AlertAction:
        """Return the action for this evaluation and advance internal state.

        ``now`` is a monotonic-style clock (seconds); only differences matter.
        """
        if status.stale:
            due = (
                self._last_alert_ts is None
                or (now - self._last_alert_ts) >= self.reminder_interval_s
            )
            self._stale_active = True
            if due:
                self._last_alert_ts = now
                return AlertAction.STALE
            return AlertAction.NONE
        # Fresh snapshot.
        if self._stale_active:
            self._stale_active = False
            self._last_alert_ts = None
            return AlertAction.RECOVERED
        return AlertAction.NONE


def _age_str(status: SnapshotStatus) -> str:
    return "unknown" if status.age_days is None else f"{status.age_days:.1f} d"


def _snap_str(status: SnapshotStatus) -> str:
    return status.snapshot_date.isoformat() if status.snapshot_date else "unknown"


def format_stale_message(status: SnapshotStatus, *, wall_now: datetime) -> str:
    """Operator message for a stale snapshot (first alert or daily reminder)."""
    return (
        "*⚠️ ClubElo snapshot is stale*\n\n"
        "The cross-league Elo feed the Champions League engine reads has not "
        "refreshed, so every CL prediction is degrading to the naive form "
        "engine until it recovers.\n\n"
        f"- File: `{status.path.name}`\n"
        f"- Snapshot date: {_snap_str(status)}\n"
        f"- Age: {_age_str(status)} (threshold {STALE_AFTER_DAYS} d)\n"
        f"- Reason: `{status.reason}`\n\n"
        "The daily http://api.clubelo.com refresh is failing upstream (their "
        "backend). No action needed unless it persists — I retry through "
        "the day, remind once daily while stale, and tell you when it recovers.\n"
        f"_Checked {eat_datetime(wall_now)}._"
    )


def format_recovered_message(status: SnapshotStatus, *, wall_now: datetime) -> str:
    """Operator message sent once when the feed comes back after being stale."""
    return (
        "*✅ ClubElo snapshot recovered*\n\n"
        "The cross-league Elo feed refreshed successfully; Champions League "
        "predictions are back on ClubElo ratings.\n\n"
        f"- File: `{status.path.name}`\n"
        f"- Snapshot date: {_snap_str(status)}\n"
        f"- Age: {_age_str(status)}\n"
        f"_Recovered {eat_datetime(wall_now)}._"
    )


async def run_clubelo_alert(
    settings,
    status: SnapshotStatus,
    alerter: ClubEloAlerter,
    *,
    notify: Callable[..., Awaitable[bool]] | None = None,
    now: float | None = None,
    wall_now: datetime | None = None,
) -> AlertAction:
    """Evaluate ``status`` against ``alerter`` and, if warranted, page the operator.

    Returns the :class:`AlertAction` taken. Never raises: a broken notifier must
    not crash the daemon tick that called it. ``notify`` defaults to
    :func:`betbot.notify.notify_operator`; ``now`` is the cadence clock and
    ``wall_now`` the display timestamp (both injectable for tests).
    """
    action = alerter.decide(status, now=time.monotonic() if now is None else now)
    if action is AlertAction.NONE:
        return action

    if notify is None:
        from betbot.notify import notify_operator

        notify = notify_operator
    wall = wall_now or datetime.now(timezone.utc)

    try:
        if action is AlertAction.STALE:
            await notify(
                settings,
                format_stale_message(status, wall_now=wall),
                kind="clubelo_stale",
                cooldown_seconds=0,
            )
        elif action is AlertAction.RECOVERED:
            await notify(
                settings,
                format_recovered_message(status, wall_now=wall),
                kind="clubelo_recovered",
                cooldown_seconds=0,
            )
    except Exception as e:  # noqa: BLE001 - alerting must never crash the caller
        log.error("clubelo_alert_send_failed", action=action.value, error=str(e))
    return action
