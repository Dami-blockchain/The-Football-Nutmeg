"""Operator alerting for a stale ClubElo snapshot.

``betbot.data.clubelo`` deliberately knows nothing about Telegram: it only makes
the degraded state loud and machine-readable via :func:`snapshot_status`. This
module is the seam that turns that status into an operator message, with cadence
control so a multi-day outage pages **once**, then reminds **at most once a day**
while it stays stale, and sends a **single** recovery note when the feed comes
back.

The decision layer (:meth:`ClubEloAlerter.decide`) does no network I/O and owns
all cadence, so it is trivially unit-testable. The async :func:`run_clubelo_alert`
wires a decision to :func:`betbot.notify.notify_operator` (or any injected
sender), formatting the message with EAT timestamps per the standing rule.

**Persistence.** The daemon runs under ``Restart=always``, so alerter state
cannot live in process memory alone: a restart mid-outage would reset the
cadence (re-firing a fresh "first alert", degrading the 1/day cap) and — worse —
drop the fact that we were stale, so the one-time recovery note would never be
sent when the feed came back. State is therefore persisted to a small JSON
sidecar next to ``clubelo_latest.csv`` and reloaded when the alerter is
constructed, and the cadence clock is wall-clock (:func:`time.time`) so a
persisted timestamp stays meaningful across processes (only differences are
used).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
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
    respect to *network* I/O — given a freshness status and the current wall
    clock it returns the action to take, mutates its own small state, and
    persists that state to :attr:`state_path` (when set) so it survives the
    ``Restart=always`` daemon bouncing mid-outage. It owns the daily reminder
    cap and the one-time recovery, so the caller can pass ``cooldown_seconds=0``
    to ``notify_operator`` and not double-suppress.

    ``state_path`` is a JSON sidecar; a missing or corrupt one is treated as a
    clean start (never raises), and a failed write logs but does not raise —
    losing persistence degrades to the old in-memory behaviour, it never crashes
    the tick.
    """

    reminder_interval_s: float = DEFAULT_REMINDER_INTERVAL_S
    state_path: Path | None = None
    _stale_active: bool = False
    _last_alert_ts: float | None = None

    def __post_init__(self) -> None:
        if self.state_path is not None:
            self.state_path = Path(self.state_path)
            self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as e:  # corrupt/unreadable
            log.warning("clubelo_alert_state_unreadable", error=str(e))
            return
        try:
            self._stale_active = bool(data.get("stale_active", False))
            ts = data.get("last_alert_wall_ts")
            self._last_alert_ts = None if ts is None else float(ts)
        except (AttributeError, TypeError, ValueError) as e:
            log.warning("clubelo_alert_state_malformed", error=str(e))
            self._stale_active = False
            self._last_alert_ts = None

    def _save(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(f".{self.state_path.name}.tmp{os.getpid()}")
            tmp.write_text(
                json.dumps(
                    {
                        "stale_active": self._stale_active,
                        "last_alert_wall_ts": self._last_alert_ts,
                    }
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self.state_path)
        except OSError as e:  # never crash the tick over a sidecar write
            log.warning("clubelo_alert_state_save_failed", error=str(e))

    # -- decision -----------------------------------------------------------
    def decide(self, status: SnapshotStatus, *, now: float) -> AlertAction:
        """Return the action for this evaluation, advance state, and persist it.

        ``now`` is a wall-clock timestamp (:func:`time.time`); only differences
        matter, which is what makes the persisted value valid across a restart.
        """
        action = self._decide(status, now)
        self._save()
        return action

    def _decide(self, status: SnapshotStatus, now: float) -> AlertAction:
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
    """Operator message for a stale snapshot (first alert or daily reminder).

    States what actually happens, which depends on the snapshot:

    * present but old — CL is still priced, just off N-day-old ratings (no hard
      cutoff; a month-old API-scale file is a smaller error than a fresh
      mis-scaled one, so we do not drop it);
    * missing/unparseable — the club list is empty, so every CL fixture falls
      back to the naive form engine.
    """
    present = status.exists and status.clubs > 0 and status.snapshot_date is not None
    if present:
        # Magnitude from the CL walk-forward on a ~6-month-stale snapshot
        # (2026-09-08). Softened to a range in the copy because the exact
        # figure is not reproducible from anything checked into the repo.
        impact = (
            f"CL is still priced, just off {_age_str(status)}-old ratings. The "
            "measured cost is small — about ~1-2% higher RPS, and it grows only "
            "slowly with age. (No hard cutoff: an aged API-scale snapshot still "
            "beats a fresh mis-scaled one.)"
        )
    else:
        impact = (
            "the snapshot is missing/unparseable, so every CL fixture is falling "
            "back to the naive form engine until it recovers."
        )
    return (
        "*⚠️ ClubElo snapshot is stale*\n\n"
        "The cross-league Elo feed the Champions League engine reads has not "
        f"refreshed, so {impact}\n\n"
        f"- File: `{status.path.name}`\n"
        f"- Snapshot date: {_snap_str(status)}\n"
        f"- Age: {_age_str(status)} (threshold {STALE_AFTER_DAYS} d)\n"
        f"- Reason: `{status.reason}`\n\n"
        "The CL engine now prices off the daily SITE-scale scrape of "
        "clubelo.com (the api.clubelo CSV/date API is gone for good). This "
        "alert means that daily scrape has not refreshed — the known failure "
        "mode is a Cloudflare 403 on the scrape. Until it recovers the engine "
        "keeps pricing off the last good site snapshot (a slowly-growing error, "
        "~1-2% RPS while it is only days old), or falls back to the naive form "
        "engine if none is present. Check the scraper if this persists.\n"
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
    :func:`betbot.notify.notify_operator`; ``now`` is the wall-clock cadence
    clock and ``wall_now`` the display timestamp (both injectable for tests).
    """
    action = alerter.decide(status, now=time.time() if now is None else now)
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
