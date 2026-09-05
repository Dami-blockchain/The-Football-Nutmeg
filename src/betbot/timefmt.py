"""Single source of truth for user-facing time rendering.

Every timestamp the Telegram bot shows a user MUST be in East Africa Time
(Africa/Nairobi, UTC+3). Storage, scheduling, cron/daemon ticks, log lines and
all DB columns stay UTC — this module is the DISPLAY layer only.

Route every user-facing surface through :func:`eat_time` / :func:`eat_datetime`
rather than converting per call site: one helper, one zone, no scattered
offsets. We use the *named* IANA zone via :mod:`zoneinfo` rather than a
hardcoded ``+3`` — Kenya observes no DST so the two agree today, but the named
zone is self-documenting and stays correct if that ever changes.

Naive datetimes are assumed UTC (SQLite hands back naive columns). Conversion
is idempotent: an already-aware datetime (even one already in EAT) is converted
to EAT exactly once, so routing an already-EAT value through here never shifts
it a second time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

#: East Africa Time. Named zone (not a raw +3 offset) on purpose — see module doc.
EAT = ZoneInfo("Africa/Nairobi")


def to_eat(dt: datetime | None) -> datetime | None:
    """A UTC (or any aware / naive-UTC) datetime as an EAT-aware datetime.

    ``None`` passes through. A naive datetime is treated as UTC before
    conversion. Idempotent for already-aware inputs.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(EAT)


def eat_time(dt: datetime | None, *, label: bool = True) -> str:
    """Clock time as ``HH:MM EAT`` (``label=False`` -> bare ``HH:MM``); '' if None."""
    d = to_eat(dt)
    if d is None:
        return ""
    s = d.strftime("%H:%M")
    return f"{s} EAT" if label else s


def eat_datetime(dt: datetime | None, *, label: bool = True) -> str:
    """Full stamp as ``YYYY-MM-DD HH:MM EAT`` (``label=False`` drops EAT); '' if None."""
    d = to_eat(dt)
    if d is None:
        return ""
    s = d.strftime("%Y-%m-%d %H:%M")
    return f"{s} EAT" if label else s
