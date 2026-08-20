"""Keep stored kickoffs — and therefore pre-match alerts — in step with upstream.

Fixtures move. football-data.org rewrites a match's ``utcDate`` when a
broadcaster or a federation shifts it, but ``predictions.kickoff`` was written
once at scoring time and never revisited. One stale column broke three things:

* **Alerts fired at the dead time.** ``plan_kickoff_alert_jobs`` reads the
  stored kickoff, so a match moved later got its pre-match alert hours early,
  and a match moved earlier got it after the whistle.
* **A phantom alert charged a real credit.** The early alert reveals-and-charges
  through the ledger. Firing it on the old day for a match nobody played burned
  the user's one paid reveal, so the alert on the day the match actually
  kicked off came back "already revealed" and free — they paid for the wrong one.
  Observed live: Celta–Osasuna, stored 2026-08-16 19:30, actually moved to
  2026-08-27 18:30.
* **Settlement chased a result that could not exist.** ``kickoff + grace`` had
  passed on paper, so every settle tick re-fetched a fixture still days away.

The re-sync re-reads the current kickoff from the SAME endpoint the scoring run
uses — one call per league per pass, not one per fixture — and writes it back.
Matches that vanish from the league window (moved clean out of it) fall back to
a single per-fixture fetch, which is bounded by how many fixtures are in the
near window at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from betbot.logging import get_logger

log = get_logger(__name__)

# football-data statuses that mean "this match is not going to be played at the
# time we have". An alert scheduled off that time must be pulled, not re-timed:
# POSTPONED keeps the original utcDate until a new date is announced, so the
# kickoff alone never reveals that anything changed.
DEAD_STATUSES = frozenset({"POSTPONED", "CANCELLED", "SUSPENDED", "AWARDED"})

# How far past the planned fire time an alert may still legitimately run — the
# scheduler can be a little late, the send itself takes a moment.
FIRE_LATE_GRACE_MINUTES = 10

# Slack on the early side of the fire-time guard. The early alert is planned at
# KO - early_lead; anything meaningfully earlier than that means the job is
# pinned to a kickoff the fixture no longer has.
FIRE_EARLY_SLACK_MINUTES = 15


@dataclass(frozen=True)
class KickoffChange:
    """One fixture whose upstream schedule no longer matches what we stored."""

    fixture_id: int
    old_kickoff: datetime
    new_kickoff: datetime | None
    status: str | None = None

    @property
    def is_dead(self) -> bool:
        """True when the match is off, not merely moved."""
        return self.new_kickoff is None or (self.status or "") in DEAD_STATUSES


def alert_job_ids(fixture_id: int) -> tuple[str, str]:
    """The two pre-match job ids ``plan_kickoff_alert_jobs`` uses for a fixture.

    Kept here so the re-sync removes exactly the ids the planner registers; a
    drift between the two would leave orphan jobs firing off dead kickoffs.
    """
    return (f"predict_early_{fixture_id}", f"predict_late_{fixture_id}")


def parse_utc(value: Any) -> datetime | None:
    """Parse a football-data ``utcDate`` into an aware UTC datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def plan_kickoff_changes(
    stored: dict[int, datetime],
    upstream: dict[int, tuple[datetime | None, str | None]],
) -> list[KickoffChange]:
    """Fixtures whose upstream kickoff or status no longer matches ``stored``.

    Pure and offline. ``stored`` is ``{fixture_id: kickoff}`` straight out of
    :func:`upcoming_prediction_kickoffs`; ``upstream`` is
    ``{fixture_id: (kickoff, status)}`` as read from football-data. A fixture
    absent from ``upstream`` is NOT reported — absence is ambiguous (it may
    simply sit outside the fetched window), and guessing "cancelled" from it
    would pull alerts off perfectly healthy matches. The caller resolves those
    with a per-fixture fetch instead.
    """
    changes: list[KickoffChange] = []
    for fixture_id, old in sorted(stored.items()):
        if fixture_id not in upstream:
            continue
        new, status = upstream[fixture_id]
        old_utc = _as_utc(old)
        new_utc = _as_utc(new)
        dead = (status or "") in DEAD_STATUSES
        if not dead and new_utc == old_utc:
            continue
        changes.append(
            KickoffChange(
                fixture_id=fixture_id,
                old_kickoff=old_utc,
                new_kickoff=None if dead else new_utc,
                status=status,
            )
        )
    return changes


def alert_still_valid(
    now: datetime,
    kickoff: datetime | None,
    status: str | None,
    *,
    early_lead_minutes: int,
) -> bool:
    """Whether a pre-match alert firing ``now`` still matches the real fixture.

    The last line of defence on the money path. Even with an hourly re-sync a
    match can move between the last pass and the fire time, and firing then
    would charge a reveal for a match at a different time — or one that is not
    being played at all. Pure, so the window logic is testable without a clock.
    """
    if kickoff is None or (status or "") in DEAD_STATUSES:
        return False
    kickoff = _as_utc(kickoff)
    now = _as_utc(now)
    if now - kickoff > timedelta(minutes=FIRE_LATE_GRACE_MINUTES):
        return False  # already kicked off (or finished) — the alert is moot
    earliest_sane = timedelta(
        minutes=early_lead_minutes + FIRE_EARLY_SLACK_MINUTES
    )
    return kickoff - now <= earliest_sane


async def fetch_upstream_kickoffs(
    client,
    leagues: Iterable[str],
    date_from: str,
    date_to: str,
) -> dict[int, tuple[datetime | None, str | None]]:
    """``{fixture_id: (kickoff, status)}`` for every match in the window.

    One call per league — the whole point of going through the competition
    endpoint rather than per-fixture. A league that fails is logged and skipped;
    a partial map only means fewer fixtures get re-synced this pass, never a
    wrong re-sync.
    """
    upstream: dict[int, tuple[datetime | None, str | None]] = {}
    for league in leagues:
        try:
            matches = await client.list_matches(league, date_from, date_to)
        except Exception as e:  # noqa: BLE001 — one bad league mustn't stop the rest
            log.warning("kickoff_resync_league_failed", league=league, error=str(e))
            continue
        for match in matches:
            fixture_id = match.get("id")
            if not isinstance(fixture_id, int):
                continue
            upstream[fixture_id] = (
                parse_utc(match.get("utcDate")),
                match.get("status"),
            )
    return upstream


async def resync_kickoffs(
    client,
    settings,
    *,
    now: datetime,
    lookahead_days: int = 3,
    backfill_days: int = 7,
    stored_fn=None,
    update_fn=None,
) -> list[KickoffChange]:
    """Re-read upstream kickoffs for upcoming fixtures and persist the changes.

    Covers both directions a fixture can move:

    * **moved within the window** — the league fetch returns it under its new
      date and ``plan_kickoff_changes`` sees the mismatch;
    * **moved clean out of the window, or postponed** — it is missing from the
      league fetch, so each stored fixture we did not see gets ONE
      ``get_match`` call to settle the question. Bounded by the number of
      fixtures actually in the window, and normally zero.

    Returns the applied changes so the caller can pull the matching alert jobs.
    """
    from betbot.storage.repos import (
        update_prediction_kickoff,
        upcoming_prediction_kickoffs,
    )

    stored_fn = stored_fn or upcoming_prediction_kickoffs
    update_fn = update_fn or update_prediction_kickoff

    now = _as_utc(now)
    window_end = now + timedelta(days=lookahead_days)
    # Reach into the PAST as well as the future. A fixture moved forward still
    # carries its ORIGINAL kickoff here, so on our clock it looks overdue rather
    # than upcoming — and a forward-only window would never look at it again.
    # That is the live shape of this bug: stored 2026-08-16, actually 2026-08-27.
    # Settled fixtures are filtered out by the query, so this stays small.
    window_start = now - timedelta(days=backfill_days)
    stored = stored_fn(window_start, window_end)
    if not stored:
        return []

    # football-data's dateTo is exclusive, hence the +1 day.
    date_from = window_start.date().isoformat()
    date_to = (window_end.date() + timedelta(days=1)).isoformat()
    upstream = await fetch_upstream_kickoffs(
        client, settings.leagues, date_from, date_to
    )

    # Anything we hold but did not see upstream: resolve it individually rather
    # than assume. This is what catches a match moved months out, which is
    # exactly the case that leaves a phantom job on the old day.
    for fixture_id in sorted(set(stored) - set(upstream)):
        try:
            match = await client.get_match(fixture_id)
        except Exception as e:  # noqa: BLE001 — best-effort, never fatal
            log.warning(
                "kickoff_resync_fetch_failed", fixture_id=fixture_id, error=str(e)
            )
            continue
        if match is None:
            continue
        upstream[fixture_id] = (
            parse_utc(match.get("utcDate")),
            match.get("status"),
        )

    changes = plan_kickoff_changes(stored, upstream)
    applied: list[KickoffChange] = []
    for change in changes:
        if change.new_kickoff is not None:
            try:
                update_fn(change.fixture_id, change.new_kickoff)
            except Exception as e:  # noqa: BLE001 — one bad row mustn't stop the rest
                log.warning(
                    "kickoff_resync_write_failed",
                    fixture_id=change.fixture_id,
                    error=str(e),
                )
                continue
        applied.append(change)
        log.info(
            "kickoff_resynced",
            fixture_id=change.fixture_id,
            old_kickoff=change.old_kickoff.isoformat(),
            new_kickoff=(
                change.new_kickoff.isoformat() if change.new_kickoff else None
            ),
            status=change.status,
        )
    if applied:
        log.info("kickoff_resync_done", changed=len(applied), checked=len(stored))
    return applied
