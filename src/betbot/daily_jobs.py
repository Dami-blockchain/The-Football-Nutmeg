"""Scheduled tipster Telegram jobs.

Two alerts replace the old 21:00 report:

* **matchday-morning** — one message per registered user with today's
  predictions, entitlement-gated (operator/trial reveal all; payers reveal up
  to their credits, the rest are locked teasers). Fires on the **Africa/Nairobi
  wall clock** via ``CronTrigger(timezone="Africa/Nairobi")`` at
  ``BETBOT_MATCHDAY_ALERT_HOUR`` (default 8).
* **~60-min-before-kickoff** — a per-fixture reminder scheduled by
  :mod:`betbot.main` (one-off DateTrigger jobs), sending that fixture's
  prediction to entitled users.

Side effects (balance-gated reveals, Telegram sends) are injected as callables
so jobs are unit-testable with fixture data and no network. ``betbot.wallet``
(web3) and the entitlement wrapper stay behind lazy imports / injected fns.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable, Sequence
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from betbot.entitlement import entitlement_for
from betbot.logging import get_logger
from betbot.storage.repos import (
    has_revealed,
    increment_predictions_consumed,
    list_users,
    predictions_for_kickoff_range,
    prediction_for_fixture,
    record_reveal,
)
from betbot.tips import format_locked, format_prediction

log = get_logger(__name__)

REPORT_TZ = "Africa/Nairobi"

# (settings, chat_id, text) -> delivered?  Matches notify.send_telegram_to.
SendFn = Callable[[object, int, str], Awaitable[bool]]


def nairobi_day_bounds(
    now: datetime | None = None,
) -> tuple[datetime, datetime, date]:
    """``(start_utc, end_utc, local_date)`` of "today" on the Nairobi clock.

    "Today" means the operator's calendar day, not the UTC day — at 08:00 EAT
    they differ by 3 hours, enough to drop or double-count fixtures if we
    naively used UTC midnight.
    """
    local = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(REPORT_TZ))
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    start = start_local.astimezone(timezone.utc)
    return start, start + timedelta(days=1), local.date()


# ----------------------------------------------------------------------
# Broadcast (operator + every registered user)
# ----------------------------------------------------------------------
def broadcast_chat_ids(settings, users) -> list[int]:
    """Operator first, then registered users, de-duplicated (the operator is
    usually also a registered user — they must not get the message twice)."""
    ids: list[int] = []
    if settings.telegram_allowed_user_id:
        ids.append(settings.telegram_allowed_user_id)
    for u in users:
        if u.telegram_user_id not in ids:
            ids.append(u.telegram_user_id)
    return ids


# ----------------------------------------------------------------------
# Entitlement-gated reveal (shared by /predictions, matchday, kickoff alerts)
# ----------------------------------------------------------------------
def _entitlement_header(ent) -> str:
    """One-line status header for a user's alert."""
    if ent.reason == "operator":
        return "🎟️ operator — unlimited predictions"
    if ent.reason == "trial":
        d = ent.trial_days_left
        return f"🎟️ free trial — {d} day{'s' if d != 1 else ''} left"
    if ent.reason == "credit":
        c = ent.credits_remaining
        return f"🎟️ {c} prediction credit{'s' if c != 1 else ''} remaining"
    return "🔒 trial ended — send 1 USDC (Polygon) per prediction to unlock"


def render_user_predictions(
    user,
    predictions,
    settings,
    *,
    now: datetime | None = None,
    entitlement_fn=entitlement_for,
    already_revealed_fn=has_revealed,
    edge_threshold: float | None = None,
) -> tuple[str, list[tuple[int, bool]]]:
    """Build one user's gated message body. **Pure — no DB writes.**

    Returns ``(text, reveals)`` where ``reveals`` is a list of
    ``(fixture_id, charged)`` for each fixture NEWLY revealed in THIS render
    (i.e. not already in the reveal ledger). The caller commits those reveals —
    and only charges credits — AFTER a confirmed Telegram send, via
    :func:`commit_reveals`. Nothing here mutates the DB, so a render whose send
    fails costs the user nothing.

    Per prediction:

    * **already revealed** (``already_revealed_fn`` is True): shown FREE, added
      to no ``reveals`` entry — it was accounted for on its first reveal.
    * **operator / trial**: shown, appended as ``(fid, False)`` — a free reveal,
      but recorded so it stays free once the trial ends.
    * **paying**: up to ``credits_remaining`` NEW fixtures are revealed and
      appended as ``(fid, True)``; the rest are locked teasers.
    * **locked** (no credit): locked teaser, nothing appended.

    ``entitlement_fn`` and ``already_revealed_fn`` are injected so tests avoid
    the network and DB.
    """
    if edge_threshold is None:
        edge_threshold = settings.edge_threshold

    ent = entitlement_fn(user, settings, now=now)
    parts = [_entitlement_header(ent)]
    reveals: list[tuple[int, bool]] = []

    if not predictions:
        parts.append("\nNo fixtures today.")
        return "\n".join(parts), reveals

    free_reason = ent.reason in ("operator", "trial")
    # Paid credits fund only NEW fixtures; already-revealed ones are free and
    # don't draw down the budget.
    credits = max(0, ent.credits_remaining) if ent.reason == "credit" else 0
    paid_revealed = 0

    for p in predictions:
        # Already paid for on a prior path/repeat — always free, never re-charged.
        if already_revealed_fn(user.telegram_user_id, p.fixture_id):
            parts.append("\n" + format_prediction(p, edge_threshold=edge_threshold))
            continue

        if free_reason:
            parts.append("\n" + format_prediction(p, edge_threshold=edge_threshold))
            reveals.append((p.fixture_id, False))
        elif paid_revealed < credits:
            parts.append("\n" + format_prediction(p, edge_threshold=edge_threshold))
            reveals.append((p.fixture_id, True))
            paid_revealed += 1
        else:
            parts.append("\n" + format_locked(p))

    return "\n".join(parts), reveals


def commit_reveals(user, reveals: list[tuple[int, bool]]) -> None:
    """Persist reveals AFTER a confirmed send; charge one credit per NEW paid one.

    ``record_reveal`` returns False if the ledger row already existed (a retried
    send), so this is idempotent — no double ledger row and, crucially, no
    double :func:`increment_predictions_consumed`. A credit is charged ONLY when
    a brand-new ``charged=True`` row is inserted, which only happens after a send
    the caller has already confirmed returned True.
    """
    for fid, charged in reveals:
        if record_reveal(user.telegram_user_id, fid, charged) and charged:
            increment_predictions_consumed(user.telegram_user_id)


# ----------------------------------------------------------------------
# Matchday-morning alert
# ----------------------------------------------------------------------
async def run_matchday_alert(
    settings,
    *,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
    fixtures_fn: Callable[[datetime, datetime], Sequence[object]] | None = None,
    entitlement_fn=entitlement_for,
    users_fn=list_users,
) -> int:
    """Send each registered user today's entitlement-gated predictions.

    Pure-ish: ``send_fn`` (Telegram), ``fixtures_fn`` (prediction source),
    ``entitlement_fn`` and ``users_fn`` are all injectable. Returns messages
    delivered. A user with no fixtures today still gets a short "no matches"
    note (the operator always sees the full list). Never raises for one bad
    send — the caller's tick wrapper is the last line of defence, but per-user
    isolation here keeps one failure from dropping the rest.
    """
    from betbot.notify import send_telegram_to

    send = send_fn or send_telegram_to
    start, end, day = nairobi_day_bounds(now)
    fixtures = (
        list(fixtures_fn(start, end))
        if fixtures_fn is not None
        else predictions_for_kickoff_range(start, end)
    )

    sent = 0
    for user in users_fn():
        text, reveals = render_user_predictions(
            user, fixtures, settings, now=now, entitlement_fn=entitlement_fn
        )
        body = f"*⚽ Matchday — {day.isoformat()}*\n\n{text}"
        try:
            if await send(settings, user.telegram_user_id, body):
                sent += 1
                # Charge + record ONLY after a confirmed send.
                commit_reveals(user, reveals)
        except Exception as e:  # noqa: BLE001 — one bad send must not drop the rest
            log.warning(
                "matchday_alert_send_failed",
                telegram_user_id=user.telegram_user_id, error=str(e),
            )

    log.info(
        "matchday_alert_sent",
        day=day.isoformat(), fixtures=len(fixtures), delivered=sent,
    )
    return sent


# ----------------------------------------------------------------------
# Per-fixture kickoff-60m alert (scheduled one-off by betbot.main)
# ----------------------------------------------------------------------
async def send_fixture_alert(
    settings,
    fixture_id: int,
    *,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
    prediction_fn=prediction_for_fixture,
    entitlement_fn=entitlement_for,
    users_fn=list_users,
) -> int:
    """Send one fixture's prediction to entitled users (~60m before kickoff).

    The freshest STORED prediction is re-sent — no confirmed-XI data exists on
    a free source, so a caveat line is appended. Returns messages delivered.
    """
    from betbot.notify import send_telegram_to

    send = send_fn or send_telegram_to
    pred = prediction_fn(fixture_id)
    if pred is None:
        log.info("kickoff_alert_no_prediction", fixture_id=fixture_id)
        return 0

    caveat = (
        "⚠️ lineup-confirmed data unavailable — model prediction as of "
        "kickoff-60m."
    )
    sent = 0
    for user in users_fn():
        text, reveals = render_user_predictions(
            user, [pred], settings, now=now, entitlement_fn=entitlement_fn
        )
        body = f"*⏰ Kickoff soon*\n\n{text}\n\n{caveat}"
        try:
            if await send(settings, user.telegram_user_id, body):
                sent += 1
                # Charge + record ONLY after a confirmed send.
                commit_reveals(user, reveals)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "kickoff_alert_send_failed",
                telegram_user_id=user.telegram_user_id,
                fixture_id=fixture_id, error=str(e),
            )
    log.info("kickoff_alert_sent", fixture_id=fixture_id, delivered=sent)
    return sent


# ----------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------
def register_daily_jobs(scheduler, settings, *, matchday_alert) -> None:
    """Register the one Nairobi-local matchday-morning cron on the daemon's
    scheduler.

    The job callable is passed in (rather than imported) so the daemon can wrap
    it in its own never-crash error handling.
    """
    scheduler.add_job(
        matchday_alert,
        trigger=CronTrigger(
            hour=settings.matchday_alert_hour, minute=0, timezone=REPORT_TZ
        ),
        id="matchday_alert",
    )


# TODO(R4b): register the weekly player-minutes refresh here. R4a ships the
# fetcher (scripts/fetch_player_minutes.py::run) + this hook; R4b wires the
# CronTrigger schedule (e.g. Monday 05:00 Nairobi) and the never-crash wrapper.
async def refresh_player_minutes_job(settings) -> None:
    """Weekly refresh of the api-football player-minutes cache (R4a fetcher).

    Kept dependency-light and best-effort: any failure is logged, never raised,
    so a scheduler tick can't crash the daemon. R4b decides the cadence.
    """
    from scripts.fetch_player_minutes import run as _refresh

    try:
        written = await _refresh(
            list(settings.leagues), settings.api_football_season
        )
        log.info("player_minutes_refreshed", written=written)
    except Exception as exc:  # noqa: BLE001 - never crash the scheduler
        log.warning("player_minutes_refresh_failed", error=str(exc))
