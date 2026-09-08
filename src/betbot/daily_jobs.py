"""Scheduled tipster Telegram jobs — the two-alert model (R4b).

Because the prediction can CHANGE once the confirmed XI is out, the morning
alert must not carry a prediction — only a heads-up. So:

* **morning heads-up** (``run_matchday_notice``) — one FREE, ungated broadcast
  listing today's fixtures: ``Home (H) v Away (A) — KO HH:MM · 🔮 prediction at
  HH:MM, confirmed-lineup update ~HH:MM``. NO probabilities, NO entitlement, NO
  credit charge. The stated early time is
  ``kickoff - early_alert_lead_minutes(competition)`` and the confirmed-lineup
  time is ``kickoff - lineup_confirm_lead_minutes()``. Fires on the
  **Africa/Nairobi wall clock** at ``BETBOT_MATCHDAY_ALERT_HOUR``.
* **pre-match prediction alerts** (``send_prediction_alert``) — the PAID
  product, fired per-fixture TWICE by :mod:`betbot.main`'s one-off DateTrigger
  jobs (the two-alert model): an EARLY model prediction at
  ``kickoff - early_alert_lead(competition)`` (XI not yet posted -> model note)
  and a LATE confirmed-XI update at ``kickoff - lineup_confirm_lead()`` (XI now
  out). Both hit the SAME function; the reveal ledger charges the fixture EXACTLY
  ONCE (early charges, late re-shows free with the updated lineup content). It
  fetches the confirmed lineup, RE-SCORES the fixture lineup-adjusted (R4a), and
  delivers the XI + adjusted prediction — ENTITLEMENT-GATED through the existing
  reveal ledger (operator/trial free; payers spend 1 credit; locked users get a
  teaser). This is where the paywall now lives.

Side effects (balance-gated reveals, Telegram sends, lineup fetch, re-scoring)
are injected as callables so jobs are unit-testable with fixture data and no
network. ``betbot.wallet`` (web3) and the entitlement wrapper stay behind lazy
imports / injected fns.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable, Sequence
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from betbot.dual_log import dual_log_audit_tick
from betbot.entitlement import entitlement_for
from betbot.logging import get_logger
from betbot.notify import notify_operator
from betbot.scheduling import add_async_job
from betbot.timefmt import to_eat
from betbot.storage.repos import (
    fixture_was_ever_revealed,
    has_revealed,
    high_conf_band_tally,
    high_conf_band_tally_sold,
    increment_predictions_consumed,
    list_users,
    mark_morning_drop_notified,
    morning_listings_pending_drop_notice,
    predictions_for_kickoff_range,
    prediction_for_fixture,
    record_morning_listing,
    record_rescore_drift,
    record_reveal,
    upsert_prediction,
)
from betbot.leagues import league_label
from betbot.tips import (
    format_locked,
    format_prediction,
    format_prediction_with_lineup,
)

log = get_logger(__name__)

REPORT_TZ = "Africa/Nairobi"

# (settings, chat_id, text) -> delivered?  Matches notify.send_telegram_to.
SendFn = Callable[[object, int, str], Awaitable[bool]]

# Rescore-drift instrumentation stage labels, keyed by the alert tag the
# scheduler fires (see betbot.main). MEASUREMENT ONLY — see RescoreDriftLog.
_DRIFT_STAGE_BY_TAG = {"early": "early_fire", "late": "kickoff_60"}


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
def notice_recipient_ids(settings, users) -> list[int]:
    """The DM recipients of the morning notice: operator first, then registered
    users, de-duplicated (the operator is usually also a registered user — they
    must not get the message twice).

    NOTE: distinct from ``settings.broadcast_chat_id`` (singular) — that is the
    GROUP target, handled separately in :func:`run_matchday_notice`.
    """
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


def high_conf_visible(
    settings,
    predictions,
    user_id: int,
    already_revealed_fn=has_revealed,
):
    """Predictions a user may be OFFERED under the high-conviction gate.

    The SINGLE definition of "what the paywall shows", shared by
    :func:`render_user_predictions` (/predictions + any daily push) and the
    chat context builder, so every surface agrees. A prediction is visible
    iff:

    * it clears the high-conviction ALERT gate
      (:func:`betbot.main.high_conf_alert_passes`) — the SAME predicate the
      pre-match and result-alert paths use, so NO fourth threshold knob is
      introduced; OR
    * the user ALREADY revealed it (``already_revealed_fn`` True) — a fixture
      paid for before this gate existed is never retroactively hidden.

    With ``settings.high_conf_alerts_only`` OFF the predicate passes every
    fixture, so the output equals ``predictions`` and behaviour is unchanged.
    Pure: no DB writes. ``already_revealed_fn`` is injected for tests.
    """
    from betbot.main import high_conf_alert_passes

    return [
        p
        for p in predictions
        if high_conf_alert_passes(settings, p)[0]
        or already_revealed_fn(user_id, p.fixture_id)
    ]


# A single newly-revealed fixture: ``(fixture_id, charged, sold_triple)`` where
# ``sold_triple`` is the ``(p_home, p_draw, p_away)`` ACTUALLY RENDERED to the
# user, or ``None`` when no triple was on screen. ``commit_reveals`` persists
# the triple with the ledger row so the sold call is recoverable after a later
# rescore overwrites the stored prediction.
Reveal = tuple[int, bool, "tuple[float, float, float] | None"]


def _rendered_triple(pred) -> tuple[float, float, float] | None:
    """The H/D/A triple as shown to the user, or ``None`` if unavailable."""
    try:
        return (float(pred.p_home), float(pred.p_draw), float(pred.p_away))
    except (AttributeError, TypeError):
        return None


def render_user_predictions(
    user,
    predictions,
    settings,
    *,
    now: datetime | None = None,
    entitlement_fn=entitlement_for,
    already_revealed_fn=has_revealed,
    edge_threshold: float | None = None,
) -> tuple[str, list[Reveal]]:
    """Build one user's gated message body. **Pure — no DB writes.**

    Returns ``(text, reveals)`` where ``reveals`` is a list of
    ``(fixture_id, charged, sold_triple)`` for each fixture NEWLY revealed in
    THIS render
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
    reveals: list[Reveal] = []

    if not predictions:
        parts.append("\nNo fixtures today.")
        return "\n".join(parts), reveals

    # HIGH-CONVICTION GATE — the SINGLE paywall choke point. Only fixtures
    # clearing high_conf_alert_passes are OFFERED: listed, revealable,
    # chargeable. Already-revealed fixtures stay visible free even if they no
    # longer clear it (paid for before this gate; never retroactively hidden).
    # Flag OFF -> passes everything -> visible == predictions (unchanged).
    visible = high_conf_visible(
        settings, predictions, user.telegram_user_id, already_revealed_fn
    )
    if not visible:
        # Fixtures existed but none cleared the bar (common at 0.65). Honest
        # message, and NO reveals -> commit_reveals charges nobody.
        parts.append("\nNo high-confidence calls today.")
        return "\n".join(parts), reveals

    free_reason = ent.reason in ("operator", "trial")
    # Paid credits fund only NEW fixtures; already-revealed ones are free and
    # don't draw down the budget.
    credits = max(0, ent.credits_remaining) if ent.reason == "credit" else 0
    paid_revealed = 0

    for p in visible:
        # Already paid for on a prior path/repeat — always free, never re-charged.
        if already_revealed_fn(user.telegram_user_id, p.fixture_id):
            parts.append("\n" + format_prediction(p, edge_threshold=edge_threshold))
            continue

        if free_reason:
            parts.append("\n" + format_prediction(p, edge_threshold=edge_threshold))
            reveals.append((p.fixture_id, False, _rendered_triple(p)))
        elif paid_revealed < credits:
            parts.append("\n" + format_prediction(p, edge_threshold=edge_threshold))
            reveals.append((p.fixture_id, True, _rendered_triple(p)))
            paid_revealed += 1
        else:
            parts.append("\n" + format_locked(p))

    return "\n".join(parts), reveals


def commit_reveals(user, reveals: list[Reveal]) -> None:
    """Persist reveals AFTER a confirmed send; charge one credit per NEW paid one.

    ``record_reveal`` returns False if the ledger row already existed (a retried
    send), so this is idempotent — no double ledger row and, crucially, no
    double :func:`increment_predictions_consumed`. A credit is charged ONLY when
    a brand-new ``charged=True`` row is inserted, which only happens after a send
    the caller has already confirmed returned True.

    Each reveal is ``(fixture_id, charged, sold_triple)``; a 2-tuple
    ``(fixture_id, charged)`` is still accepted (legacy callers) and stores no
    triple. The sold triple lands on the ledger row only when the row is
    brand-new — first reveal wins.
    """
    for rev in reveals:
        fid, charged = rev[0], rev[1]
        triple = rev[2] if len(rev) > 2 else None
        p_home, p_draw, p_away = triple if triple else (None, None, None)
        inserted = record_reveal(
            user.telegram_user_id, fid, charged,
            p_home=p_home, p_draw=p_draw, p_away=p_away,
        )
        if inserted and charged:
            increment_predictions_consumed(user.telegram_user_id)


# ----------------------------------------------------------------------
# Morning heads-up notice (FREE, ungated) — NO prediction, only a schedule
# ----------------------------------------------------------------------
def render_matchday_notice(settings, fixtures, day) -> str | None:
    """Build the FREE morning heads-up body, or ``None`` when there are none.

    With the high-conviction gate ON (``settings.high_conf_alerts_only``) this
    lists ONLY the fixtures that clear it (option a): a call line reads
    ``*Home (H) v Away (A)* — KO HH:MM · 🔮 prediction at HH:MM,
    confirmed-lineup update ~HH:MM`` and sub-threshold fixtures do not appear at
    all. If NOTHING clears the gate (common at 0.65) an honest
    ``No high-confidence calls today`` message is returned instead of a bare
    schedule. The gate is :func:`betbot.main.high_conf_alert_passes` — the SAME
    predicate the scheduler and paywall use, so NO new threshold knob is
    introduced. Times are the
    **Africa/Nairobi wall clock** (EAT); the early time is
    ``kickoff - early_alert_lead_minutes(competition)`` and the confirmed-lineup
    time is ``kickoff - lineup_confirm_lead_minutes()`` — the SAME leads the
    scheduler fires on. Deliberately carries NO probabilities/edge/xG.

    With ``settings.high_conf_alerts_only`` OFF the gate passes every fixture,
    so every line carries the prediction time and the footer is byte-identical
    to the pre-gate notice.
    """
    if not fixtures:
        return None
    from betbot.main import high_conf_alert_passes

    gate_on = getattr(settings, "high_conf_alerts_only", False)
    header = (
        f"*⚽ High-confidence calls — {day.isoformat()}*" if gate_on
        else f"*⚽ Today's fixtures — {day.isoformat()}*"
    )
    lines = [header, ""]
    qualifying = 0
    for f in fixtures:
        ko = f.kickoff
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        code = getattr(f, "competition_code", None)
        early_lead = settings.early_alert_lead_minutes(code)
        late_lead = settings.lineup_confirm_lead_minutes()
        ko_local = to_eat(ko)
        early_local = to_eat(ko - timedelta(minutes=early_lead))
        late_local = to_eat(ko - timedelta(minutes=late_lead))
        # Option (a): list ONLY fixtures that clear the high-conviction gate.
        # Flag OFF -> the predicate passes every fixture, so nothing is skipped
        # and this loop is byte-identical to the pre-gate notice. Flag ON -> a
        # sub-threshold fixture is dropped entirely (not even a KO-only line).
        if not high_conf_alert_passes(settings, f)[0]:
            continue
        qualifying += 1
        league = league_label(getattr(f, "competition_code", None))
        league_tag = f" · {league}" if league else ""
        lines.append(
            f"*{f.home_team} (H) v {f.away_team} (A)*{league_tag} — "
            f"KO {ko_local:%H:%M} · 🔮 prediction at {early_local:%H:%M}, "
            f"confirmed-lineup update ~{late_local:%H:%M}"
        )
    if gate_on and not qualifying:
        # Gate ON with an all-sub-threshold card (common at 0.65): send an
        # honest empty-hand message, never a bare/empty schedule.
        return (
            f"{header}\n\n"
            "No high-confidence calls today. Predictions are sent only for "
            "matches that clear our confidence bar, and nothing in today's "
            "card does."
        )
    lines.append("")
    if not gate_on:
        # Flag OFF: byte-identical to the pre-gate notice.
        lines.append("_Times EAT. An early model prediction is sent per match, then "
                     "a confirmed-XI update once the lineup is out._")
    else:
        lines.append("_Times EAT. These are today's high-confidence calls only. An "
                     "early model prediction is sent per match, then a confirmed-XI "
                     "update once the lineup is out._")
    return "\n".join(lines)


async def run_matchday_notice(
    settings,
    *,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
    fixtures_source: Callable[[datetime, datetime], Sequence[object]] | None = None,
    users_fn=list_users,
) -> int:
    """Broadcast today's FREE heads-up (fixture list) to every registered user.

    Pure-ish: ``send_fn`` (Telegram), ``fixtures_source`` (fixture rows) and
    ``users_fn`` are injectable. NO entitlement, NO reveal ledger, NO credit
    charge — this is a schedule, not a prediction. Returns messages delivered.
    With no fixtures today, nothing is sent (the day is simply quiet). One bad
    send never drops the rest.
    """
    from betbot.notify import send_telegram_to

    # `is not None`, not `or`: a falsy-but-valid injected sender (a callable
    # object defining __len__) would otherwise be swapped for the real
    # Telegram transport.
    send = send_fn if send_fn is not None else send_telegram_to
    start, end, day = nairobi_day_bounds(now)
    fixtures = (
        list(fixtures_source(start, end))
        if fixtures_source is not None
        else predictions_for_kickoff_range(start, end)
    )

    body = render_matchday_notice(settings, fixtures, day)
    if body is None:
        log.info("matchday_notice_no_fixtures", day=day.isoformat())
        return 0

    # Persist WHICH fixtures this notice NAMED (gate ON only — the sole regime
    # where the alert path can later SUPPRESS a listed fixture). The notice
    # evaluates the gate on the EARLY stored triple; upsert_prediction overwrites
    # that triple IN PLACE on every later rescore, so this set CANNOT be
    # re-derived once a fixture drifts — it must be captured now. Same predicate
    # as render_matchday_notice, so the recorded set == the named set. Recorded
    # BEFORE the send loop (we record the PROMISE, not delivery). Best-effort:
    # a listing write must never break the broadcast.
    if getattr(settings, "high_conf_alerts_only", False):
        from betbot.main import high_conf_alert_passes
        for f in fixtures:
            try:
                if not high_conf_alert_passes(settings, f)[0]:
                    continue
                ko = f.kickoff
                if ko.tzinfo is None:
                    ko = ko.replace(tzinfo=timezone.utc)
                record_morning_listing(
                    f.fixture_id,
                    getattr(f, "competition_code", "") or "",
                    f.home_team, f.away_team, ko, day.isoformat(),
                )
            except Exception as e:  # noqa: BLE001 — listing must not break the notice
                log.warning(
                    "morning_listing_record_failed",
                    fixture_id=getattr(f, "fixture_id", None), error=str(e),
                )

    sent = 0
    for uid in notice_recipient_ids(settings, users_fn()):
        try:
            if await send(settings, uid, body):
                sent += 1
        except Exception as e:  # noqa: BLE001 — one bad send must not drop the rest
            log.warning(
                "matchday_notice_send_failed",
                telegram_user_id=uid, error=str(e),
            )

    # Group broadcast: the SAME body ALSO goes to the configured GROUP target
    # (settings.broadcast_chat_id) when set — identical treatment to the
    # prematch high-conf broadcast. NO entitlement, reveal ledger, credit charge
    # or registration; the notice already lists only qualifying 0.65 calls, so
    # this exposes nothing the group would not receive via the alert path. A
    # failed group send must NEVER affect the user DMs above: it is caught,
    # logged distinctly, and kept OUT of `sent` (which counts user-DM
    # deliveries). Unset -> skipped, so behaviour is byte-identical to before.
    broadcast_chat_id = getattr(settings, "broadcast_chat_id", None)
    if broadcast_chat_id:
        try:
            if await send(settings, int(broadcast_chat_id), body):
                log.info(
                    "matchday_notice_broadcast_sent",
                    chat_id=int(broadcast_chat_id), day=day.isoformat(),
                )
        except Exception as e:  # noqa: BLE001 — group send must never break DMs
            log.warning(
                "matchday_notice_broadcast_failed",
                chat_id=int(broadcast_chat_id), error=str(e),
            )

    log.info(
        "matchday_notice_sent",
        day=day.isoformat(), fixtures=len(fixtures), delivered=sent,
    )
    return sent


# ----------------------------------------------------------------------
# "Dropped below the bar" notice — reconciles the morning high-conf list
# ----------------------------------------------------------------------
def render_morning_drop_notice(listing) -> str:
    """The short, plain, non-alarming notice for a fixture that was NAMED in the
    morning high-confidence notice but whose promised call has since drifted
    below the bar and will NOT be sent.

    Distinct from HIGH_CONF_DOWNGRADE_NOTE (which rides ALONGSIDE a call still
    being sent): here there is NO call, so the copy says so plainly and closes on
    a BOLD NO BET. Carries NO probabilities/edge. Routes the league name through
    ``league_label`` so it matches every other fixture-naming surface.
    """
    league = league_label(getattr(listing, "competition_code", None))
    league_tag = f" \u00b7 {league}" if league else ""
    return (
        "*\u26bd High-confidence list \u2014 update*\n\n"
        f"*{listing.home_team} (H) v {listing.away_team} (A)*{league_tag}\n\n"
        "\u2139\ufe0f This was on this morning's high-confidence list, but a "
        "fresh model run has since eased it below our confidence bar \u2014 so "
        "we're not sending a call on it.\n\n"
        "*NO BET.*"
    )


async def run_morning_drop_notices(
    settings,
    *,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
    fixture_ids: list[int] | None = None,
    listings_fn=None,
    prediction_fn=prediction_for_fixture,
    users_fn=list_users,
    ever_revealed_fn=None,
    mark_fn=None,
) -> int:
    """Tell the morning notice's audience when a NAMED high-confidence call has
    since dropped below the bar and will NOT be sent — instead of silence.

    The morning notice advertises fixtures that cleared the 0.65 gate on their
    EARLY stored triple. The alert gate is re-evaluated later (plan time, fire
    time) on the LIVE stored triple; a fixture that has drifted below is SILENTLY
    suppressed and nothing is sent. This reconciler closes that gap.

    TWO entry points, both landing here:
      * EVENT-DRIVEN (``fixture_ids`` set): called the instant the confirmed-XI
        (late) alert is SUPPRESSED for a fixture, so the notice lands at ~KO-10
        rather than on a clock sweep. There is no race with a drift-back-up: a
        fixture that climbed back above 0.65 by then PASSES the late gate and
        fires the normal alert, so it never reaches this call.
      * SWEEP (``fixture_ids`` None): the periodic RETRY safety-net — retries a
        send that failed and catches any listing whose late job never fired,
        bounded to listings whose late-alert time (KO - lineup_confirm_lead) has
        passed.

    For each pending listing it decides —

      * HONOURED (the call went out): the live stored row still clears the gate,
        OR the fixture was ever revealed to a user (the drift-below-then-back-up
        case, where the confirmed-XI alert fired). CONSUME it (mark handled),
        send nothing — exactly the ``passes OR ever_revealed`` predicate the
        RESULT path uses one surface later.
      * DROPPED (the promised call silently vanished): below the bar now AND
        never revealed. SEND the short drop notice to the SAME audience as the
        notice — the DM recipients (operator + registered users) AND the group
        broadcast target when set.

    FREE and READ-ONLY on the money path: NO PredictionReveal, NO charge, NO
    free-limit draw, NO registration — this is information about a call we
    already advertised for free. Idempotent: ``drop_notified`` flips True ONLY
    after an actual successful send (or a deliberate consume), so a total send
    failure leaves the fixture pending for the next tick's retry inside the
    bounded window. A failed send to one recipient never blocks the others.
    Returns messages delivered (DM copies). Injected fns keep it testable.
    """
    # Gate OFF: the alert path suppresses nothing, so no listed fixture can
    # silently vanish and there is nothing to reconcile. (Listings are recorded
    # gate-ON only, so pending is empty anyway — short-circuit for clarity and to
    # touch no DB when the flag is off.)
    if not getattr(settings, "high_conf_alerts_only", False):
        return 0

    from betbot.main import high_conf_alert_passes
    from betbot.notify import send_telegram_to

    send = send_fn if send_fn is not None else send_telegram_to
    ever_revealed_fn = ever_revealed_fn or fixture_was_ever_revealed
    mark_fn = mark_fn or mark_morning_drop_notified
    now = now or datetime.now(timezone.utc)

    if fixture_ids is not None:
        # Event-driven: reconcile exactly the fixture(s) whose late alert just
        # got suppressed. Scoped by id, no kickoff-window clause (the suppression
        # proves the late-alert lifecycle is over); already-notified -> no-op.
        from betbot.storage.repos import morning_listings_pending_by_ids
        pending = list(morning_listings_pending_by_ids(fixture_ids))
    else:
        # Sweep: bounded to listings whose late-alert time (KO - lead) has passed.
        lead = settings.lineup_confirm_lead_minutes()
        _fetch = listings_fn or morning_listings_pending_drop_notice
        pending = list(_fetch(now, lead))
    if not pending:
        return 0

    recipients = notice_recipient_ids(settings, users_fn())
    group_id = getattr(settings, "broadcast_chat_id", None)

    sent = 0
    consumed = 0
    for listing in pending:
        pred = prediction_fn(listing.fixture_id)
        # HONOURED? Same predicate as the result path: the live stored row still
        # clears the gate (still qualifying / drifted back up and the late alert
        # fired), OR the fixture was ever revealed to a user (it alerted at some
        # fire). Either way the promised call went out — CONSUME, send nothing.
        gate_passes = pred is not None and high_conf_alert_passes(settings, pred)[0]
        if gate_passes or ever_revealed_fn(listing.fixture_id):
            mark_fn(listing.fixture_id)
            consumed += 1
            log.info(
                "morning_drop_notice_consumed_honoured",
                fixture_id=listing.fixture_id, gate_passes=gate_passes,
            )
            continue

        # DROPPED: below the bar now AND never alerted -> owe the audience a
        # notice. No audience at all (no operator/users AND no group): the notice
        # can never be delivered, so CONSUME it rather than log a failure forever.
        if not recipients and not group_id:
            mark_fn(listing.fixture_id)
            log.info("morning_drop_notice_no_audience", fixture_id=listing.fixture_id)
            continue

        body = render_morning_drop_notice(listing)
        any_success = False
        for uid in recipients:
            try:
                if await send(settings, uid, body):
                    sent += 1
                    any_success = True
            except Exception as e:  # noqa: BLE001 — one bad send must not drop the rest
                log.warning(
                    "morning_drop_notice_send_failed",
                    telegram_user_id=uid, fixture_id=listing.fixture_id, error=str(e),
                )
        # Group broadcast (SAME body, ONCE) when set — identical treatment to the
        # morning notice's group copy. A failed group send must NEVER affect the
        # DMs above: caught, logged distinctly, kept OUT of ``sent`` (DM count).
        group_success = False
        if group_id:
            try:
                if await send(settings, int(group_id), body):
                    group_success = True
                    log.info(
                        "morning_drop_notice_broadcast_sent",
                        fixture_id=listing.fixture_id, chat_id=int(group_id),
                    )
            except Exception as e:  # noqa: BLE001 — group send must never break DMs
                log.warning(
                    "morning_drop_notice_broadcast_failed",
                    fixture_id=listing.fixture_id, chat_id=int(group_id), error=str(e),
                )

        # Flag ONLY once at least one recipient (DM or group) actually got it, so
        # drop_notified never lies. On a TOTAL send failure it is left False and
        # the fixture stays pending for the next tick's retry inside the bounded
        # window — better a retry than a "sent" that nobody received.
        if any_success or group_success:
            mark_fn(listing.fixture_id)
            log.info(
                "morning_drop_notice_sent",
                fixture_id=listing.fixture_id,
                dm_recipients=len(recipients), group=bool(group_success),
            )
        else:
            log.warning(
                "morning_drop_notice_all_sends_failed",
                fixture_id=listing.fixture_id,
                note="left un-notified for retry on the next tick",
            )
    if consumed:
        log.info("morning_drop_notices_consumed_total", count=consumed)
    return sent


# ----------------------------------------------------------------------
# Pre-match lineup-adjusted prediction alert (scheduled one-off by betbot.main)
# ----------------------------------------------------------------------
def render_user_lineup_prediction(
    user,
    pred,
    lineup,
    settings,
    *,
    now: datetime | None = None,
    adj_note: str | None = None,
    absences: str | None = None,
    entitlement_fn=entitlement_for,
    already_revealed_fn=has_revealed,
    edge_threshold: float | None = None,
    high_conf_body: str | None = None,
) -> tuple[str, list[Reveal]]:
    """One user's gated body for a SINGLE fixture's lineup-adjusted prediction.

    Same entitlement + reveal-ledger semantics as
    :func:`render_user_predictions` (operator/trial free & recorded, payer
    charged once per NEW fixture, locked -> teaser), but the revealed body
    carries the confirmed XIs via :func:`format_prediction_with_lineup`. Pure —
    no DB writes; the caller commits reveals only after a confirmed send.

    When ``high_conf_body`` is supplied (the high-conviction alert path is ON)
    it REPLACES the standing revealed body — the entitlement/reveal-ledger
    semantics are untouched, so LOCKED users still see only the teaser and the
    Model triple stays behind the paywall exactly as before. With it ``None``
    (flag OFF) the output is byte-identical to before.
    """
    if edge_threshold is None:
        edge_threshold = settings.edge_threshold
    ent = entitlement_fn(user, settings, now=now)
    header = _entitlement_header(ent)
    fid = pred.fixture_id

    def _revealed_body() -> str:
        if high_conf_body is not None:
            return high_conf_body
        return format_prediction_with_lineup(
            pred, lineup, edge_threshold=edge_threshold,
            adj_note=adj_note, absences=absences,
        )

    triple = _rendered_triple(pred)
    # Already paid for on a prior path/repeat — always free, never re-charged.
    if already_revealed_fn(user.telegram_user_id, fid):
        return header + "\n\n" + _revealed_body(), []
    if ent.reason in ("operator", "trial"):
        return header + "\n\n" + _revealed_body(), [(fid, False, triple)]
    if ent.reason == "credit" and ent.credits_remaining >= 1:
        return header + "\n\n" + _revealed_body(), [(fid, True, triple)]
    # Locked: teaser only, nothing revealed or charged.
    return header + "\n\n" + format_locked(pred), []


# ----------------------------------------------------------------------
# Lineup-gap reporting (the "silent degradation" alarm)
# ----------------------------------------------------------------------
#: Minutes before kickoff past which a missing XI stops being "too early" and
#: starts being a broken feed. The LATE alert exists solely to show a confirmed
#: XI, so if one is not available by then the feature is not working.
LINEUP_EXPECTED_BY_MINUTES = 20


def lineup_gap_is_notable(alert_tag: str, kickoff, now) -> bool:
    """Whether a missing XI at this moment is worth waking the operator for.

    The EARLY alert deliberately fires before the XI is posted — flagging that
    would be an alarm on normal operation, and an alarm that cries wolf is one
    the operator learns to ignore. Inside
    ``LINEUP_EXPECTED_BY_MINUTES`` of kickoff, though, a missing XI means the
    feed is not delivering what the late alert was built to show. Pure, so the
    boundary is testable without a clock.
    """
    if alert_tag != "late":
        return False
    if kickoff is None:
        return True  # can't place it; report rather than swallow
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    return (kickoff - now) <= timedelta(minutes=LINEUP_EXPECTED_BY_MINUTES)


async def report_lineup_gap(
    settings,
    baseline,
    *,
    alert_tag: str,
    error: str | None = None,
    now=None,
    send_fn=None,
) -> bool:
    """Telegram the operator when a confirmed XI could not be fetched.

    Returns True iff a message was sent. Deduped per fixture per alert, so three
    matches on the same evening produce three flags rather than one — each is a
    distinct event the operator asked to see — while a retry of the same alert
    stays quiet.
    """
    now = now or datetime.now(timezone.utc)
    if not lineup_gap_is_notable(alert_tag, baseline.kickoff, now):
        return False
    fixture = f"{baseline.home_team} v {baseline.away_team}"
    reason = f"`{error}`" if error else "the feed returned no starting XI"
    body = (
        "*\u26a0\ufe0f Lineup unavailable*\n\n"
        f"{fixture} ({baseline.competition_code}) kicks off soon and the "
        f"confirmed XI could not be fetched — {reason}.\n\n"
        "The pre-match alert was sent on the MODEL prediction only, with no "
        "lineup adjustment. If this repeats across fixtures the lineup feed is "
        "down, not merely late."
    )
    log.warning(
        "lineup_gap",
        fixture_id=baseline.fixture_id,
        fixture=fixture,
        competition=baseline.competition_code,
        alert=alert_tag,
        error=error,
    )
    return await notify_operator(
        settings,
        body,
        kind="lineup_gap",
        dedupe_key=f"lineup_gap:{baseline.fixture_id}:{alert_tag}",
        send_fn=send_fn,
    )


#: Plain, non-alarming note appended to a pre-match alert whose call has
#: rescored BELOW the high-confidence bar since it was first alerted (the
#: "display drift" case). The result is still sent — see run_result_alerts,
#: which honours the alert-time promise via fixture_was_ever_revealed.
HIGH_CONF_DOWNGRADE_NOTE = (
    "ℹ️ Update: since this call was first flagged, a fresh model run has eased "
    "it below our high-confidence bar. We're still sending it, and the "
    "full-time result will follow."
)


async def send_prediction_alert(
    settings,
    fixture_id: int,
    *,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
    prediction_fn=prediction_for_fixture,
    lineup_fn: Callable | None = None,
    rescore_fn: Callable | None = None,
    entitlement_fn=entitlement_for,
    users_fn=list_users,
    alert_tag: str = "early",
    operator_send_fn: Callable | None = None,
) -> int:
    """Pre-match: fetch the confirmed XI, re-score lineup-adjusted, send gated.

    Steps (each side-effecting bit is injectable so tests need no network):

    1. Load the STORED baseline prediction (``prediction_fn``). None -> skip.
    2. Fetch the confirmed lineup + ``(home_adj, away_adj)`` via
       ``lineup_fn(baseline) -> (lineup, home_adj, away_adj, absences)``.
    3. ALWAYS RE-SCORE fresh via ``rescore_fn(fixture_id, home_adj, away_adj)
       -> (Prediction, kickoff)`` (the adjustment may be 0) and persist it
       (``upsert_prediction``) so the alert never ships a stale stored row. On a
       re-score failure we fall back to the stored baseline. If lineups aren't
       out the fresh (adj == 0) prediction is still sent, with a caveat.
    4. Per user, build ``(text, reveals)`` via
       :func:`render_user_lineup_prediction` and, ONLY after a confirmed send,
       ``commit_reveals`` — so the EXISTING ledger prevents double-charge across
       repeat views. Returns messages delivered.
    """
    from betbot.notify import send_telegram_to

    # `is not None`, not `or`: a falsy-but-valid injected sender (a callable
    # object defining __len__) would otherwise be swapped for the real
    # Telegram transport.
    send = send_fn if send_fn is not None else send_telegram_to
    baseline = prediction_fn(fixture_id)
    if baseline is None:
        log.info("prematch_alert_no_prediction", fixture_id=fixture_id)
        return 0

    # --- confirmed lineup + adjustments (default: production lineup service) ---
    lineup = None
    home_adj = away_adj = 0.0
    absences: str | None = None
    lineup_error: str | None = None
    if lineup_fn is None:
        lineup_fn = _default_lineup_fn(settings)
    try:
        lineup, home_adj, away_adj, absences = await lineup_fn(baseline)
    except Exception as e:  # noqa: BLE001 — lineup data is best-effort
        lineup_error = str(e)
        log.warning("prematch_lineup_failed", fixture_id=fixture_id, error=str(e))
    # EVERY outcome is logged, found or not. This path degraded to its "not yet
    # confirmed" caveat on every single fixture for weeks and said NOTHING —
    # not one warning in the daemon log — so the only way to discover the feed
    # was dead was for the operator to ask the bot. A fallback that silent is
    # indistinguishable from a feature that works.
    log.info(
        "prematch_lineup_result",
        fixture_id=fixture_id,
        alert=alert_tag,
        found=bool(lineup),
        home_adj=home_adj,
        away_adj=away_adj,
        error=lineup_error,
    )
    if not lineup:
        await report_lineup_gap(
            settings,
            baseline,
            alert_tag=alert_tag,
            error=lineup_error,
            send_fn=operator_send_fn,
        )

    # --- ALWAYS re-score fresh at alert time (or fall back to the baseline) ----
    # Previously this only re-scored when the lineup adjustment was nonzero, so
    # when the player-minutes cache was empty (adj == 0) we shipped the STALE
    # stored row (observed live: Celta stored H87/D8/A6 while a fresh score gave
    # H50/D23/A27). We now re-score unconditionally — the adjustment may be 0 —
    # so the freshest ratings/DC/market drive the alert. If re-scoring fails
    # (e.g. network), we gracefully keep the stored baseline rather than skipping
    # the alert, and the money path (entitlement + reveal ledger) is untouched.
    pred = baseline
    adj_note: str | None = None
    if rescore_fn is not None:
        try:
            rescored, kickoff = await rescore_fn(fixture_id, home_adj, away_adj)
            if rescored is not None:
                upsert_prediction(rescored, kickoff=kickoff)
                # Re-read so the persisted row (with any paper_bet) drives the
                # standing format; fall back to the stored baseline on a miss.
                pred = prediction_fn(fixture_id) or baseline
        except Exception as e:  # noqa: BLE001 — re-score is best-effort
            log.warning("prematch_rescore_failed", fixture_id=fixture_id, error=str(e))
    if not lineup:
        adj_note = "⚠️ lineup not yet confirmed — model prediction"

    # Snapshot the post-rescore triple ACTUALLY in force at this fire, tagged
    # by stage, so early-vs-late drift is measurable later. Best-effort and
    # match-level (once per fire, outside the per-user loop); never blocks the
    # alert. MEASUREMENT ONLY — nothing here touches gating or thresholds.
    try:
        _drift_stage = _DRIFT_STAGE_BY_TAG.get(alert_tag, alert_tag)
        record_rescore_drift(
            fixture_id, _drift_stage, pred.p_home, pred.p_draw, pred.p_away
        )
        log.info(
            "rescore_drift_observed",
            fixture_id=fixture_id,
            stage=_drift_stage,
            p_home=round(pred.p_home, 4),
            p_draw=round(pred.p_draw, 4),
            p_away=round(pred.p_away, 4),
        )
    except Exception as e:  # noqa: BLE001 — instrumentation never blocks a send
        log.warning("rescore_drift_log_failed", fixture_id=fixture_id, error=str(e))

    # High-conviction alert format (BETBOT_HIGH_CONF_ALERTS_ONLY). Built ONCE
    # per fixture (match-level, same for every user) and only when the flag is
    # ON. The live-season tally is read fresh here at send time from the settled
    # ledger (club-only, current season, World Cup excluded) so the copy can
    # never quote a stale streak. Best-effort: a ledger read failure degrades to
    # no live tally rather than dropping the alert.
    high_conf_body: str | None = None
    if getattr(settings, "high_conf_alerts_only", False):
        # Lazy import: betbot.main imports this module, so a top-level import
        # would be circular. By call time main is fully loaded.
        from betbot.main import high_conf_alert_passes
        from betbot.notify import format_high_conf_alert

        # Re-check the gate on the FINAL (post-rescore) row that is actually
        # shown. The alert ALWAYS re-scores at fire time, and a fixture stored
        # above the bar can rescore below it (the live Celta 87->50 case). We
        # gate on stored p, and after the rescore ``pred`` IS the freshest
        # stored row — so a body still wearing "HIGH-CONFIDENCE" over a sub-band
        # Model line would be self-contradictory. On drift we DROP the high-conf
        # framing and fall back to the STANDARD body rather than suppress: the
        # fixture already cleared the gate at planning and again at fire time,
        # and the reveal ledger is engaged for this very send, so making it
        # vanish this late would be worse than sending it without a banner it no
        # longer earns. The tally read is skipped on drift (no band is quoted).
        passes, _pick, _p = high_conf_alert_passes(settings, pred)
        if not passes:
            log.info(
                "high_conf_display_drift",
                fixture_id=fixture_id,
                note="stored row cleared the gate but the rescored row did not;"
                     " sending the standard body without the high-conf banner",
            )
            # Tell the reader plainly that the call has slipped below the bar
            # since the earlier alert, and that the result still follows. Plumbed
            # through the existing ``adj_note`` (appended, so a lineup caveat is
            # kept) — the standard body already carries the "NO BET — below our
            # confidence bar" line when the filter is live, and this reads
            # coherently above it.
            adj_note = (
                f"{adj_note}\n{HIGH_CONF_DOWNGRADE_NOTE}"
                if adj_note
                else HIGH_CONF_DOWNGRADE_NOTE
            )
        else:
            # Prefer the SOLD-basis tally (band membership judged on the triple
            # actually sold, not the post-rescore triple) — it does not silently
            # drop a call whose stored triple drifted below the bar after sale.
            # It falls back to the standard tally when the sold ledger has no
            # rows yet (legacy reveals carry NULL triples), and the alert is
            # labelled honestly for whichever basis is used. Same club-only /
            # current-season scope either way (enforced inside the repo helpers).
            tally = None
            tally_sold = False
            try:
                min_p = settings.high_conf_alert_min_p
                sold_hits, sold_n = high_conf_band_tally_sold(min_p)
                if sold_n > 0:
                    tally, tally_sold = (sold_hits, sold_n), True
                else:
                    tally = high_conf_band_tally(min_p)
            except Exception as e:  # noqa: BLE001 — never block the alert on a read
                log.warning("high_conf_tally_failed", fixture_id=fixture_id, error=str(e))
                tally = None
            high_conf_body = format_high_conf_alert(
                pred, settings, market=None,
                live_tally=tally, live_tally_sold=tally_sold,
            )

    sent = 0
    for user in users_fn():
        text, reveals = render_user_lineup_prediction(
            user, pred, lineup, settings, now=now,
            adj_note=adj_note, absences=absences,
            entitlement_fn=entitlement_fn,
            high_conf_body=high_conf_body,
        )
        # Honest header: the EARLY alert fires before the XI is posted, so
        # only claim "confirmed lineup" when one is actually attached.
        label = "confirmed lineup" if lineup else "model prediction"
        body = f"*⏰ Pre-match — {label}*\n\n{text}"
        try:
            if await send(settings, user.telegram_user_id, body):
                sent += 1
                # Charge + record ONLY after a confirmed send.
                commit_reveals(user, reveals)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "prematch_alert_send_failed",
                telegram_user_id=user.telegram_user_id,
                fixture_id=fixture_id, error=str(e),
            )
    # --- Optional BROADCAST copy of the high-confidence call --------------
    # When BETBOT_BROADCAST_CHAT_ID is set AND this fixture produced a
    # high-confidence alert body, send the SAME rendered alert ONCE to that
    # chat (a group/channel), IN ADDITION to the per-user DMs above, which
    # remain the primary target and are untouched. BROADCAST-ONLY: groups are
    # not an approved paid surface (the paywall keys on telegram_user_id), so
    # this path deliberately does NOT build a per-user body, and creates NO
    # reveal row, charge, or free-limit draw -- it never calls commit_reveals.
    # It is also kept OUT of ``sent`` so the returned delivered-count is
    # unchanged. Default unset -> this block is skipped entirely and behaviour
    # is byte-identical to before.
    broadcast_chat_id = getattr(settings, "broadcast_chat_id", None)
    if high_conf_body is not None and broadcast_chat_id:
        label = "confirmed lineup" if lineup else "model prediction"
        broadcast_body = f"*\u23f0 Pre-match \u2014 {label}*\n\n{high_conf_body}"
        try:
            if await send(settings, int(broadcast_chat_id), broadcast_body):
                log.info(
                    "high_conf_broadcast_sent",
                    fixture_id=fixture_id,
                    chat_id=int(broadcast_chat_id),
                )
        except Exception as e:  # noqa: BLE001 -- broadcast must never break DMs
            log.warning(
                "high_conf_broadcast_failed",
                fixture_id=fixture_id,
                error=str(e),
            )

    log.info("prematch_alert_sent", fixture_id=fixture_id, delivered=sent)
    return sent


# Process-wide shared LineupService (budget fix). Each pre-match alert used to
# build a FRESH LineupService, whose per-(league,date) /matches cache started
# empty every time — so N alerts for the same league+day fired N /matches calls
# (~4 calls/fixture/day), and a heavy Saturday neared the 100/day Highlightly
# cap. One long-lived instance shares that cache across the whole alert batch:
# 1 /matches per league/day + 1 /lineups per fixture. Rebuilt only if settings
# change (never in production; only tests inject a new settings object).
_LINEUP_SERVICE = None
_LINEUP_SERVICE_SETTINGS = None


def _lineup_service(settings):
    """Return the shared :class:`LineupService`, creating it once and reusing it.

    The instance's ``/matches``-per-(league,date) cache is what makes repeated
    fixtures on the same day reuse a SINGLE ``/matches`` fetch. Keyed by the
    settings object so a test with a different settings gets its own service.
    """
    global _LINEUP_SERVICE, _LINEUP_SERVICE_SETTINGS
    if _LINEUP_SERVICE is None or _LINEUP_SERVICE_SETTINGS is not settings:
        from betbot.data.lineup_service import LineupService

        _LINEUP_SERVICE = LineupService(settings)
        _LINEUP_SERVICE_SETTINGS = settings
    return _LINEUP_SERVICE


# ----------------------------------------------------------------------
# End-of-match RESULT ALERT (free; sent only to users who saw the prediction)
# ----------------------------------------------------------------------
async def run_result_alerts(
    settings,
    *,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
    outcomes_fn=None,
    prediction_fn=prediction_for_fixture,
    users_fn=list_users,
    already_revealed_fn=has_revealed,
    mark_notified_fn=None,
) -> int:
    """Broadcast full-time RESULT ALERTS for recently-settled fixtures.

    FREE and READ-ONLY on the money path: no reveal ledger write, no credit
    charge. For each un-notified scored outcome it sends
    :func:`betbot.tips.format_result` ONLY to users who had that fixture's
    prediction REVEALED (``has_revealed`` True) — the operator always. After the
    batch for a fixture completes it is flagged ``result_notified`` so it never
    re-sends. Returns total messages delivered. Injected fns keep it testable.

    HIGH-CONVICTION GATE (result path). The old claim that this path agreed
    with the PRE-MATCH path "by construction" was FALSE: the alert fires on the
    row as it stood at alert time, but ``upsert_prediction`` overwrites
    ``p_home/p_draw/p_away`` IN PLACE on every later rescore, so a fixture that
    cleared the 0.65 bar when we alerted can read BELOW it by settlement (real
    cases 2026-09-04: Stuttgart 0.654->0.617, Ipswich v Liverpool AWAY
    0.691->0.639 — both alerted, both correct, both silently dropped their
    result). The governing product rule is "high confidence AT ALERT TIME": if
    we alerted it, we owe the outcome. So each pending fixture passes iff::

        high_conf_alert_passes(live stored row)  OR  fixture_was_ever_revealed

    exactly the ``passes OR already_revealed`` shape :func:`high_conf_visible`
    already uses. NO second threshold, hysteresis band, or grace margin is
    introduced — the ONE 0.65 gate, plus an honest memory of what we alerted.

      * ``settings.high_conf_alerts_only`` OFF: the predicate returns ``True``
        without touching the prediction, so EVERY settled fixture alerts and a
        missing prediction is harmless — byte-identical to the legacy behaviour.
      * ON: a fixture alerts iff its live stored top-pick probability clears
        ``settings.high_conf_alert_min_p`` and is NOT the draw, OR the fixture
        was ever revealed to any user (the drift case above). A fixture that
        was never revealed AND whose stored row is below the bar (or has no
        stored prediction — it could not have cleared the pre-match gate) is
        SUPPRESSED (never dereferenced). Each honoured-on-drift fixture logs
        ``result_alert_honoured_prior_reveal`` so the drift population stays
        countable.

    NOTIFIED-FLAG DECISION: a SUPPRESSED fixture is STILL flagged
    ``result_notified`` (it is consumed, just not sent). Leaving it unflagged
    would re-queue it on every run and grow an unbounded pending backlog that is
    re-filtered forever — the silent-death failure mode this path must avoid.
    Marking it also mirrors the pre-match path, where a suppressed fixture is
    simply absent from the plan and the coverage watchdog treats it as COVERED
    (not a missing alert). So here ``result_notified`` means "handled by the
    result path" — a DELIBERATE SUPPRESSION, or a broadcast in which AT LEAST
    ONE recipient actually received the result. On a TOTAL send failure (every
    recipient errored) the flag is left False so the 2-hourly pass retries the
    fixture inside the 3-day pending window rather than recording a result
    "sent" that nobody got. A consequence:
    a fixture suppressed while the flag was ON is not retroactively alerted if
    the flag is later turned OFF (its outcome is already consumed) — symmetric
    with the pre-match path, whose scheduled fire time is likewise long gone.
    Every suppression is logged (``result_alert_suppressed_low_conf`` per
    fixture + a ``result_alerts_suppressed_total`` count) so it can never die
    silently the way the alert scheduler once did.
    """
    from betbot.main import high_conf_alert_passes
    from betbot.notify import send_telegram_to
    from betbot.storage.repos import (
        fixture_was_ever_revealed,
        mark_result_notified,
        outcomes_pending_result_alert,
        sold_triple,
    )
    from betbot.tips import format_result

    # `is not None`, not `or`: a falsy-but-valid injected sender (a callable
    # object defining __len__) would otherwise be swapped for the real
    # Telegram transport.
    send = send_fn if send_fn is not None else send_telegram_to
    outcomes_fn = outcomes_fn or outcomes_pending_result_alert
    mark_notified_fn = mark_notified_fn or mark_result_notified

    pending = list(outcomes_fn())
    if not pending:
        return 0

    users = users_fn()
    operator_id = settings.telegram_allowed_user_id
    sent = 0
    suppressed = 0
    for row in pending:
        pred = prediction_fn(row.fixture_id)

        # Gate the RESULT path on the LIVE stored row, but honour the alert-time
        # promise: upsert_prediction overwrites p_* in place on rescore, so the
        # live row can have drifted below the bar since we alerted. When
        # high_conf_alerts_only is OFF the predicate returns True WITHOUT reading
        # pred, so a missing prediction still alerts (legacy behaviour). When ON,
        # a fixture with no stored prediction could not have cleared the
        # pre-match gate (a None pred must not raise here). The OR-clause
        # (fixture_was_ever_revealed) mirrors high_conf_visible and rescues a
        # drifted-but-alerted fixture — see the HIGH-CONVICTION GATE docstring.
        if pred is None:
            gate_passes = not getattr(settings, "high_conf_alerts_only", False)
        else:
            gate_passes = high_conf_alert_passes(settings, pred)[0]
        ever_revealed = fixture_was_ever_revealed(row.fixture_id)
        if not gate_passes and ever_revealed:
            # Below the bar now, but we alerted it — the promise is "high
            # confidence AT ALERT TIME", so we still owe the result. Distinct
            # event so the drift population is countable later. A revealed
            # fixture with no stored prediction row lands here too (has_pred
            # False): logged, never dereferenced, never raised.
            log.info(
                "result_alert_honoured_prior_reveal",
                fixture_id=row.fixture_id,
                has_pred=pred is not None,
            )
        if not gate_passes and not ever_revealed:
            # Never alerted AND below the bar (or no stored prediction): consume
            # it (mark notified) so it is never re-queued — see the
            # NOTIFIED-FLAG DECISION in the docstring.
            mark_notified_fn(row.fixture_id)
            suppressed += 1
            log.info("result_alert_suppressed_low_conf", fixture_id=row.fixture_id)
            continue

        # Snapshot the RESULT-stage triple (what settlement scored) for the
        # alerted population, so drift from the alert-time triple is
        # measurable. Best-effort; never affects the result broadcast.
        try:
            record_rescore_drift(
                row.fixture_id, "result",
                row.predicted_home, row.predicted_draw, row.predicted_away,
            )
            log.info(
                "rescore_drift_observed",
                fixture_id=row.fixture_id,
                stage="result",
                p_home=round(row.predicted_home, 4),
                p_draw=round(row.predicted_draw, 4),
                p_away=round(row.predicted_away, 4),
            )
        except Exception as e:  # noqa: BLE001 — instrumentation never blocks
            log.warning(
                "rescore_drift_log_failed", fixture_id=row.fixture_id, error=str(e)
            )

        home = pred.home_team if pred is not None else "Home"
        away = pred.away_team if pred is not None else "Away"
        # Quote the triple ACTUALLY SOLD when we have it (None for legacy/never-
        # revealed fixtures), so the result echoes what the user paid for rather
        # than the post-rescore triple on the outcome row.
        body = "*⚽ Result*\n\n" + format_result(
            row, home, away,
            competition_code=getattr(row, "competition_code", None),
            sold_triple=sold_triple(row.fixture_id),
        )

        # Audience: the operator (always) + every user who saw this prediction.
        audience: list[int] = []
        if operator_id:
            audience.append(operator_id)
        for u in users:
            if u.telegram_user_id in audience:
                continue
            if already_revealed_fn(u.telegram_user_id, row.fixture_id):
                audience.append(u.telegram_user_id)

        if not audience:
            # No operator and nobody revealed it: the audience cannot grow after
            # settlement, so retrying is pointless. Consume it (mark notified) so
            # it never re-queues, rather than logging all_sends_failed every 2h
            # for 3 days over a send that can never happen.
            mark_notified_fn(row.fixture_id)
            log.info("result_alert_no_audience", fixture_id=row.fixture_id)
            continue

        any_success = False
        for uid in audience:
            try:
                if await send(settings, uid, body):
                    sent += 1
                    any_success = True
            except Exception as e:  # noqa: BLE001 — one bad send mustn't drop the rest
                log.warning(
                    "result_alert_send_failed",
                    telegram_user_id=uid, fixture_id=row.fixture_id, error=str(e),
                )
        # Flag ONLY once at least one recipient actually received it, so
        # ``result_notified`` never lies. On a TOTAL send failure the flag is
        # left False and the fixture stays pending for the 2-hourly retry inside
        # the 3-day window (outcomes_pending_result_alert) — better a retry than
        # a result marked "sent" that nobody got.
        if any_success:
            mark_notified_fn(row.fixture_id)
            log.info(
                "result_alert_sent", fixture_id=row.fixture_id, delivered=len(audience),
            )
        else:
            log.warning(
                "result_alert_all_sends_failed",
                fixture_id=row.fixture_id, audience=len(audience),
                note="left un-notified for retry on the next 2-hourly pass",
            )
    if suppressed:
        log.info("result_alerts_suppressed_total", count=suppressed)
    return sent


def _default_lineup_fn(settings):
    """Build the production ``lineup_fn`` closure over the SHARED LineupService.

    Returns ``async (baseline) -> (lineup, home_adj, away_adj, absences)``.
    Reuses the process-wide :func:`_lineup_service` so its per-(league,date)
    ``/matches`` cache spans the whole alert batch (the budget fix — no fresh
    caches, no per-alert re-fetch). Any gap yields ``(None, 0.0, 0.0, None)`` —
    the caller then sends the baseline with a "lineup not yet confirmed" caveat.
    """
    async def _fn(baseline):
        svc = _lineup_service(settings)
        code = baseline.competition_code
        ko = baseline.kickoff
        ko_date = (ko.date().isoformat() if ko is not None else "")
        match_id = await svc.resolve_match_id(
            code, baseline.home_team, baseline.away_team, ko_date
        )
        if match_id is None:
            return None, 0.0, 0.0, None
        # One /lineups call, reused for both the display XI and the adj.
        lineup = await svc.get_confirmed_xi(
            code, baseline.home_team, baseline.away_team, ko_date,
            match_id=match_id,
        )
        home_adj, away_adj = await svc.adjustments_for_fixture(
            code, baseline.home_team, baseline.away_team, ko_date,
            match_id=match_id, lineups=lineup,
        )
        absences = _absence_summary(lineup, home_adj, away_adj)
        return lineup, home_adj, away_adj, absences

    return _fn


def _absence_summary(lineup, home_adj: float, away_adj: float) -> str | None:
    """Short 'who is notably out' line, only when a side is materially weakened.

    We don't have a per-player importance readout at the message layer (the
    penalty is aggregate), so this states WHICH SIDE is weakened and by how much
    in Glicko points — enough for the user to gauge the adjustment without
    leaking the model internals.
    """
    if not (home_adj or away_adj):
        return None
    parts: list[str] = []
    if home_adj:
        parts.append(f"home {home_adj:+.0f}")
    if away_adj:
        parts.append(f"away {away_adj:+.0f}")
    return "rating shift " + ", ".join(parts) if parts else None


# ----------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------
# --- Prior-season player-minutes backfill (budget-paced, one league / day) ----
#
# Only the currently-fetched PRIOR season (the newest api-football FREE-tier
# season, 2024) carries usable minutes; the current season (2026) is empty at
# season start. Fetching all five domestic leagues at once would blow the
# 100 req/day free budget, so instead a DAILY tick fills exactly ONE missing
# domestic league's ``<CODE>_<PRIOR>.json`` per run — all four fill within ~4
# days, each run well under the cap. Once every league is populated it no-ops.
#
# Prior season = api_football_season - 2 (2026 -> 2024): the immediate prior
# (2025) is unavailable on the free tier, matching lineup_service's fallback.
_PRIOR_SEASON_OFFSET = 2
# Domestic top-5 only; CL squads overlap these leagues and its own minutes are
# tiny, so it is excluded from the backfill (mirrors fetch_player_minutes' CL skip).
_BACKFILL_LEAGUES = ("PL", "PD", "BL1", "SA", "FL1")
# A cache file this small (``{}`` == 2 bytes, or absent) counts as unpopulated.
_EMPTY_CACHE_MAX_BYTES = 2


def prior_minutes_season(settings) -> int:
    """The completed season we backfill player-minutes for (free tier: 2024)."""
    return settings.api_football_season - _PRIOR_SEASON_OFFSET


def pick_league_to_backfill(
    season: int, *, minutes_dir=None, leagues: Sequence[str] = _BACKFILL_LEAGUES
) -> str | None:
    """Return the FIRST domestic league whose ``<CODE>_<season>.json`` cache is
    missing or empty (<= 2 bytes), else ``None`` (all populated).

    Pure/offline: only stats the filesystem, no network. Used by the daily tick
    to pick a single league to fetch, and unit-tested against a temp dir.
    """
    from betbot.data.lineup_service import PLAYER_MINUTES_DIR

    base = minutes_dir if minutes_dir is not None else PLAYER_MINUTES_DIR
    for code in leagues:
        path = base / f"{code.upper()}_{season}.json"
        try:
            populated = path.exists() and path.stat().st_size > _EMPTY_CACHE_MAX_BYTES
        except OSError:
            populated = False
        if not populated:
            return code.upper()
    return None


async def backfill_one_league_minutes_tick(settings, *, repo_root=None) -> None:
    """Daily: fetch ONE missing prior-season domestic league's player minutes.

    Budget-paced — one league per run keeps each day well under the 100 req/day
    api-football free cap; the four domestic leagues self-complete over ~4 days.
    No-ops once every league is populated. Runs ``fetch_player_minutes.py`` in a
    subprocess (isolation) and is best-effort: any failure is logged, never
    raised, so a bad fetch can't crash the daemon.
    """
    import asyncio
    import subprocess
    from pathlib import Path

    season = prior_minutes_season(settings)
    code = pick_league_to_backfill(season)
    if code is None:
        log.info("player_minutes_backfill_complete", season=season)
        return

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    def _run() -> None:
        args = [
            ".venv/bin/python", "scripts/fetch_player_minutes.py",
            "--league", code, "--season", str(season),
        ]
        subprocess.run(
            args, cwd=str(root), timeout=1800, check=True, capture_output=True,
        )

    try:
        await asyncio.to_thread(_run)
        log.info("player_minutes_backfilled", code=code, season=season)
    except Exception as exc:  # noqa: BLE001 — never crash the daemon
        log.warning(
            "player_minutes_backfill_failed", code=code, season=season, error=str(exc)
        )


def register_daily_jobs(scheduler, settings, *, matchday_notice) -> None:
    """Register the Nairobi-local morning heads-up cron on the daemon's scheduler.

    The job callable is passed in (rather than imported) so the daemon can wrap
    it in its own never-crash error handling.
    """
    add_async_job(
        scheduler,
        matchday_notice,
        trigger=CronTrigger(
            hour=settings.matchday_alert_hour, minute=0, timezone=REPORT_TZ
        ),
        id="matchday_notice",
    )
    # Daily 05:15 UTC (before the 05:xx alert reschedule / scoring): fill ONE
    # missing prior-season domestic league's player-minutes cache. Budget-paced;
    # no-ops once all four are populated. Self-contained + best-effort.
    #
    # ``settings`` is bound with args=, NOT a sync ``lambda: tick(settings)``:
    # that lambda shape made APScheduler call-and-discard the coroutine, so
    # this backfill never ran either (same root cause as the pre-match alert
    # outage). add_async_job now rejects it at registration time.
    add_async_job(
        scheduler,
        backfill_one_league_minutes_tick,
        args=(settings,),
        trigger=CronTrigger.from_crontab("15 5 * * *", timezone=timezone.utc),
        id="player_minutes_backfill",
    )
    # Daily 05:30 UTC: is the challenger dual-log actually accumulating?
    #
    # model_predictions sat frozen from 2026-07-17 to 2026-08-22 while the
    # roadmap waited on it to reach a sample size, and nothing said a word.
    # Read-only (three aggregate SELECTs) and best-effort, so it cannot affect
    # scoring, settlement or alerts. See betbot/dual_log.py.
    add_async_job(
        scheduler,
        dual_log_audit_tick,
        args=(settings,),
        trigger=CronTrigger.from_crontab("30 5 * * *", timezone=timezone.utc),
        id="challenger_dual_log_audit",
    )


# The weekly player-minutes refresh is wired in betbot.main.run_daemon as a
# Monday cron running scripts/fetch_player_minutes.py via subprocess (mirroring
# _club_refresh_tick), so a bad refresh can never corrupt the daemon. This hook
# remains as an in-process alternative / for manual invocation.
async def refresh_player_minutes_job(settings) -> None:
    """Weekly refresh of the api-football player-minutes cache (R4a fetcher).

    Kept dependency-light and best-effort: any failure is logged, never raised,
    so a scheduler tick can't crash the daemon.
    """
    from scripts.fetch_player_minutes import run as _refresh

    try:
        written = await _refresh(
            list(settings.leagues), settings.api_football_season
        )
        log.info("player_minutes_refreshed", written=written)
    except Exception as exc:  # noqa: BLE001 - never crash the scheduler
        log.warning("player_minutes_refresh_failed", error=str(exc))
