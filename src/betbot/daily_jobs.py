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

from betbot.entitlement import entitlement_for
from betbot.logging import get_logger
from betbot.scheduling import add_async_job
from betbot.storage.repos import (
    has_revealed,
    increment_predictions_consumed,
    list_users,
    predictions_for_kickoff_range,
    prediction_for_fixture,
    record_reveal,
    upsert_prediction,
)
from betbot.tips import (
    format_locked,
    format_prediction,
    format_prediction_with_lineup,
)

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
# Morning heads-up notice (FREE, ungated) — NO prediction, only a schedule
# ----------------------------------------------------------------------
def render_matchday_notice(settings, fixtures, day) -> str | None:
    """Build the FREE morning heads-up body, or ``None`` when there are none.

    One line per fixture, stating BOTH pre-match alert times (the two-alert
    model): ``*Home (H) v Away (A)* — KO HH:MM · 🔮 prediction at HH:MM,
    confirmed-lineup update ~HH:MM``. Times are the **Africa/Nairobi wall
    clock** (EAT). The early prediction time is
    ``kickoff - early_alert_lead_minutes(competition)`` and the confirmed-lineup
    time is ``kickoff - lineup_confirm_lead_minutes()`` — the SAME leads the
    scheduler uses to fire the two alerts, so the stated times are the real
    ones. Deliberately carries NO probabilities/edge/xG: the prediction can
    still change once the confirmed XI is out.
    """
    if not fixtures:
        return None
    tz = ZoneInfo(REPORT_TZ)
    lines = [f"*⚽ Today's fixtures — {day.isoformat()}*", ""]
    for f in fixtures:
        ko = f.kickoff
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        code = getattr(f, "competition_code", None)
        early_lead = settings.early_alert_lead_minutes(code)
        late_lead = settings.lineup_confirm_lead_minutes()
        ko_local = ko.astimezone(tz)
        early_local = (ko - timedelta(minutes=early_lead)).astimezone(tz)
        late_local = (ko - timedelta(minutes=late_lead)).astimezone(tz)
        lines.append(
            f"*{f.home_team} (H) v {f.away_team} (A)* — "
            f"KO {ko_local:%H:%M} · 🔮 prediction at {early_local:%H:%M}, "
            f"confirmed-lineup update ~{late_local:%H:%M}"
        )
    lines.append("")
    lines.append("_Times EAT. An early model prediction is sent per match, then "
                 "a confirmed-XI update once the lineup is out._")
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

    send = send_fn or send_telegram_to
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

    sent = 0
    for uid in broadcast_chat_ids(settings, users_fn()):
        try:
            if await send(settings, uid, body):
                sent += 1
        except Exception as e:  # noqa: BLE001 — one bad send must not drop the rest
            log.warning(
                "matchday_notice_send_failed",
                telegram_user_id=uid, error=str(e),
            )

    log.info(
        "matchday_notice_sent",
        day=day.isoformat(), fixtures=len(fixtures), delivered=sent,
    )
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
) -> tuple[str, list[tuple[int, bool]]]:
    """One user's gated body for a SINGLE fixture's lineup-adjusted prediction.

    Same entitlement + reveal-ledger semantics as
    :func:`render_user_predictions` (operator/trial free & recorded, payer
    charged once per NEW fixture, locked -> teaser), but the revealed body
    carries the confirmed XIs via :func:`format_prediction_with_lineup`. Pure —
    no DB writes; the caller commits reveals only after a confirmed send.
    """
    if edge_threshold is None:
        edge_threshold = settings.edge_threshold
    ent = entitlement_fn(user, settings, now=now)
    header = _entitlement_header(ent)
    fid = pred.fixture_id

    def _revealed_body() -> str:
        return format_prediction_with_lineup(
            pred, lineup, edge_threshold=edge_threshold,
            adj_note=adj_note, absences=absences,
        )

    # Already paid for on a prior path/repeat — always free, never re-charged.
    if already_revealed_fn(user.telegram_user_id, fid):
        return header + "\n\n" + _revealed_body(), []
    if ent.reason in ("operator", "trial"):
        return header + "\n\n" + _revealed_body(), [(fid, False)]
    if ent.reason == "credit" and ent.credits_remaining >= 1:
        return header + "\n\n" + _revealed_body(), [(fid, True)]
    # Locked: teaser only, nothing revealed or charged.
    return header + "\n\n" + format_locked(pred), []


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

    send = send_fn or send_telegram_to
    baseline = prediction_fn(fixture_id)
    if baseline is None:
        log.info("prematch_alert_no_prediction", fixture_id=fixture_id)
        return 0

    # --- confirmed lineup + adjustments (default: production lineup service) ---
    lineup = None
    home_adj = away_adj = 0.0
    absences: str | None = None
    if lineup_fn is None:
        lineup_fn = _default_lineup_fn(settings)
    try:
        lineup, home_adj, away_adj, absences = await lineup_fn(baseline)
    except Exception as e:  # noqa: BLE001 — lineup data is best-effort
        log.warning("prematch_lineup_failed", fixture_id=fixture_id, error=str(e))

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

    sent = 0
    for user in users_fn():
        text, reveals = render_user_lineup_prediction(
            user, pred, lineup, settings, now=now,
            adj_note=adj_note, absences=absences,
            entitlement_fn=entitlement_fn,
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
    """
    from betbot.notify import send_telegram_to
    from betbot.storage.repos import (
        mark_result_notified,
        outcomes_pending_result_alert,
    )
    from betbot.tips import format_result

    send = send_fn or send_telegram_to
    outcomes_fn = outcomes_fn or outcomes_pending_result_alert
    mark_notified_fn = mark_notified_fn or mark_result_notified

    pending = list(outcomes_fn())
    if not pending:
        return 0

    users = users_fn()
    operator_id = settings.telegram_allowed_user_id
    sent = 0
    for row in pending:
        pred = prediction_fn(row.fixture_id)
        home = pred.home_team if pred is not None else "Home"
        away = pred.away_team if pred is not None else "Away"
        body = "*⚽ Result*\n\n" + format_result(row, home, away)

        # Audience: the operator (always) + every user who saw this prediction.
        audience: list[int] = []
        if operator_id:
            audience.append(operator_id)
        for u in users:
            if u.telegram_user_id in audience:
                continue
            if already_revealed_fn(u.telegram_user_id, row.fixture_id):
                audience.append(u.telegram_user_id)

        for uid in audience:
            try:
                if await send(settings, uid, body):
                    sent += 1
            except Exception as e:  # noqa: BLE001 — one bad send mustn't drop the rest
                log.warning(
                    "result_alert_send_failed",
                    telegram_user_id=uid, fixture_id=row.fixture_id, error=str(e),
                )
        # Flag AFTER attempting the whole audience so a fixture is broadcast once.
        mark_notified_fn(row.fixture_id)
        log.info(
            "result_alert_sent", fixture_id=row.fixture_id, delivered=len(audience),
        )
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
