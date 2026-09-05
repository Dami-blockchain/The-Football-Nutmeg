"""Outbound Telegram notifications (proactive bot -> operator messages).

Uses the Bot API directly so any process (the daemon's arb watcher) can push a
message to the operator without running the polling bot. The operator must have
/start-ed the bot once (they have).

Two layers live here:

* :func:`send_telegram_to` — the raw transport. One chat id, one message.
* :func:`notify_operator` — THE way code tells the human something. It picks
  the operator's chat id off settings, rate-limits per message kind so a
  repeating fault can't send 24 identical pushes a day, degrades to plain text
  when Markdown is rejected, and never, ever raises into its caller — a failed
  notification must not take down the daemon job that was trying to report.

The rule this exists to enforce: **silence is not health.** A fault that only
reaches the log file is a fault nobody sees. The bot ran for days with its
pre-match alerts dead and said nothing.
"""

from __future__ import annotations

import re
import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import httpx

from betbot.logging import get_logger
from betbot.timefmt import eat_datetime, eat_time

log = get_logger(__name__)


async def send_telegram(settings, text: str) -> bool:
    """Push a message to the operator (the original single-recipient path)."""
    return await send_telegram_to(settings, settings.telegram_allowed_user_id, text)


async def send_telegram_to(
    settings, chat_id: int, text: str, parse_mode: str | None = "Markdown"
) -> bool:
    """Push a message to one chat id — used to broadcast the daily reports to
    the operator AND every registered user, not just the operator.

    ``parse_mode=None`` sends the text verbatim. Telegram rejects a message
    whose Markdown does not parse (a lone ``_`` in a team name is enough), so
    callers that pass through operator-supplied text can retry unformatted.
    """
    token = settings.telegram_bot_token
    if not token or not chat_id:
        log.warning("telegram_notify_skipped", reason="no bot token or chat id")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 — a failed notification shouldn't crash callers
        # NEVER log str(e) here: httpx renders its errors as
        # "... for url 'https://api.telegram.org/bot<TOKEN>/sendMessage'",
        # so this line would print the bot token on every failure -- and the
        # Markdown->plain retry is TRIGGERED by a 400, so it fires often.
        # Log the SHAPE of the failure, never its text.
        status = getattr(getattr(e, "response", None), "status_code", None)
        log.warning(
            "telegram_notify_failed",
            error_type=type(e).__name__,
            status_code=status,
        )
        return False


#: Telegram embeds the bot token in the URL path, so ANY error text that
#: quotes a request URL carries the credential. On the production path
#: send_telegram_to returns a bool and never raises, so the logs below are
#: already safe -- but an injected send_fn could raise a raw httpx error,
#: and this file exists because that token leaked once. Redact by shape so
#: the class of bug is gone, not just today's instance.
# No \b before the digits: the token appears as "bot<digits>:<secret>" in the
# URL, and a word boundary cannot match between "t" and a digit.
_TOKEN_RE = re.compile(r"\d{6,}:[A-Za-z0-9_-]{20,}")


def _redact(exc: BaseException) -> str:
    """Error text with any bot-token-shaped substring removed."""
    return _TOKEN_RE.sub("<REDACTED>", str(exc))


# --------------------------------------------------------------------------
# High-conviction alert message format (BETBOT_HIGH_CONF_ALERTS_ONLY)
# --------------------------------------------------------------------------
#
# The band statistics below are a fixed table derived from ONE measured
# walk-forward run: top-5 leagues 2022–2026, n = 7,082 settled fixtures, gating
# on the model top-pick probability. They are baked in as constants (not
# recomputed live) so the copy is stable and auditable.
#
# HONESTY: every number here is an ACCURACY KPI — hit rate on short-priced
# favourites. It is NOT edge, NOT +EV, NOT beating the market. On gated subsets
# the measured ROI was −1.3% [−4.3, +1.9]. This table must never be captioned
# as profit. The live-season tally rendered alongside it is computed fresh from
# the settled ledger so a hot streak on a handful of games cannot masquerade as
# the track record.
HIGH_CONF_WALKFORWARD_N = 7082
#: ``threshold -> (keep_fraction, hit_pct, ci_low_pct, ci_high_pct)``. ``n`` per
#: band is derived as ``round(keep_fraction * HIGH_CONF_WALKFORWARD_N)`` so the
#: fixture count shown is always consistent with the keep fraction (at 0.65:
#: round(0.147 * 7082) = 1,041).
HIGH_CONF_BANDS: dict[float, tuple[float, float, float, float]] = {
    0.55: (0.342, 66.4, 64.5, 68.3),
    0.60: (0.234, 70.2, 68.1, 72.3),
    0.65: (0.147, 72.8, 70.1, 75.4),
    0.70: (0.076, 76.5, 73.0, 80.1),
}
#: Below this many settled in-season fixtures the live tally is meaningless and
#: is captioned as such — a hot streak on n=6 is noise, not a track record.
HIGH_CONF_MIN_MEANINGFUL_N = 30


def band_fixture_count(threshold: float) -> int:
    """Walk-forward fixture count for a band = round(keep_fraction * N)."""
    keep, *_ = HIGH_CONF_BANDS[threshold]
    return round(keep * HIGH_CONF_WALKFORWARD_N)


def _band_for(min_p: float) -> tuple[float, tuple[float, float, float, float]]:
    """Return ``(threshold, stats)`` for ``min_p``.

    Exact table match wins; otherwise the largest defined threshold at or below
    ``min_p`` (report the WEAKER, wider band rather than overclaim), and below
    the smallest defined threshold the smallest band.
    """
    if min_p in HIGH_CONF_BANDS:
        return min_p, HIGH_CONF_BANDS[min_p]
    below = [t for t in HIGH_CONF_BANDS if t <= min_p]
    t = max(below) if below else min(HIGH_CONF_BANDS)
    return t, HIGH_CONF_BANDS[t]


def format_band_line(min_p: float, live_tally: tuple[int, int] | None) -> str:
    """One-line band record + honest live-season tally.

    ``live_tally`` is ``(hits, n)`` from
    :func:`betbot.storage.repos.high_conf_band_tally` (club-only, current
    season, World Cup excluded) or ``None`` when it could not be read. The band
    stats come from the fixed :data:`HIGH_CONF_BANDS` table; the live tally is
    captioned "too few to mean anything yet" until it reaches
    :data:`HIGH_CONF_MIN_MEANINGFUL_N`.
    """
    threshold, (_keep, hit, lo, hi) = _band_for(min_p)
    n_wf = band_fixture_count(threshold)
    band = (
        f"Band record: p>={threshold:g} hits {hit:.1f}% "
        f"[{lo:.1f}–{hi:.1f}] on {n_wf:,} walk-forward fixtures (2022–26)."
    )
    if live_tally is None:
        live = "This season live: not available."
    else:
        hits, n = live_tally
        if n == 0:
            live = "This season live: none settled yet."
        elif n < HIGH_CONF_MIN_MEANINGFUL_N:
            live = f"This season live: {hits}/{n} — too few to mean anything yet."
        else:
            live = f"This season live: {hits}/{n} ({hits / n:.0%})."
    return f"{band} {live}"


def _kickoff_eat(pred) -> str:
    """Stored (UTC) kickoff as bare ``HH:MM`` in EAT; '' if absent.

    Bare (no label) because the one caller appends " EAT" itself. Uses the
    shared named-zone helper rather than a hardcoded +3 offset.
    """
    return eat_time(getattr(pred, "kickoff", None), label=False)


def _model_triple_line(pred, top_pick: str, market: str) -> str:
    """``Model: HOME 71% / draw 18% / away 11%   Market: <market>`` with the
    model's top pick's side token upper-cased (home/away designation always
    present)."""
    home_tok = "HOME" if top_pick == "HOME" else "home"
    away_tok = "AWAY" if top_pick == "AWAY" else "away"
    draw_tok = "DRAW" if top_pick == "DRAW" else "draw"
    model = (
        f"Model: {home_tok} {pred.p_home:.0%} / "
        f"{draw_tok} {pred.p_draw:.0%} / {away_tok} {pred.p_away:.0%}"
    )
    return f"{model}   Market: {market}"


def format_high_conf_alert(
    pred,
    settings,
    *,
    market: tuple[str, float, float] | None = None,
    live_tally: tuple[int, int] | None = None,
) -> str:
    """The high-conviction alert body for one fixture.

    Standing format rules: home/away designation on both teams, market
    anchoring, a BOLD bet/no-bet call defaulting to NO BET, and honest band
    stats. ``market`` is ``(side, implied_prob, decimal_price)`` when a quote is
    available; the Polymarket matching path has been dead for weeks, so it is
    normally ``None`` and the Market field says so HONESTLY rather than
    fabricating a price. ``live_tally`` is passed straight to
    :func:`format_band_line`.

    Gates/probabilities are read off the stored triple carried on ``pred``; this
    is display only and performs no gating itself (the caller has already
    decided the fixture clears the bar).
    """
    home, away = pred.home_team, pred.away_team
    code = getattr(pred, "competition_code", "") or ""
    triples = [("HOME", pred.p_home), ("DRAW", pred.p_draw), ("AWAY", pred.p_away)]
    top_pick, _top_p = max(triples, key=lambda kv: kv[1])

    ko = _kickoff_eat(pred)
    header = f"\U0001f3af *HIGH-CONFIDENCE ALERT* — {home} (HOME) v {away} (AWAY)"
    if code:
        header += f", {code}"
    if ko:
        header += f", KO {ko} EAT"

    if market is None:
        market_str = "unavailable (no live quote)"
    else:
        m_side, m_prob, m_price = market
        market_str = f"{m_side} {m_prob:.0%} ({m_price:.2f})"

    # Default NO BET, in BOLD, per the standing rule: the gate is a SELECTION on
    # short-priced favourites (higher hit rate) and is NOT an edge/value claim,
    # so the call defaults to NO BET rather than backing the favourite blind.
    # With no live quote the reason must NOT claim an edge-vs-price comparison
    # that never happened — the Market field already says the price is missing.
    call = (
        "*NO BET* (default; no live price to assess edge)"
        if market is None
        else "*NO BET* (default; edge vs price below threshold)"
    )

    min_p = float(getattr(settings, "high_conf_alert_min_p", 0.65))
    return "\n".join([
        header,
        _model_triple_line(pred, top_pick, market_str),
        call,
        format_band_line(min_p, live_tally),
    ])


# --------------------------------------------------------------------------
# Operator notifications
# --------------------------------------------------------------------------

#: Fallback cooldown for a kind with no explicit entry below. Six hours: long
#: enough that an hourly watchdog on a stuck fault sends 4 messages a day
#: instead of 24, short enough that the operator is re-reminded within a
#: working day. A fault nobody has fixed by tomorrow deserves another ping.
DEFAULT_COOLDOWN_SECONDS: float = 6 * 3600.0

#: Per-kind cooldowns, in seconds. ``0`` means "never suppress".
#:
#: * ``announce`` — a human/agent deliberately flagging a change. Every one is
#:   a distinct intentional statement; suppressing one would be the bug.
#: * ``alert_coverage_gap`` — fired by the HOURLY watchdog. 6h floor.
#: * ``scheduler_jobs_not_awaitable`` — fires once per daemon start. A daily
#:   floor keeps a restart loop from becoming a message loop.
#: * ``kill_switch_tripped`` — re-evaluated on every 2h settlement pass while
#:   the switch stays tripped. Daily floor; the first one is what matters.
#: * ``lineup_gap`` — the dedupe_key carries the fixture id, so three matches
#:   in an evening send three flags while a RETRY of the same fixture stays
#:   quiet. The cooldown must therefore be non-zero (0 disables suppression
#:   entirely and every retry would resend); 6h comfortably spans one fixture.
COOLDOWN_SECONDS: dict[str, float] = {
    "announce": 0.0,
    "alert_coverage_gap": 6 * 3600.0,
    "scheduler_jobs_not_awaitable": 24 * 3600.0,
    "kill_switch_tripped": 24 * 3600.0,
    "lineup_gap": 6 * 3600.0,
    # ``challenger_dual_log_stale`` — a DAILY audit reporting a condition that
    # persists until someone writes code (no challenger writes
    # model_predictions in this build). A 6h floor would turn a true, standing
    # fact into four pushes a day, and an operator who learns to swipe this
    # away is an operator who will swipe away the real staleness alarm it
    # becomes once a challenger is wired. Weekly: loud, not nagging.
    "challenger_dual_log_stale": 7 * 24 * 3600.0,
    # ``telegram_bot_not_polling`` — re-evaluated by the bot's 15-minute
    # heartbeat. Nothing supervises the bot, so this needs a MANUAL restart and
    # will keep firing until the operator acts; 6h keeps it present without
    # sending 96 pushes a day.
    "telegram_bot_not_polling": 6 * 3600.0,
}

#: ``(kind, dedupe_key) -> monotonic timestamp of the last SUCCESSFUL send``.
#:
#: Deliberately in-memory rather than on disk. The daemon is long-lived, so the
#: hourly-watchdog case this exists for is fully covered; and a process that
#: has just restarted SHOULD re-report a fault that is still present, because
#: the restart may well have been the operator's attempted fix.
_last_sent: dict[tuple[str, str], float] = {}


def reset_operator_notify_cooldowns() -> None:
    """Forget every recorded send. For tests and for a deliberate re-arm."""
    _last_sent.clear()


def cooldown_for(kind: str, override: float | None = None) -> float:
    """Cooldown in seconds for ``kind`` (``override`` wins when given)."""
    if override is not None:
        return max(0.0, float(override))
    return COOLDOWN_SECONDS.get(kind, DEFAULT_COOLDOWN_SECONDS)


def _suppressed(kind: str, dedupe_key: str, cooldown: float, now: float) -> float | None:
    """Seconds remaining on the cooldown, or ``None`` if the send may proceed."""
    if cooldown <= 0:
        return None
    last = _last_sent.get((kind, dedupe_key))
    if last is None:
        return None
    elapsed = now - last
    if elapsed >= cooldown:
        return None
    return cooldown - elapsed


async def notify_operator(
    settings,
    text: str,
    *,
    kind: str = "general",
    dedupe_key: str | None = None,
    cooldown_seconds: float | None = None,
    send_fn: Callable[..., Awaitable[bool]] | None = None,
    now: float | None = None,
) -> bool:
    """Tell the operator something. Returns True iff a message was sent.

    ``kind`` groups messages for rate limiting; ``dedupe_key`` defaults to
    ``kind``, so by default a kind is capped at one message per cooldown no
    matter how the wording drifts. Pass a distinct ``dedupe_key`` when two
    messages of the same kind are genuinely different events.

    Guarantees, in order of how much they matter:

    1. **Never raises.** Any exception — transport, formatting, a caller's own
       broken ``send_fn`` — is caught. Callers are daemon jobs.
    2. **Never fails silently.** A send that does not land logs ERROR
       (``operator_notify_failed``). A notifier that swallowed its own failure
       would recreate the exact outage this module was written for.
    3. A Markdown rejection is retried once as plain text before it counts as
       a failure.
    """
    kind = kind or "general"
    key = dedupe_key or kind
    cooldown = cooldown_for(kind, cooldown_seconds)
    stamp = time.monotonic() if now is None else now

    try:
        remaining = _suppressed(kind, key, cooldown, stamp)
        if remaining is not None:
            log.info(
                "operator_notify_suppressed",
                kind=kind,
                dedupe_key=key,
                cooldown_s=int(cooldown),
                retry_in_s=int(remaining),
            )
            return False

        chat_id = getattr(settings, "telegram_allowed_user_id", None)
        if not chat_id:
            log.error(
                "operator_notify_failed",
                kind=kind,
                reason="no operator chat id configured "
                "(TELEGRAM_ALLOWED_USER_ID) — this message reached nobody",
            )
            return False

        # `is not None`, not `or`: an injected sender can be falsy (a callable
        # object that defines __len__/__bool__), and `or` would silently swap it
        # for the real Telegram transport — a test would 'pass' having sent
        # nothing, or worse, having sent for real.
        send = send_fn if send_fn is not None else send_telegram_to
        ok = await send(settings, chat_id, text)
        if not ok:
            # Most likely cause of a rejected-but-reachable send is Markdown
            # that does not parse. Retry verbatim before giving up on the human.
            ok = await _retry_plain(send, settings, chat_id, text, kind)
        if ok:
            _last_sent[(kind, key)] = stamp
            log.info("operator_notified", kind=kind, dedupe_key=key)
            return True
        log.error(
            "operator_notify_failed",
            kind=kind,
            dedupe_key=key,
            reason="telegram send returned false",
            preview=text[:120],
        )
        return False
    except Exception as e:  # noqa: BLE001 — a notifier must not crash its caller
        log.error(
            "operator_notify_failed",
            kind=kind,
            dedupe_key=key,
            error=_redact(e),
            error_type=type(e).__name__,
            preview=text[:120],
        )
        return False


async def _retry_plain(send, settings, chat_id: int, text: str, kind: str) -> bool:
    """Second attempt with no parse_mode. False if it isn't supported/works."""
    try:
        return bool(await send(settings, chat_id, text, None))
    except TypeError:
        # An injected send_fn that only takes three arguments — nothing to
        # retry with, the first attempt was the only attempt.
        return False
    except Exception as e:  # noqa: BLE001
        log.warning("operator_notify_plain_retry_failed", kind=kind, error=_redact(e))
        return False


def notify_operator_sync(settings, text: str, **kwargs) -> bool:
    """Blocking :func:`notify_operator`, for CLI commands and sync callers.

    Refuses (loudly, without raising) if a loop is already running — from
    async code, ``await notify_operator(...)`` instead.
    """
    kind = kwargs.get("kind", "general")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no loop: the normal CLI case
    else:
        log.error(
            "operator_notify_failed",
            kind=kind,
            reason="notify_operator_sync called from a running event loop; "
            "await notify_operator() instead",
        )
        return False
    try:
        return asyncio.run(notify_operator(settings, text, **kwargs))
    except Exception as e:  # noqa: BLE001 — never raise into a caller
        log.error(
            "operator_notify_failed", kind=kind, error=_redact(e),
            error_type=type(e).__name__,
        )
        return False


# --------------------------------------------------------------------------
# Change announcements
# --------------------------------------------------------------------------

NO_ROLLBACK_STATED = "NOT STATED — re-send with a rollback"


def format_change_announcement(
    what: str,
    *,
    rollback: str = "",
    who: str = "",
    when: datetime | None = None,
) -> str:
    """The message an operator/agent sends BEFORE touching anything.

    Standing rule: the operator is flagged on Telegram *before* a change is
    committed or a flag is flipped — never after, never only in a log. The
    rollback is part of the format because "how do we undo this" is the
    question that gets skipped at exactly the wrong moment.
    """
    ts = eat_datetime(when or datetime.now(timezone.utc))
    lines = [
        "*\U0001f6e0 Change announcement*",
        "",
        "*Changing:*",
        (what or "").strip() or "(nothing described — this is a bug)",
        "",
        "*Rollback:*",
        (rollback or "").strip() or NO_ROLLBACK_STATED,
        "",
        f"_Announced {ts}" + (f" by {who}" if who else "") + " — NOT yet applied._",
    ]
    return "\n".join(lines)


def announce_change(
    settings,
    what: str,
    *,
    rollback: str = "",
    who: str = "",
    send_fn: Callable[..., Awaitable[bool]] | None = None,
) -> bool:
    """Format + send a change announcement. Blocking. Never rate-limited."""
    return notify_operator_sync(
        settings,
        format_change_announcement(what, rollback=rollback, who=who),
        kind="announce",
        send_fn=send_fn,
    )
