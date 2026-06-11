"""Scheduled daily Telegram jobs: 09:00 arb digest + 21:00 daily report.

Both jobs fire on the **Africa/Nairobi wall clock** via APScheduler
``CronTrigger(timezone="Africa/Nairobi")`` — a named zone, NOT a hardcoded
UTC offset, so "09:00" and "exactly 9pm" stay true to the operator's clock
even if the zone's rules ever change. Hours are overridable via
``BETBOT_ARB_DIGEST_HOUR`` / ``BETBOT_DAILY_REPORT_HOUR``.

Side effects (arb scanning, balance reads, Telegram sends) are injected as
callables so every job is unit-testable with fixture data and no network.
``betbot.wallet`` (which drags in web3) is only imported inside the default
balance collector, keeping this module importable on the base install — the
same convention :mod:`betbot.main` follows.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Sequence
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from betbot.logging import get_logger
from betbot.reports import (
    ArbLine,
    BalanceLine,
    BetLine,
    DailyReport,
    format_arb_digest,
    format_daily_report,
)
from betbot.storage.repos import (
    count_arb_opportunities_between,
    cumulative_realized_pnl_usd,
    list_bets_created_between,
    list_bets_settled_between,
    list_users,
    record_arb_opportunity,
)

log = get_logger(__name__)

REPORT_TZ = "Africa/Nairobi"
_OUTCOME_ORDER = ("HOME", "DRAW", "AWAY")

# (settings, chat_id, text) -> delivered?  Matches notify.send_telegram_to.
SendFn = Callable[[object, int, str], Awaitable[bool]]


def nairobi_day_bounds(
    now: datetime | None = None,
) -> tuple[datetime, datetime, date]:
    """``(start_utc, end_utc, local_date)`` of "today" on the Nairobi clock.

    "Today" in every report means the operator's calendar day, not the UTC
    day — at 21:00 EAT those differ by 3 hours, enough to drop or double-count
    evening trades if we naively used UTC midnight.
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


async def _broadcast(settings, text: str, send_fn: SendFn | None) -> int:
    from betbot.notify import send_telegram_to

    send = send_fn or send_telegram_to
    sent = 0
    for chat_id in broadcast_chat_ids(settings, list_users()):
        if await send(settings, chat_id, text):
            sent += 1
    return sent


# ----------------------------------------------------------------------
# 09:00 — arbitrage digest
# ----------------------------------------------------------------------
def arb_line_from_opportunity(opp) -> ArbLine:
    """Reduce an :class:`ArbOpportunity` to the digest's display line."""
    legs = tuple(
        f"{ov}: {opp.legs[ov].exchange.value} @ {opp.legs[ov].price:.3f}"
        for ov in _OUTCOME_ORDER
        if ov in opp.legs
    )
    return ArbLine(
        match=f"{opp.home_team} v {opp.away_team}",
        margin=opp.margin,
        price_sum=opp.price_sum,
        legs=legs,
    )


def persist_arb_opportunities(opportunities: Sequence, *, scanned_at=None) -> None:
    """Record each hit so the 21:00 report can count today's opportunities."""
    for o in opportunities:
        record_arb_opportunity(
            home_team=o.home_team,
            away_team=o.away_team,
            venues="+".join(sorted({lg.exchange.value for lg in o.legs.values()})),
            margin=o.margin,
            price_sum=o.price_sum,
            scanned_at=scanned_at,
        )


async def run_arb_digest(settings, scan_fn, *, send_fn: SendFn | None = None) -> int:
    """One cross-venue scan + digest broadcast. Returns messages delivered.

    ``scan_fn`` is the injected scanner (the daemon passes the same
    ``_run_arb_scan`` the arb watcher uses); the digest persists every hit
    before formatting so the day's count survives even if Telegram is down.
    """
    opportunities = await scan_fn()
    persist_arb_opportunities(opportunities)
    _, _, day = nairobi_day_bounds()
    text = format_arb_digest(
        [arb_line_from_opportunity(o) for o in opportunities], day
    )
    sent = await _broadcast(settings, text, send_fn)
    log.info("arb_digest_sent", opportunities=len(opportunities), delivered=sent)
    return sent


# ----------------------------------------------------------------------
# 21:00 — full daily report
# ----------------------------------------------------------------------
def _bet_line(bet, home: str, away: str) -> BetLine:
    return BetLine(
        match=f"{home} v {away}",
        outcome=bet.outcome,
        stake_usd=bet.stake_usd,
        market_price=bet.market_price,
        settled_outcome=bet.settled_outcome,
        pnl_usd=bet.pnl_usd,
    )


def collect_balances(settings) -> tuple[BalanceLine, ...]:
    """Agent wallet + every registered user's USDC on Polygon and Base.

    Default (network-touching) balance source — production only. Tests inject
    a ``balances_fn`` into :func:`collect_daily_report` instead. web3 is
    imported lazily here for the same reason as in :mod:`betbot.main`.
    """
    from betbot.wallet import all_balances, get_or_create_address

    lines: list[BalanceLine] = []
    if Path(settings.wallet_keyfile).exists():
        address = get_or_create_address(settings.wallet_keyfile)
        lines.append(_balance_line("agent", address, all_balances(address, settings)))
    seen = {ln.address for ln in lines}
    for u in list_users():
        # The operator's user row maps onto the agent wallet — skip the dup.
        if u.wallet_address in seen:
            continue
        seen.add(u.wallet_address)
        lines.append(
            _balance_line(u.name, u.wallet_address, all_balances(u.wallet_address, settings))
        )
    return tuple(lines)


def _balance_line(owner: str, address: str, balances) -> BalanceLine:
    by_chain = {b.chain: (b.usdc if b.ok else None) for b in balances}
    return BalanceLine(
        owner=owner,
        address=address,
        polygon_usdc=by_chain.get("polygon"),
        base_usdc=by_chain.get("base"),
    )


def collect_daily_report(
    settings,
    *,
    balances_fn: Callable[[], Sequence[BalanceLine]] | None = None,
    now: datetime | None = None,
) -> DailyReport:
    """Gather everything the 21:00 report shows. Pure given its inputs:
    storage queries + the injected balance source, no formatting."""
    start, end, day = nairobi_day_bounds(now)
    settled = list_bets_settled_between(start, end)
    balances = balances_fn() if balances_fn is not None else collect_balances(settings)
    return DailyReport(
        day=day,
        trades=tuple(_bet_line(*row) for row in list_bets_created_between(start, end)),
        settlements=tuple(_bet_line(*row) for row in settled),
        realised_today_usd=float(sum(b.pnl_usd or 0.0 for b, _, _ in settled)),
        realised_cumulative_usd=cumulative_realized_pnl_usd(),
        arb_count_today=count_arb_opportunities_between(start, end),
        balances=tuple(balances),
    )


async def run_daily_report(
    settings,
    *,
    balances_fn: Callable[[], Sequence[BalanceLine]] | None = None,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
) -> int:
    """Build + broadcast the full daily report. Returns messages delivered."""
    report = collect_daily_report(settings, balances_fn=balances_fn, now=now)
    sent = await _broadcast(settings, format_daily_report(report), send_fn)
    log.info(
        "daily_report_sent",
        day=report.day.isoformat(),
        trades=len(report.trades),
        settled=len(report.settlements),
        delivered=sent,
    )
    return sent


# ----------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------
def register_daily_jobs(scheduler, settings, *, arb_digest, daily_report) -> None:
    """Add both Nairobi-local cron jobs to the daemon's scheduler.

    The job callables are passed in (rather than imported) so the daemon can
    wrap them in its own never-crash error handling, mirroring its arb tick.
    """
    if settings.arb_digest_enabled:
        scheduler.add_job(
            arb_digest,
            trigger=CronTrigger(
                hour=settings.arb_digest_hour, minute=0, timezone=REPORT_TZ
            ),
            id="arb_digest",
        )
    if settings.daily_report_enabled:
        scheduler.add_job(
            daily_report,
            trigger=CronTrigger(
                hour=settings.daily_report_hour, minute=0, timezone=REPORT_TZ
            ),
            id="daily_report",
        )
