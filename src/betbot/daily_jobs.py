"""Scheduled daily Telegram job: the 21:00 daily report.

The job fires on the **Africa/Nairobi wall clock** via APScheduler
``CronTrigger(timezone="Africa/Nairobi")`` — a named zone, NOT a hardcoded
UTC offset, so "exactly 9pm" stays true to the operator's clock even if the
zone's rules ever change. The hour is overridable via
``BETBOT_DAILY_REPORT_HOUR``.

Side effects (balance reads, Telegram sends) are injected as callables so the
job is unit-testable with fixture data and no network. ``betbot.wallet``
(which drags in web3) is only imported inside the default balance collector,
keeping this module importable on the base install — the same convention
:mod:`betbot.main` follows.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Sequence
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from betbot.logging import get_logger
from betbot.reports import (
    BalanceLine,
    BetLine,
    DailyReport,
    format_daily_report,
    format_user_daily_report,
)
from betbot.storage.repos import (
    cumulative_realized_pnl_usd,
    list_bets_created_between,
    list_bets_settled_between,
    list_users,
)

log = get_logger(__name__)

REPORT_TZ = "Africa/Nairobi"

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


# ----------------------------------------------------------------------
# 21:00 — full daily report
# ----------------------------------------------------------------------
def _bet_line(
    bet, home: str, away: str,
    home_xg: float | None = None, away_xg: float | None = None,
) -> BetLine:
    return BetLine(
        match=f"{home} v {away}",
        outcome=bet.outcome,
        stake_usd=bet.stake_usd,
        market_price=bet.market_price,
        settled_outcome=bet.settled_outcome,
        pnl_usd=bet.pnl_usd,
        home_xg=home_xg,
        away_xg=away_xg,
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
        realised_today_usd=float(sum(row[0].pnl_usd or 0.0 for row in settled)),
        realised_cumulative_usd=cumulative_realized_pnl_usd(),
        balances=tuple(balances),
    )


async def run_daily_report(
    settings,
    *,
    balances_fn: Callable[[], Sequence[BalanceLine]] | None = None,
    send_fn: SendFn | None = None,
    now: datetime | None = None,
) -> int:
    """Build + send the daily report, scoped per recipient.

    Privacy boundary (the bot is public — anyone can register): the FULL
    report (every user's name + balances, the agent wallet, cumulative P&L)
    goes ONLY to the operator chat id. Each registered user receives a
    scoped report containing the day's shared trading activity and ONLY
    their own wallet's balances. Returns messages delivered.
    """
    from betbot.notify import send_telegram_to

    send = send_fn or send_telegram_to
    report = collect_daily_report(settings, balances_fn=balances_fn, now=now)
    sent = 0

    operator_id = settings.telegram_allowed_user_id
    if operator_id:
        if await send(settings, operator_id, format_daily_report(report)):
            sent += 1
    else:
        log.warning(
            "daily_report_no_operator",
            note="TELEGRAM_ALLOWED_USER_ID unset — full report has no recipient",
        )

    by_address = {ln.address: ln for ln in report.balances}
    for user in list_users():
        if user.telegram_user_id == operator_id:
            continue  # the operator already got the full report
        text = format_user_daily_report(
            report, by_address.get(user.wallet_address)
        )
        if await send(settings, user.telegram_user_id, text):
            sent += 1

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
def register_daily_jobs(scheduler, settings, *, daily_report) -> None:
    """Add the Nairobi-local daily-report cron job to the daemon's scheduler.

    The job callable is passed in (rather than imported) so the daemon can
    wrap it in its own never-crash error handling.
    """
    if settings.daily_report_enabled:
        scheduler.add_job(
            daily_report,
            trigger=CronTrigger(
                hour=settings.daily_report_hour, minute=0, timezone=REPORT_TZ
            ),
            id="daily_report",
        )
