"""Daily Telegram jobs — golden formats, scheduling, and empty-day report.

No network: the arb scan, balance reads, and Telegram sends are all injected
fakes; storage runs against a throwaway SQLite file.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from apscheduler.triggers.cron import CronTrigger

from betbot.config import Settings
from betbot.daily_jobs import (
    arb_line_from_opportunity,
    broadcast_chat_ids,
    collect_daily_report,
    nairobi_day_bounds,
    register_daily_jobs,
    run_arb_digest,
    run_daily_report,
)
from betbot.data.models import MatchOutcome
from betbot.exchanges.arbitrage import ArbOpportunity, Leg
from betbot.exchanges.base import ExchangeName, MarketRef
from betbot.reports import (
    ArbLine,
    BalanceLine,
    BetLine,
    DailyReport,
    format_arb_digest,
    format_daily_report,
    format_user_daily_report,
)
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    count_arb_opportunities_between,
    get_or_create_user,
    insert_paper_bet,
    list_recent_paper_bets,
    record_settlement,
    upsert_prediction,
)
from betbot.strategy.engine import BetDecision, Outcome, Prediction


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "daily_jobs.sqlite")
    yield


def _tg_settings(tmp_path) -> Settings:
    return Settings(
        FOOTBALL_DATA_API_KEY="fake-test-key",
        TELEGRAM_BOT_TOKEN="fake-token",
        TELEGRAM_ALLOWED_USER_ID=111,
        BETBOT_WALLET_KEYFILE=str(tmp_path / "agent.key"),
    )


def _opportunity() -> ArbOpportunity:
    def leg(outcome: MatchOutcome, venue: ExchangeName, price: float) -> Leg:
        market = MarketRef(venue, "m1", "Arsenal v Chelsea", {})
        return Leg(outcome, venue, price, 100.0, market)

    legs = {
        "HOME": leg(MatchOutcome.HOME, ExchangeName.POLYMARKET, 0.45),
        "DRAW": leg(MatchOutcome.DRAW, ExchangeName.POLYMARKET, 0.28),
        "AWAY": leg(MatchOutcome.AWAY, ExchangeName.LIMITLESS, 0.24),
    }
    return ArbOpportunity("Arsenal", "Chelsea", legs, 0.97, 0.0)


# ----------------------------------------------------------------------
# Golden message formats
# ----------------------------------------------------------------------
def test_arb_digest_golden_format():
    lines = [
        ArbLine(
            match="Arsenal v Chelsea",
            margin=0.03,
            price_sum=0.97,
            legs=(
                "HOME: POLYMARKET @ 0.450",
                "DRAW: POLYMARKET @ 0.280",
                "AWAY: LIMITLESS @ 0.240",
            ),
        )
    ]
    assert format_arb_digest(lines, date(2026, 6, 11)) == (
        "*Arb digest — 2026-06-11*\n"
        "\n"
        "Found *1* opportunity(ies):\n"
        "\n"
        "*Arsenal v Chelsea* — margin *+3.0%* (cost 0.970)\n"
        "  HOME: POLYMARKET @ 0.450\n"
        "  DRAW: POLYMARKET @ 0.280\n"
        "  AWAY: LIMITLESS @ 0.240"
    )


def test_arb_digest_clean_no_arb_message():
    assert format_arb_digest([], date(2026, 6, 11)) == (
        "*Arb digest — 2026-06-11*\n"
        "\n"
        "No arb found today — cross-venue prices never summed below $1."
    )


def test_daily_report_golden_format():
    report = DailyReport(
        day=date(2026, 6, 11),
        trades=(
            BetLine("Arsenal v Chelsea", "HOME", 10.0, 0.45),
            BetLine("Inter v Milan", "DRAW", 10.0, None),
        ),
        settlements=(BetLine("Lyon v Nice", "AWAY", 10.0, 0.40, "AWAY", 15.0),),
        realised_today_usd=15.0,
        realised_cumulative_usd=-4.5,
        arb_count_today=2,
        balances=(
            BalanceLine("agent", "0xabc", 120.0, 55.5),
            BalanceLine("Alice", "0xdef", 10.0, None),  # Base RPC read failed
        ),
    )
    assert format_daily_report(report) == (
        "*Daily report — 2026-06-11*\n"
        "\n"
        "*Trades placed today (2)*\n"
        "```\n"
        "match              out   stake  price\n"
        "Arsenal v Chelsea  HOME  10.00   0.45\n"
        "Inter v Milan      DRAW  10.00      -\n"
        "```\n"
        "*Settled today (1)*\n"
        "```\n"
        "match        out   result     pnl\n"
        "Lyon v Nice  AWAY  AWAY    +15.00\n"
        "```\n"
        "*Realised P&L:* today +15.00 USD | cumulative -4.50 USD\n"
        "*Arb opportunities today:* 2\n"
        "*Balances (USDC)*\n"
        "```\n"
        "owner  polygon   base   total\n"
        "agent   120.00  55.50  175.50\n"
        "Alice    10.00    err   10.00\n"
        "```"
    )


def test_user_daily_report_golden_format():
    """The per-user 21:00 message: shared activity + ONLY the user's wallet.
    No other users, no agent wallet, no cumulative P&L."""
    report = DailyReport(
        day=date(2026, 6, 11),
        trades=(BetLine("Arsenal v Chelsea", "HOME", 10.0, 0.45),),
        settlements=(),
        realised_today_usd=0.0,
        realised_cumulative_usd=-4.5,
        arb_count_today=2,
        balances=(
            BalanceLine("agent", "0xabc", 120.0, 55.5),
            BalanceLine("Alice", "0xdef", 10.0, None),
        ),
    )
    assert format_user_daily_report(report, report.balances[1]) == (
        "*Daily report — 2026-06-11*\n"
        "\n"
        "*Trades placed today (1)*\n"
        "```\n"
        "match              out   stake  price\n"
        "Arsenal v Chelsea  HOME  10.00   0.45\n"
        "```\n"
        "*Settled today:* none\n"
        "*Realised P&L today:* +0.00 USD\n"
        "*Arb opportunities today:* 2\n"
        "*Your balances (USDC)*\n"
        "```\n"
        "owner  polygon  base  total\n"
        "Alice    10.00   err  10.00\n"
        "```"
    )


def test_user_daily_report_without_balance_line():
    report = DailyReport(date(2026, 6, 11), (), (), 0.0, 0.0, 0, ())
    text = format_user_daily_report(report, None)
    assert "*Your balances:* unavailable" in text
    assert "cumulative" not in text


def test_daily_report_empty_day_format():
    report = DailyReport(date(2026, 6, 11), (), (), 0.0, 0.0, 0, ())
    assert format_daily_report(report) == (
        "*Daily report — 2026-06-11*\n"
        "\n"
        "*Trades placed today:* none\n"
        "*Settled today:* none\n"
        "*Realised P&L:* today +0.00 USD | cumulative +0.00 USD\n"
        "*Arb opportunities today:* 0\n"
        "*Balances:* unavailable"
    )


# ----------------------------------------------------------------------
# Scheduler registration
# ----------------------------------------------------------------------
class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, object] = {}

    def add_job(self, func, *, trigger, id):  # noqa: A002 — APScheduler's kwarg
        self.jobs[id] = trigger


async def _noop() -> None:  # pragma: no cover — never fired in tests
    pass


def test_both_cron_jobs_registered_with_nairobi_timezone(settings):
    sched = FakeScheduler()
    register_daily_jobs(sched, settings, arb_digest=_noop, daily_report=_noop)
    assert set(sched.jobs) == {"arb_digest", "daily_report"}
    for job_id, hour in (("arb_digest", 9), ("daily_report", 21)):
        trig = sched.jobs[job_id]
        assert isinstance(trig, CronTrigger)
        assert str(trig.timezone) == "Africa/Nairobi"
        fields = {f.name: str(f) for f in trig.fields}
        assert fields["hour"] == str(hour)
        assert fields["minute"] == "0"


def test_hour_overrides_and_toggles(tmp_path):
    s = Settings(
        FOOTBALL_DATA_API_KEY="fake-test-key",
        BETBOT_ARB_DIGEST_HOUR=7,
        BETBOT_DAILY_REPORT_ENABLED=False,
    )
    sched = FakeScheduler()
    register_daily_jobs(sched, s, arb_digest=_noop, daily_report=_noop)
    assert set(sched.jobs) == {"arb_digest"}
    fields = {f.name: str(f) for f in sched.jobs["arb_digest"].fields}
    assert fields["hour"] == "7"


# ----------------------------------------------------------------------
# Day bounds + broadcast recipients
# ----------------------------------------------------------------------
def test_nairobi_day_bounds_anchor_the_local_calendar_day():
    # 22:30 UTC on June 10 is already 01:30 June 11 in Nairobi (UTC+3).
    now = datetime(2026, 6, 10, 22, 30, tzinfo=timezone.utc)
    start, end, day = nairobi_day_bounds(now)
    assert day == date(2026, 6, 11)
    assert start == datetime(2026, 6, 10, 21, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 6, 11, 21, 0, tzinfo=timezone.utc)


def test_broadcast_includes_operator_and_users_deduplicated(tmp_path):
    class U:
        def __init__(self, tid):
            self.telegram_user_id = tid

    s = _tg_settings(tmp_path)
    ids = broadcast_chat_ids(s, [U(222), U(111), U(333)])
    assert ids == [111, 222, 333]  # operator first, no duplicate 111


# ----------------------------------------------------------------------
# Jobs end-to-end (fakes for scan / balances / send; real storage)
# ----------------------------------------------------------------------
async def test_run_arb_digest_persists_and_broadcasts(db, tmp_path):
    s = _tg_settings(tmp_path)
    get_or_create_user(222, "Alice", secrets_dir=str(tmp_path / "secrets"))
    sent: list[tuple[int, str]] = []

    async def fake_send(settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    async def fake_scan():
        return [_opportunity()]

    delivered = await run_arb_digest(s, fake_scan, send_fn=fake_send)

    assert delivered == 2
    assert [chat_id for chat_id, _ in sent] == [111, 222]
    assert "Arsenal v Chelsea" in sent[0][1]
    assert "*+3.0%*" in sent[0][1]
    start, end, _ = nairobi_day_bounds()
    assert count_arb_opportunities_between(start, end) == 1


def test_arb_line_from_opportunity_orders_legs_home_draw_away():
    line = arb_line_from_opportunity(_opportunity())
    assert line.match == "Arsenal v Chelsea"
    assert line.legs == (
        "HOME: POLYMARKET @ 0.450",
        "DRAW: POLYMARKET @ 0.280",
        "AWAY: LIMITLESS @ 0.240",
    )
    assert line.margin == pytest.approx(0.03)


def _log_settled_bet() -> None:
    """One market bet placed AND settled now (inside today's Nairobi day)."""
    now = datetime.now(timezone.utc)
    pred = Prediction(
        fixture_id=1, competition_code="PL",
        home_team="Arsenal", away_team="Chelsea",
        p_home=0.5, p_draw=0.3, p_away=0.2,
        home_score=2.0, away_score=1.0, draw_score=2.4,
    )
    pred_id = upsert_prediction(pred, kickoff=now)
    decision = BetDecision(
        fixture_id=1, competition_code="PL",
        home_team="Arsenal", away_team="Chelsea",
        outcome=Outcome.HOME, our_probability=0.5, market_price=0.45,
        edge=0.05, stake_usd=10.0, rationale="test",
    )
    assert insert_paper_bet(decision, pred_id)
    bet = list_recent_paper_bets(days=1)[0]
    # win at 0.45: stake * (1/price - 1) ≈ +12.22
    record_settlement(bet.id, "HOME", 12.22, settled_at=now)


async def test_run_daily_report_with_activity(db, tmp_path):
    s = _tg_settings(tmp_path)
    _log_settled_bet()
    balances = [BalanceLine("agent", "0xabc", 100.0, 25.0)]
    sent: list[tuple[int, str]] = []

    async def fake_send(settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    delivered = await run_daily_report(
        s, balances_fn=lambda: balances, send_fn=fake_send
    )

    assert delivered == 1  # operator only — no other registered users
    text = sent[0][1]
    assert "Arsenal v Chelsea" in text
    assert "today +12.22 USD" in text
    assert "cumulative +12.22 USD" in text
    assert "agent   100.00  25.00  125.00" in text


async def test_daily_report_full_version_goes_to_operator_only(db, tmp_path):
    """Privacy boundary of the public bot: each registered user receives ONLY
    their own balances; the multi-user table, agent wallet and cumulative
    P&L go exclusively to the operator chat id."""
    s = _tg_settings(tmp_path)  # operator chat id = 111
    alice = get_or_create_user(222, "Alice", secrets_dir=str(tmp_path / "sec"))
    bob = get_or_create_user(333, "Bob", secrets_dir=str(tmp_path / "sec"))
    balances = [
        BalanceLine("agent", "0xagent", 100.0, 25.0),
        BalanceLine("Alice", alice.wallet_address, 50.0, 0.0),
        BalanceLine("Bob", bob.wallet_address, 7.5, 0.0),
    ]
    sent: dict[int, str] = {}

    async def fake_send(settings, chat_id, text):
        sent[chat_id] = text
        return True

    delivered = await run_daily_report(
        s, balances_fn=lambda: balances, send_fn=fake_send
    )
    assert delivered == 3
    assert set(sent) == {111, 222, 333}

    operator = sent[111]  # full report: everyone + agent + cumulative P&L
    for fragment in ("agent", "Alice", "Bob", "cumulative"):
        assert fragment in operator

    alice_text = sent[222]
    assert "Alice" in alice_text and "50.00" in alice_text
    for leaked in ("Bob", "agent", "cumulative", "100.00", "7.50"):
        assert leaked not in alice_text

    bob_text = sent[333]
    assert "Bob" in bob_text and "7.50" in bob_text
    for leaked in ("Alice", "agent", "cumulative", "100.00", "50.00"):
        assert leaked not in bob_text


async def test_daily_report_operator_user_row_not_double_sent(db, tmp_path):
    """The operator is usually also a registered user — they must get the
    full report exactly once, never a second scoped copy."""
    s = _tg_settings(tmp_path)
    get_or_create_user(111, "Operator", secrets_dir=str(tmp_path / "sec"))
    sent: list[tuple[int, str]] = []

    async def fake_send(settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    delivered = await run_daily_report(
        s,
        balances_fn=lambda: [BalanceLine("agent", "0xabc", 10.0, 0.0)],
        send_fn=fake_send,
    )
    assert delivered == 1
    assert [chat_id for chat_id, _ in sent] == [111]
    assert "cumulative" in sent[0][1]  # the FULL report


def test_collect_daily_report_empty_day(db, tmp_path):
    s = _tg_settings(tmp_path)
    report = collect_daily_report(s, balances_fn=lambda: [])
    assert report.trades == ()
    assert report.settlements == ()
    assert report.realised_today_usd == 0.0
    assert report.realised_cumulative_usd == 0.0
    assert report.arb_count_today == 0
    text = format_daily_report(report)
    assert "*Trades placed today:* none" in text
    assert "*Settled today:* none" in text
