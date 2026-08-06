"""Tipster Telegram jobs — gated reveals, matchday + kickoff alerts, cron.

No network: entitlement + Telegram sends are injected fakes; storage runs
against a throwaway SQLite file. The reports.py daily-report formatters are
still exercised by test_reports_format below (kept as regression cover — those
formatters are unused by the daemon now but remain importable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pytest
from apscheduler.triggers.cron import CronTrigger

from betbot.config import Settings
from betbot.daily_jobs import (
    broadcast_chat_ids,
    nairobi_day_bounds,
    register_daily_jobs,
    render_user_predictions,
    run_matchday_alert,
    send_fixture_alert,
)
from betbot.entitlement import Entitlement
from betbot.storage.db import init_engine


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
        BETBOT_EDGE_THRESHOLD=0.05,
    )


# ----------------------------------------------------------------------
# Fixtures / fakes
# ----------------------------------------------------------------------
@dataclass
class _Bet:
    outcome: str
    market_price: float | None
    edge: float | None


@dataclass
class _Pred:
    fixture_id: int = 1
    home_team: str = "Man City"
    away_team: str = "Arsenal"
    p_home: float = 0.39
    p_draw: float = 0.31
    p_away: float = 0.30
    home_xg: float | None = 1.44
    away_xg: float | None = 1.17
    kickoff: datetime = datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc)
    paper_bets: list = field(default_factory=list)


@dataclass
class _User:
    telegram_user_id: int
    wallet_address: str = "0xabc"
    predictions_consumed: int = 0
    created_at: datetime = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _ent(reason, *, trial=0, credits=0):
    allowed = reason in ("operator", "trial", "credit")
    return Entitlement(allowed=allowed, reason=reason,
                       trial_days_left=trial, credits_remaining=credits)


# ----------------------------------------------------------------------
# render_user_predictions — the gating heart
# ----------------------------------------------------------------------
def test_operator_reveals_all():
    preds = [_Pred(fixture_id=1), _Pred(fixture_id=2, home_team="Spurs")]
    text, revealed = render_user_predictions(
        _User(111), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("operator"),
    )
    assert revealed == 2
    assert "operator" in text
    assert "🔒" not in text


def test_trial_reveals_all():
    preds = [_Pred(fixture_id=1), _Pred(fixture_id=2)]
    text, revealed = render_user_predictions(
        _User(5), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("trial", trial=3),
    )
    assert revealed == 2
    assert "3 days left" in text
    assert "🔒" not in text


def test_credits_reveal_then_lock():
    preds = [_Pred(fixture_id=i) for i in range(3)]
    consumed: list[int] = []
    text, revealed = render_user_predictions(
        _User(5), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("credit", credits=2),
        consume_fn=lambda tid: consumed.append(tid),
    )
    assert revealed == 2  # only 2 credits
    assert consumed == [5, 5]  # charged once per reveal
    assert text.count("🔒") == 1  # the third is locked


def test_locked_user_gets_all_teasers():
    preds = [_Pred(fixture_id=1), _Pred(fixture_id=2)]
    consumed: list[int] = []
    text, revealed = render_user_predictions(
        _User(5), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("locked"),
        consume_fn=lambda tid: consumed.append(tid),
    )
    assert revealed == 0
    assert consumed == []
    # One lock in the header + one teaser per prediction (2) = 3.
    assert text.count("🔒") == 3
    assert text.count("send 1 USDC (Polygon) to unlock this prediction") == 2
    assert "%" not in text  # no probabilities leak


def test_no_fixtures_message():
    text, revealed = render_user_predictions(
        _User(5), [], _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("trial", trial=3),
    )
    assert revealed == 0
    assert "No fixtures today" in text


def _tg_settings_stub() -> Settings:
    return Settings(FOOTBALL_DATA_API_KEY="x", BETBOT_EDGE_THRESHOLD=0.05)


# ----------------------------------------------------------------------
# run_matchday_alert
# ----------------------------------------------------------------------
async def test_run_matchday_alert_delivers_per_user(tmp_path):
    s = _tg_settings(tmp_path)
    users = [_User(111), _User(222)]
    preds = [_Pred(fixture_id=1)]
    sent: list[tuple[int, str]] = []

    async def fake_send(settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    def ent_fn(u, settings, now=None):
        return _ent("operator") if u.telegram_user_id == 111 else _ent("trial", trial=2)

    delivered = await run_matchday_alert(
        s, send_fn=fake_send,
        fixtures_fn=lambda a, b: preds,
        entitlement_fn=ent_fn,
        users_fn=lambda: users,
    )
    assert delivered == 2
    assert {cid for cid, _ in sent} == {111, 222}
    assert all("Matchday" in t for _, t in sent)


async def test_matchday_one_bad_send_does_not_drop_others(tmp_path):
    s = _tg_settings(tmp_path)
    users = [_User(111), _User(222)]

    async def fake_send(settings, chat_id, text):
        if chat_id == 111:
            raise RuntimeError("telegram down")
        return True

    delivered = await run_matchday_alert(
        s, send_fn=fake_send,
        fixtures_fn=lambda a, b: [_Pred()],
        entitlement_fn=lambda u, se, now=None: _ent("trial", trial=1),
        users_fn=lambda: users,
    )
    assert delivered == 1  # 222 still got theirs


# ----------------------------------------------------------------------
# send_fixture_alert (kickoff-60m)
# ----------------------------------------------------------------------
async def test_send_fixture_alert_includes_lineup_caveat(tmp_path):
    s = _tg_settings(tmp_path)
    sent: list[str] = []

    async def fake_send(settings, chat_id, text):
        sent.append(text)
        return True

    delivered = await send_fixture_alert(
        s, 1, send_fn=fake_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        entitlement_fn=lambda u, se, now=None: _ent("trial", trial=1),
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 1
    assert "lineup-confirmed data unavailable" in sent[0]
    assert "Kickoff soon" in sent[0]


async def test_send_fixture_alert_no_prediction_is_noop(tmp_path):
    s = _tg_settings(tmp_path)
    delivered = await send_fixture_alert(
        s, 999, send_fn=None,
        prediction_fn=lambda fid: None,
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 0


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


def test_matchday_cron_registered_with_nairobi_timezone(settings):
    sched = FakeScheduler()
    register_daily_jobs(sched, settings, matchday_alert=_noop)
    assert set(sched.jobs) == {"matchday_alert"}
    trig = sched.jobs["matchday_alert"]
    assert isinstance(trig, CronTrigger)
    assert str(trig.timezone) == "Africa/Nairobi"
    fields = {f.name: str(f) for f in trig.fields}
    assert fields["hour"] == "8"  # default matchday_alert_hour
    assert fields["minute"] == "0"


# ----------------------------------------------------------------------
# Day bounds + broadcast recipients
# ----------------------------------------------------------------------
def test_nairobi_day_bounds_anchor_the_local_calendar_day():
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
# reports.py formatters — still importable (unused by daemon now)
# ----------------------------------------------------------------------
def test_reports_daily_formatter_still_importable():
    from betbot.reports import DailyReport, format_daily_report

    text = format_daily_report(DailyReport(date(2026, 6, 11), (), (), 0.0, 0.0, ()))
    assert "Daily report" in text


# keep timedelta import used (kickoff scheduling reference)
_ = timedelta
