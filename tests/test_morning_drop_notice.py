"""Morning high-confidence list -> "dropped below the bar" reconciliation.

The morning notice advertises fixtures that cleared the 0.65 gate on their early
stored triple. The alert gate is re-evaluated later on the live (rescored) triple
and a drifted fixture is SILENTLY suppressed. ``run_morning_drop_notices`` closes
that gap: it tells the SAME audience (DMs + group) when a NAMED call has dropped
below the bar, exactly once, without any paid side effect.

No network: Telegram sends are injected fakes; storage is a throwaway SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from betbot import daily_jobs
from betbot.daily_jobs import render_morning_drop_notice, run_matchday_notice
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import MorningNoticeListing
from betbot.storage.repos import record_morning_listing, record_reveal


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "morning_drop.sqlite")
    yield


def _gate_on(settings, min_p=0.65, operator=999):
    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", min_p)
    object.__setattr__(settings, "telegram_allowed_user_id", operator)


@dataclass
class _User:
    telegram_user_id: int


@dataclass
class _Pred:
    """Live stored-prediction stand-in carrying the H/D/A triple + names."""
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    competition_code: str = "PL"


@dataclass
class _Fixture:
    fixture_id: int
    home_team: str
    away_team: str
    kickoff: datetime
    competition_code: str = "PL"
    p_home: float = 0.72
    p_draw: float = 0.18
    p_away: float = 0.10


def _list(fixture_id, ko, code="PL", home="Man City", away="Arsenal"):
    record_morning_listing(fixture_id, code, home, away, ko, ko.date().isoformat())


def _drop_notified(fixture_id) -> bool | None:
    with session_scope() as s:
        row = (
            s.query(MorningNoticeListing)
            .filter(MorningNoticeListing.fixture_id == fixture_id)
            .one_or_none()
        )
        return None if row is None else row.drop_notified


# ----------------------------------------------------------------------
# Copy
# ----------------------------------------------------------------------
def test_copy_names_teams_league_and_closes_on_bold_no_bet():
    body = render_morning_drop_notice(
        _Pred("Man City", "Arsenal", 0.6, 0.2, 0.2, competition_code="PL")
    )
    assert "Man City (H) v Arsenal (A)" in body   # home/away designation
    assert "Premier League" in body               # league label, not the raw code
    assert "PL" not in body
    assert "*NO BET.*" in body                    # BOLD no-bet call
    assert "below our confidence bar" in body
    # No probabilities / edge leaked.
    assert "0." not in body


def test_copy_omits_league_tag_when_code_unknown():
    body = render_morning_drop_notice(
        _Pred("A", "B", 0.6, 0.2, 0.2, competition_code=None)
    )
    assert "·" not in body  # no dangling league separator


# ----------------------------------------------------------------------
# Recording: run_matchday_notice persists the NAMED set (gate ON only)
# ----------------------------------------------------------------------
async def test_matchday_notice_records_only_named_high_conf_fixtures(db, settings):
    _gate_on(settings)
    ko = datetime(2026, 9, 8, 19, 30, tzinfo=timezone.utc)
    fixtures = [
        _Fixture(1, "Man City", "Arsenal", ko, p_home=0.72, p_draw=0.18, p_away=0.10),
        _Fixture(2, "Spurs", "Everton", ko, p_home=0.40, p_draw=0.33, p_away=0.27),  # sub-bar
    ]

    async def fake_send(s, cid, text):
        return True

    await run_matchday_notice(
        settings, send_fn=fake_send,
        fixtures_source=lambda a, b: fixtures,
        users_fn=lambda: [_User(999)],
    )
    # Only the qualifying fixture was recorded; the sub-threshold one was not.
    assert _drop_notified(1) is False   # recorded, not yet dropped
    assert _drop_notified(2) is None    # never listed


async def test_matchday_notice_gate_off_records_nothing(db, settings):
    assert settings.high_conf_alerts_only is False
    ko = datetime(2026, 9, 8, 19, 30, tzinfo=timezone.utc)
    fixtures = [_Fixture(3, "Man City", "Arsenal", ko)]

    async def fake_send(s, cid, text):
        return True

    await run_matchday_notice(
        settings, send_fn=fake_send,
        fixtures_source=lambda a, b: fixtures,
        users_fn=lambda: [_User(999)],
    )
    assert _drop_notified(3) is None  # gate off -> no listing rows at all


# ----------------------------------------------------------------------
# Reconciliation
# ----------------------------------------------------------------------
async def test_listed_then_drifted_sends_one_notice_to_both_surfaces(db, settings):
    _gate_on(settings)
    object.__setattr__(settings, "broadcast_chat_id", -1002)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(10, now - timedelta(minutes=1))  # kickoff just passed -> lifecycle over

    # Live stored row has drifted below 0.65, and nobody was ever revealed it.
    preds = {10: _Pred("Man City", "Arsenal", 0.60, 0.25, 0.15)}
    sent: list[tuple[int, str]] = []

    async def fake_send(s, cid, text):
        sent.append((cid, text))
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    # DM count = operator(999) + user(111); group is out of `n` but did receive.
    assert n == 2
    assert {cid for cid, _ in sent} == {999, 111, -1002}
    assert sum(1 for cid, _ in sent if cid == -1002) == 1
    # Same body to every surface (one renderer).
    assert len({t for _, t in sent}) == 1
    assert _drop_notified(10) is True

    # Idempotent: a second tick sends nothing.
    sent.clear()
    n2 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now + timedelta(minutes=15),
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n2 == 0 and sent == []


async def test_listed_and_still_qualifying_no_drop_notice(db, settings):
    _gate_on(settings)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(11, now - timedelta(minutes=1))
    # Live row STILL clears the bar -> the normal alert went out -> consume.
    preds = {11: _Pred("Man City", "Arsenal", 0.72, 0.18, 0.10)}
    sent: list[int] = []

    async def fake_send(s, cid, text):
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert _drop_notified(11) is True  # consumed, never re-queues


async def test_never_listed_subthreshold_fixture_is_silent(db, settings):
    _gate_on(settings)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    # No listing recorded at all for fixture 12.
    preds = {12: _Pred("C", "D", 0.55, 0.25, 0.20)}
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must never fire
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert _drop_notified(12) is None  # was never in the table


async def test_drift_below_then_back_up_and_alerted_no_drop_notice(db, settings):
    """Listed -> below by the early alert -> back ABOVE by confirmed-XI (the late
    alert fired and revealed it). By kickoff the live row may read either way, but
    the reveal proves the promised call went out -> NO drop notice."""
    _gate_on(settings)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(13, now - timedelta(minutes=1))
    record_reveal(111, 13, charged=False)  # the late alert reached a user
    # Live stored row happens to read below the bar now (drifted again post-alert).
    preds = {13: _Pred("Man City", "Arsenal", 0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, cid, text):
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []          # honoured by the prior reveal
    assert _drop_notified(13) is True     # consumed


async def test_total_send_failure_leaves_unnotified_for_retry(db, settings):
    _gate_on(settings)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(14, now - timedelta(minutes=1))
    preds = {14: _Pred("Man City", "Arsenal", 0.60, 0.25, 0.15)}

    async def boom(s, cid, text):
        raise RuntimeError("telegram down")

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=boom, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0
    assert _drop_notified(14) is False  # NOT marked -> will retry

    # Next tick: Telegram recovered -> the notice is delivered and now marked.
    sent: list[int] = []

    async def ok(s, cid, text):
        sent.append(cid)
        return True

    n2 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=ok, now=now + timedelta(minutes=15),
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n2 == 2 and set(sent) == {999, 111}
    assert _drop_notified(14) is True


async def test_group_unset_users_still_notified(db, settings):
    _gate_on(settings)
    assert settings.broadcast_chat_id is None
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(15, now - timedelta(minutes=1))
    preds = {15: _Pred("Man City", "Arsenal", 0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, cid, text):
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 2 and set(sent) == {999, 111}  # no group id, no extra send
    assert _drop_notified(15) is True


async def test_group_failure_does_not_block_user_dms(db, settings):
    _gate_on(settings)
    object.__setattr__(settings, "broadcast_chat_id", -1002)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(16, now - timedelta(minutes=1))
    preds = {16: _Pred("Man City", "Arsenal", 0.60, 0.25, 0.15)}
    delivered: list[int] = []

    async def fake_send(s, cid, text):
        if cid == -1002:
            raise RuntimeError("group send boom")
        delivered.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 2 and set(delivered) == {999, 111}  # user DMs unaffected
    # At least one DM succeeded, so it is marked (group failure alone mustn't retry).
    assert _drop_notified(16) is True


async def test_lifecycle_not_over_is_deferred(db, settings):
    """A listed fixture still BEFORE kickoff must not get a drop notice yet — it
    may still drift back above and alert. It stays pending for a later tick."""
    _gate_on(settings)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(17, now + timedelta(hours=2))  # kickoff in the future
    preds = {17: _Pred("Man City", "Arsenal", 0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must not fire pre-KO
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert _drop_notified(17) is False  # still pending, not consumed


async def test_gate_off_short_circuits(db, settings):
    assert settings.high_conf_alerts_only is False
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
    _list(18, now - timedelta(minutes=1))  # a stray listing
    preds = {18: _Pred("Man City", "Arsenal", 0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert _drop_notified(18) is False  # untouched when the flag is off
