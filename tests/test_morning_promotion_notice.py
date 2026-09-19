"""Morning high-confidence list -> "risen above the bar" PROMOTION notice.

The mirror of the drop notice. The morning notice advertises the fixtures that
cleared the 0.65 gate on their early stored triple. A fixture that was NOT named
but has since climbed above the bar would otherwise get a silent call the
operator was never told to expect. ``run_morning_promotion_notices`` closes that
gap: it tells the SAME audience (DMs + group), exactly once, without any paid
side effect — and only after the morning notice has actually run.

No network: Telegram sends are injected fakes; storage is a throwaway SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from betbot import daily_jobs
from betbot.daily_jobs import (
    nairobi_day_bounds,
    render_morning_promotion_notice,
    run_matchday_notice,
    run_morning_drop_notices,
)
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    mark_matchday_notice_ran,
    matchday_notice_ran,
    promotion_notice_sent,
    record_morning_listing,
    record_reveal,
)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "promotion.sqlite")
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
    """Stored-prediction stand-in carrying the H/D/A triple, names + kickoff."""
    fixture_id: int
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    kickoff: datetime
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


NOW = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)


def _mark_ran(now=NOW):
    _start, _end, day = nairobi_day_bounds(now)
    mark_matchday_notice_ran(day.isoformat())


def _above(fid, ko_offset_min=120, home="Man City", away="Arsenal"):
    return _Pred(
        fid, home, away, 0.72, 0.18, 0.10,
        NOW + timedelta(minutes=ko_offset_min),
    )


def _below(fid, ko_offset_min=120, home="Man City", away="Arsenal"):
    return _Pred(
        fid, home, away, 0.55, 0.25, 0.20,
        NOW + timedelta(minutes=ko_offset_min),
    )


# ----------------------------------------------------------------------
# Copy
# ----------------------------------------------------------------------
def test_copy_names_teams_league_and_closes_on_bold_call():
    body = render_morning_promotion_notice(_above(1))
    assert "Man City (H) v Arsenal (A)" in body       # home/away designation
    assert "Premier League" in body                   # league label, not the code
    assert "PL" not in body
    assert "*BET ON — full call incoming.*" in body  # BOLD call
    assert "NOT on this morning's high-confidence list" in body
    assert "above our confidence bar" in body
    # No probabilities / edge leaked (the pick is the paid reveal).
    assert "0." not in body


def test_copy_omits_league_tag_when_code_unknown():
    p = _above(1)
    p.competition_code = None
    body = render_morning_promotion_notice(p)
    assert "·" not in body  # no dangling league separator


# ----------------------------------------------------------------------
# run_matchday_notice records the "ran today" marker
# ----------------------------------------------------------------------
async def test_matchday_notice_records_ran_marker(db, settings):
    _gate_on(settings)
    ko = datetime(2026, 9, 8, 19, 30, tzinfo=timezone.utc)
    fixtures = [_Fixture(1, "Man City", "Arsenal", ko)]

    async def fake_send(s, cid, text):
        return True

    _start, _end, day = nairobi_day_bounds(datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc))
    assert matchday_notice_ran(day.isoformat()) is False
    await run_matchday_notice(
        settings, send_fn=fake_send,
        now=datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc),
        fixtures_source=lambda a, b: fixtures,
        users_fn=lambda: [_User(999)],
    )
    assert matchday_notice_ran(day.isoformat()) is True


async def test_matchday_notice_records_ran_marker_even_with_no_fixtures(db, settings):
    """REQUIRED (Fable): the ran-marker must be stamped even when the day has NO
    stored fixtures (render_matchday_notice returns None and the notice returns
    early). Otherwise a failed/overrunning 08:00 scoring run followed by a
    zero-row 09:00 notice leaves no marker, and fixtures scored later that day
    that clear the bar get alerts while the promotion failsafe stays silent all
    day. FAILS before the fix (marker was stamped after the early return)."""
    _gate_on(settings)
    now = datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc)
    _start, _end, day = nairobi_day_bounds(now)

    async def fake_send(s, cid, text):  # pragma: no cover - nothing to send
        return True

    assert matchday_notice_ran(day.isoformat()) is False
    n = await run_matchday_notice(
        settings, send_fn=fake_send, now=now,
        fixtures_source=lambda a, b: [],   # no stored fixtures for the day
        users_fn=lambda: [_User(999)],
    )
    assert n == 0  # nothing sent
    assert matchday_notice_ran(day.isoformat()) is True  # but it RAN


# ----------------------------------------------------------------------
# Firing / gating
# ----------------------------------------------------------------------
async def test_promotion_fires_once_and_only_once(db, settings):
    _gate_on(settings)
    object.__setattr__(settings, "broadcast_chat_id", -1002)
    _mark_ran()
    preds = {40: _above(40)}
    sent: list[int] = []

    async def fake_send(s, cid, text):
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[40],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 2 and set(sent) == {999, 111, -1002}  # DMs count 2, group extra
    assert promotion_notice_sent(40) is True

    # Second pass: no-op.
    sent.clear()
    n2 = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW + timedelta(minutes=15),
        fixture_ids=[40],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n2 == 0 and sent == []


async def test_does_not_fire_for_fixture_on_morning_list(db, settings):
    _gate_on(settings)
    _mark_ran()
    ko = NOW + timedelta(minutes=120)
    record_morning_listing(41, "PL", "Man City", "Arsenal", ko, ko.date().isoformat())
    preds = {41: _above(41)}  # still above the bar, but it WAS listed
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must not fire
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[41],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert promotion_notice_sent(41) is False


async def test_does_not_fire_before_morning_notice_ran(db, settings):
    _gate_on(settings)
    # Deliberately DO NOT mark the notice as run today.
    preds = {42: _above(42)}
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must not fire
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[42],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert promotion_notice_sent(42) is False


async def test_does_not_fire_after_kickoff(db, settings):
    _gate_on(settings)
    _mark_ran()
    preds = {43: _above(43, ko_offset_min=-1)}  # kickoff just passed
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must not fire
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[43],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert promotion_notice_sent(43) is False


async def test_does_not_fire_when_alert_already_sent(db, settings):
    _gate_on(settings)
    _mark_ran()
    record_reveal(111, 44, charged=False)  # the call already went out
    preds = {44: _above(44)}
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must not fire
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[44],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert promotion_notice_sent(44) is False


async def test_below_bar_fixture_never_promoted(db, settings):
    _gate_on(settings)
    _mark_ran()
    preds = {45: _below(45)}  # never rose above the bar
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must not fire
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[45],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []


# ----------------------------------------------------------------------
# Collision with the recovery note (listed -> dropped -> recovered)
# ----------------------------------------------------------------------
async def test_recovered_listed_fixture_excluded_no_double_message(db, settings):
    """A fixture that was NAMED in the morning notice, dropped (drop notice sent),
    then recovered is owned by the recovery-note path (HIGH_CONF_RECOVERY_NOTE on
    the late alert). Because it WAS on the morning list, the promotion notice must
    never also fire for it."""
    _gate_on(settings)
    _mark_ran()
    ko = NOW + timedelta(minutes=120)
    # It was listed this morning (and, in production, later drop-noticed then rose).
    record_morning_listing(46, "PL", "Man City", "Arsenal", ko, ko.date().isoformat())
    preds = {46: _above(46)}  # recovered above the bar
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover - must not fire
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[46],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert promotion_notice_sent(46) is False


# ----------------------------------------------------------------------
# Promote then demote: no ping-pong
# ----------------------------------------------------------------------
async def test_promote_then_demote_no_drop_notice_ping_pong(db, settings):
    """A promoted fixture (never on the morning list) that later demotes must NOT
    trigger a drop notice — the drop path only reconciles morning listings, and
    this fixture has none — so the operator gets one "call is coming" then silence,
    never a contradictory pair."""
    _gate_on(settings)
    _mark_ran()
    above = {50: _above(50)}
    sent: list[int] = []

    async def fake_send(s, cid, text):
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[50],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: above.get(fid),
    )
    assert n == 2 and promotion_notice_sent(50) is True
    sent.clear()

    # It now demotes. The drop path is scoped to morning listings; fixture 50 has
    # none, so it is a hard no-op — no ping-pong.
    below = {50: _below(50)}
    n_drop = await run_morning_drop_notices(
        settings, send_fn=fake_send, now=NOW + timedelta(minutes=20),
        fixture_ids=[50],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: below.get(fid),
    )
    assert n_drop == 0 and sent == []

    # And the promotion path does not re-fire either (already sent + below bar).
    n2 = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW + timedelta(minutes=20),
        fixture_ids=[50],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: below.get(fid),
    )
    assert n2 == 0 and sent == []


# ----------------------------------------------------------------------
# Sweep mode + audience + robustness
# ----------------------------------------------------------------------
async def test_sweep_mode_scans_todays_predictions(db, settings):
    _gate_on(settings)
    _mark_ran()
    preds = [_above(60), _below(61)]  # 60 promotes, 61 stays down
    by_id = {p.fixture_id: p for p in preds}
    sent: list[int] = []

    async def fake_send(s, cid, text):
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW,  # fixture_ids None -> sweep
        predictions_source=lambda a, b: preds,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: by_id.get(fid),
    )
    assert n == 2 and set(sent) == {999, 111}
    assert promotion_notice_sent(60) is True
    assert promotion_notice_sent(61) is False


async def test_gate_off_short_circuits(db, settings):
    assert settings.high_conf_alerts_only is False
    _mark_ran()
    preds = {62: _above(62)}
    sent: list[int] = []

    async def fake_send(s, cid, text):  # pragma: no cover
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[62],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []
    assert promotion_notice_sent(62) is False


async def test_total_send_failure_leaves_unrecorded_for_retry(db, settings):
    _gate_on(settings)
    _mark_ran()
    preds = {63: _above(63)}

    async def boom(s, cid, text):
        raise RuntimeError("telegram down")

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=boom, now=NOW, fixture_ids=[63],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0
    assert promotion_notice_sent(63) is False  # NOT recorded -> will retry

    sent: list[int] = []

    async def ok(s, cid, text):
        sent.append(cid)
        return True

    n2 = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=ok, now=NOW + timedelta(minutes=15), fixture_ids=[63],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n2 == 2 and set(sent) == {999, 111}
    assert promotion_notice_sent(63) is True


async def test_group_failure_does_not_block_user_dms(db, settings):
    _gate_on(settings)
    object.__setattr__(settings, "broadcast_chat_id", -1002)
    _mark_ran()
    preds = {64: _above(64)}
    delivered: list[int] = []

    async def fake_send(s, cid, text):
        if cid == -1002:
            raise RuntimeError("group send boom")
        delivered.append(cid)
        return True

    n = await daily_jobs.run_morning_promotion_notices(
        settings, send_fn=fake_send, now=NOW, fixture_ids=[64],
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 2 and set(delivered) == {999, 111}
    assert promotion_notice_sent(64) is True  # at least one DM landed -> recorded


async def test_concurrent_hook_and_sweep_do_not_double_send(db, settings):
    """The plan hook (fixture_ids) and the 15-min sweep (no ids) overlap on the
    one daemon loop. The module lock must serialise them so the fixture is
    promotion-noticed exactly ONCE across both surfaces."""
    import asyncio

    _gate_on(settings)
    object.__setattr__(settings, "broadcast_chat_id", -1002)
    _mark_ran()
    preds = [_above(777)]
    by_id = {p.fixture_id: p for p in preds}
    sent: list[int] = []

    async def slow_send(s, cid, text):
        await asyncio.sleep(0.05)
        sent.append(cid)
        return True

    r1, r2 = await asyncio.gather(
        daily_jobs.run_morning_promotion_notices(
            settings, send_fn=slow_send, now=NOW, fixture_ids=[777],
            users_fn=lambda: [_User(111)], prediction_fn=lambda fid: by_id.get(fid),
        ),
        daily_jobs.run_morning_promotion_notices(
            settings, send_fn=slow_send, now=NOW,
            predictions_source=lambda a, b: preds,
            users_fn=lambda: [_User(111)], prediction_fn=lambda fid: by_id.get(fid),
        ),
    )
    assert sorted(sent) == [-1002, 111, 999]  # one delivery per surface
    assert {r1, r2} == {2, 0}
    assert promotion_notice_sent(777) is True
