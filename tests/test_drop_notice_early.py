"""Drop notice at PLAN time (the true drop moment) + a recovery line that keys
off a notice ACTUALLY SENT.

FIX 1: the primary trigger is the PLAN-time suppression (a sub-threshold fixture
gets no alert job, so a fire-time hook never runs for it). The fire-time hook is
now the secondary catch (passed-at-plan, drifted-at-fire); the sweep is the final
backstop. run_morning_drop_notices(fixture_ids=...) is the shared entry point the
plan hook calls, exercised directly here.

FIX 2: the LATE-alert recovery line keys off ``drop_notice_sent_at`` (set only on
a real send), NOT ``drop_notified`` (also set by a HONOURED / ever-revealed
consume) — so a fixture merely consumed never carries a spurious "this was below
our bar in the earlier notice" line. Tested THROUGH the send path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot import daily_jobs
from betbot.daily_jobs import HIGH_CONF_RECOVERY_NOTE, send_prediction_alert
from betbot.storage.repos import (
    morning_listing_drop_notice_sent,
    record_morning_listing,
    record_reveal,
)

from tests.test_daily_jobs import _Pred, _User, _ent, _lineup_fn_stub, _rescore_stub, _tg_settings
from tests.test_high_conf_alert import _hc

KO = datetime(2026, 9, 8, 19, 30, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    from betbot.storage.db import init_engine

    init_engine(tmp_path / "drop_early.sqlite")
    yield


def _clearing():
    # Clears 0.65 and not draw-topped -> a high_conf_body is built -> late-alert
    # recovery-line injection point is reached.
    return _Pred(fixture_id=1, p_home=0.72, p_draw=0.18, p_away=0.10, kickoff=KO)


def _mk_pred(ph, pd, pa, fixture_id=99):
    from types import SimpleNamespace

    return SimpleNamespace(
        fixture_id=fixture_id, home_team="Man City", away_team="Arsenal",
        p_home=ph, p_draw=pd, p_away=pa, competition_code="PL",
    )


# ----------------------------------------------------------------------
# The pure selection helper the plan hook uses (unit-testable extraction)
# ----------------------------------------------------------------------
def test_suppressed_fixture_ids_selects_gate_failures(settings):
    from betbot.main import suppressed_fixture_ids

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    preds = [
        _mk_pred(0.72, 0.18, 0.10, fixture_id=1),  # passes
        _mk_pred(0.60, 0.25, 0.15, fixture_id=2),  # fails: below bar
        _mk_pred(0.20, 0.70, 0.10, fixture_id=3),  # fails: draw-topped
    ]
    assert suppressed_fixture_ids(settings, preds) == [2, 3]


def test_suppressed_fixture_ids_empty_when_gate_off(settings):
    from betbot.main import suppressed_fixture_ids

    object.__setattr__(settings, "high_conf_alerts_only", False)
    preds = [_mk_pred(0.40, 0.35, 0.25, fixture_id=1)]
    assert suppressed_fixture_ids(settings, preds) == []


def _drop_notified(fixture_id):
    from betbot.storage.db import session_scope
    from betbot.storage.models import MorningNoticeListing

    with session_scope() as s:
        row = (
            s.query(MorningNoticeListing)
            .filter(MorningNoticeListing.fixture_id == fixture_id)
            .one_or_none()
        )
        return None if row is None else row.drop_notified


async def _run_alert(settings, *, alert_tag, capture, fixture_id=1):
    async def fake_send(_s, cid, txt):
        capture.append((cid, txt))
        return True

    clearing = _clearing()
    return await send_prediction_alert(
        settings, fixture_id, send_fn=fake_send, alert_tag=alert_tag,
        prediction_fn=lambda fid: clearing,
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(clearing),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )


async def _send_real_drop_notice(settings, fixture_id, *, pred):
    """Drive a REAL drop notice through the send path (stamps drop_notice_sent_at)
    when ``pred`` is below the bar, or a HONOURED consume when it clears."""
    record_morning_listing(fixture_id, "PL", "Man City", "Arsenal", KO, KO.date().isoformat())

    async def sink(_s, cid, txt):
        return True

    return await daily_jobs.run_morning_drop_notices(
        settings, fixture_ids=[fixture_id], send_fn=sink,
        prediction_fn=lambda fid: pred, users_fn=lambda: [_User(111)],
    )


# ----------------------------------------------------------------------
# FIX 2 — recovery line keys off a notice ACTUALLY SENT
# ----------------------------------------------------------------------
async def test_late_alert_carries_recovery_line_when_notice_sent(db, tmp_path):
    s = _hc(_tg_settings(tmp_path)).model_copy(update={"broadcast_chat_id": -1002})
    # A real drop notice went out (fixture dropped below the bar earlier)...
    await _send_real_drop_notice(s, 1, pred=_mk_pred(0.60, 0.25, 0.15))
    assert morning_listing_drop_notice_sent(1) is True

    # ...and it has since recovered by the late fire -> alert carries the line.
    sent: list[tuple[int, str]] = []
    delivered = await _run_alert(s, alert_tag="late", capture=sent)
    assert delivered >= 1
    assert all(HIGH_CONF_RECOVERY_NOTE in t for _, t in sent)
    assert any(cid == -1002 for cid, _ in sent)  # group copy carries it too


async def test_no_recovery_line_when_listing_only_consumed(db, tmp_path):
    # THE FIX 2 regression: gate still clears at reconcile time -> HONOURED
    # consume sets drop_notified but sends NOTHING. The late alert must NOT
    # claim a notice was sent.
    s = _hc(_tg_settings(tmp_path))
    await _send_real_drop_notice(s, 1, pred=_mk_pred(0.72, 0.18, 0.10))  # clears -> consume
    assert _drop_notified(1) is True                     # flag set by consume
    assert morning_listing_drop_notice_sent(1) is False  # but NO notice sent

    sent: list[tuple[int, str]] = []
    await _run_alert(s, alert_tag="late", capture=sent)
    assert sent
    assert all(HIGH_CONF_RECOVERY_NOTE not in t for _, t in sent)


async def test_no_recovery_line_when_consumed_via_ever_revealed(db, tmp_path):
    # ever_revealed consume: a user saw it via /predictions earlier -> the hook
    # consumes (drop_notified True) without sending. No spurious recovery line.
    s = _hc(_tg_settings(tmp_path))
    record_reveal(111, 1, charged=False)
    await _send_real_drop_notice(s, 1, pred=_mk_pred(0.60, 0.25, 0.15))  # below, but revealed
    assert _drop_notified(1) is True
    assert morning_listing_drop_notice_sent(1) is False

    sent: list[tuple[int, str]] = []
    await _run_alert(s, alert_tag="late", capture=sent)
    assert all(HIGH_CONF_RECOVERY_NOTE not in t for _, t in sent)


async def test_late_alert_no_recovery_line_when_never_listed(db, tmp_path):
    s = _hc(_tg_settings(tmp_path))
    sent: list[tuple[int, str]] = []
    await _run_alert(s, alert_tag="late", capture=sent)
    assert sent
    assert all(HIGH_CONF_RECOVERY_NOTE not in t for _, t in sent)


async def test_early_alert_never_carries_recovery_line(db, tmp_path):
    # Recovery is a LATE-alert acknowledgement only.
    s = _hc(_tg_settings(tmp_path))
    await _send_real_drop_notice(s, 1, pred=_mk_pred(0.60, 0.25, 0.15))
    assert morning_listing_drop_notice_sent(1) is True
    sent: list[tuple[int, str]] = []
    await _run_alert(s, alert_tag="early", capture=sent)
    assert sent
    assert all(HIGH_CONF_RECOVERY_NOTE not in t for _, t in sent)


# ----------------------------------------------------------------------
# FIX 1 — plan-time drop notice (the shared fixture_ids entry point) + coherence
# ----------------------------------------------------------------------
async def test_plan_time_hook_sends_one_notice_for_suppressed_fixtures(db, settings):
    # What _schedule_kickoff_alerts_locked now calls after planning: the
    # sub-threshold fixture ids go straight to run_morning_drop_notices.
    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    object.__setattr__(settings, "broadcast_chat_id", -1002)
    record_morning_listing(30, "PL", "Man City", "Arsenal", KO, KO.date().isoformat())
    below = {30: _mk_pred(0.5949, 0.25, 0.1551)}  # today's Sporting-v-Gala drop
    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    n1 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, fixture_ids=[30],
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: below.get(fid),
    )
    assert n1 == 2 and set(sent) == {999, 111, -1002}
    assert morning_listing_drop_notice_sent(30) is True

    # Re-planning (hourly) with the same suppressed set is a no-op — one notice.
    sent.clear()
    n2 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, fixture_ids=[30],
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: below.get(fid),
    )
    assert n2 == 0 and sent == []


async def test_sweep_safety_net_delivers_when_plan_hook_never_ran(db, settings):
    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)  # after late-alert time
    record_morning_listing(40, "PL", "Man City", "Arsenal",
                           now - timedelta(minutes=1),
                           (now - timedelta(minutes=1)).date().isoformat())
    below = {40: _mk_pred(0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: below.get(fid),
    )
    assert n == 2 and set(sent) == {999, 111}
    sent.clear()
    n2 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now + timedelta(minutes=15),
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: below.get(fid),
    )
    assert n2 == 0 and sent == []
