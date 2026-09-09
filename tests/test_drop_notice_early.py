"""Move the morning drop notice to the EARLY gate (KO-55, KO-70 PL).

The primary trigger is now the EARLY suppression; the late suppression is an
idempotent backstop and the 15-min sweep the final one (all exercised in
tests/test_morning_drop_notice.py, which must stay green). Covered HERE:
  * the RECOVERY line on the LATE high-confidence alert when a fixture that got
    an early drop notice has climbed back above the bar — and its absence on the
    early alert and when no drop notice was sent;
  * exactly one drop notice per fixture across an early send then a late recovery
    (idempotent — no second notice);
  * the safety net: if the early trigger never ran, the sweep still delivers once.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot import daily_jobs
from betbot.daily_jobs import HIGH_CONF_RECOVERY_NOTE, send_prediction_alert
from betbot.storage.repos import mark_morning_drop_notified, record_morning_listing

from tests.test_daily_jobs import _Pred, _User, _ent, _lineup_fn_stub, _rescore_stub, _tg_settings
from tests.test_high_conf_alert import _hc

KO = datetime(2026, 9, 8, 19, 30, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    from betbot.storage.db import init_engine

    init_engine(tmp_path / "drop_early.sqlite")
    yield


def _clearing():
    # A triple that clears 0.65 and is not draw-topped -> high_conf_body built.
    return _Pred(fixture_id=1, p_home=0.72, p_draw=0.18, p_away=0.10, kickoff=KO)


async def _run(settings, *, alert_tag, capture):
    async def fake_send(_s, cid, txt):
        capture.append((cid, txt))
        return True

    clearing = _clearing()
    return await send_prediction_alert(
        settings, 1, send_fn=fake_send, alert_tag=alert_tag,
        prediction_fn=lambda fid: clearing,
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(clearing),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )


def _list_and_drop(fixture_id):
    record_morning_listing(fixture_id, "PL", "Man City", "Arsenal", KO, KO.date().isoformat())
    mark_morning_drop_notified(fixture_id)  # simulate the early drop notice sent


# ----------------------------------------------------------------------
# Recovery line on the LATE alert
# ----------------------------------------------------------------------
async def test_late_alert_carries_recovery_line_when_drop_notified(db, tmp_path):
    s = _hc(_tg_settings(tmp_path)).model_copy(update={"broadcast_chat_id": -1002})
    _list_and_drop(1)  # got an early drop notice, has since recovered
    sent: list[tuple[int, str]] = []

    delivered = await _run(s, alert_tag="late", capture=sent)

    assert delivered >= 1
    bodies = [t for _, t in sent]
    # Both the operator DM and the group broadcast carry the recovery line.
    assert all(HIGH_CONF_RECOVERY_NOTE in b for b in bodies)
    assert any(cid == -1002 for cid, _ in sent)  # group got it too


async def test_late_alert_no_recovery_line_when_not_drop_notified(db, tmp_path):
    s = _hc(_tg_settings(tmp_path))
    # Listed but NOT drop-notified (never dropped) -> no recovery acknowledgement.
    record_morning_listing(1, "PL", "Man City", "Arsenal", KO, KO.date().isoformat())
    sent: list[tuple[int, str]] = []

    await _run(s, alert_tag="late", capture=sent)

    assert sent, "the alert should still fire"
    assert all(HIGH_CONF_RECOVERY_NOTE not in t for _, t in sent)


async def test_early_alert_never_carries_recovery_line(db, tmp_path):
    # Recovery is a LATE-alert acknowledgement only; the early alert never has it
    # (even in the pathological case where a listing were already drop-notified).
    s = _hc(_tg_settings(tmp_path))
    _list_and_drop(1)
    sent: list[tuple[int, str]] = []

    await _run(s, alert_tag="early", capture=sent)

    assert sent
    assert all(HIGH_CONF_RECOVERY_NOTE not in t for _, t in sent)


# ----------------------------------------------------------------------
# Exactly one drop notice: early send then a late recovery
# ----------------------------------------------------------------------
async def test_early_notice_then_recovery_sends_no_second_notice(db, settings):
    # Listed, drops at the EARLY gate -> one notice; recovers by late -> the
    # event hook fires again at the late backstop but finds it already notified.
    from betbot.storage.repos import record_morning_listing as _rec

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    object.__setattr__(settings, "broadcast_chat_id", -1002)
    now = datetime(2026, 9, 8, 18, 35, tzinfo=timezone.utc)  # ~KO-55
    _rec(30, "PL", "Man City", "Arsenal", KO, KO.date().isoformat())

    below = {30: _mk_pred(0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    # EARLY suppression -> event hook (fixture_ids). Sends the one notice.
    n1 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now, fixture_ids=[30],
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: below.get(fid),
    )
    assert n1 == 2 and set(sent) == {999, 111, -1002}

    # LATE backstop fires the hook again (idempotent) -> no second notice.
    sent.clear()
    recovered = {30: _mk_pred(0.71, 0.19, 0.10)}
    n2 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now + timedelta(minutes=45), fixture_ids=[30],
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: recovered.get(fid),
    )
    assert n2 == 0 and sent == []


# ----------------------------------------------------------------------
# Safety net: early trigger never ran -> the sweep still delivers once
# ----------------------------------------------------------------------
async def test_sweep_safety_net_delivers_when_early_never_ran(db, settings):
    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    now = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)  # after late-alert time
    from betbot.storage.repos import record_morning_listing as _rec
    _rec(40, "PL", "Man City", "Arsenal", now - timedelta(minutes=1),
         (now - timedelta(minutes=1)).date().isoformat())
    below = {40: _mk_pred(0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    # No event hook ever fired; the SWEEP (fixture_ids None) catches it.
    n = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now,
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: below.get(fid),
    )
    assert n == 2 and set(sent) == {999, 111}
    # And exactly once: a second sweep is idempotent.
    sent.clear()
    n2 = await daily_jobs.run_morning_drop_notices(
        settings, send_fn=fake_send, now=now + timedelta(minutes=15),
        users_fn=lambda: [_User(111)], prediction_fn=lambda fid: below.get(fid),
    )
    assert n2 == 0 and sent == []


def _mk_pred(ph, pd, pa):
    from types import SimpleNamespace

    return SimpleNamespace(
        home_team="Man City", away_team="Arsenal",
        p_home=ph, p_draw=pd, p_away=pa, competition_code="PL",
    )
