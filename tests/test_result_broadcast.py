"""DEFECT 2 — result alerts must ALSO broadcast to the configured group.

The pre-match high-conf alert broadcasts to ``settings.broadcast_chat_id``; the
RESULT alert never did (it built an operator+revealed-user DM audience only), so
on 2026-09-08 the group got the pre-match call but NOT the full-time result.
These pin the group copy onto ``run_result_alerts`` with the SAME contract as
the pre-match broadcast: one extra copy of the SAME body, no reveal/charge, not
counted in the delivered total, group failure isolated from DM delivery, and
only for fixtures that actually alert (a suppressed fixture broadcasts nothing).
"""

from __future__ import annotations

import pytest

from betbot import daily_jobs
from betbot.storage.repos import record_reveal

from tests.test_outcome_loop import _GatePred, _User, _seed_outcome

BROADCAST_ID = -1003880403502  # the live Nutmeg broadcast supergroup shape


@pytest.fixture
def db(tmp_path):
    from betbot.storage.db import init_engine

    init_engine(tmp_path / "result_broadcast.sqlite")
    yield


def _set(settings, **kw):
    for k, v in kw.items():
        object.__setattr__(settings, k, v)


async def test_result_broadcast_off_by_default(db, settings):
    """broadcast_chat_id unset -> only DMs, byte-identical legacy behaviour."""
    _set(settings, telegram_allowed_user_id=999)
    assert settings.broadcast_chat_id is None
    _seed_outcome(404)
    record_reveal(111, 404, charged=False)

    sent: list[tuple[int, str]] = []

    async def fake_send(s, cid, txt):
        sent.append((cid, txt))
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send, users_fn=lambda: [_User(111)]
    )
    recipients = {cid for cid, _ in sent}
    assert recipients == {999, 111}          # operator + revealed user only
    assert BROADCAST_ID not in recipients    # no group copy when unset
    assert n == len(sent) == 2


async def test_result_broadcast_sends_one_copy_when_configured(db, settings):
    """broadcast set -> exactly ONE extra copy of the SAME body to the group,
    NOT counted in the delivered total."""
    _set(settings, telegram_allowed_user_id=999, broadcast_chat_id=BROADCAST_ID)
    _seed_outcome(404)  # HOME 2-0, correct
    record_reveal(111, 404, charged=False)

    sent: list[tuple[int, str]] = []

    async def fake_send(s, cid, txt):
        sent.append((cid, txt))
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send, users_fn=lambda: [_User(111)]
    )
    group = [txt for cid, txt in sent if cid == BROADCAST_ID]
    dms = [txt for cid, txt in sent if cid != BROADCAST_ID]
    assert len(group) == 1                       # exactly one broadcast copy
    assert {cid for cid, _ in sent if cid != BROADCAST_ID} == {999, 111}
    # Delivered counts DMs only; the broadcast copy is deliberately excluded.
    assert n == 2
    assert len(sent) == 3
    # SAME body as the DM (no reveal/charge markup difference).
    assert group[0] == dms[0]
    assert "Result" in group[0]
    assert "Full time" in group[0]


async def test_result_broadcast_failure_isolated_from_dms(db, settings):
    """A group send that RAISES must not drop DM delivery, change the delivered
    total, or leave the fixture un-notified for a pointless retry."""
    from betbot.storage.db import session_scope
    from betbot.storage.models import PredictionOutcome

    _set(settings, telegram_allowed_user_id=999, broadcast_chat_id=BROADCAST_ID)
    _seed_outcome(404)
    record_reveal(111, 404, charged=False)

    dm_sent: list[int] = []

    async def flaky_send(s, cid, txt):
        if cid == BROADCAST_ID:
            raise RuntimeError("group send boom")
        dm_sent.append(cid)
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=flaky_send, users_fn=lambda: [_User(111)]
    )
    assert set(dm_sent) == {999, 111}   # DMs unaffected by the group failure
    assert n == 2                        # delivered total unchanged
    with session_scope() as s:
        row = (
            s.query(PredictionOutcome)
            .filter(PredictionOutcome.fixture_id == 404)
            .one()
        )
        assert row.result_notified is True  # flagged off DM success, not group


async def test_suppressed_fixture_broadcasts_nothing(db, settings):
    """Gate ON + below-bar + never-revealed fixture is SUPPRESSED -> the group
    must NOT receive a result for a call it never heard about."""
    _set(
        settings,
        telegram_allowed_user_id=999,
        broadcast_chat_id=BROADCAST_ID,
        high_conf_alerts_only=True,
        high_conf_alert_min_p=0.65,
    )
    _seed_outcome(701)  # not revealed to anyone
    preds = {701: _GatePred("A", "B", 0.30, 0.45, 0.25)}  # DRAW-topped, p<0.65

    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert sent == []          # nothing sent at all
    assert n == 0
    assert BROADCAST_ID not in sent
