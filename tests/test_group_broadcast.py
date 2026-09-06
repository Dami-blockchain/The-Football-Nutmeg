"""TASK F — group chat-id capture + high-confidence broadcast plumbing.

Covers:
  1. the non-private chat-id capture handler (logs id/type/title; ignores
     private chats; registered in its OWN handler group so existing dispatch
     is byte-identical);
  2. the high-confidence alert broadcast: OFF by default (byte-identical),
     sends ONE extra copy when BETBOT_BROADCAST_CHAT_ID is set, and carries NO
     paywall side effects (no reveal row / charge) on the broadcast path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import betbot.telegram_bot as tb
from betbot.storage.db import init_engine

from tests.test_high_conf_alert import _hc, _stateful_pred_fn
from tests.test_daily_jobs import (
    _Pred, _User, _ent, _lineup_fn_stub, _rescore_stub, _tg_settings,
)

BROADCAST_ID = -1001234567890  # supergroup-shaped negative id


# ----------------------------------------------------------------------
# 1. Non-private chat-id capture handler
# ----------------------------------------------------------------------
def _update(chat_type: str, *, chat_id=BROADCAST_ID, title="Nutmeg VIPs"):
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type, title=title)
    )


def test_capture_logs_nonprivate_chat_id_type_title(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(tb.log, "info", lambda evt, **kw: seen.append((evt, kw)))
    asyncio.run(tb.log_group_chat(_update("supergroup"), None))
    assert seen, "a non-private chat must be logged"
    evt, kw = seen[0]
    assert evt == "telegram_group_chat_seen"
    assert kw["chat_id"] == BROADCAST_ID
    assert kw["chat_type"] == "supergroup"
    assert kw["chat_title"] == "Nutmeg VIPs"


def test_capture_ignores_private_chat(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(tb.log, "info", lambda evt, **kw: seen.append((evt, kw)))
    asyncio.run(tb.log_group_chat(_update("private", chat_id=111, title=None), None))
    assert seen == [], "a private chat must NOT be logged by the capture handler"


def test_capture_survives_missing_chat(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(tb.log, "info", lambda evt, **kw: seen.append((evt, kw)))
    asyncio.run(tb.log_group_chat(SimpleNamespace(effective_chat=None), None))
    assert seen == []


def test_capture_handler_registered_in_its_own_group(settings):
    """Registered in group 1 so it runs ALONGSIDE (not instead of) the group-0
    command/chat handlers — existing dispatch is unchanged."""
    settings.telegram_bot_token = "123456:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEF"
    app = tb.build_application(settings)
    group1 = app.handlers.get(1, [])
    assert any(h.callback is tb.log_group_chat for h in group1), \
        "capture handler must live in its own handler group (1)"
    # Group 0 still carries the original command/chat handlers untouched.
    assert app.handlers.get(0), "the original handlers must remain in group 0"


# ----------------------------------------------------------------------
# 2. High-confidence broadcast plumbing
# ----------------------------------------------------------------------
def _clearing_pred_fn():
    # stored clears the bar, rescored still clears -> high_conf_body is built.
    return _stateful_pred_fn(
        _Pred(fixture_id=1, p_home=0.72, p_draw=0.18, p_away=0.10),
        _Pred(fixture_id=1, p_home=0.71, p_draw=0.18, p_away=0.11),
    )


async def _run_alert(s, capture):
    from betbot.daily_jobs import send_prediction_alert

    async def fake_send(_se, cid, txt):
        capture.append((cid, txt))
        return True

    return await send_prediction_alert(
        s, 1, send_fn=fake_send,
        prediction_fn=_clearing_pred_fn(),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )


async def test_broadcast_off_by_default(tmp_path):
    """Unset BETBOT_BROADCAST_CHAT_ID -> only the user DM is sent (byte-identical)."""
    init_engine(tmp_path / "off.sqlite")
    s = _hc(_tg_settings(tmp_path))  # high-conf ON, broadcast unset
    assert s.broadcast_chat_id is None
    capture: list[tuple] = []
    delivered = await _run_alert(s, capture)
    assert delivered == 1
    targets = {cid for cid, _ in capture}
    assert targets == {111}, "no broadcast copy when the setting is unset"


async def test_broadcast_sends_one_copy_when_configured(tmp_path):
    init_engine(tmp_path / "on.sqlite")
    s = _hc(_tg_settings(tmp_path)).model_copy(update={"broadcast_chat_id": BROADCAST_ID})
    capture: list[tuple] = []
    delivered = await _run_alert(s, capture)
    # Delivered counts only user DMs; broadcast is deliberately excluded.
    assert delivered == 1
    bc = [txt for cid, txt in capture if cid == BROADCAST_ID]
    assert len(bc) == 1, "exactly one broadcast copy to the configured chat"
    body = bc[0]
    assert "HIGH-CONFIDENCE" in body           # same rendered high-conf alert
    assert "Pre-match" in body                  # same header as the DM
    # The operator DM is still sent and untouched (primary target).
    assert 111 in {cid for cid, _ in capture}


async def test_broadcast_has_no_reveal_or_charge_side_effects(tmp_path, monkeypatch):
    import betbot.daily_jobs as dj

    init_engine(tmp_path / "nocharge.sqlite")
    s = _hc(_tg_settings(tmp_path)).model_copy(update={"broadcast_chat_id": BROADCAST_ID})

    commits: list[int] = []
    monkeypatch.setattr(
        dj, "commit_reveals",
        lambda user, reveals: commits.append(user.telegram_user_id),
    )
    capture: list[tuple] = []
    await _run_alert(s, capture)

    # Two sends happened (one user DM + one broadcast) ...
    assert len(capture) == 2
    # ... but commit_reveals fired ONLY for the user, never the broadcast chat.
    assert commits == [111]
    assert BROADCAST_ID not in commits


# ----------------------------------------------------------------------
# 3. SECURITY: group-0 command/message handlers are private-chat only
# ----------------------------------------------------------------------
def _tg_update(chat_type: str, chat_id: int, text: str = "/predictions"):
    import datetime
    from telegram import Chat, Message, Update
    from telegram import User as TgUser

    chat = Chat(id=chat_id, type=chat_type)
    user = TgUser(id=999, is_bot=False, first_name="X")
    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.timezone.utc),
        chat=chat,
        from_user=user,
        text=text,
    )
    return Update(update_id=1, message=msg)


def _built_app():
    from betbot.config import Settings

    s = Settings(TELEGRAM_BOT_TOKEN="123456:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEF")
    return tb.build_application(s)


def test_no_group0_handler_accepts_a_group_update():
    """Once the bot is in a group, NO command/message handler may fire — else a
    member's /predictions posts operator reveals into the group and writes rows."""
    app = _built_app()
    grp = _tg_update("supergroup", -1001234567890, "/predictions")
    accepting = [
        getattr(h, "commands", None) or h.callback.__name__
        for h in app.handlers[0]
        if bool(h.filters.check_update(grp))
    ]
    assert accepting == [], f"group-0 handlers must reject group chats, got {accepting}"


def test_predictions_command_private_only():
    app = _built_app()
    pred = next(
        h for h in app.handlers[0]
        if getattr(h, "commands", None) and "predictions" in h.commands
    )
    assert not pred.filters.check_update(_tg_update("supergroup", -100, "/predictions"))
    assert pred.filters.check_update(_tg_update("private", 999, "/predictions"))


def test_free_text_chat_handler_private_only():
    app = _built_app()
    chat_h = next(h for h in app.handlers[0] if getattr(h, "commands", None) is None)
    assert not chat_h.filters.check_update(_tg_update("supergroup", -100, "hello"))
    assert chat_h.filters.check_update(_tg_update("private", 999, "hello"))


def test_capture_handler_still_sees_groups_not_private():
    app = _built_app()
    cap = next(h for h in app.handlers[1] if h.callback is tb.log_group_chat)
    assert cap.filters.check_update(_tg_update("supergroup", -100, "hi"))
    assert not cap.filters.check_update(_tg_update("private", 999, "hi"))


def test_group_predictions_writes_no_reveal_rows(tmp_path):
    """A group /predictions never dispatches, so the reveal ledger stays empty."""
    from sqlalchemy import func, select

    from betbot.storage.db import init_engine, session_scope
    from betbot.storage.models import PredictionReveal

    init_engine(tmp_path / "grp.sqlite")
    app = _built_app()
    grp = _tg_update("supergroup", -1001234567890, "/predictions")
    # No group-0 handler accepts it -> predictions_cmd is never invoked.
    assert not any(h.filters.check_update(grp) for h in app.handlers[0])
    with session_scope() as s:
        n = s.execute(select(func.count()).select_from(PredictionReveal)).scalar_one()
    assert n == 0, "a group message must create no reveal rows"
