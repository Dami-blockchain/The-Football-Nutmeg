"""Read-only command subset in approved group chats.

Invariant under test: /record and /title may run in an APPROVED group and show
only public aggregate state, while every money/entitlement/reveal/LLM surface
stays DM-only. No group path may register a user, create a PredictionReveal,
charge, or consume a free draw; an unapproved group gets nothing.
"""

from __future__ import annotations

import asyncio
import datetime
from types import SimpleNamespace

from sqlalchemy import func, select

import betbot.telegram_bot as tb
from betbot.config import Settings
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionReveal

FAKE_TOKEN = "123456:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEF"
BROADCAST_ID = -1002222222222
OTHER_GROUP = -1003333333333


# ---------------------------------------------------------------- config
def test_group_command_ids_default_to_broadcast_only():
    s = Settings(TELEGRAM_BOT_TOKEN=FAKE_TOKEN, BETBOT_BROADCAST_CHAT_ID=BROADCAST_ID)
    assert s.group_command_chat_ids == {BROADCAST_ID}


def test_group_command_ids_empty_when_nothing_set():
    s = Settings(TELEGRAM_BOT_TOKEN=FAKE_TOKEN)
    assert s.group_command_chat_ids == set()


def test_explicit_list_overrides_broadcast_default():
    s = Settings(
        TELEGRAM_BOT_TOKEN=FAKE_TOKEN,
        BETBOT_BROADCAST_CHAT_ID=BROADCAST_ID,
        BETBOT_GROUP_COMMAND_CHAT_IDS=str(OTHER_GROUP),
    )
    # explicit list wins; broadcast group is NOT auto-included -> can disable it
    assert s.group_command_chat_ids == {OTHER_GROUP}


# ---------------------------------------------------------------- filters
def _tg_update(chat_type: str, chat_id: int, text: str):
    from telegram import Chat, Message, MessageEntity, Update
    from telegram import User as TgUser

    chat = Chat(id=chat_id, type=chat_type)
    user = TgUser(id=999, is_bot=False, first_name="X")
    # A CommandHandler only matches when a BOT_COMMAND entity sits at offset 0.
    entities = ()
    if text.startswith("/"):
        cmd_len = len(text.split()[0])
        entities = (MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=cmd_len),)
    msg = Message(
        message_id=1,
        date=datetime.datetime.now(datetime.timezone.utc),
        chat=chat,
        from_user=user,
        text=text,
        entities=entities,
    )
    return Update(update_id=1, message=msg)


def _app_with_group():
    s = Settings(TELEGRAM_BOT_TOKEN=FAKE_TOKEN, BETBOT_BROADCAST_CHAT_ID=BROADCAST_ID)
    return tb.build_application(s)


def _fires(app, update, command=None):
    """Callbacks that would fire for `update`, computed WITHOUT a bot (the full
    handler.check_update needs one). A CommandHandler fires when its command
    name matches AND its chat filter passes; the free-text MessageHandler fires
    on its filter alone."""
    out = []
    for h in app.handlers[0]:
        cmds = getattr(h, "commands", None)
        if cmds is not None:
            if command in cmds and bool(h.filters.check_update(update)):
                out.append(h.callback)
        elif bool(h.filters.check_update(update)):
            out.append(h.callback)
    return out


def test_record_and_title_accept_the_approved_group():
    app = _app_with_group()
    for cmd, group_cb in (("record", tb.record_group_cmd), ("title", tb.title_group_cmd)):
        upd = _tg_update("supergroup", BROADCAST_ID, f"/{cmd}")
        fires = _fires(app, upd, cmd)
        assert group_cb in fires, f"/{cmd} must be served in the approved group"
        # and it is the GROUP handler, not the private one
        assert tb.record_cmd not in fires and tb.title_cmd not in fires


def test_approved_group_command_rejected_in_unapproved_group():
    app = _app_with_group()
    for cmd in ("record", "title"):
        upd = _tg_update("supergroup", OTHER_GROUP, f"/{cmd}")
        assert _fires(app, upd, cmd) == [], \
            f"/{cmd} must not fire in an unapproved group"


def test_dm_only_commands_never_fire_in_the_approved_group():
    app = _app_with_group()
    for cmd in ("predictions", "balance", "status", "start", "help", "guide"):
        upd = _tg_update("supergroup", BROADCAST_ID, f"/{cmd}")
        assert _fires(app, upd, cmd) == [], \
            f"/{cmd} must be DM-only even in an approved group"


def test_free_text_llm_never_fires_in_a_group():
    app = _app_with_group()
    upd = _tg_update("supergroup", BROADCAST_ID, "who wins tonight?")
    assert tb.chat_handler not in _fires(app, upd, command=None)


def test_private_record_and_title_still_serve_dms():
    app = _app_with_group()
    for cmd, priv_cb in (("record", tb.record_cmd), ("title", tb.title_cmd)):
        upd = _tg_update("private", 999, f"/{cmd}")
        assert priv_cb in _fires(app, upd, cmd)


def test_default_settings_register_no_group_handlers():
    s = Settings(TELEGRAM_BOT_TOKEN=FAKE_TOKEN)  # no broadcast, no group ids
    app = tb.build_application(s)
    cbs = [h.callback for h in app.handlers[0]]
    assert tb.record_group_cmd not in cbs
    assert tb.title_group_cmd not in cbs
    # and the existing invariant holds: no handler fires for a group update
    upd = _tg_update("supergroup", BROADCAST_ID, "/record")
    assert _fires(app, upd, "record") == []


# ---------------------------------------------------------------- behaviour
class _Msg:
    def __init__(self):
        self.sent: list[str] = []

    async def reply_text(self, text, **_kw):
        self.sent.append(text)


def _fake_update(chat_id: int, chat_type: str = "supergroup"):
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_user=SimpleNamespace(id=999, full_name="X", username="x"),
        message=_Msg(),
    )


def _ctx():
    return SimpleNamespace(args=[])


def _approve(monkeypatch, tmp_path):
    """Init an empty DB and point tb.get_settings at a Settings whose only
    approved group is BROADCAST_ID."""
    init_engine(tmp_path / "gc.sqlite")
    tb._group_cmd_last.clear()
    s = Settings(TELEGRAM_BOT_TOKEN=FAKE_TOKEN, BETBOT_BROADCAST_CHAT_ID=BROADCAST_ID)
    assert s.group_command_chat_ids == {BROADCAST_ID}
    monkeypatch.setattr(tb, "get_settings", lambda: s)
    return s


def _count_reveals() -> int:
    with session_scope() as sess:
        return sess.scalar(select(func.count()).select_from(PredictionReveal)) or 0


def test_record_group_replies_with_no_side_effects(tmp_path, monkeypatch):
    _approve(monkeypatch, tmp_path)
    # any registration attempt on a group path is a bug
    monkeypatch.setattr(tb, "_register",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("registered!")))
    before = _count_reveals()
    upd = _fake_update(BROADCAST_ID)
    asyncio.run(tb.record_group_cmd(upd, _ctx()))
    assert upd.message.sent, "an approved group must get a /record reply"
    assert "Track record" in upd.message.sent[0]
    assert _count_reveals() == before == 0
    assert tb.get_user(999) is None  # no wallet created for the group member


def test_title_group_replies_and_never_registers(tmp_path, monkeypatch):
    _approve(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(tb, "_register", lambda *a, **k: calls.append(1))
    upd = _fake_update(BROADCAST_ID)
    asyncio.run(tb.title_group_cmd(upd, _ctx()))
    assert upd.message.sent, "an approved group must get a /title reply"
    assert calls == [], "/title in a group must NOT register the caller"
    assert _count_reveals() == 0
    assert tb.get_user(999) is None


def test_unapproved_group_gets_nothing(tmp_path, monkeypatch):
    _approve(monkeypatch, tmp_path)  # approves only BROADCAST_ID
    upd = _fake_update(OTHER_GROUP)
    asyncio.run(tb.record_group_cmd(upd, _ctx()))
    asyncio.run(tb.title_group_cmd(upd, _ctx()))
    assert upd.message.sent == [], "an unapproved group must receive no reply"


def test_private_chat_rejected_by_the_group_guard(tmp_path, monkeypatch):
    _approve(monkeypatch, tmp_path)
    upd = _fake_update(999, chat_type="private")
    asyncio.run(tb.record_group_cmd(upd, _ctx()))
    assert upd.message.sent == []


def test_group_command_debounced_per_chat(tmp_path, monkeypatch):
    _approve(monkeypatch, tmp_path)
    upd1 = _fake_update(BROADCAST_ID)
    upd2 = _fake_update(BROADCAST_ID)
    asyncio.run(tb.record_group_cmd(upd1, _ctx()))
    asyncio.run(tb.record_group_cmd(upd2, _ctx()))
    assert upd1.message.sent, "first call in a chat replies"
    assert upd2.message.sent == [], "a rapid second call in the same chat is debounced"


# -------------------------------------------------- auth ordering (open reg OFF)
def _settings_open_reg_off(tmp_path):
    return Settings(
        TELEGRAM_BOT_TOKEN=FAKE_TOKEN,
        BETBOT_BROADCAST_CHAT_ID=BROADCAST_ID,
        TELEGRAM_OPEN_REGISTRATION=False,
        BETBOT_WALLET_KEYFILE=str(tmp_path / "secrets" / "agent_wallet.key"),
    )


def _count_users() -> int:
    from betbot.storage.models import User
    with session_scope() as sess:
        return sess.scalar(select(func.count()).select_from(User)) or 0


def test_group_title_never_reads_per_user_state_even_with_open_reg_off(tmp_path, monkeypatch):
    """The security bug (FIX 2): with @_authed on the shared body, an
    approved-group /title from an unregistered member ran _allowed -> get_user
    and could post 'Not authorized' INTO the group, disclosing that member's
    registration status. The group path must NEVER touch per-user state; it
    serves the public projection to anyone in an approved group."""
    init_engine(tmp_path / "gc.sqlite")
    tb._group_cmd_last.clear()
    s = _settings_open_reg_off(tmp_path)
    monkeypatch.setattr(tb, "get_settings", lambda: s)
    calls: list = []
    monkeypatch.setattr(tb, "get_user", lambda *a, **k: calls.append(1))
    upd = _fake_update(BROADCAST_ID)
    asyncio.run(tb.title_group_cmd(upd, _ctx()))
    assert calls == [], "the group path must never call get_user (the disclosure vector)"
    assert upd.message.sent, "an approved group still gets the public projection"
    assert "Not authorized" not in upd.message.sent[0]
    assert _count_users() == 0


def test_private_title_auth_precedes_register_with_open_reg_off(tmp_path, monkeypatch):
    """Regression (FIX 2): title_cmd must auth FIRST. With open registration off,
    an unauthorized DM /title must be refused WITHOUT creating a User row or a
    wallet keyfile first."""
    init_engine(tmp_path / "priv.sqlite")
    s = _settings_open_reg_off(tmp_path)
    monkeypatch.setattr(tb, "get_settings", lambda: s)
    upd = _fake_update(999, chat_type="private")
    asyncio.run(tb.title_cmd(upd, _ctx()))
    assert _count_users() == 0, "auth must precede register: no user row for an unauthorized DM"
    assert upd.message.sent and "Not authorized" in upd.message.sent[0]
