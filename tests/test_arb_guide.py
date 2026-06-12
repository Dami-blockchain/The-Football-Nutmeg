"""Cross-venue arbitrage onboarding: /start button → opt-in → guide, and
/linklimitless credential capture. All telegram objects are faked; no network."""

from __future__ import annotations

import json
import stat

import pytest

import betbot.telegram_bot as tb
from betbot.config import Settings
from betbot.storage.db import init_engine
from betbot.storage.repos import get_user


# ----------------------------------------------------------------------
# Fake telegram plumbing
# ----------------------------------------------------------------------
class _FakeTGUser:
    def __init__(self, user_id: int):
        self.id = user_id
        self.full_name = "New Person"
        self.username = "newperson"


class _FakeTGMessage:
    def __init__(self, text: str = "", chat_id: int = 999):
        self.text = text
        self.chat_id = chat_id
        self.replies: list[str] = []
        self.markups: list[object] = []
        self.deleted = False

    async def reply_text(self, text: str, **kwargs) -> None:
        self.replies.append(text)
        self.markups.append(kwargs.get("reply_markup"))

    async def delete(self) -> None:
        self.deleted = True


class _FakeChat:
    def __init__(self, chat_id: int = 999):
        self.id = chat_id


class _FakeCallbackQuery:
    def __init__(self, user_id: int, chat_id: int = 999):
        self.message = _FakeTGMessage(chat_id=chat_id)
        self.answered = False

    async def answer(self, *a, **k) -> None:
        self.answered = True


class _FakeBot:
    def __init__(self, *, video_raises: bool = False):
        self.messages: list[tuple[int, str]] = []
        self.videos: list[tuple[int, object]] = []
        self._video_raises = video_raises

    async def send_message(self, chat_id, text, **kwargs) -> None:
        self.messages.append((chat_id, text))

    async def send_video(self, chat_id, video=None, **kwargs) -> None:
        if self._video_raises:
            raise RuntimeError("bad video url")
        self.videos.append((chat_id, video))


class _FakeContext:
    def __init__(self, bot, args=None):
        self.bot = bot
        self.args = args


class _FakeUpdateMsg:
    def __init__(self, user_id: int, text: str = "", chat_id: int = 999):
        self.effective_user = _FakeTGUser(user_id)
        self.message = _FakeTGMessage(text, chat_id=chat_id)
        self.effective_chat = _FakeChat(chat_id)
        self.callback_query = None


class _FakeUpdateCallback:
    def __init__(self, user_id: int, chat_id: int = 999):
        self.effective_user = _FakeTGUser(user_id)
        self.message = None
        self.callback_query = _FakeCallbackQuery(user_id, chat_id)


def _settings(**overrides) -> Settings:
    base = dict(TELEGRAM_BOT_TOKEN="123:TESTTOKEN", TELEGRAM_OPEN_REGISTRATION=True)
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def tg_env(tmp_path, monkeypatch):
    init_engine(tmp_path / "bot.sqlite")
    s = _settings(BETBOT_WALLET_KEYFILE=str(tmp_path / "secrets" / "agent.key"))
    monkeypatch.setattr(tb, "get_settings", lambda: s)
    return s, tmp_path


# ----------------------------------------------------------------------
# /start shows the inline button
# ----------------------------------------------------------------------
async def test_start_shows_arb_inline_button(tg_env):
    update = _FakeUpdateMsg(user_id=10)

    await tb.start_cmd(update, ctx=None)

    [markup] = update.message.markups
    assert markup is not None
    button = markup.inline_keyboard[0][0]
    assert button.callback_data == tb.ARB_INTEREST_CB
    assert "arbitrage" in button.text.lower()


# ----------------------------------------------------------------------
# arb_interest callback: flag + explainer + video/fallback + guide
# ----------------------------------------------------------------------
async def test_arb_interest_sets_flag_and_sends_full_written_guide(tg_env):
    # No video URL configured → the detailed written walkthrough IS the guide;
    # no "coming soon" placeholder, no send_video call.
    update = _FakeUpdateCallback(user_id=20)
    bot = _FakeBot()
    ctx = _FakeContext(bot)

    await tb.arb_interest_cb(update, ctx)

    assert update.callback_query.answered is True
    assert get_user(20).arb_interest is True

    texts = [t for _, t in bot.messages]
    # explainer + every walkthrough part, in order.
    assert texts == [tb.ARB_EXPLAINER, *tb.LIMITLESS_GUIDE_PARTS]
    assert bot.videos == []  # no URL → never tries send_video
    # The final part carries the /linklimitless call-to-action.
    assert "/linklimitless" in texts[-1]
    # The walkthrough is genuinely detailed (multi-step).
    assert len(tb.LIMITLESS_GUIDE_PARTS) >= 4


async def test_arb_interest_sends_video_then_full_guide_when_url_set(tmp_path, monkeypatch):
    init_engine(tmp_path / "bot.sqlite")
    s = _settings(
        BETBOT_WALLET_KEYFILE=str(tmp_path / "secrets" / "agent.key"),
        BETBOT_ARB_GUIDE_VIDEO_URL="https://example.com/arb.mp4",
    )
    monkeypatch.setattr(tb, "get_settings", lambda: s)

    update = _FakeUpdateCallback(user_id=21)
    bot = _FakeBot()
    ctx = _FakeContext(bot)

    await tb.arb_interest_cb(update, ctx)

    assert bot.videos == [(update.callback_query.message.chat_id, "https://example.com/arb.mp4")]
    texts = [t for _, t in bot.messages]
    # explainer + the full written walkthrough (video is an extra, not a swap).
    assert texts == [tb.ARB_EXPLAINER, *tb.LIMITLESS_GUIDE_PARTS]


async def test_arb_interest_bad_video_still_sends_full_guide(tmp_path, monkeypatch):
    init_engine(tmp_path / "bot.sqlite")
    s = _settings(
        BETBOT_WALLET_KEYFILE=str(tmp_path / "secrets" / "agent.key"),
        BETBOT_ARB_GUIDE_VIDEO_URL="https://bad/url.mp4",
    )
    monkeypatch.setattr(tb, "get_settings", lambda: s)

    update = _FakeUpdateCallback(user_id=22)
    bot = _FakeBot(video_raises=True)
    ctx = _FakeContext(bot)

    await tb.arb_interest_cb(update, ctx)  # must not raise

    texts = [t for _, t in bot.messages]
    # A broken video URL is swallowed; the written walkthrough still arrives.
    assert texts == [tb.ARB_EXPLAINER, *tb.LIMITLESS_GUIDE_PARTS]
    assert get_user(22).arb_interest is True


# ----------------------------------------------------------------------
# /linklimitless
# ----------------------------------------------------------------------
async def test_linklimitless_stores_creds_securely(tg_env):
    _, tmp_path = tg_env
    update = _FakeUpdateMsg(user_id=30)
    ctx = _FakeContext(_FakeBot(), args=["MYKEY", "MYSECRET"])

    await tb.linklimitless_cmd(update, ctx)

    creds = tmp_path / "secrets" / "users" / "30.limitless.json"
    assert creds.exists()
    # Under the user's secrets dir (sibling of the per-user wallet key).
    assert creds.parent == (tmp_path / "secrets" / "users")
    # 0600
    assert stat.S_IMODE(creds.stat().st_mode) == 0o600
    # Contents correct
    data = json.loads(creds.read_text())
    assert data == {"api_key": "MYKEY", "api_secret": "MYSECRET"}
    # The secret-bearing message is scrubbed from the chat.
    assert update.message.deleted is True
    # Confirmation goes to the chat (message is gone), NOT as a reply, and
    # never echoes the key or secret.
    [(_chat_id, reply)] = ctx.bot.messages
    assert "MYSECRET" not in reply
    assert "MYKEY" not in reply
    assert "linked" in reply.lower()


async def test_linklimitless_missing_args_shows_usage(tg_env):
    update = _FakeUpdateMsg(user_id=31)
    ctx = _FakeContext(_FakeBot(), args=["only_one"])

    await tb.linklimitless_cmd(update, ctx)

    [reply] = update.message.replies
    assert "/linklimitless <api_key> <api_secret>" in reply
    assert "direct message only" in reply.lower()
    # Nothing stored.
    _, tmp_path = tg_env
    assert not (tmp_path / "secrets" / "users" / "31.limitless.json").exists()


# ----------------------------------------------------------------------
# Handler registration ordering
# ----------------------------------------------------------------------
def test_build_application_registers_new_handlers_before_catchall(tg_env):
    from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

    s, _ = tg_env
    app = tb.build_application(s)
    handlers = app.handlers[0]

    assert any(isinstance(h, CallbackQueryHandler) for h in handlers)
    assert any(
        isinstance(h, CommandHandler) and "linklimitless" in h.commands
        for h in handlers
    )
    # Catch-all free-text MessageHandler stays LAST.
    assert isinstance(handlers[-1], MessageHandler)
