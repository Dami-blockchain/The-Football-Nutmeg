"""LLM assistant + public-bot onboarding. All httpx traffic is mocked."""

from __future__ import annotations

import httpx
import pytest

import betbot.llm_agent as llm_agent
import betbot.telegram_bot as tb
from betbot.config import Settings
from betbot.llm_agent import (
    ANTHROPIC_API_URL,
    MAX_HISTORY_MESSAGES,
    NO_KEY_MESSAGE,
    RATE_LIMIT_MESSAGE,
    LLMAgent,
)
from betbot.storage.db import init_engine
from betbot.storage.repos import get_user


# ----------------------------------------------------------------------
# Fake httpx plumbing (no network, ever)
# ----------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=None
            )

    def json(self) -> dict:
        return self._payload


def _install_fake_httpx(monkeypatch, payload: dict, status_code: int = 200):
    """Replace httpx.AsyncClient inside llm_agent; returns captured calls."""
    calls: list[dict] = []

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            calls.append({"url": url, "headers": headers, "json": json})
            return _FakeResponse(payload, status_code)

    monkeypatch.setattr(llm_agent.httpx, "AsyncClient", _FakeClient)
    return calls


def _settings(**overrides) -> Settings:
    base = dict(
        ANTHROPIC_API_KEY="test-key",
        BETBOT_LLM_MODEL="claude-haiku-4-5-20251001",
        BETBOT_LLM_MAX_TOKENS=500,
        BETBOT_LLM_DAILY_LIMIT_PER_USER=20,
        TELEGRAM_BOT_TOKEN="123:TESTTOKEN",
    )
    base.update(overrides)
    return Settings(**base)


_OK_PAYLOAD = {
    "content": [{"type": "text", "text": "You deposit USDC to your wallet."}],
    "stop_reason": "end_turn",
}


# ----------------------------------------------------------------------
# LLMAgent
# ----------------------------------------------------------------------
async def test_happy_path_calls_messages_api(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings())

    reply = await agent.answer(1, "how do I deposit?")

    assert reply == "You deposit USDC to your wallet."
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == ANTHROPIC_API_URL
    assert call["headers"]["x-api-key"] == "test-key"
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    body = call["json"]
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert body["max_tokens"] == 500
    assert "10 USDC" in body["system"]  # system prompt carries the user guide
    assert body["messages"][-1] == {"role": "user", "content": "how do I deposit?"}


async def test_conversation_memory_is_sent_and_capped(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings(BETBOT_LLM_DAILY_LIMIT_PER_USER=100))

    await agent.answer(1, "first question")
    await agent.answer(1, "second question")

    # Second request must include the first exchange before the new message.
    msgs = calls[1]["json"]["messages"]
    assert msgs[0] == {"role": "user", "content": "first question"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[-1] == {"role": "user", "content": "second question"}

    # Memory stays bounded at ~6 exchanges.
    for i in range(20):
        await agent.answer(1, f"q{i}")
    assert len(agent._history[1]) == MAX_HISTORY_MESSAGES
    # Oldest-first and starts with a user turn (valid Messages API shape).
    assert agent._history[1][0]["role"] == "user"


async def test_memory_is_per_user(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings())

    await agent.answer(1, "alice question")
    await agent.answer(2, "bob question")

    bob_msgs = calls[1]["json"]["messages"]
    assert all("alice" not in m["content"] for m in bob_msgs)


async def test_daily_rate_limit_refuses_politely(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings(BETBOT_LLM_DAILY_LIMIT_PER_USER=1))

    first = await agent.answer(1, "hello")
    second = await agent.answer(1, "hello again")

    assert first == "You deposit USDC to your wallet."
    assert second == RATE_LIMIT_MESSAGE
    assert len(calls) == 1  # no API call once exhausted

    # The limit is per-user: another user is unaffected.
    assert await agent.answer(2, "hi") != RATE_LIMIT_MESSAGE


async def test_no_api_key_falls_back_gracefully(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings(ANTHROPIC_API_KEY=""))

    reply = await agent.answer(1, "hello?")

    assert reply == NO_KEY_MESSAGE
    assert calls == []  # never even tries the network


async def test_api_error_returns_friendly_message(monkeypatch):
    _install_fake_httpx(monkeypatch, {}, status_code=500)
    agent = LLMAgent(_settings())

    reply = await agent.answer(1, "hello?")

    assert reply == llm_agent.ERROR_MESSAGE
    # The failed exchange is not committed to memory.
    assert len(agent._history[1]) == 0


# ----------------------------------------------------------------------
# Telegram wiring: open registration + onboarding
# ----------------------------------------------------------------------
class _FakeTGUser:
    def __init__(self, user_id: int):
        self.id = user_id
        self.full_name = "New Person"
        self.username = "newperson"


class _FakeTGMessage:
    def __init__(self, text: str = ""):
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **kwargs) -> None:
        self.replies.append(text)


class _FakeTGUpdate:
    def __init__(self, user_id: int, text: str = ""):
        self.effective_user = _FakeTGUser(user_id)
        self.message = _FakeTGMessage(text)


@pytest.fixture
def tg_env(tmp_path, monkeypatch):
    """Isolated DB + settings wired into the telegram_bot module."""
    init_engine(tmp_path / "bot.sqlite")
    s = _settings(
        BETBOT_WALLET_KEYFILE=str(tmp_path / "secrets" / "agent.key"),
        TELEGRAM_OPEN_REGISTRATION=True,
    )
    monkeypatch.setattr(tb, "get_settings", lambda: s)
    monkeypatch.setattr(tb, "_llm_agent", None)
    return s


def test_open_registration_defaults_on():
    # Field default (not env-dependent): the bot is public out of the box.
    assert Settings.model_fields["telegram_open_registration"].default is True
    assert (
        Settings.model_fields["llm_model"].default == "claude-haiku-4-5-20251001"
    )
    assert Settings.model_fields["llm_max_tokens"].default == 500
    assert Settings.model_fields["llm_daily_limit_per_user"].default == 20


async def test_start_registers_unknown_user_and_creates_wallet(tg_env, tmp_path):
    update = _FakeTGUpdate(user_id=4242)
    assert get_user(4242) is None  # brand new, not allowlisted anywhere

    await tb.start_cmd(update, ctx=None)

    u = get_user(4242)
    assert u is not None
    assert u.wallet_address.startswith("0x")
    assert (tmp_path / "secrets" / "users" / "4242.key").exists()


async def test_start_onboarding_content(tg_env):
    update = _FakeTGUpdate(user_id=777)

    await tb.start_cmd(update, ctx=None)

    [reply] = update.message.replies
    u = get_user(777)
    assert u.wallet_address in reply  # personal deposit address
    assert "10 USDC" in reply  # minimum deposit to begin
    assert "bridge" in reply.lower()  # bridging explained
    assert "9pm" in reply  # daily report (the 9am arb digest is gone)
    for cmd in ("/deposit", "/balance", "/status", "/bets", "/help"):
        assert cmd in reply
    assert "not financial advice" in reply.lower()  # risk disclaimer
    assert "lose money" in reply.lower()


async def test_free_text_registers_user_and_replies_via_llm(
    tg_env, tmp_path, monkeypatch
):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    update = _FakeTGUpdate(user_id=555, text="hi, what is this bot?")

    await tb.chat_handler(update, ctx=None)

    assert get_user(555) is not None  # "hi" alone registers + creates wallet
    assert (tmp_path / "secrets" / "users" / "555.key").exists()
    assert update.message.replies == ["You deposit USDC to your wallet."]
    assert calls[0]["json"]["messages"][-1]["content"] == "hi, what is this bot?"


async def test_empty_text_is_ignored(tg_env, monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    update = _FakeTGUpdate(user_id=556, text="   ")

    await tb.chat_handler(update, ctx=None)

    assert update.message.replies == []
    assert calls == []


def test_build_application_wires_text_handler(tg_env):
    from telegram.ext import MessageHandler

    app = tb.build_application(tg_env)
    handlers = app.handlers[0]
    assert any(isinstance(h, MessageHandler) for h in handlers)
    # Free-text handler is registered after every command handler.
    assert isinstance(handlers[-1], MessageHandler)
