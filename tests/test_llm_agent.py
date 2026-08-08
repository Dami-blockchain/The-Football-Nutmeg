"""Interactive chat assistant (Groq) + public-bot onboarding. httpx is mocked."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

import betbot.llm_agent as llm_agent
import betbot.telegram_bot as tb
from betbot.config import Settings
from betbot.llm_agent import (
    MAX_HISTORY_MESSAGES,
    NO_KEY_MESSAGE,
    RATE_LIMIT_MESSAGE,
    LLMAgent,
    build_prediction_context,
)
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_or_create_user,
    get_user,
    record_reveal,
    upsert_prediction,
)


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
        GROQ_API_KEY="test-groq-key",
        BETBOT_GROQ_MODEL="llama-3.3-70b-versatile",
        BETBOT_LLM_MAX_TOKENS=500,
        BETBOT_LLM_DAILY_LIMIT_PER_USER=20,
        TELEGRAM_BOT_TOKEN="123:TESTTOKEN",
    )
    base.update(overrides)
    return Settings(**base)


_OK_PAYLOAD = {
    "choices": [
        {"message": {"role": "assistant",
                     "content": "You deposit USDC to your wallet."}}
    ],
}


# ----------------------------------------------------------------------
# Groq call shape
# ----------------------------------------------------------------------
async def test_happy_path_calls_groq_api(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings())

    reply = await agent.answer(1, "how do I deposit?")

    assert reply == "You deposit USDC to your wallet."
    assert len(calls) == 1
    call = calls[0]
    # Groq URL + OpenAI-compatible completions path.
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    # Bearer key AND the REQUIRED browser User-Agent (Cloudflare 403s without it).
    assert call["headers"]["Authorization"] == "Bearer test-groq-key"
    assert "Mozilla/5.0" in call["headers"]["User-Agent"]
    body = call["json"]
    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["max_tokens"] == 500
    # OpenAI schema: a system message carries the prompt, last message is the user.
    assert body["messages"][0]["role"] == "system"
    assert "Football Nutmeg Bot" in body["messages"][0]["content"]
    assert body["messages"][-1] == {"role": "user", "content": "how do I deposit?"}


async def test_conversation_memory_is_sent_and_capped(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings(BETBOT_LLM_DAILY_LIMIT_PER_USER=100))

    await agent.answer(1, "first question")
    await agent.answer(1, "second question")

    # Second request: system, then the first exchange, then the new message.
    msgs = calls[1]["json"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "first question"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[-1] == {"role": "user", "content": "second question"}

    for i in range(20):
        await agent.answer(1, f"q{i}")
    assert len(agent._history[1]) == MAX_HISTORY_MESSAGES
    assert agent._history[1][0]["role"] == "user"


async def test_memory_is_per_user(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings())

    await agent.answer(1, "alice question")
    await agent.answer(2, "bob question")

    bob_msgs = calls[1]["json"]["messages"]
    assert all(
        "alice" not in m["content"] for m in bob_msgs if m["role"] != "system"
    )


async def test_daily_rate_limit_refuses_politely(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings(BETBOT_LLM_DAILY_LIMIT_PER_USER=1))

    first = await agent.answer(1, "hello")
    second = await agent.answer(1, "hello again")

    assert first == "You deposit USDC to your wallet."
    assert second == RATE_LIMIT_MESSAGE
    assert len(calls) == 1  # no API call once exhausted
    assert await agent.answer(2, "hi") != RATE_LIMIT_MESSAGE


async def test_no_api_key_falls_back_gracefully(monkeypatch):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    agent = LLMAgent(_settings(GROQ_API_KEY=""))

    reply = await agent.answer(1, "hello?")

    assert reply == NO_KEY_MESSAGE
    assert calls == []  # never even tries the network


async def test_http_error_falls_back_to_fallback_model_then_message(monkeypatch):
    """HTTP 500 → tries primary then fallback model, then friendly message."""
    calls = _install_fake_httpx(monkeypatch, {}, status_code=500)
    agent = LLMAgent(_settings())

    reply = await agent.answer(1, "hello?")

    assert reply == llm_agent.ERROR_MESSAGE
    # Primary model failed → one retry on the fallback model.
    assert len(calls) == 2
    assert calls[0]["json"]["model"] == "llama-3.3-70b-versatile"
    assert calls[1]["json"]["model"] == llm_agent._FALLBACK_MODEL
    assert len(agent._history[1]) == 0  # failed exchange not committed


async def test_malformed_body_returns_friendly_message(monkeypatch):
    # 200 but no 'choices' → parse KeyError → graceful fallback, no exception.
    _install_fake_httpx(monkeypatch, {"nonsense": True})
    agent = LLMAgent(_settings())

    reply = await agent.answer(1, "hello?")

    assert reply == llm_agent.ERROR_MESSAGE


# ----------------------------------------------------------------------
# Paywall context — the critical tests
# ----------------------------------------------------------------------
_KO = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)  # inside Nairobi "today"


def _mk_prediction(fixture_id, home, away, *, p_home, p_draw, p_away):
    from betbot.strategy.engine import Prediction

    return Prediction(
        fixture_id=fixture_id,
        competition_code="PL",
        home_team=home,
        away_team=away,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        home_score=1.5,
        away_score=1.0,
        draw_score=2.4,
    )


@pytest.fixture
def preds_db(tmp_path, monkeypatch):
    """DB with two fixtures kicking off 'today', plus a frozen-now helper."""
    init_engine(tmp_path / "bot.sqlite")
    # Two distinct fixtures today.
    upsert_prediction(
        _mk_prediction(9001, "Arsenal", "Chelsea",
                       p_home=0.55, p_draw=0.25, p_away=0.20),
        kickoff=_KO,
    )
    upsert_prediction(
        _mk_prediction(9002, "Liverpool", "Everton",
                       p_home=0.60, p_draw=0.22, p_away=0.18),
        kickoff=_KO,
    )
    return _KO


def _user(tmp_path, uid, *, created_days_ago, consumed=0):
    get_or_create_user(
        uid, f"user{uid}", secrets_dir=str(tmp_path / "secrets"), keyfile=None
    )
    # Age the account + set consumed by mutating the persisted row.
    from betbot.storage.db import session_scope
    from betbot.storage.models import User
    from sqlalchemy import select

    with session_scope() as s:
        row = s.execute(
            select(User).where(User.telegram_user_id == uid)
        ).scalar_one()
        row.created_at = _KO - timedelta(days=created_days_ago)
        row.predictions_consumed = consumed
    return get_user(uid)


def _no_balance(monkeypatch):
    """Force the on-chain USDC read to 0 (no credits) for payer/locked tests."""
    class _CB:
        ok = True
        usdc = 0.0

    monkeypatch.setattr(
        "betbot.wallet.usdc_balance", lambda *a, **k: _CB(), raising=False
    )


def test_context_operator_sees_all_full_detail(preds_db, tmp_path, monkeypatch):
    s = _settings(TELEGRAM_ALLOWED_USER_ID=1111)
    u = _user(tmp_path, 1111, created_days_ago=99)  # long past trial, but operator

    ctx = build_prediction_context(u, s, now=_KO)

    assert "Arsenal" in ctx and "Chelsea" in ctx
    assert "Liverpool" in ctx and "Everton" in ctx
    assert "55%" in ctx and "60%" in ctx  # both probability triples present
    assert "LOCKED" not in ctx


def test_context_trial_sees_all_full_detail(preds_db, tmp_path, monkeypatch):
    s = _settings()
    u = _user(tmp_path, 2222, created_days_ago=1)  # inside 7-day trial

    ctx = build_prediction_context(u, s, now=_KO)

    assert "Arsenal" in ctx and "Liverpool" in ctx
    assert "55%" in ctx and "60%" in ctx
    assert "LOCKED" not in ctx


def test_context_payer_reveals_only_A_B_stays_locked(
    preds_db, tmp_path, monkeypatch
):
    _no_balance(monkeypatch)
    s = _settings()
    u = _user(tmp_path, 3333, created_days_ago=99)  # trial ended, no credits
    # User has revealed fixture A (9001) but NOT B (9002).
    record_reveal(3333, 9001, charged=True)

    ctx = build_prediction_context(u, s, now=_KO)

    # A is fully revealed.
    assert "Arsenal" in ctx and "Chelsea" in ctx
    assert "55%" in ctx  # A's p_home
    # B appears only as a LOCKED line — NO pick, NO probability leaked.
    assert "Liverpool v Everton — LOCKED" in ctx
    assert "60%" not in ctx  # B's p_home absent
    assert "18%" not in ctx  # B's p_away absent
    assert "*Bet:" not in ctx.split("LOCKED")[1]  # nothing after B leaks a pick


def test_context_locked_user_all_locked_no_probabilities(
    preds_db, tmp_path, monkeypatch
):
    _no_balance(monkeypatch)
    s = _settings()
    u = _user(tmp_path, 4444, created_days_ago=99)  # trial ended, no reveals

    ctx = build_prediction_context(u, s, now=_KO)

    assert "Arsenal v Chelsea — LOCKED" in ctx
    assert "Liverpool v Everton — LOCKED" in ctx
    # No probabilities anywhere.
    for pct in ("55%", "60%", "25%", "22%", "20%", "18%"):
        assert pct not in ctx


def test_context_no_fixtures(tmp_path, monkeypatch):
    init_engine(tmp_path / "empty.sqlite")
    s = _settings(TELEGRAM_ALLOWED_USER_ID=5555)
    u = _user(tmp_path, 5555, created_days_ago=1)

    ctx = build_prediction_context(u, s, now=_KO)

    assert "no fixtures today" in ctx


# ----------------------------------------------------------------------
# Chat is FREE + READ-ONLY: never charges / consumes / records a reveal
# ----------------------------------------------------------------------
async def test_answer_never_charges_or_records_reveal(
    preds_db, tmp_path, monkeypatch
):
    _no_balance(monkeypatch)
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    s = _settings()
    u = _user(tmp_path, 6666, created_days_ago=99)  # a payer

    def _boom(*a, **k):  # any money mutation is a bug in the chat path
        raise AssertionError("chat must not charge/consume/record a reveal")

    monkeypatch.setattr(llm_agent, "has_revealed", lambda *a, **k: False)
    # If the agent (or its context builder) ever tried to write money state,
    # these would fire. They are monkeypatched on the modules the chat path
    # could reach.
    import betbot.storage.repos as repos

    monkeypatch.setattr(repos, "record_reveal", _boom)
    monkeypatch.setattr(repos, "increment_predictions_consumed", _boom)

    agent = LLMAgent(s)
    reply = await agent.answer(u, "what's the Arsenal game?")

    assert reply == "You deposit USDC to your wallet."
    assert len(calls) == 1  # it did chat
    # Nothing was charged.
    assert get_user(6666).predictions_consumed == 0


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
    assert Settings.model_fields["telegram_open_registration"].default is True
    assert (
        Settings.model_fields["groq_model"].default == "llama-3.3-70b-versatile"
    )
    assert (
        Settings.model_fields["groq_base_url"].default
        == "https://api.groq.com/openai/v1"
    )
    assert Settings.model_fields["llm_max_tokens"].default == 500
    assert Settings.model_fields["llm_daily_limit_per_user"].default == 20


async def test_start_registers_unknown_user_and_creates_wallet(tg_env, tmp_path):
    update = _FakeTGUpdate(user_id=4242)
    assert get_user(4242) is None

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
    assert u.wallet_address in reply
    assert "7 days" in reply.lower()
    assert "1 USDC" in reply
    for cmd in ("/predictions", "/balance", "/status", "/help"):
        assert cmd in reply
    assert "not financial advice" in reply.lower()
    assert "lose money" in reply.lower()
    assert "today's matches" in reply.lower()  # chat invite
    assert "bridge" not in reply.lower()
    assert "/deposit" not in reply
    assert "/bets" not in reply


async def test_free_text_registers_user_and_replies_via_llm(
    tg_env, tmp_path, monkeypatch
):
    calls = _install_fake_httpx(monkeypatch, _OK_PAYLOAD)
    update = _FakeTGUpdate(user_id=555, text="hi, what is this bot?")

    await tb.chat_handler(update, ctx=None)

    assert get_user(555) is not None
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
    assert isinstance(handlers[-1], MessageHandler)
