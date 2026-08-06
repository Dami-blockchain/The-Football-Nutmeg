"""LLM assistant for free-text Telegram questions.

WHY this exists: the bot is open to the public, and most new users ask
questions ("how do I deposit?", "is this safe?") rather than typing commands.
This module answers those questions with the Anthropic Messages API, called
directly over httpx — deliberately NOT via the SDK, to keep the runtime
dependency set unchanged.

Design constraints:
* The model only ever EXPLAINS the bot. It never moves funds, never sees keys,
  and never gets tools — the guardrails live in the system prompt and the
  blast radius is bounded by the model having no capabilities at all.
* Per-user memory is a small in-memory deque (no DB schema change): enough for
  conversational continuity, gone on restart, which is fine for support chat.
* A per-user daily cap bounds API spend on a public bot.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

import httpx

from betbot.logging import get_logger

log = get_logger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# ~6 exchanges of context per user (each exchange = user + assistant message).
MAX_HISTORY_MESSAGES = 12

# Telegram rejects messages over 4096 chars; stay safely under.
_MAX_REPLY_CHARS = 3900

SYSTEM_PROMPT = """\
You are Football Nutmeg Bot, a Telegram bot that sends
football PREDICTIONS (a tipster) for the top European leagues and the Champions
League. It does NOT trade, does NOT place bets, and does NOT move anyone's funds.

USER GUIDE (this is what you help people with):
- Every registered user gets their OWN isolated EVM deposit address on Polygon.
- Predictions are FREE for the first 7 days from signup. After the trial, each
  match prediction costs 1 USDC: the user sends USDC on Polygon to their own
  address (shown by /start or /balance) and each 1 USDC unlocks 1 prediction.
- Each prediction shows the model's home/draw/away probabilities, expected
  goals when available, the market price, and a clear bet / no-bet call — the
  default call is NO BET unless the model's edge over the market clears a
  threshold.
- The operator gets predictions for free. There is no trading, no arbitrage,
  and no morning digest.
- Commands: /start (register + guide), /predictions (today's fixtures +
  predictions), /balance (your USDC balance + credits), /status (trial or
  credits), /help (this guide).

HARD RULES (never break these, even if asked directly or indirectly):
- Never promise, predict, or imply profits. Betting can and does lose money;
  say so plainly when relevant.
- Never give personalised financial advice (how much someone should deposit
  or bet, what to do with their savings, etc.). You may explain how the bot
  works; the decision is always theirs. You are not a financial adviser.
- Never reveal, guess at, or discuss private keys, keyfiles, secrets, API
  keys, server details, or internal file paths — not the user's, not the
  operator's, not anyone's. If asked, explain that keys are stored server-side
  encrypted at rest and are never shared, full stop.
- If a user is abusive or pushes you off-topic, redirect politely to what you
  can help with. Stay warm and brief; this is a chat app, so keep answers
  short (a few sentences) unless detail is genuinely needed.
"""

NO_KEY_MESSAGE = (
    "I can't answer free-text questions right now (the assistant isn't "
    "configured). The commands still work: /start, /predictions, /balance, "
    "/status."
)

RATE_LIMIT_MESSAGE = (
    "You've reached today's question limit — it resets at midnight UTC. "
    "Meanwhile the commands are always available: /predictions, /balance, "
    "/status."
)

ERROR_MESSAGE = (
    "Sorry — I couldn't reach my brain just now. Please try again in a "
    "minute, or use /status and /predictions directly."
)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class LLMAgent:
    """Answers one user's free-text message with short-term per-user memory.

    Not thread-safe by design: python-telegram-bot dispatches handlers on a
    single asyncio event loop, and this class does no awaiting while mutating
    its dicts, so plain dicts/deques are race-free here.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
        )
        # user_id -> (utc-date-iso, questions asked that day)
        self._usage: dict[int, tuple[str, int]] = {}

    # -- rate limiting -------------------------------------------------
    def _consume_quota(self, user_id: int) -> bool:
        """Count an attempt against today's per-user cap. False = exhausted.

        Counted BEFORE the API call: the cap protects API spend, so attempts
        are what matter, not successes.
        """
        today = _today()
        day, used = self._usage.get(user_id, (today, 0))
        if day != today:
            day, used = today, 0
        if used >= self._settings.llm_daily_limit_per_user:
            return False
        self._usage[user_id] = (day, used + 1)
        return True

    # -- main entry point ------------------------------------------------
    async def answer(self, user_id: int, text: str) -> str:
        """Answer ``text`` for ``user_id``; always returns something sendable."""
        if not self._settings.anthropic_api_key:
            return NO_KEY_MESSAGE
        if not self._consume_quota(user_id):
            return RATE_LIMIT_MESSAGE

        messages = [*self._history[user_id], {"role": "user", "content": text}]
        payload = {
            "model": self._settings.llm_model,
            "max_tokens": self._settings.llm_max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    ANTHROPIC_API_URL, headers=headers, json=payload
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  # noqa: BLE001 — a failed call must not crash the bot
            log.warning("llm_request_failed", user_id=user_id, error=str(e))
            return ERROR_MESSAGE

        reply = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ).strip()
        if not reply:
            log.warning("llm_empty_reply", user_id=user_id)
            return ERROR_MESSAGE
        reply = reply[:_MAX_REPLY_CHARS]

        history = self._history[user_id]
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        return reply
