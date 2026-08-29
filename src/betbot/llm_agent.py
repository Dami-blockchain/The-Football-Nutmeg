"""Interactive chat assistant for the Football Nutmeg Bot (Groq / free).

WHY this exists: users DM the bot and converse about its predictions, like
chatting with an assistant. Since most public users type free text rather than
commands, this module answers them — grounded in the predictions they are
ENTITLED to, and never leaking a locked (unpaid) pick.

Backend: the FREE Groq API, which is OpenAI-compatible
(``POST /openai/v1/chat/completions``, ``Authorization: Bearer <key>``). Called
directly over httpx — no SDK — to keep the runtime dependency set unchanged.

Design constraints:
* CRITICAL — Groq sits behind Cloudflare, which 403s ("error code: 1010") bare
  Python HTTP clients. Every request MUST carry a browser ``User-Agent`` header
  (:data:`_BROWSER_UA`); a missing UA is a guaranteed 403.
* The model only ever DISCUSSES predictions the user may see. The paywall lives
  in :func:`build_prediction_context`: a payer's un-revealed fixtures are shown
  only as ``LOCKED`` with NO pick or probabilities, so chatting can never bypass
  the 1-USDC-per-prediction paywall. Chat is FREE and READ-ONLY — it never
  charges, consumes a credit, or records a reveal.
* Per-user memory is a small in-memory deque (no DB schema change): enough for
  conversational continuity, gone on restart, which is fine for support chat.
* A per-user daily cap bounds request volume on a public bot.
* Any HTTP/parse error degrades to a friendly fallback string — a failed call
  never raises into the Telegram handler.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

import httpx

from betbot.entitlement import entitlement_for
from betbot.logging import get_logger
from betbot.storage.repos import (
    get_user,
    has_revealed,
    predictions_for_kickoff_range,
)
from betbot.tips import format_prediction

log = get_logger(__name__)

# Groq is OpenAI-compatible; the model constant lives in settings.
# CRITICAL: Groq is fronted by Cloudflare, which 403s ("error code: 1010")
# clients without a browser UA. This header is REQUIRED on every request.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)
_FALLBACK_MODEL = "openai/gpt-oss-20b"

# ~6 exchanges of context per user (each exchange = user + assistant message).
MAX_HISTORY_MESSAGES = 12

# Telegram rejects messages over 4096 chars; stay safely under.
_MAX_REPLY_CHARS = 3900

SYSTEM_PROMPT = (
    "You are Football Nutmeg Bot, a football match-prediction assistant on "
    "Telegram. Be concise, friendly, and conversational. You may ONLY discuss "
    "the predictions listed as available below. Respect the ACCESS line: if the "
    "user has full access (operator or trial) NEVER tell them anything is locked "
    "or that they must pay. Only for a fixture explicitly marked LOCKED do you "
    "withhold the pick and mention unlocking it for 1 USDC via the pre-match "
    "alert. If NO predictions are listed (no fixtures today), simply say there "
    "are no matches today — do NOT invent predictions and do NOT mention paying "
    "or unlocking. This is not financial advice."
)

NO_KEY_MESSAGE = (
    "I can't chat right now (the assistant isn't configured). The commands "
    "still work: /start, /predictions, /balance, /status."
)

RATE_LIMIT_MESSAGE = (
    "You've reached today's chat limit — it resets at midnight UTC. "
    "Meanwhile the commands are always available: /predictions, /balance, "
    "/status."
)

ERROR_MESSAGE = (
    "Sorry — I couldn't reach my brain just now. Please try again in a "
    "minute, or use /status and /predictions directly."
)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ----------------------------------------------------------------------
# Prediction-aware, paywall-respecting context
# ----------------------------------------------------------------------
def build_prediction_context(user, settings, *, now: datetime | None = None) -> str:
    """A compact text block of the predictions THIS user may discuss.

    Built fresh per message because entitlement (trial expiry, on-chain
    balance) and the reveal ledger change over time.

    * operator / trial → entitled to ALL of today's predictions → full detail
      (via :func:`betbot.tips.format_prediction`).
    * otherwise (payer / locked) → full detail ONLY for fixtures already in the
      reveal ledger (:func:`has_revealed`); every other fixture is a LOCKED line
      carrying NO pick or probability, so the paywall is never bypassed.

    READ-ONLY: never records a reveal, charges a credit, or mutates anything.
    """
    now = now or datetime.now(timezone.utc)
    from betbot.daily_jobs import nairobi_day_bounds
    from betbot.storage.repos import track_record

    # Rolling accuracy line so the assistant can answer "how accurate have you
    # been?" truthfully. This is ACCURACY, not proof of profit/edge/CLV.
    tr = track_record(30)
    # SCOPE: track_record -> prediction_outcomes_since applies four filters
    # (ledger epoch, non-degenerate triple, CLUB competition, season start), so
    # this figure is THIS SEASON'S CLUB football only — never internationals or
    # the World Cup, and never last season. The label must say so; "ALL matches"
    # here means "every club fixture this season", NOT "every match ever
    # predicted", and is stated that way so the assistant cannot overclaim.
    if tr["n"] == 0:
        record_line = (
            "Track record (last 30 days, club competitions this season): no "
            "settled predictions in scope yet — too small a sample to quote an "
            "accuracy."
        )
    else:
        small = " (small sample — treat as provisional)" if tr["n"] < 10 else ""
        # All-match accuracy and called-pick hit rate are DIFFERENT metrics
        # and are stated separately so the assistant can never conflate them.
        called = tr.get("called") or {}
        called_line = ""
        if called.get("enabled") and called.get("n"):
            called_line = (
                f" On the subset the confidence filter CALLS as a bet "
                f"({called['call_rate']:.0%} of matches): {called['hits']}/"
                f"{called['n']} ({called['hit_rate']:.0%}, 95% CI "
                f"{called['ci_lo']:.0%}-{called['ci_hi']:.0%}). Those are "
                "short-priced favourites, so a higher hit rate there is "
                "expected and is NOT an edge, NOT +EV and NOT beating the "
                "market — never present it as such."
            )
        record_line = (
            f"Track record (last 30 days), all club fixtures this season "
            f"(n={tr['n']}): {tr['hits']}/{tr['n']} correct "
            f"({tr['hit_rate']:.0%}), mean RPS {tr['mean_rps']:.2f}{small}. "
            "Scope is this season's club competitions only — no internationals "
            "or World Cup. This is prediction ACCURACY, not a profit/edge "
            "guarantee."
            + called_line
        )

    start, end, _day = nairobi_day_bounds(now)
    preds = predictions_for_kickoff_range(start, end)
    # Same high-conviction gate as the paywall (single shared predicate) so
    # the chat never offers a low-confidence fixture as buyable — "nothing
    # more". Already-revealed fixtures stay visible; flag OFF -> unchanged.
    from betbot.daily_jobs import high_conf_visible

    visible = high_conf_visible(settings, preds, user.telegram_user_id, has_revealed)

    ent = entitlement_for(user, settings, now=now)
    all_free = ent.reason in ("operator", "trial")
    if ent.reason == "operator":
        access_note = (
            "ACCESS: this user is the OPERATOR — unlimited free access. Nothing "
            "is ever locked for them; never mention paying or unlocking."
        )
    elif ent.reason == "trial":
        access_note = (
            "ACCESS: this user is on their free trial — all of today's "
            "predictions are unlocked for them; do not mention paying."
        )
    else:
        access_note = (
            "ACCESS: this user is a payer — only fixtures shown with a pick are "
            "unlocked; fixtures marked LOCKED need 1 USDC to reveal."
        )

    if not visible:
        if preds:
            empty_note = (
                "(no HIGH-CONFIDENCE calls today — the bot only offers "
                "fixtures it is highly confident in, and none qualify today). "
                "Tell the user plainly there are no high-confidence calls "
                "today; do NOT mention locking or paying."
            )
        else:
            empty_note = (
                "(no fixtures scheduled today). Tell the user plainly there "
                "are no matches today; do NOT mention locking or paying."
            )
        return "\n".join([
            record_line, "", access_note, "",
            "Today's predictions available to this user: " + empty_note,
        ])

    lines = [record_line, "", access_note, "", "Today's predictions available to this user:"]
    for p in visible:
        entitled = all_free or has_revealed(
            user.telegram_user_id, p.fixture_id
        )
        if entitled:
            lines.append("")
            lines.append(format_prediction(p, edge_threshold=settings.edge_threshold))
        else:
            lines.append(
                f"{p.home_team} v {p.away_team} — LOCKED (user must pay 1 USDC "
                "to unlock; do NOT reveal the pick)"
            )
    return "\n".join(lines)


class LLMAgent:
    """Answers one user's chat message with short-term per-user memory.

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

        Counted BEFORE the API call: the cap protects request volume, so
        attempts are what matter, not successes. (This is the CHAT quota — it
        has nothing to do with the prediction paywall / credits.)
        """
        today = _today()
        day, used = self._usage.get(user_id, (today, 0))
        if day != today:
            day, used = today, 0
        if used >= self._settings.llm_daily_limit_per_user:
            return False
        self._usage[user_id] = (day, used + 1)
        return True

    # -- Groq HTTP call --------------------------------------------------
    async def _call_groq(self, model: str, messages: list[dict]) -> str | None:
        """POST one completion to Groq; return the reply text or None on error.

        Sends the REQUIRED browser User-Agent (missing UA = Cloudflare 403).
        Never raises — any HTTP/parse error is logged and returns None so the
        caller can fall back gracefully.
        """
        url = f"{self._settings.groq_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self._settings.llm_max_tokens,
            "temperature": 0.6,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": _BROWSER_UA,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # noqa: BLE001 — a failed call must not crash the bot
            log.warning("groq_request_failed", model=model, error=str(e))
            return None

    # -- main entry point ------------------------------------------------
    async def answer(self, user, text: str) -> str:
        """Answer ``text`` for ``user``; always returns something sendable.

        ``user`` may be a persisted :class:`~betbot.storage.models.User` or a
        raw Telegram user id (int) — an int is resolved via :func:`get_user`.
        The reply is grounded in :func:`build_prediction_context`, so the model
        can only discuss predictions this user is entitled to.
        """
        if not self._settings.groq_api_key:
            return NO_KEY_MESSAGE

        # Resolve the persisted user (chat_handler passes the DB row after
        # registering, but accept an id for convenience / older callers). A
        # lookup failure (e.g. no DB) degrades to id-only, no context.
        if isinstance(user, int):
            try:
                db_user = get_user(user)
            except Exception:  # noqa: BLE001 — chit-chat must survive a missing DB
                db_user = None
            user_id = db_user.telegram_user_id if db_user is not None else user
        else:
            db_user = user
            user_id = db_user.telegram_user_id

        if not self._consume_quota(user_id):
            return RATE_LIMIT_MESSAGE

        # Fresh per-turn context: entitlement and reveals change over time.
        context = ""
        if db_user is not None:
            try:
                context = build_prediction_context(db_user, self._settings)
            except Exception as e:  # noqa: BLE001 — never block chat on a context error
                log.warning("prediction_context_failed", user_id=user_id, error=str(e))

        system = SYSTEM_PROMPT + ("\n\n" + context if context else "")
        messages = [
            {"role": "system", "content": system},
            *self._history[user_id],
            {"role": "user", "content": text},
        ]

        reply = await self._call_groq(self._settings.groq_model, messages)
        if not reply:
            # Retry once on the smaller/faster fallback model.
            reply = await self._call_groq(_FALLBACK_MODEL, messages)
        if not reply:
            return ERROR_MESSAGE

        reply = reply[:_MAX_REPLY_CHARS]
        history = self._history[user_id]
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        return reply
