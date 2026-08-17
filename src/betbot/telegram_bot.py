"""Telegram bot — multi-user football-prediction TIPSTER.

Each user registers with /start and gets their OWN isolated deposit wallet (key
stored server-side under .secrets/users/<id>.key). The bot sends football
predictions — it places NO orders and moves NO funds. Predictions are FREE for
7 days from signup; after that each individual match prediction costs 1 USDC,
paid by sending USDC on Polygon to the user's own address. The operator
(TELEGRAM_ALLOWED_USER_ID) is always free/unlimited.

Access: a user may register if (a) open registration is on, (b) their id is in
the allowlist, or (c) they're already registered. Run with TELEGRAM_BOT_TOKEN
set:  python -m betbot.telegram_bot
"""

from __future__ import annotations

from datetime import datetime, timezone

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from betbot.config import get_settings
from betbot.daily_jobs import (
    commit_reveals,
    nairobi_day_bounds,
    render_user_predictions,
)
from betbot.entitlement import entitlement_for
from betbot.llm_agent import LLMAgent
from betbot.logging import get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_or_create_user,
    get_user,
    predictions_for_kickoff_range,
)

log = get_logger(__name__)


def _allowed(update: Update) -> bool:
    s = get_settings()
    user = update.effective_user
    if user is None:
        return False
    if s.telegram_open_registration:
        return True
    if user.id in s.allowed_telegram_ids:
        return True
    return get_user(user.id) is not None  # already registered


def _register(update: Update):
    s = get_settings()
    u = update.effective_user
    name = (u.full_name or u.username or str(u.id)) if u else "user"
    # The operator maps onto the existing agent wallet (where their prior
    # deposit sits); everyone else gets a fresh per-user wallet.
    keyfile = str(s.wallet_keyfile) if u.id == s.telegram_allowed_user_id else None
    return get_or_create_user(u.id, name, secrets_dir=s.secrets_dir, keyfile=keyfile)


def _authed(handler):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _allowed(update):
            log.warning("telegram_unauthorized", user_id=getattr(update.effective_user, "id", None))
            if update.message:
                await update.message.reply_text("⛔ Not authorized. Ask the operator to add you.")
            return
        return await handler(update, ctx)

    return wrapper


def build_onboarding_guide(user, settings) -> str:
    """The step-by-step onboarding guide shown on /start and /guide.

    Pure string builder (no I/O) so it is unit-testable. Branches on whether
    ``user`` is the operator (``telegram_allowed_user_id``): the operator sees
    an always-free line instead of the trial/pricing pitch. Everyone else sees
    their real per-user Polygon deposit address and the 1-USDC pricing.
    """
    is_operator = bool(
        settings.telegram_allowed_user_id
        and user.telegram_user_id == settings.telegram_allowed_user_id
    )

    if is_operator:
        pricing = (
            "*Pricing*\n"
            "• You're the *operator* — all predictions are *always free*, "
            "unlimited, no trial or payment.\n"
            "• Check your account any time with /balance.\n\n"
        )
    else:
        pricing = (
            "*Pricing*\n"
            "• Your first *7 days are FREE* — every prediction, no charge.\n"
            "• After that it's *1 USDC per prediction*, paid in USDC on "
            "*Polygon* (1 USDC unlocks 1 prediction).\n"
            "• Top up by sending USDC (on *Polygon*) to your personal deposit "
            "address below — I detect it on-chain automatically.\n"
            f"• *Your deposit address (Polygon):*\n`{user.wallet_address}`\n"
            "• Check your credits any time with /balance.\n\n"
        )

    return (
        "⚽ *Football Nutmeg Bot*\n\n"
        "Welcome! I send *data-driven predictions* for the big-5 European "
        "leagues (England, Spain, Italy, Germany, France) plus the Champions "
        "League. I never place bets and never move your funds.\n\n"
        "*How it works — for each match you get:*\n"
        "1️⃣ A *morning heads-up* of the day's fixtures and when the "
        "predictions will land.\n"
        "2️⃣ About *1 hour before kickoff*, the *model prediction* — "
        "the model's *win / draw / loss probabilities* (home / draw / away) "
        "plus the expected-goals (xG) readout.\n"
        "3️⃣ About *10 minutes before kickoff*, the *confirmed starting XI* "
        "and a *lineup-adjusted* prediction.\n"
        "4️⃣ At *full time*, the *result vs the model's pick* (✅ / ❌).\n\n"
        "*Just chat with me* — message me any time to ask about a match: why a "
        "pick, the xG, form, anything about today's matches.\n\n"
        f"{pricing}"
        "*Commands*\n"
        "/predictions – today's fixtures + picks\n"
        "/title – who's winning La Liga? (or /title PL)\n"
        "/balance – your balance + credits\n"
        "/status – trial / credits status\n"
        "/record – how accurate I've been\n"
        "/guide – show this guide again\n"
        "/help – show this guide\n\n"
        "⚠️ *Not financial advice.* Prediction markets / betting can "
        "lose money. Bet responsibly — only ever stake what you can afford to "
        "lose, and past results never guarantee future ones."
    )


@_authed
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    u = _register(update)
    await update.message.reply_text(
        build_onboarding_guide(u, get_settings()),
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def guide_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-send the same onboarding guide on demand (also backs /help)."""
    u = _register(update)
    await update.message.reply_text(
        build_onboarding_guide(u, get_settings()),
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def predictions_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_settings()
    u = _register(update)
    now = datetime.now(timezone.utc)
    start, end, day = nairobi_day_bounds(now)
    preds = predictions_for_kickoff_range(start, end)
    text, reveals = render_user_predictions(u, preds, s, now=now)
    # reply_text raises on a failed send; only commit (charge) once it returns.
    await update.message.reply_text(
        f"*⚽ Today's predictions — {day.isoformat()}*\n\n{text}",
        parse_mode=ParseMode.MARKDOWN,
    )
    commit_reveals(u, reveals)


@_authed
async def balance_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_settings()
    u = _register(update)
    ent = entitlement_for(u, s)
    from betbot.wallet import usdc_balance

    cb = usdc_balance(u.wallet_address, "polygon", s.polygon_rpc_url)
    bal = f"{cb.usdc:.2f} USDC" + ("" if cb.ok else " (read failed)")

    if ent.reason == "operator":
        status = "operator — unlimited predictions"
    elif ent.reason == "trial":
        status = f"free trial — {ent.trial_days_left} day(s) left"
    elif ent.reason == "credit":
        status = f"{ent.credits_remaining} prediction credit(s) remaining"
    else:
        status = "trial ended — send 1 USDC per prediction to unlock"

    await update.message.reply_text(
        f"*Your account*\n\n"
        f"Polygon balance: {bal}\n"
        f"Status: {status}\n\n"
        f"*Payment address (Polygon):*\n`{u.wallet_address}`\n\n"
        "Send *1 USDC per prediction* on Polygon to unlock more.",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_settings()
    u = _register(update)
    ent = entitlement_for(u, s)
    is_operator = ent.reason == "operator"
    if is_operator:
        access = "unlimited (operator)"
    elif ent.reason == "trial":
        access = f"free trial — {ent.trial_days_left} day(s) left"
    elif ent.reason == "credit":
        access = f"{ent.credits_remaining} prediction credit(s)"
    else:
        access = "locked — send 1 USDC per prediction"
    await update.message.reply_text(
        f"*Status*\n\n"
        f"Access: {access}\n"
        f"Operator: {'yes' if is_operator else 'no'}\n"
        f"Predictions consumed: {u.predictions_consumed}",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def record_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Rolling prediction accuracy (free). Accuracy — NOT a profit/edge claim."""
    from betbot.storage.repos import track_record

    tr = track_record(30)
    if tr["n"] == 0:
        body = (
            "*Track record — last 30 days*\n\n"
            "No settled predictions yet. Once matches finish I'll show how many "
            "of my calls were right."
        )
    else:
        small = (
            "\n\n_Small sample so far — treat as provisional._"
            if tr["n"] < 10 else ""
        )
        body = (
            "*Track record — last 30 days*\n\n"
            f"Correct: {tr['hits']}/{tr['n']} ({tr['hit_rate']:.0%})\n"
            f"Mean RPS: {tr['mean_rps']:.2f}  ·  Mean Brier: {tr['mean_brier']:.2f}\n\n"
            "_This is prediction accuracy, not proof of profit or betting edge._"
            + small
        )
    await update.message.reply_text(body, parse_mode=ParseMode.MARKDOWN)


def _format_title_race(result: dict, code: str) -> str:
    """Render a cached season-title projection for /title (Markdown)."""
    from betbot.season_service import LEAGUE_NAMES

    name = LEAGUE_NAMES.get(code, code)
    md = result.get("matchday")
    played = result.get("played")
    remaining = result.get("remaining")
    gen = (result.get("generated_at") or "")[:10]
    table = result.get("table") or []

    lines = [f"*🏆 {name} title race*"]
    ctx = []
    if md:
        ctx.append(f"after matchday {md}")
    if played is not None and remaining is not None:
        ctx.append(f"{played} played / {remaining} left")
    if ctx:
        lines.append("_" + ", ".join(ctx) + "_")
    lines.append("")
    for i, row in enumerate(table[:6], 1):
        lines.append(
            f"{i}. *{row['team']}* — {row['p_title']*100:.0f}% "
            f"(exp {row['exp_points']:.0f} pts)"
        )
    if not table:
        lines.append("_No projection available yet._")
    early = (played or 0) < 8
    note = (
        "\n\n_Model projection from a Monte-Carlo of the run-in — not a "
        "guarantee. "
        + ("Very early season, so this will move a lot. " if early else "")
        + (f"Sim run {gen}." if gen else "")
        + "_"
    )
    return "\n".join(lines) + note


@_authed
async def title_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Season-title race for a league (default La Liga). FREE, cached."""
    from betbot.season_service import LEAGUE_NAMES, load_cache

    _register(update)
    s = get_settings()
    arg = (ctx.args[0].upper() if ctx.args else "PD")
    code = arg if arg in LEAGUE_NAMES else "PD"
    result = load_cache(s, code)
    if not result:
        await update.message.reply_text(
            f"The {LEAGUE_NAMES.get(code, code)} title projection is still being "
            "computed — check back shortly. (Tip: /title PL, /title SA, "
            "/title BL1, /title FL1.)"
        )
        return
    await update.message.reply_text(
        _format_title_race(result, code), parse_mode=ParseMode.MARKDOWN,
    )


# Lazily constructed so importing this module never needs settings; tests
# inject their own agent by assigning to ``_llm_agent``.
_llm_agent: LLMAgent | None = None


def _get_llm_agent() -> LLMAgent:
    global _llm_agent
    if _llm_agent is None:
        _llm_agent = LLMAgent(get_settings())
    return _llm_agent


@_authed
async def chat_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Any non-command text becomes a question for the LLM assistant.

    Registers the user first (same flow as /start) so a brand-new user who
    opens with "hi" still gets a wallet and counts as registered.
    """
    if update.message is None or not (update.message.text or "").strip():
        return
    user = _register(update)
    # Chat is FREE and READ-ONLY: it discusses only predictions the user is
    # entitled to (paywall enforced in the agent's context builder) and never
    # charges a credit or records a reveal.
    reply = await _get_llm_agent().answer(user, update.message.text.strip())
    await update.message.reply_text(reply)


def build_application(settings) -> Application:
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", guide_cmd))
    app.add_handler(CommandHandler("guide", guide_cmd))
    app.add_handler(CommandHandler("predictions", predictions_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("record", record_cmd))
    app.add_handler(CommandHandler("title", title_cmd))
    # Free-text → LLM assistant. Added LAST so commands keep priority.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
    return app


def main() -> None:
    settings = get_settings()
    from betbot.logging import configure_logging

    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env — cannot start the bot.")
    log.info(
        "telegram_bot_starting",
        open_registration=settings.telegram_open_registration,
        allowlisted=len(settings.allowed_telegram_ids),
    )
    app = build_application(settings)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
