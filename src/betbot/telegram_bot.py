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


@_authed
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    u = _register(update)
    await update.message.reply_text(
        "⚽ *Football Nutmeg Bot*\n\n"
        "I send you football *predictions* for the top European leagues and the "
        "Champions League: model probabilities, expected goals, and a clear "
        "bet / no-bet call anchored to the market. I never place bets and never "
        "move your funds.\n\n"
        "*You're registered.*\n\n"
        "*How it works*\n"
        "• Predictions are *FREE for your first 7 days*.\n"
        "• After the trial, each match prediction costs *1 USDC on Polygon* — "
        "1 USDC unlocks 1 prediction.\n"
        "• Pay by sending USDC (Polygon) to your personal address below; I "
        "detect it on-chain automatically.\n\n"
        f"*Your payment address (Polygon):*\n`{u.wallet_address}`\n\n"
        "*Commands*\n"
        "/predictions – today's fixtures + predictions\n"
        "/balance – your USDC balance + credits\n"
        "/status – trial / credits\n"
        "/help – this guide\n\n"
        "You can also just *ask me anything* in plain text.\n\n"
        "⚠️ *Not financial advice.* Predictions can be wrong and betting can "
        "lose money — only ever stake what you can afford to lose. Past results "
        "never guarantee future ones.",
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
    _register(update)
    reply = await _get_llm_agent().answer(
        update.effective_user.id, update.message.text.strip()
    )
    await update.message.reply_text(reply)


def build_application(settings) -> Application:
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", start_cmd))
    app.add_handler(CommandHandler("predictions", predictions_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
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
