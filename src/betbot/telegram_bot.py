"""Telegram bot — multi-user, non-custodial.

Each user registers with /start and gets their OWN isolated wallet (key stored
server-side under .secrets/users/<id>.key). Funds stay in each user's own wallet
— nothing is pooled. The bot reads its own functions in-process; it never moves
one user's funds into another's.

Access: a user may register if (a) open registration is on, (b) their id is in
the allowlist, or (c) they're already registered. Run with TELEGRAM_BOT_TOKEN
set:  python -m betbot.telegram_bot
"""

from __future__ import annotations

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from betbot.backtest import backtest_stored
from betbot.config import get_settings
from betbot.gate import evaluate_gate
from betbot.llm_agent import LLMAgent
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import get_or_create_user, get_user, list_recent_paper_bets
from betbot.wallet import all_balances

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
        "⚽ *The Football Smart Manager*\n\n"
        "I trade football prediction markets on Polymarket (Polygon) and "
        "Limitless (Base) using a probability model, and I only bet when my "
        "edge over the market price clears a threshold.\n\n"
        "*You're registered.* You have your own isolated wallet — your funds "
        "stay yours and are never pooled with anyone else's.\n\n"
        f"*Your personal deposit address:*\n`{u.wallet_address}`\n\n"
        "*Getting started*\n"
        "1. Deposit at least *10 USDC* to your personal wallet address above "
        "to begin.\n"
        "2. The same address works on *Polygon* and *Base* — send USDC on "
        "either chain (bridge between them with any standard bridge if "
        "needed). /balance confirms arrival.\n"
        "3. Trading starts in *paper mode*; live trading only switches on "
        "once the performance gate passes.\n"
        "4. Daily reports (Nairobi time): arbitrage digest at *9am*, "
        "performance report at *9pm*.\n\n"
        "*Commands*\n"
        "/deposit – your wallet address (Polygon + Base)\n"
        "/balance – your USDC balance\n"
        "/status – mode, gate, performance\n"
        "/bets – recent bets\n"
        "/help – this guide\n\n"
        "You can also just *ask me anything* in plain text — how deposits "
        "work, what the bot is doing, what a report means.\n\n"
        "⚠️ *Risk disclaimer:* prediction-market trading can lose money, "
        "including everything you deposit. Past performance never guarantees "
        "future results. This is not financial advice — only deposit what "
        "you can afford to lose.",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def deposit_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    u = _register(update)
    await update.message.reply_text(
        "*Deposit USDC to your wallet*\n\n"
        f"`{u.wallet_address}`\n\n"
        "Same address on *Polygon* (Polymarket) and *Base* (Limitless). Send "
        "USDC on either chain; /balance confirms it. Your funds stay in your own "
        "wallet — never pooled with anyone else's.",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def balance_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_settings()
    u = _register(update)
    balances = all_balances(u.wallet_address, s)
    lines = "\n".join(
        f"• {b.label}: {b.usdc:.2f} USDC" + ("" if b.ok else " (read failed)")
        for b in balances
    )
    total = sum(b.usdc for b in balances if b.ok)
    await update.message.reply_text(
        f"*Your balance* — total {total:.2f} USDC\n\n{lines}\n\n`{u.wallet_address}`",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_settings()
    g = evaluate_gate(s)
    r = backtest_stored()
    gate_line = "PASS ✅" if g.passed else "FAIL ❌"
    await update.message.reply_text(
        f"*Status*\n\nMode: `{s.mode}`\nLive-trading gate: {gate_line}\n"
        f"Settled bets: {r.n} (hit {r.hit_rate:.0%}, ROI {r.roi:+.1%})\n"
        f"Cross-venue arb alerts: on (every {s.arb_scan_interval_min} min)",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def bets_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    rows = list_recent_paper_bets(days=7)
    if not rows:
        await update.message.reply_text("No bets in the last 7 days.")
        return
    lines = []
    for b in rows[:15]:
        res = b.settled_outcome or "—"
        pnl = f"{b.pnl_usd:+.2f}" if b.pnl_usd is not None else "—"
        lines.append(f"#{b.fixture_id} {b.outcome} p={b.our_probability:.2f} → {res} ({pnl})")
    await update.message.reply_text("*Recent bets*\n" + "\n".join(lines), parse_mode=ParseMode.MARKDOWN)


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
    app.add_handler(CommandHandler("deposit", deposit_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("bets", bets_cmd))
    # Free-text → LLM assistant. Added LAST so commands keep priority.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))
    return app


def main() -> None:
    settings = get_settings()
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
