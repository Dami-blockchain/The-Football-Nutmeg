"""Telegram bot — the operator's chat interface to The Football Smart Manager.

Locked to a single Telegram user id (``TELEGRAM_ALLOWED_USER_ID``) because it
exposes the agent's deposit wallet and status. It reads the bot's own functions
in-process (wallet balances, gate, P&L) — it never moves funds. "Depositing"
means the operator sends USDC from their own wallet to the address this bot
shows; the bot only displays the address and confirms the balance.

Run it (after setting TELEGRAM_BOT_TOKEN + TELEGRAM_ALLOWED_USER_ID in .env):
    python -m betbot.telegram_bot
"""

from __future__ import annotations

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from betbot.backtest import backtest_stored
from betbot.config import get_settings
from betbot.gate import evaluate_gate
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_kill_switch,
    list_recent_paper_bets,
    settled_pnl_window,
)
from betbot.wallet import wallet_summary

log = get_logger(__name__)


def _authorized(update: Update) -> bool:
    allowed = get_settings().telegram_allowed_user_id
    user = update.effective_user
    return allowed == 0 or (user is not None and user.id == allowed)


async def _deny(update: Update) -> None:
    log.warning(
        "telegram_unauthorized",
        user_id=getattr(update.effective_user, "id", None),
    )
    if update.message:
        await update.message.reply_text("⛔ Not authorized.")


def _authed(handler):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _authorized(update):
            return await _deny(update)
        return await handler(update, ctx)

    return wrapper


@_authed
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "⚽ *The Football Smart Manager*\n\n"
        "I run a football prediction-market bot. Commands:\n"
        "/deposit – the agent wallet address to fund with USDC\n"
        "/balance – current USDC balance (Polygon + Base)\n"
        "/status – mode, gate, P&L, kill switch\n"
        "/bets – recent paper bets\n",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def deposit_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    w = wallet_summary(get_settings())
    await update.message.reply_text(
        "*Deposit USDC to the agent wallet*\n\n"
        f"`{w['address']}`\n\n"
        "Send *USDC* on *Polygon* (Polymarket) or *Base* (Limitless) to this "
        "address — it's the same address on both chains.\n\n"
        "Then use /balance to confirm it arrived.\n\n"
        "⚠️ Only send USDC. Live trading stays OFF until the paper-trading gate "
        "passes — depositing does not start real-money betting.",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def balance_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    w = wallet_summary(get_settings())
    lines = "\n".join(
        f"• {b['label']}: {b['usdc']:.2f} USDC" + ("" if b["ok"] else " (read failed)")
        for b in w["balances"]
    )
    await update.message.reply_text(
        f"*USDC balance* — total {w['total_usdc']:.2f}\n\n{lines}\n\n"
        f"`{w['address']}`",
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    s = get_settings()
    ks = get_kill_switch()
    g = evaluate_gate(s)
    r = backtest_stored()
    pnl, staked = settled_pnl_window(s.drawdown_window_days)
    ks_line = "TRIPPED ⛔" if ks.tripped_at else "clear ✅"
    gate_line = "PASS ✅" if g.passed else "FAIL ❌"
    await update.message.reply_text(
        f"*Status*\n\n"
        f"Mode: `{s.mode}`\n"
        f"Kill switch: {ks_line}\n"
        f"Live-trading gate: {gate_line}\n"
        f"Settled bets: {r.n}  (hit {r.hit_rate:.0%}, ROI {r.roi:+.1%})\n"
        f"Trailing {s.drawdown_window_days}d P&L: ${pnl:+.2f} on ${staked:.0f}\n"
        + ("" if g.passed else "\nGate blockers:\n" + "\n".join(f"– {x}" for x in g.reasons)),
        parse_mode=ParseMode.MARKDOWN,
    )


@_authed
async def bets_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    rows = list_recent_paper_bets(days=7)
    if not rows:
        await update.message.reply_text("No paper bets in the last 7 days.")
        return
    lines = []
    for b in rows[:15]:
        res = b.settled_outcome or "—"
        pnl = f"{b.pnl_usd:+.2f}" if b.pnl_usd is not None else "—"
        lines.append(f"#{b.fixture_id} {b.outcome} p={b.our_probability:.2f} → {res} ({pnl})")
    await update.message.reply_text("*Recent bets*\n" + "\n".join(lines),
                                    parse_mode=ParseMode.MARKDOWN)


def build_application(settings) -> Application:
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", start_cmd))
    app.add_handler(CommandHandler("deposit", deposit_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("bets", bets_cmd))
    return app


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env — cannot start the bot.")
    if settings.telegram_allowed_user_id == 0:
        log.warning("telegram_open", note="TELEGRAM_ALLOWED_USER_ID=0 — bot accepts ANY user")
    log.info("telegram_bot_starting")
    app = build_application(settings)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
