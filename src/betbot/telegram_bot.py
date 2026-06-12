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

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
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
from betbot.storage.repos import (
    get_or_create_user,
    get_user,
    list_recent_paper_bets,
    set_arb_interest,
)
from betbot.wallet import all_balances, store_limitless_creds

log = get_logger(__name__)

# Callback data for the /start cross-venue-arbitrage opt-in button.
ARB_INTEREST_CB = "arb_interest"

# Cross-venue arbitrage explainer, sent when a user opts in.
ARB_EXPLAINER = (
    "🔁 *Cross-venue arbitrage*\n\n"
    "Sometimes the same match is priced differently on Polymarket and "
    "Limitless. When the gap is big enough, I can buy *both* sides — one on "
    "each venue — so you lock a profit no matter who wins. I only fire when "
    "the locked margin clears *2%*, size within strict caps, and never chase "
    "a half-filled trade.\n\n"
    "To include you, I trade the Polymarket leg from your wallet "
    "automatically. The *Limitless* leg needs you to open a Limitless account "
    "once and link an API key — here's how. 👇"
)

# Very detailed step-by-step Limitless onboarding, sent after the explainer
# (and after the optional video). Split into a few Telegram messages so each
# step is easy to read on a phone and the whole thing isn't one wall of text.
# These are sent in order; LIMITLESS_GUIDE_PARTS[-1] carries the /linklimitless
# call-to-action.
LIMITLESS_GUIDE_PARTS = (
    # Part 1 — what you're about to do + the one prerequisite.
    "📋 *Set up Limitless — full walkthrough*\n"
    "_About 5 minutes. Follow it with app.limitless.exchange open beside this "
    "chat. You only ever do this once._\n\n"
    "*What you'll end up with:* a Limitless account funded with USDC on the "
    "Base network, and an API key you paste to me so I can place the Limitless "
    "side of an arbitrage for you.\n\n"
    "*Before you start, you need:* a crypto wallet (e.g. MetaMask, Rabby, or "
    "any wallet app) — OR just an email address, since Limitless lets you sign "
    "in with email and creates a wallet for you.",
    # Part 2 — open the account.
    "*Step 1 — Open your Limitless account*\n"
    "1. On your phone or computer, go to *app.limitless.exchange*\n"
    "2. Tap *Connect* (or *Sign in*) in the top corner.\n"
    "3. Choose how to sign in:\n"
    "   • *Email* — type your email, then the code they send you. Limitless "
    "makes a wallet for you automatically.\n"
    "   • *Wallet* — pick your wallet app (MetaMask etc.) and approve the "
    "connection.\n"
    "4. You're in when you can see your account/balance in the corner. ✅",
    # Part 3 — fund it (the part people get wrong).
    "*Step 2 — Add money (USDC on Base)*\n"
    "Limitless runs on the *Base* network and trades in *USDC*. Your money "
    "must be USDC *on Base* specifically — not Ethereum, not Polygon.\n\n"
    "• If you bought USDC on an exchange (Coinbase, Binance…), withdraw it and "
    "choose the *Base* network when asked, sending to your Limitless wallet "
    "address.\n"
    "• If your USDC is on another network, use any bridge (e.g. *bridge.base.org*) "
    "to move it to Base first.\n"
    "• Start small — even *10–20 USDC* is enough to take part.\n\n"
    "⚠️ Picking the wrong network is the #1 mistake — double-check it says "
    "*Base* before you confirm.",
    # Part 4 — create the API key.
    "*Step 3 — Create your API key*\n"
    "1. In Limitless, open *Settings* (often a gear icon or your profile menu).\n"
    "2. Find *API* (sometimes labelled *Developer* or *API keys*).\n"
    "3. Tap *Create API key*. If it asks for a *scope* or *permission*, choose "
    "*trading*.\n"
    "4. Limitless now shows you an *API key* and an *API secret*.\n\n"
    "🔑 *Copy BOTH immediately* — the *secret is shown only once*. If you lose "
    "it you just make a new key (which replaces the old one). Paste them "
    "somewhere safe for the next step.",
    # Part 5 — hand it to the bot (CTA + security).
    "*Step 4 — Link it to me*\n"
    "Send me this message, pasting in your two values:\n\n"
    "`/linklimitless YOUR_API_KEY YOUR_API_SECRET`\n\n"
    "_Example:_ `/linklimitless lmts_ab12... sk_9f8e...`\n\n"
    "🔒 *Your safety:* your key is saved only in your own isolated profile, "
    "file-locked, never shown again, and used solely to place *your* Limitless "
    "arbitrage orders. The moment you send it, I *delete your message* so the "
    "key isn't left sitting in this chat. Only ever send it here, in this "
    "direct chat — never in a group.\n\n"
    "That's it — once you've also deposited, you're in the arbitrage pool. 🎉",
)

# OPTIONAL extra: an operator can record the ~90s screen-capture (script below)
# and set BETBOT_ARB_GUIDE_VIDEO_URL. The written walkthrough above is the
# primary, self-sufficient guide — the video is a nice-to-have, never a missing
# piece, so there is no "coming soon" placeholder when it's unset.
#
# Operator: record ~90s screen capture covering:
#   (0-12s)  What arb is: same match, two venues, different prices → buy both
#            sides → profit locked regardless of result.
#   (12-28s) How the bot does it: scans Polymarket + Limitless every cycle,
#            only fires at >=2% locked margin, sizes within caps,
#            both-legs-or-abort.
#   (28-42s) What you need: deposit >=10 USDC; the Polymarket leg is automatic;
#            the Limitless leg needs your own Limitless account + API key.
#   (42-80s) Open Limitless: app.limitless.exchange → connect wallet (Privy) →
#            fund with USDC on Base → Settings/API → Create key (trading scope)
#            → copy key + secret (shown once).
#   (80-90s) Link it: send /linklimitless <key> <secret> to the bot. Done —
#            you're in the arb pool.


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
        "⚽ *The Football Nutmeg Agent*\n\n"
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
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🔁 Cross-venue arbitrage — tell me more",
                        callback_data=ARB_INTEREST_CB,
                    )
                ]
            ]
        ),
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


@_authed
async def arb_interest_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """User tapped the cross-venue-arbitrage opt-in button on /start.

    Persists arb_interest=True, answers the callback, then sends three things
    in order: the explainer, a how-to video (or a graceful fallback note), and
    the Limitless onboarding guide.
    """
    s = get_settings()
    query = update.callback_query
    user = update.effective_user
    # Make sure the user exists (and so the flag has somewhere to live) — a
    # callback can in principle arrive before any other handler registered them.
    _register(update)
    set_arb_interest(user.id, True)
    await query.answer()

    chat_id = query.message.chat_id
    # (a) explainer
    await ctx.bot.send_message(
        chat_id, ARB_EXPLAINER, parse_mode=ParseMode.MARKDOWN
    )
    # (b) OPTIONAL video — only when the operator configured one. The written
    # walkthrough below is the primary, complete guide, so there is no
    # "coming soon" placeholder when no video is set. A bad URL/file_id must
    # never break onboarding.
    if s.arb_guide_video_url:
        try:
            await ctx.bot.send_video(chat_id, video=s.arb_guide_video_url)
        except Exception as e:  # noqa: BLE001 — a bad URL can't break onboarding
            log.warning("arb_guide_video_failed", error=str(e))
    # (c) the detailed Limitless walkthrough, one message per step.
    for part in LIMITLESS_GUIDE_PARTS:
        await ctx.bot.send_message(chat_id, part, parse_mode=ParseMode.MARKDOWN)


@_authed
async def linklimitless_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Capture + store a user's OWN Limitless API key/secret, 0600, never echoed.

    Usage: ``/linklimitless <api_key> <api_secret>``.

    NOTE: this CAPTURES + STORES the per-user Limitless key only. WIRING it into
    live order placement (make_signer_adapters) is a deliberate follow-up — the
    stored key is not yet used to place any order. Do not claim it trades yet.
    """
    s = get_settings()
    u = _register(update)
    args = ctx.args if ctx is not None else None
    if not args or len(args) < 2:
        await update.message.reply_text(
            "Usage: `/linklimitless <api_key> <api_secret>`\n\n"
            "🔒 Send this in a direct message only; the key is stored "
            "encrypted-at-rest on your isolated profile and never shown again.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    api_key, api_secret = args[0], args[1]
    # Stored under the user's own secrets dir; values are NEVER logged or echoed.
    store_limitless_creds(s.secrets_dir, u.telegram_user_id, api_key, api_secret)
    # The user's message contains their live API secret — scrub it from the
    # Telegram chat history so it doesn't linger in scrollback or on Telegram's
    # servers. Best-effort: requires no special admin right in a private chat,
    # but never fail the link if deletion is refused.
    deleted = False
    try:
        await update.message.delete()
        deleted = True
    except Exception as e:  # noqa: BLE001 — deletion is a nicety, not the contract
        log.info("linklimitless_msg_delete_failed", error=str(e))
    confirm = (
        "✅ Limitless API key linked. It's stored in your own isolated "
        "profile (file-locked, never shown again)"
        + (" and I deleted your message with the key." if deleted
           else ". Tip: delete your message above so the key isn't left in "
                "the chat.")
        + " You're in the arbitrage pool once you deposit."
    )
    # Send to the chat directly — update.message may be gone after delete().
    await ctx.bot.send_message(chat_id=update.effective_chat.id, text=confirm)


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
    app.add_handler(CommandHandler("linklimitless", linklimitless_cmd))
    # Inline-button opt-in for cross-venue arbitrage. Registered before the
    # catch-all MessageHandler so it's never shadowed.
    app.add_handler(CallbackQueryHandler(arb_interest_cb, pattern=f"^{ARB_INTEREST_CB}$"))
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
