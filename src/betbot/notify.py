"""Outbound Telegram notifications (proactive bot -> operator messages).

Uses the Bot API directly so any process (the daemon's arb watcher) can push a
message to the operator without running the polling bot. The operator must have
/start-ed the bot once (they have).
"""

from __future__ import annotations

import httpx

from betbot.logging import get_logger

log = get_logger(__name__)


async def send_telegram(settings, text: str) -> bool:
    """Push a message to the operator (the original single-recipient path)."""
    return await send_telegram_to(settings, settings.telegram_allowed_user_id, text)


async def send_telegram_to(settings, chat_id: int, text: str) -> bool:
    """Push a message to one chat id — used to broadcast the daily reports to
    the operator AND every registered user, not just the operator."""
    token = settings.telegram_bot_token
    if not token or not chat_id:
        log.warning("telegram_notify_skipped", reason="no bot token or chat id")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            )
            resp.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 — a failed notification shouldn't crash callers
        log.warning("telegram_notify_failed", error=str(e))
        return False
