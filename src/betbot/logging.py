"""structlog setup. Console-renderer in dev, JSON in production later."""

from __future__ import annotations

import logging
import sys

import structlog

from betbot.redaction import install_log_redaction, structlog_redactor


def configure_logging(level: str = "INFO") -> None:
    """Idempotent — safe to call multiple times."""
    level_num = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level_num,
    )
    # httpx logs "HTTP Request: POST <url> ..." at INFO, and the Telegram
    # API embeds the bot token IN the URL path -- so leaving httpx at INFO
    # prints the credential to stdout/journal on EVERY successful send.
    # Not hypothetical: that is how the token leaked into an agent
    # transcript on 2026-08-20. WARNING keeps genuine transport failures
    # visible without ever rendering a request line.
    for _noisy in ("httpx", "httpcore"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    # Level-pinning httpx only fixes the leak we already know about. Library
    # code can still put a credential in a log line -- python-telegram-bot
    # raises InvalidToken("The token `<token>` was rejected by the server.")
    # and logs that traceback itself, from inside its own retry loop, before
    # the exception ever reaches our main(). A filter on the root *handler*
    # is the only hook that gets in front of that.
    install_log_redaction()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            # betbot's own logs bypass stdlib logging entirely (PrintLogger
            # writes straight to stdout), so the handler filter never sees
            # them. This covers that half.
            structlog_redactor,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
