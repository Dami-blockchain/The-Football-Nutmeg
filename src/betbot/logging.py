"""structlog setup. Console-renderer in dev, JSON in production later."""

from __future__ import annotations

import logging
import sys

import structlog


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
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
