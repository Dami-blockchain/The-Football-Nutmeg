"""Credential redaction — one regex, one home.

The live Telegram bot token leaked into the logs on 2026-08-20 by three
independent routes:

1. ``httpx`` logged ``HTTP Request: POST <url>`` at INFO and the token sits in
   the Telegram URL *path*, so every successful send printed it.
2. :mod:`betbot.notify` logged ``str(e)`` and httpx renders the request URL
   into its error text.
3. python-telegram-bot puts the token in an exception *message* —
   ``InvalidToken("The token `<token>` was rejected by the server.")`` at
   ``telegram/_bot.py:860`` — and ``telegram.ext._utils.networkloop`` logs
   that traceback with ``LOGGER.exception()`` before re-raising.

Three symptoms, one bug: a credential living inside a string that nobody
thought of as a credential. This module is the single place that knows what a
secret looks like, so route four is covered before anybody finds it.
"""

from __future__ import annotations

import logging
import re

#: What a redacted secret is replaced with. Deliberately visible: an operator
#: reading a log should be able to tell that something *was* here.
REDACTED = "<REDACTED>"

#: Telegram bot token: ``<bot_id_digits>:<secret>``.
#:
#: No ``\b`` before the digits. The token appears as ``bot<digits>:<secret>``
#: in the API URL, and a word boundary cannot match between "t" and a digit —
#: a first attempt at this regex used ``\b`` and silently matched nothing on
#: exactly the string it existed to catch. Do not "tidy" this.
TOKEN_RE = re.compile(r"\d{6,}:[A-Za-z0-9_-]{20,}")


def redact_text(text: str) -> str:
    """Return ``text`` with any credential-shaped substring removed."""
    return TOKEN_RE.sub(REDACTED, text)


def redact(exc: BaseException) -> str:
    """Error text with any bot-token-shaped substring removed."""
    return redact_text(str(exc))


class RedactingFilter(logging.Filter):
    """Strip credential-shaped substrings from every record reaching a handler.

    Attach this to the *handlers* on the root logger, not to a logger. A filter
    on a logger only sees records logged directly to that logger; a filter on a
    handler sees everything that propagates up to it — including tracebacks
    logged by third-party libraries we do not control.

    That distinction is the whole point. When Telegram rejects a token,
    python-telegram-bot logs the traceback *itself* before re-raising, so
    catching the exception in our own ``main()`` is already too late: the
    credential is in the file. Only a handler-level filter gets in front of it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:  # pragma: no cover - a broken record must still log
            rendered = str(record.msg)
        if TOKEN_RE.search(rendered):
            # Collapse msg+args into the redacted rendering. Done only when a
            # secret is actually present, so ordinary records keep their
            # structure for any handler that wants msg and args apart.
            record.msg = redact_text(rendered)
            record.args = None

        # exc_info is rendered by the Formatter later, from the traceback
        # object — a filter cannot rewrite a traceback in place. But
        # Formatter.format() skips formatException() when record.exc_text is
        # already set, so we render it ourselves, redacted, and the formatter
        # uses ours. This is the line that closes the library path.
        if record.exc_info and not record.exc_text:
            record.exc_text = redact_text(
                logging.Formatter().formatException(record.exc_info)
            )
        elif record.exc_text:
            record.exc_text = redact_text(record.exc_text)

        if record.stack_info:
            record.stack_info = redact_text(record.stack_info)
        return True


def install_log_redaction(logger: logging.Logger | None = None) -> None:
    """Attach a :class:`RedactingFilter` to every handler on ``logger``.

    Defaults to the root logger, which is where stdlib-logging output from
    ``telegram``, ``asyncio`` and ``httpx`` ends up. Idempotent — a handler
    that already carries one is left alone, so this is safe to call from an
    idempotent ``configure_logging``.
    """
    target = logger if logger is not None else logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())


def structlog_redactor(_logger, _method_name, event_dict):
    """structlog processor: redact credential-shaped values in the event dict.

    betbot's own logs go through structlog's PrintLogger, which writes to
    stdout *without* passing through stdlib logging — so the handler filter
    above never sees them. This covers that half.
    """
    for key, value in list(event_dict.items()):
        if isinstance(value, str) and TOKEN_RE.search(value):
            event_dict[key] = redact_text(value)
    return event_dict
