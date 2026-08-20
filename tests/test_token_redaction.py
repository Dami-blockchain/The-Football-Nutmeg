"""The bot token must never reach a log, stdout or stderr.

The token leaked for real on 2026-08-20. Two of the three routes were closed at
the call site; the third lives in library code — python-telegram-bot raises
``InvalidToken("The token `<token>` was rejected by the server.")`` and logs
that traceback itself before re-raising. These tests assert the *absence of the
secret*, not the presence of a friendly message: a test that only checks the
nice error text would still pass while the token sat three lines below it.

Every token in this file is synthetic. Nothing here reads .env.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from betbot import notify
from betbot.redaction import (
    REDACTED,
    RedactingFilter,
    install_log_redaction,
    redact,
    redact_text,
)

#: Synthetic, token-shaped, never valid. Matches the real Telegram layout
#: (``<bot_id>:<secret>``) so it exercises the same regex a real one would.
FAKE_TOKEN = "7654321:AAHfakeTOKENforTESTS_not-real-000"
#: The half that is actually secret. Asserted separately so a partial render
#: (secret without the numeric prefix) cannot slip through the whole-match sub.
FAKE_SECRET = "AAHfakeTOKENforTESTS_not-real-000"

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# The regex itself
# ---------------------------------------------------------------------------


def test_redacts_token_embedded_in_a_url_path():
    """The regression that made the first attempt useless.

    The token appears as ``bot<digits>:<secret>`` in the API URL. A ``\\b``
    before the digits cannot match between "t" and a digit, so an anchored
    pattern silently matched nothing on the exact string it existed to catch.
    """
    url = f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"
    cleaned = redact_text(url)
    assert FAKE_TOKEN not in cleaned
    assert FAKE_SECRET not in cleaned
    assert REDACTED in cleaned


def test_redact_strips_token_from_exception_text():
    exc = RuntimeError(f"The token `{FAKE_TOKEN}` was rejected by the server.")
    cleaned = redact(exc)
    assert FAKE_TOKEN not in cleaned
    assert FAKE_SECRET not in cleaned


def test_notify_reuses_the_shared_redactor_rather_than_a_second_regex():
    """Structural: one regex, one home. Two copies drift, and the stale copy
    is always the one guarding the path that leaks."""
    assert notify._redact is redact


# ---------------------------------------------------------------------------
# The logging filter — the part that gets in front of library code
# ---------------------------------------------------------------------------


def test_filter_redacts_a_logged_traceback(caplog):
    """`.exception()` renders the traceback from the traceback *object*, so the
    token is not in record.msg at all — it is produced later by the Formatter.
    The filter has to pre-render it to win that race."""
    record = logging.LogRecord(
        name="telegram.ext", level=logging.ERROR, pathname=__file__,
        lineno=1, msg="Invalid token. Aborting retry loop.", args=(),
        exc_info=None,
    )
    try:
        raise ValueError(f"The token `{FAKE_TOKEN}` was rejected by the server.")
    except ValueError:
        record.exc_info = sys.exc_info()

    assert RedactingFilter().filter(record) is True
    rendered = logging.Formatter("%(message)s").format(record)
    assert FAKE_TOKEN not in rendered
    assert FAKE_SECRET not in rendered
    assert REDACTED in rendered


def test_filter_redacts_a_token_passed_as_a_lazy_arg():
    """``log.error("failed: %s", exc)`` never puts the token in record.msg."""
    record = logging.LogRecord(
        name="telegram", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="request failed: %s", args=(f"bot{FAKE_TOKEN}/getMe",),
        exc_info=None,
    )
    RedactingFilter().filter(record)
    assert FAKE_SECRET not in record.getMessage()


def test_install_log_redaction_is_idempotent():
    logger = logging.getLogger("betbot.test.idempotent")
    logger.handlers = [logging.NullHandler()]
    install_log_redaction(logger)
    install_log_redaction(logger)
    filters = logger.handlers[0].filters
    assert sum(isinstance(f, RedactingFilter) for f in filters) == 1


# ---------------------------------------------------------------------------
# End to end: run a real process and read everything it wrote
# ---------------------------------------------------------------------------

#: Driven entirely from argv so nothing needs templating into it.
_CHILD = '''
import logging
import sys

SRC, TOKEN = sys.argv[1], sys.argv[2]
sys.path.insert(0, SRC)

from telegram.error import InvalidToken

import betbot.telegram_bot as tb


class _Settings:
    telegram_bot_token = TOKEN
    telegram_open_registration = False
    allowed_telegram_ids = ()
    log_level = "INFO"
    db_path = ":memory:"


class _App:
    """Reproduces python-telegram-bot 22.7 on a rejected token.

    Bot.initialize() re-raises with the token in the message (_bot.py:860) and
    telegram.ext._utils.networkloop logs that traceback with LOGGER.exception()
    before re-raising (networkloop.py:182) -- i.e. the library leaks it before
    our own except block ever runs.
    """

    def run_polling(self, **kwargs):
        try:
            try:
                raise InvalidToken("Unauthorized")
            except InvalidToken as inner:
                raise InvalidToken(
                    "The token `" + TOKEN + "` was rejected by the server."
                ) from inner
        except InvalidToken:
            logging.getLogger("telegram.ext._utils.networkloop").exception(
                "Bootstrap Initialize Application Invalid token. Aborting retry loop."
            )
            raise


tb.get_settings = lambda: _Settings()
tb.init_engine = lambda *a, **k: None
tb.build_application = lambda settings: _App()
tb.main()
'''


@pytest.fixture()
def child_script(tmp_path):
    script = tmp_path / "run_bot_with_bad_token.py"
    script.write_text(_CHILD)
    return script


def test_rejected_token_never_appears_in_process_output(child_script):
    """The load-bearing test: run it for real, capture *everything* the process
    wrote to stdout and stderr, and assert the secret is not in there."""
    env = dict(os.environ)
    # Pin the child to THIS checkout's src, not whatever is pip-installed.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, str(child_script), str(SRC_DIR), FAKE_TOKEN],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    combined = proc.stdout + proc.stderr

    # 1. The secret is absent. This is the assertion that matters.
    assert FAKE_TOKEN not in combined
    assert FAKE_SECRET not in combined

    # 2. The library DID try to log it, and was redacted rather than silenced.
    #    Without this the test would also pass if logging broke entirely.
    assert REDACTED in combined

    # 3. A supervisor sees a failure.
    assert proc.returncode != 0

    # 4. The operator gets something actionable.
    assert "TELEGRAM_BOT_TOKEN" in combined

    # 5. No chained traceback dumped the original exception's message.
    assert "was rejected by the server" not in proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr
