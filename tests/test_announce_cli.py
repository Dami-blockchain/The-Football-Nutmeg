"""``tfsm announce`` — the change-announcement entry point.

The standing rule is that the operator is flagged on Telegram BEFORE a change
is committed or a flag is flipped. That rule has to be usable from a shell
script or an agent, not just from Python, so it is a CLI command — and it has
to fail loudly (non-zero exit) when the message did not land, or "announce
then change" would quietly degrade to "change".
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from betbot import main as cli_main
from betbot import notify

runner = CliRunner()

OPERATOR_ID = 559715


@pytest.fixture()
def op_settings(settings, monkeypatch):
    settings.telegram_allowed_user_id = OPERATOR_ID
    settings.telegram_bot_token = "test-token"
    monkeypatch.setattr(cli_main, "get_settings", lambda: settings)
    return settings


@pytest.fixture()
def outbox(monkeypatch):
    """Intercept the real transport, so the CLI path is exercised end to end."""
    sent: list[tuple[int, str]] = []

    async def _send(_settings, chat_id, text, parse_mode="Markdown"):
        sent.append((chat_id, text))
        return True

    monkeypatch.setattr(notify, "send_telegram_to", _send)
    return sent


def test_announce_sends_the_change_and_the_rollback(op_settings, outbox):
    result = runner.invoke(
        cli_main.app,
        [
            "announce",
            "merge feat/operator-notify to main",
            "--rollback",
            "git revert abc123 && systemctl restart tfsm",
            "--who",
            "ronaldo",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(outbox) == 1
    chat_id, body = outbox[0]
    assert chat_id == OPERATOR_ID
    assert "merge feat/operator-notify to main" in body
    assert "git revert abc123 && systemctl restart tfsm" in body
    assert "ronaldo" in body
    assert "NOT yet applied" in body


def test_announce_without_a_rollback_still_sends_but_says_so(op_settings, outbox):
    result = runner.invoke(cli_main.app, ["announce", "flip a flag"])
    assert result.exit_code == 0, result.output
    _chat, body = outbox[0]
    assert notify.NO_ROLLBACK_STATED in body


def test_announce_exits_non_zero_when_telegram_is_down(op_settings, monkeypatch):
    """`tfsm announce ... && git merge` must NOT proceed on a failed send."""

    async def _dead(*_a, **_kw):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(notify, "send_telegram_to", _dead)
    result = runner.invoke(
        cli_main.app, ["announce", "something", "--rollback", "undo it"]
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_announce_exits_non_zero_when_no_operator_is_configured(
    op_settings, outbox
):
    op_settings.telegram_allowed_user_id = 0
    result = runner.invoke(cli_main.app, ["announce", "x", "--rollback", "y"])
    assert result.exit_code == 1
    assert outbox == []


def test_repeated_announcements_are_not_rate_limited(op_settings, outbox):
    for i in range(3):
        result = runner.invoke(
            cli_main.app, ["announce", f"change {i}", "--rollback", "revert"]
        )
        assert result.exit_code == 0, result.output
    assert len(outbox) == 3
