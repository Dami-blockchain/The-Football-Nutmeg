"""The operator notification layer.

These tests exist because the failure mode being guarded against is *silence*.
A notifier that raises takes down the daemon job that was trying to report a
fault; a notifier that swallows its own failure recreates the multi-day outage
where pre-match alerts were dead and nothing said so. Both are asserted here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot import notify
from betbot.notify import (
    NO_ROLLBACK_STATED,
    announce_change,
    cooldown_for,
    format_change_announcement,
    notify_operator,
    notify_operator_sync,
    reset_operator_notify_cooldowns,
)

OPERATOR_ID = 559715


@pytest.fixture()
def op_settings(settings):
    settings.telegram_allowed_user_id = OPERATOR_ID
    settings.telegram_bot_token = "test-token"
    return settings


class _Recorder:
    """A stand-in transport that records what it was asked to send."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[tuple[int, str, object]] = []

    async def __call__(self, _settings, chat_id, text, parse_mode="Markdown"):
        self.sent.append((chat_id, text, parse_mode))
        return self.ok


class _LogSpy:
    """Captures structlog calls so "did it log ERROR?" is directly assertable.

    caplog would not do: structlog only feeds stdlib logging once
    configure_logging() has run, which the test process does not guarantee.
    """

    def __init__(self) -> None:
        self.errors: list[tuple[str, dict]] = []
        self.warnings: list[tuple[str, dict]] = []
        self.infos: list[tuple[str, dict]] = []

    def error(self, event, **kw):
        self.errors.append((event, kw))

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def info(self, event, **kw):
        self.infos.append((event, kw))

    @property
    def error_events(self) -> list[str]:
        return [e for e, _ in self.errors]


@pytest.fixture()
def logspy(monkeypatch):
    spy = _LogSpy()
    monkeypatch.setattr(notify, "log", spy)
    return spy


# --- it reaches the operator ------------------------------------------------

async def test_notify_operator_sends_to_the_operator_chat_id(op_settings):
    send = _Recorder()
    assert await notify_operator(op_settings, "hello", kind="t_send", send_fn=send)
    assert len(send.sent) == 1
    chat_id, body, _mode = send.sent[0]
    assert chat_id == OPERATOR_ID
    assert body == "hello"


async def test_notify_operator_reports_failure_when_no_operator_is_configured(
    op_settings, logspy
):
    op_settings.telegram_allowed_user_id = 0
    send = _Recorder()
    assert await notify_operator(op_settings, "hi", kind="t_nochat", send_fn=send) is False
    assert send.sent == []
    assert "operator_notify_failed" in logspy.error_events


# --- it never raises, and never fails quietly -------------------------------

async def test_notify_operator_never_raises_when_the_transport_explodes(op_settings):
    """A dead Telegram must not propagate into a daemon job."""

    async def _boom(*_a, **_kw):
        raise RuntimeError("telegram down")

    assert await notify_operator(op_settings, "x", kind="t_boom", send_fn=_boom) is False


async def test_notify_operator_logs_error_when_the_transport_explodes(
    op_settings, logspy
):
    async def _boom(*_a, **_kw):
        raise RuntimeError("telegram down")

    await notify_operator(op_settings, "x", kind="t_boomlog", send_fn=_boom)
    assert "operator_notify_failed" in logspy.error_events
    assert logspy.errors[0][1]["error"] == "telegram down"


async def test_notify_operator_logs_error_when_the_send_returns_false(
    op_settings, logspy
):
    """Returning False is a failure too — it must not pass as success."""
    send = _Recorder(ok=False)
    assert await notify_operator(
        op_settings, "x", kind="t_false", send_fn=send
    ) is False
    assert "operator_notify_failed" in logspy.error_events


async def test_a_failed_send_does_not_arm_the_cooldown(op_settings):
    """A fault the operator never heard about must be retried, not suppressed."""
    dead = _Recorder(ok=False)
    await notify_operator(op_settings, "x", kind="t_retry", send_fn=dead)
    live = _Recorder(ok=True)
    assert await notify_operator(op_settings, "x", kind="t_retry", send_fn=live) is True


# --- markdown fallback ------------------------------------------------------

async def test_a_markdown_rejection_is_retried_as_plain_text(op_settings):
    """Unbalanced ``_`` in operator text must not silently eat the message."""
    calls: list[object] = []

    async def _send(_settings, _chat, _text, parse_mode="Markdown"):
        calls.append(parse_mode)
        return parse_mode is None  # Telegram 400s the Markdown attempt

    assert await notify_operator(
        op_settings, "a_b_c", kind="t_md", send_fn=_send
    ) is True
    assert calls == ["Markdown", None]


# --- cooldown ---------------------------------------------------------------

def test_cooldown_defaults_are_the_documented_ones():
    assert cooldown_for("announce") == 0.0
    assert cooldown_for("alert_coverage_gap") == 6 * 3600.0
    assert cooldown_for("scheduler_jobs_not_awaitable") == 24 * 3600.0
    assert cooldown_for("kill_switch_tripped") == 24 * 3600.0
    assert cooldown_for("something_new") == notify.DEFAULT_COOLDOWN_SECONDS
    assert cooldown_for("alert_coverage_gap", 30) == 30.0


async def test_a_repeating_fault_is_sent_once_per_cooldown(op_settings):
    """The hourly watchdog on a stuck fault must not send 24 messages a day."""
    send = _Recorder()
    hour = 3600.0
    sent_at = []
    for h in range(24):
        if await notify_operator(
            op_settings,
            f"2 alert jobs missing (hour {h})",
            kind="alert_coverage_gap",
            send_fn=send,
            now=h * hour,
        ):
            sent_at.append(h)
    # 6h cooldown over a 24h stuck fault -> hours 0, 6, 12, 18.
    assert sent_at == [0, 6, 12, 18]
    assert len(send.sent) == 4


async def test_the_cooldown_expires(op_settings):
    send = _Recorder()
    assert await notify_operator(
        op_settings, "a", kind="t_cd", cooldown_seconds=100, send_fn=send, now=0.0
    )
    assert not await notify_operator(
        op_settings, "a", kind="t_cd", cooldown_seconds=100, send_fn=send, now=99.0
    )
    assert await notify_operator(
        op_settings, "a", kind="t_cd", cooldown_seconds=100, send_fn=send, now=100.0
    )
    assert len(send.sent) == 2


async def test_different_kinds_do_not_share_a_cooldown(op_settings):
    send = _Recorder()
    assert await notify_operator(op_settings, "a", kind="t_k1", send_fn=send, now=0.0)
    assert await notify_operator(op_settings, "b", kind="t_k2", send_fn=send, now=0.0)
    assert len(send.sent) == 2


async def test_an_explicit_dedupe_key_separates_events_of_one_kind(op_settings):
    """Same kind, genuinely different events -> both go through."""
    send = _Recorder()
    assert await notify_operator(
        op_settings, "a", kind="t_dk", dedupe_key="fixture-1", send_fn=send, now=0.0
    )
    assert await notify_operator(
        op_settings, "b", kind="t_dk", dedupe_key="fixture-2", send_fn=send, now=0.0
    )
    assert not await notify_operator(
        op_settings, "c", kind="t_dk", dedupe_key="fixture-1", send_fn=send, now=1.0
    )
    assert len(send.sent) == 2


async def test_a_zero_cooldown_never_suppresses(op_settings):
    send = _Recorder()
    for _ in range(3):
        assert await notify_operator(
            op_settings, "go", kind="announce", send_fn=send, now=0.0
        )
    assert len(send.sent) == 3


async def test_reset_rearms_the_cooldown(op_settings):
    send = _Recorder()
    assert await notify_operator(op_settings, "a", kind="t_rst", send_fn=send, now=0.0)
    assert not await notify_operator(op_settings, "a", kind="t_rst", send_fn=send, now=1.0)
    reset_operator_notify_cooldowns()
    assert await notify_operator(op_settings, "a", kind="t_rst", send_fn=send, now=1.0)


# --- sync wrapper -----------------------------------------------------------

def test_notify_operator_sync_sends(op_settings):
    send = _Recorder()
    assert notify_operator_sync(op_settings, "hi", kind="t_sync", send_fn=send) is True
    assert send.sent[0][0] == OPERATOR_ID


async def test_notify_operator_sync_refuses_inside_a_running_loop(op_settings, logspy):
    """asyncio.run() inside a loop raises — it must be reported, not raised."""
    send = _Recorder()
    assert notify_operator_sync(
        op_settings, "hi", kind="t_syncloop", send_fn=send
    ) is False
    assert "operator_notify_failed" in logspy.error_events
    assert send.sent == []


# --- change announcements ---------------------------------------------------

def test_the_announcement_carries_the_change_and_the_rollback():
    body = format_change_announcement(
        "flip BETBOT_MOV_FIX to true",
        rollback="unset BETBOT_MOV_FIX; systemctl restart tfsm",
        who="ronaldo",
        when=datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc),
    )
    assert "flip BETBOT_MOV_FIX to true" in body
    assert "unset BETBOT_MOV_FIX; systemctl restart tfsm" in body
    assert "Rollback" in body
    assert "ronaldo" in body
    assert "2026-08-20 12:30 EAT" in body  # 09:30 UTC shown in EAT
    assert "NOT yet applied" in body


def test_a_missing_rollback_is_called_out_rather_than_hidden():
    body = format_change_announcement("merge the branch")
    assert NO_ROLLBACK_STATED in body


def test_announce_change_sends_what_it_formats(op_settings):
    send = _Recorder()
    assert announce_change(
        op_settings, "merge feat/x", rollback="git revert abc", send_fn=send
    ) is True
    _chat, body, _mode = send.sent[0]
    assert "merge feat/x" in body
    assert "git revert abc" in body


def test_announcements_are_never_rate_limited(op_settings):
    send = _Recorder()
    for i in range(3):
        assert announce_change(op_settings, f"change {i}", rollback="revert", send_fn=send)
    assert len(send.sent) == 3
