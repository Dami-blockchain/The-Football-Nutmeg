"""The Telegram bot must not be able to become a silent husk.

On 2026-08-22 /tmp/bot.log had been frozen for ~11h, ending in an unhandled
``telegram.error.NetworkError: Bad Gateway`` with "No error handlers are
registered, logging exception." above it. The process was still alive. From
outside there was no way to tell a recovered updater from a dead one — the
same "process up != working" trap as the APScheduler outage.

(It had in fact recovered: python-telegram-bot's network_retry_loop treats
NetworkError as retryable with backoff and default max_retries=-1, and the
process still held an ESTABLISHED socket to 149.154.166.110:443. The bug was
that nothing said so.)
"""

from __future__ import annotations

import types

import pytest

import betbot.telegram_bot as tb


class _FakeUpdater:
    def __init__(self, running: bool):
        self.running = running


class _FakeApp:
    def __init__(self, running: bool):
        self.updater = _FakeUpdater(running)


def _ctx(*, error=None, running=True):
    return types.SimpleNamespace(error=error, application=_FakeApp(running))


# ---------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_error_handler_logs_shape_not_message(monkeypatch):
    """The live log said 'No error handlers are registered'. Now one exists —
    and it must log the SHAPE of the failure, never the text."""
    seen: list[tuple] = []
    monkeypatch.setattr(
        tb.log, "warning", lambda evt, **kw: seen.append((evt, kw))
    )

    from telegram.error import NetworkError

    await tb._log_network_error(None, _ctx(error=NetworkError("Bad Gateway")))

    assert seen, "a polling error must be logged"
    evt, kw = seen[0]
    assert evt == "telegram_polling_error"
    assert kw["error_type"] == "NetworkError"


@pytest.mark.asyncio
async def test_error_handler_never_logs_a_url_bearing_token(monkeypatch):
    """Telegram puts the bot token in the URL path.

    httpx renders errors as "... for url
    'https://api.telegram.org/bot<TOKEN>/getUpdates'", and this handler fires
    on every network blip — logging str(error) would reintroduce the leak
    fixed in 2bd1830, continuously.
    """
    seen: list[dict] = []
    monkeypatch.setattr(tb.log, "warning", lambda evt, **kw: seen.append(kw))

    token = "8000000000:AAH-THIS-IS-A-FAKE-SECRET-TOKEN-VALUE"
    leaky = RuntimeError(
        f"Server error for url 'https://api.telegram.org/bot{token}/getUpdates'"
    )
    await tb._log_network_error(None, _ctx(error=leaky))

    blob = repr(seen)
    assert token not in blob
    assert "api.telegram.org" not in blob


# ---------------------------------------------------------------------
# Heartbeat — telling "alive" from "alive but deaf"
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_heartbeat_reports_polling_when_the_updater_is_running(monkeypatch):
    seen: list[tuple] = []
    monkeypatch.setattr(tb.log, "info", lambda evt, **kw: seen.append((evt, kw)))

    await tb._heartbeat(_ctx(running=True))

    assert ("telegram_bot_heartbeat", {"polling": True}) in seen


@pytest.mark.asyncio
async def test_heartbeat_errors_and_pages_when_the_updater_has_stopped(monkeypatch):
    """The zombie case: live process, dead poll loop, nothing to restart it."""
    errors: list[str] = []
    monkeypatch.setattr(tb.log, "error", lambda evt, **kw: errors.append(evt))

    sent: list[str] = []

    async def _fake_notify(_settings, text, **_kw):
        sent.append(text)
        return True

    import betbot.notify as notify

    monkeypatch.setattr(notify, "notify_operator", _fake_notify)

    await tb._heartbeat(_ctx(running=False))

    assert "telegram_bot_not_polling" in errors
    assert sent and "NOT POLLING" in sent[0]
    assert "manual restart" in sent[0]


@pytest.mark.asyncio
async def test_heartbeat_never_raises_when_notification_fails(monkeypatch):
    """A monitor that can crash the bot is worse than the gap it watches."""
    monkeypatch.setattr(tb.log, "error", lambda evt, **kw: None)

    async def _boom(*_a, **_kw):
        raise RuntimeError("telegram down")

    import betbot.notify as notify

    monkeypatch.setattr(notify, "notify_operator", _boom)

    await tb._heartbeat(_ctx(running=False))  # must not raise


# ---------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------
def test_build_application_registers_an_error_handler_and_heartbeat(settings):
    settings.telegram_bot_token = "123456:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEF"
    app = tb.build_application(settings)

    assert app.error_handlers, "PTB logged 'No error handlers are registered'"
    assert tb._log_network_error in app.error_handlers

    names = {j.name for j in app.job_queue.jobs()}
    assert "telegram_heartbeat" in names


def test_heartbeat_interval_is_short_enough_to_be_a_monitor(settings):
    """An 11-hour silence must be impossible to mistake for health."""
    assert tb.HEARTBEAT_SECONDS <= 30 * 60
