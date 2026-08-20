"""The loud paths now reach a human, not just the log file.

Three ERROR paths were logging into the void. Each is wired to
notify_operator here, and each wiring is asserted: an ERROR log that nobody
reads is the failure mode the whole notification layer exists to end.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot import main
from betbot import settlement as settlement_mod
from betbot.settlement import SettlementWatcher

OPERATOR_ID = 424242


@pytest.fixture()
def op_settings(settings):
    return settings.model_copy(
        update={
            "telegram_allowed_user_id": OPERATOR_ID,
            "telegram_bot_token": "test-token",
        }
    )


class _RecordingScheduler:
    """Minimal APScheduler stand-in: remembers what was registered."""

    def __init__(self) -> None:
        self.jobs: list = []

    def add_job(self, func, trigger=None, id=None, **kw):  # noqa: A002
        job = type("_Job", (), {"func": func, "id": id, "args": (), "kwargs": {}})()
        self.jobs.append(job)
        return job

    def get_jobs(self):
        return list(self.jobs)


class _Outbox:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[tuple[int, str]] = []

    async def __call__(self, _settings, chat_id, text, parse_mode="Markdown"):
        self.sent.append((chat_id, text))
        return self.ok


# ----------------------------------------------------------------------
# 1. The hourly alert-coverage watchdog
# ----------------------------------------------------------------------
async def test_a_coverage_gap_reaches_the_operator(op_settings):
    sched = _RecordingScheduler()
    when = datetime.now(timezone.utc) + timedelta(hours=1)
    out = _Outbox()

    missing = await main.report_alert_coverage(
        sched, [("predict_early_1", when)], settings=op_settings, send_fn=out
    )
    assert missing == ["predict_early_1"]
    chat_id, body = out.sent[0]
    assert chat_id == OPERATOR_ID
    assert "predict_early_1" in body


async def test_a_persistent_gap_is_not_re_sent_every_hour(op_settings):
    """The watchdog runs hourly. 24 identical pushes a day would train the
    operator to ignore the exact signal it exists to send."""
    sched = _RecordingScheduler()
    when = datetime.now(timezone.utc) + timedelta(hours=1)
    out = _Outbox()

    for _hour in range(6):
        await main.report_alert_coverage(
            sched, [("predict_early_1", when)], settings=op_settings, send_fn=out
        )
    assert len(out.sent) == 1


async def test_a_coverage_gap_that_fails_to_send_never_crashes_the_pass(op_settings):
    sched = _RecordingScheduler()
    when = datetime.now(timezone.utc) + timedelta(hours=1)

    async def _boom(*_a, **_kw):
        raise RuntimeError("telegram down")

    missing = await main.report_alert_coverage(
        sched, [("predict_late_1", when)], settings=op_settings, send_fn=_boom
    )
    assert missing == ["predict_late_1"]


# ----------------------------------------------------------------------
# 2. The unawaitable-jobs sweep at daemon start
# ----------------------------------------------------------------------
class _StopBeforeStart(Exception):
    pass


def test_the_unawaitable_jobs_sweep_notifies_the_operator(monkeypatch, op_settings):
    """A job APScheduler would drop is a SILENTLY dead feature. Tell the human.

    The sweep already logged ERROR; this asserts the ERROR is now also a
    Telegram push, using the same run_daemon harness as the sweep's own test.
    """
    sent: list[dict] = []

    async def _notify(_settings, text, **kw):
        sent.append({"text": text, **kw})
        return True

    class _Sched(_RecordingScheduler):
        def start(self):
            raise _StopBeforeStart

        def shutdown(self, *_a, **_kw):
            return None

    monkeypatch.setattr(main, "AsyncIOScheduler", lambda *_a, **_kw: _Sched())
    monkeypatch.setattr(main, "init_engine", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "get_settings", lambda: op_settings)
    monkeypatch.setattr(main, "unawaitable_jobs", lambda _s: ["ghost_job"])
    monkeypatch.setattr(main, "notify_operator", _notify)

    with pytest.raises(_StopBeforeStart):
        main.run_daemon()

    assert len(sent) == 1
    assert sent[0]["kind"] == "scheduler_jobs_not_awaitable"
    assert "ghost_job" in sent[0]["text"]
    assert "NEVER RUN" in sent[0]["text"]


def test_the_sweep_is_silent_when_every_job_is_awaitable(monkeypatch, op_settings):
    sent: list[dict] = []

    async def _notify(_settings, text, **kw):  # pragma: no cover - must not fire
        sent.append({"text": text, **kw})
        return True

    class _Sched(_RecordingScheduler):
        def start(self):
            raise _StopBeforeStart

        def shutdown(self, *_a, **_kw):
            return None

    monkeypatch.setattr(main, "AsyncIOScheduler", lambda *_a, **_kw: _Sched())
    monkeypatch.setattr(main, "init_engine", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "get_settings", lambda: op_settings)
    monkeypatch.setattr(main, "notify_operator", _notify)

    with pytest.raises(_StopBeforeStart):
        main.run_daemon()

    assert sent == []


# ----------------------------------------------------------------------
# 3. The drawdown kill switch
# ----------------------------------------------------------------------
class _FakeFD:
    async def get_match(self, _fixture_id):  # pragma: no cover - no bets due
        return None


async def test_a_tripped_kill_switch_reaches_the_operator(monkeypatch, op_settings):
    """Betting has STOPPED. That cannot live only in a log line."""
    sent: list[dict] = []

    async def _notify(_settings, text, **kw):
        sent.append({"text": text, **kw})
        return True

    monkeypatch.setattr(settlement_mod, "notify_operator", _notify)
    monkeypatch.setattr(settlement_mod, "list_unsettled_bets_due", lambda *_a: [])
    monkeypatch.setattr(
        SettlementWatcher, "_score_outcomes", lambda self, now: _zero()
    )
    monkeypatch.setattr(
        SettlementWatcher,
        "_evaluate_kill_switch",
        lambda self: (True, -250.0, 1000.0),
    )

    summary = await SettlementWatcher(_FakeFD(), op_settings).settle_due(
        now=datetime.now(timezone.utc)
    )
    assert summary.kill_switch_tripped is True
    assert len(sent) == 1
    assert sent[0]["kind"] == "kill_switch_tripped"
    assert "TRIPPED" in sent[0]["text"]
    assert "-250" in sent[0]["text"] or "250.00" in sent[0]["text"]
    assert "kill-switch reset" in sent[0]["text"]


async def test_a_clear_kill_switch_says_nothing(monkeypatch, op_settings):
    sent: list[dict] = []

    async def _notify(_settings, text, **kw):  # pragma: no cover - must not fire
        sent.append({"text": text, **kw})
        return True

    monkeypatch.setattr(settlement_mod, "notify_operator", _notify)
    monkeypatch.setattr(settlement_mod, "list_unsettled_bets_due", lambda *_a: [])
    monkeypatch.setattr(
        SettlementWatcher, "_score_outcomes", lambda self, now: _zero()
    )
    monkeypatch.setattr(
        SettlementWatcher, "_evaluate_kill_switch", lambda self: (False, 10.0, 100.0)
    )

    await SettlementWatcher(_FakeFD(), op_settings).settle_due(
        now=datetime.now(timezone.utc)
    )
    assert sent == []


async def _zero() -> int:
    return 0
