"""Guards against the "coroutine ... was never awaited" scheduler bug.

APScheduler's asyncio executor awaits a job ONLY when
:func:`apscheduler.util.iscoroutinefunction_partial` says its callable is a
coroutine function. Register an ``async def`` behind a SYNC ``lambda`` and the
executor calls the lambda, receives a coroutine object, and drops it — the job
silently never runs. That killed ``reschedule_kickoff_alerts`` (the daily
re-scan that schedules every pre-match + confirmed-lineup alert) and
``player_minutes_backfill`` in production for days, with no error: the only
trace was a ``RuntimeWarning`` emitted by the garbage collector.

These tests walk the REAL registration path — ``run_daemon`` and
``register_daily_jobs`` — rather than asserting against a hand-written list, so
they catch the NEXT occurrence on any job, not just the two that broke.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone

import pytest

import betbot.main as main
from betbot.daily_jobs import register_daily_jobs
from betbot.scheduling import add_async_job, is_async_job, unawaitable_jobs


# ----------------------------------------------------------------------
# Harness: capture every add_job the daemon makes, without starting anything
# ----------------------------------------------------------------------
class _StopBeforeStart(Exception):
    """Raised from the fake ``scheduler.start()`` to unwind ``run_daemon``.

    Every registration happens BEFORE ``scheduler.start()``, so raising there
    captures the complete job set while skipping the daemon's scoring tick,
    its network calls and its ``asyncio.Event().wait()`` forever-block.
    """


class _FakeJob:
    def __init__(self, func, job_id, args, kwargs, trigger):
        self.func = func
        self.id = job_id
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.trigger = trigger


class _RecordingScheduler:
    """Minimal stand-in for AsyncIOScheduler that records registrations."""

    def __init__(self, *_a, **_kw):
        self.jobs: list[_FakeJob] = []

    def add_job(
        self, func, trigger=None, *, id=None, args=None, kwargs=None, **_rest
    ):
        job = _FakeJob(func, id, args, kwargs, trigger)
        self.jobs = [j for j in self.jobs if j.id != id]
        self.jobs.append(job)
        return job

    def get_jobs(self):
        return list(self.jobs)

    def start(self):
        raise _StopBeforeStart

    def shutdown(self, wait=False):  # pragma: no cover - never reached
        pass


def _daemon_jobs(monkeypatch, settings) -> list[_FakeJob]:
    """Every job ``run_daemon`` registers, captured just before start()."""
    captured: dict[str, _RecordingScheduler] = {}

    def _factory(*_a, **_kw):
        sched = _RecordingScheduler()
        captured["scheduler"] = sched
        return sched

    monkeypatch.setattr(main, "AsyncIOScheduler", _factory)
    monkeypatch.setattr(main, "init_engine", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    with pytest.raises(_StopBeforeStart):
        main.run_daemon()

    jobs = captured["scheduler"].jobs
    assert jobs, "run_daemon registered no jobs — harness is broken"
    return jobs


# ----------------------------------------------------------------------
# 1. The specific regression: the daily alert re-scan must be awaited
# ----------------------------------------------------------------------
def test_reschedule_kickoff_alerts_callable_is_a_coroutine_function(
    monkeypatch, settings
):
    """The exact bug: this job was a sync lambda, so it never ran.

    Fails against the pre-fix code, where ``job.func`` was
    ``lambda: _schedule_kickoff_alerts(scheduler)``.
    """
    jobs = _daemon_jobs(monkeypatch, settings)
    job = next((j for j in jobs if j.id == "reschedule_kickoff_alerts"), None)
    assert job is not None, "the daily alert re-scan job is not registered at all"
    assert asyncio.iscoroutinefunction(job.func), (
        "reschedule_kickoff_alerts is registered as a SYNC callable. "
        "APScheduler will call it, get a coroutine back and discard it, so the "
        "daily pre-match/lineup alert re-scan will never run."
    )
    # ``scheduler`` must be bound through args=, not captured by a wrapper.
    assert job.args, "the scheduler must be bound via args=, not a lambda"


def test_reschedule_kickoff_alerts_actually_schedules_when_invoked(
    monkeypatch, settings
):
    """Awaiting the registered callable really does register alert jobs.

    Coroutine-ness alone would be satisfied by an ``async def`` that does
    nothing; this pins the end-to-end behaviour the operator cares about.
    """
    jobs = _daemon_jobs(monkeypatch, settings)
    job = next(j for j in jobs if j.id == "reschedule_kickoff_alerts")

    now = datetime.now(timezone.utc)
    ko = now + timedelta(hours=6)

    class _Pred:
        fixture_id = 559715
        competition_code = "FL1"
        kickoff = ko

    monkeypatch.setattr(
        main, "predictions_for_kickoff_range", lambda _s, _e: [_Pred()]
    )
    sched = _RecordingScheduler()
    asyncio.run(job.func(sched, *job.args[1:]))

    ids = {j.id for j in sched.jobs}
    assert "predict_early_559715" in ids
    assert "predict_late_559715" in ids
    for j in sched.jobs:
        assert is_async_job(j.func), f"{j.id} would never be awaited"


# ----------------------------------------------------------------------
# 2. The generic guard: NO registered job may be a sync coroutine factory
# ----------------------------------------------------------------------
def test_no_daemon_job_is_a_sync_callable(monkeypatch, settings):
    """Every job ``run_daemon`` registers must be one APScheduler awaits."""
    jobs = _daemon_jobs(monkeypatch, settings)
    offenders = [j.id for j in jobs if not is_async_job(j.func)]
    assert offenders == [], (
        f"jobs registered as sync callables: {offenders}. APScheduler's "
        "asyncio executor only awaits coroutine functions (and partials of "
        "them); anything else is called and its return value discarded."
    )


def test_no_daemon_job_returns_an_unawaited_coroutine(monkeypatch, settings):
    """Dynamic form of the guard: call any sync job and inspect its return.

    This is the shape of the failure that actually shipped — a callable that
    LOOKS fine to the scheduler but hands back a coroutine nobody awaits.
    """
    jobs = _daemon_jobs(monkeypatch, settings)
    for job in jobs:
        if is_async_job(job.func):
            continue
        result = job.func(*job.args, **job.kwargs)
        if inspect.isawaitable(result):
            result.close() if inspect.iscoroutine(result) else None
            pytest.fail(
                f"job {job.id!r} is a sync callable returning {result!r}; "
                "APScheduler discards it and the job never runs"
            )


def test_register_daily_jobs_registers_only_awaitable_callables(settings):
    """``register_daily_jobs`` is a second registration site — guard it too."""

    async def _notice() -> None:  # pragma: no cover - never invoked
        return None

    sched = _RecordingScheduler()
    register_daily_jobs(sched, settings, matchday_notice=_notice)

    assert {j.id for j in sched.jobs} == {"matchday_notice", "player_minutes_backfill"}
    assert unawaitable_jobs(sched) == []


def test_daemon_scheduler_reports_no_unawaitable_jobs(monkeypatch, settings):
    """The production-code sweep agrees with the tests."""
    captured: dict[str, _RecordingScheduler] = {}

    def _factory(*_a, **_kw):
        sched = _RecordingScheduler()
        captured["scheduler"] = sched
        return sched

    monkeypatch.setattr(main, "AsyncIOScheduler", _factory)
    monkeypatch.setattr(main, "init_engine", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with pytest.raises(_StopBeforeStart):
        main.run_daemon()
    assert unawaitable_jobs(captured["scheduler"]) == []


# ----------------------------------------------------------------------
# 3. add_async_job refuses the broken shape at REGISTRATION time
# ----------------------------------------------------------------------
def test_add_async_job_rejects_a_sync_lambda_wrapping_a_coroutine():
    """The exact pre-fix line must now be impossible to write."""

    async def _job(_arg) -> None:  # pragma: no cover - never invoked
        return None

    sched = _RecordingScheduler()
    with pytest.raises(TypeError, match="coroutine function"):
        add_async_job(sched, lambda: _job("x"), id="broken")
    assert sched.jobs == [], "a rejected job must not be registered"


def test_add_async_job_accepts_a_coroutine_function_and_a_partial():
    import functools

    async def _job(_arg=None) -> None:  # pragma: no cover - never invoked
        return None

    sched = _RecordingScheduler()
    add_async_job(sched, _job, args=("x",), id="ok")
    add_async_job(sched, functools.partial(_job, "x"), id="ok_partial")
    assert unawaitable_jobs(sched) == []


def test_add_async_job_rejects_a_plain_sync_function():
    def _job() -> None:  # pragma: no cover - never invoked
        return None

    sched = _RecordingScheduler()
    with pytest.raises(TypeError):
        add_async_job(sched, _job, id="sync")


# ----------------------------------------------------------------------
# 4. Runtime self-check: a coverage gap is LOUD, not silent
# ----------------------------------------------------------------------
def _operator_settings(settings, uid: int = 424242):
    return settings.model_copy(update={"telegram_allowed_user_id": uid})


def test_audit_alert_coverage_lists_only_missing_jobs():
    sched = _RecordingScheduler()

    async def _noop() -> None:  # pragma: no cover - never invoked
        return None

    when = datetime.now(timezone.utc) + timedelta(hours=1)
    sched.add_job(_noop, id="predict_early_1")
    plan = [("predict_early_1", when), ("predict_late_1", when)]
    assert main.audit_alert_coverage(sched, plan) == ["predict_late_1"]


async def test_report_alert_coverage_telegrams_the_operator_on_a_gap(settings):
    s = _operator_settings(settings)
    sched = _RecordingScheduler()
    when = datetime.now(timezone.utc) + timedelta(hours=1)
    plan = [("predict_early_559715", when), ("predict_late_559715", when)]

    sent: list[tuple[int, str]] = []

    async def _send(_settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    missing = await main.report_alert_coverage(
        sched, plan, settings=s, send_fn=_send
    )
    assert missing == ["predict_early_559715", "predict_late_559715"]
    assert len(sent) == 1
    chat_id, body = sent[0]
    assert chat_id == s.telegram_allowed_user_id
    assert "predict_early_559715" in body


async def test_report_alert_coverage_is_quiet_when_coverage_is_complete(settings):
    s = _operator_settings(settings)
    sched = _RecordingScheduler()

    async def _noop() -> None:  # pragma: no cover - never invoked
        return None

    when = datetime.now(timezone.utc) + timedelta(hours=1)
    sched.add_job(_noop, id="predict_early_1")
    sched.add_job(_noop, id="predict_late_1")

    sent: list[tuple[int, str]] = []

    async def _send(_settings, chat_id, text):  # pragma: no cover - must not fire
        sent.append((chat_id, text))
        return True

    missing = await main.report_alert_coverage(
        sched,
        [("predict_early_1", when), ("predict_late_1", when)],
        settings=s,
        send_fn=_send,
    )
    assert missing == []
    assert sent == []


async def test_report_alert_coverage_survives_a_failing_telegram_send(settings):
    """The alert must never crash the scheduling pass it is auditing."""
    s = _operator_settings(settings)
    sched = _RecordingScheduler()
    when = datetime.now(timezone.utc) + timedelta(hours=1)

    async def _send(*_a, **_kw):
        raise RuntimeError("telegram down")

    missing = await main.report_alert_coverage(
        sched, [("predict_late_1", when)], settings=s, send_fn=_send
    )
    assert missing == ["predict_late_1"]


def test_scheduling_pass_reports_a_gap_when_jobs_vanish(monkeypatch, settings):
    """End-to-end: a pass that plans jobs but registers none alerts the operator.

    This is the production failure reduced to a unit: the plan was non-empty,
    nothing landed on the scheduler, and NOTHING said so.
    """
    s = _operator_settings(settings)
    jobs = _daemon_jobs(monkeypatch, s)
    rescan = next(j for j in jobs if j.id == "reschedule_kickoff_alerts")

    now = datetime.now(timezone.utc)

    class _Pred:
        fixture_id = 559715
        competition_code = "FL1"
        kickoff = now + timedelta(hours=6)

    monkeypatch.setattr(
        main, "predictions_for_kickoff_range", lambda _st, _en: [_Pred()]
    )

    class _DroppingScheduler(_RecordingScheduler):
        def add_job(self, func, trigger=None, *, id=None, **_rest):
            return _FakeJob(func, id, (), {}, trigger)  # registers nothing

    sent: list[tuple[int, str]] = []

    async def _send(_settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    from betbot import notify

    monkeypatch.setattr(notify, "send_telegram_to", _send)
    # Sync test on purpose: run_daemon() (via _daemon_jobs) calls asyncio.run(),
    # which refuses to nest inside an already-running loop.
    asyncio.run(rescan.func(_DroppingScheduler(), *rescan.args[1:]))

    assert sent, "a scheduling pass that registered nothing stayed silent"
    assert "predict_early_559715" in sent[0][1]
