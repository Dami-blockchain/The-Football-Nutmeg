"""Scheduler registration helpers — make the "never awaited" bug impossible.

APScheduler decides ONCE, at fire time, whether to ``await`` a job or just
call it: :class:`~apscheduler.executors.asyncio.AsyncIOExecutor` awaits the
result only when :func:`apscheduler.util.iscoroutinefunction_partial` says the
callable is a coroutine function. Wrap an ``async def`` in a SYNC ``lambda``
and the executor calls the lambda, gets a coroutine object back, and throws it
away — the job silently never runs. Python's only complaint is a
``RuntimeWarning: coroutine '...' was never awaited`` emitted from the garbage
collector, far from the guilty line.

That is not a hypothetical: it took out the daily pre-match/lineup alert
re-scan (``reschedule_kickoff_alerts``) and the player-minutes backfill for
days, in production, with no error and no missing-alert signal.

So registration goes through :func:`add_async_job`, which REFUSES a callable
the executor would drop, at registration time (daemon start) rather than at
fire time (whenever, silently). Bind arguments with ``args=``/``kwargs=`` —
never with a wrapping lambda.
"""

from __future__ import annotations

from typing import Any, Callable

try:  # the executor's own predicate — unwraps functools.partial
    from apscheduler.util import iscoroutinefunction_partial as _iscoro
except ImportError:  # pragma: no cover — apscheduler <3.6 / future rename
    from asyncio import iscoroutinefunction as _iscoro


def is_async_job(func: Callable[..., Any]) -> bool:
    """True iff APScheduler's asyncio executor will AWAIT ``func``.

    This is deliberately the executor's OWN predicate rather than a
    re-implementation, so the guard can never drift from the runtime
    behaviour it is guarding. ``functools.partial`` of a coroutine function
    passes; a sync ``lambda`` returning a coroutine does not.
    """
    return bool(_iscoro(func))


def add_async_job(scheduler, func: Callable[..., Any], **kwargs):
    """``scheduler.add_job`` restricted to callables APScheduler will await.

    Raises :class:`TypeError` — loudly, at registration time — for a callable
    the executor would call-and-drop. Bind arguments through APScheduler's own
    ``args=`` / ``kwargs=`` parameters; a ``lambda: coro_fn(x)`` wrapper is
    exactly the bug this exists to prevent.
    """
    if not is_async_job(func):
        raise TypeError(
            f"add_async_job requires a coroutine function, got {func!r}. "
            "APScheduler would CALL this and discard the coroutine it returns, "
            "so the job would silently never run. Pass the 'async def' itself "
            "and bind arguments with args=/kwargs= instead of wrapping it in a "
            "sync lambda."
        )
    return scheduler.add_job(func, **kwargs)


def unawaitable_jobs(scheduler) -> list[str]:
    """Ids of registered jobs whose callable APScheduler would NOT await.

    Empty list means every job on the scheduler will actually run. Used by the
    regression guard so a job registered through a raw ``add_job`` — bypassing
    :func:`add_async_job` — is still caught.
    """
    return [job.id for job in scheduler.get_jobs() if not is_async_job(job.func)]
