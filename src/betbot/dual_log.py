"""Health of the challenger dual-log (``model_predictions``).

WHAT THIS TABLE IS FOR
----------------------
``model_predictions`` is the head-to-head ledger: one row per fixture holding
what each competing model predicted BEFORE kickoff, plus each model's RPS once
the result lands. It is the only evidence that can retire or promote a
challenger, and the roadmap's "accumulate until n is about 30 settled, then
re-run the gate" step reads exactly this table.

WHY IT WENT COLD — AND WHY THAT WAS NOT A BUG
---------------------------------------------
On 2026-08-22 the table was found frozen: 104 rows, last write
``2026-07-17 08:01:02``, while ``predictions`` and ``paper_bets`` had both
written that morning. Five weeks of apparent silent failure.

It was not a silent failure. Every one of the 104 rows is an INTERNATIONAL
fixture (the last is Spain v Argentina, the 2026 World Cup final), because the
dual-log only ever existed inside the World Cup engine. Writes stopped on
2026-07-17 because THE TOURNAMENT ENDED. Three weeks later ``6abc132``
("strip: club-focused core — remove WC machinery") deleted the machinery
outright: ``international_engine``, ``model_select``/Hedge, and both the
dispersion and MOV challengers, along with their ``BETBOT_DISPERSION_FIX`` /
``BETBOT_MOV_FIX`` flags. That is why those flags are absent from ``.env``:
nothing reads them any more.

So the club engine has never written a dual-log row, and never will until one
is built for it. The real defect is not a broken write — it is that a table the
roadmap depends on could go permanently cold without anything saying so. The
bot's own rule applies: **silence is not health.**

WHAT THIS MODULE DOES
---------------------
It states the situation out loud, on a schedule, instead of letting the
operator infer it from a stale ``max(created_at)`` five weeks late:

* while :data:`CHALLENGER_DUAL_LOG_ENABLED` is ``False`` it reports that
  accumulation is FROZEN, and at what counts, so "we're gathering data" can
  never quietly mean "we are gathering nothing";
* once a challenger is wired for the club engine and the flag flips, the same
  audit arms as a genuine staleness alarm: main path writing + dual log not
  writing is then a real regression, and it pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from betbot.logging import get_logger
from betbot.notify import notify_operator
from betbot.storage.db import session_scope
from betbot.storage.models import ModelPrediction, PredictionRow

log = get_logger(__name__)

#: Does ANY code path in this build write a challenger row to
#: ``model_predictions``?
#:
#: It does not. ``6abc132`` removed the dispersion challenger, the MOV
#: challenger and the Hedge model-selection layer together with the World Cup
#: engine that hosted them; ``grep -rn "model_predictions" src`` now finds only
#: the ORM model and its migrations.
#:
#: This is a BUILD FACT and is deliberately not a ``Settings`` field: no ``.env``
#: edit can make a deleted challenger start logging. Whoever rebuilds the
#: dual-log for the club engine flips this in the SAME commit that adds the
#: write, which is what arms the staleness alarm below.
CHALLENGER_DUAL_LOG_ENABLED: bool = False

#: How far the dual log may lag the main prediction path before it is stale.
#: Only meaningful once :data:`CHALLENGER_DUAL_LOG_ENABLED` is True. Generous
#: on purpose — an international break or a blank midweek is not a fault.
STALE_AFTER_DAYS: float = 3.0


@dataclass(frozen=True)
class DualLogAudit:
    """What the challenger ledger currently holds, and whether that is OK."""

    enabled: bool
    rows: int
    settled: int
    last_dual_write: datetime | None
    last_main_write: datetime | None
    lag_days: float | None
    healthy: bool
    reason: str

    @property
    def summary(self) -> str:
        last = (
            self.last_dual_write.strftime("%Y-%m-%d %H:%M")
            if self.last_dual_write
            else "never"
        )
        return (
            f"model_predictions: {self.rows} rows, {self.settled} settled, "
            f"last write {last} ({self.reason})"
        )


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; treat them as UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def audit_dual_log(*, now: datetime | None = None) -> DualLogAudit:
    """Read-only snapshot of the challenger ledger's health.

    Touches nothing: three aggregate SELECTs. Never raises for an empty table —
    "never written" is a legitimate state that must still be reportable.
    """
    now = now or datetime.now(timezone.utc)
    with session_scope() as session:
        rows = session.execute(
            select(func.count()).select_from(ModelPrediction)
        ).scalar_one()
        settled = session.execute(
            select(func.count())
            .select_from(ModelPrediction)
            .where(ModelPrediction.outcome.is_not(None))
        ).scalar_one()
        last_dual = session.execute(
            select(func.max(ModelPrediction.created_at))
        ).scalar_one()
        last_main = session.execute(
            select(func.max(PredictionRow.created_at))
        ).scalar_one()

    last_dual = _as_utc(last_dual)
    last_main = _as_utc(last_main)
    lag_days = (
        (last_main - last_dual).total_seconds() / 86400.0
        if last_main and last_dual
        else None
    )

    if not CHALLENGER_DUAL_LOG_ENABLED:
        # Not a fault — a DECLARED gap. Reported anyway, because a roadmap step
        # that reads this table needs to know it is not filling.
        return DualLogAudit(
            enabled=False,
            rows=rows,
            settled=settled,
            last_dual_write=last_dual,
            last_main_write=last_main,
            lag_days=lag_days,
            healthy=True,
            reason="no challenger wired — accumulation frozen",
        )

    # From here on the build claims to dual-log, so silence IS a fault.
    if last_main is None:
        return DualLogAudit(
            enabled=True,
            rows=rows,
            settled=settled,
            last_dual_write=last_dual,
            last_main_write=last_main,
            lag_days=lag_days,
            healthy=True,
            reason="main prediction path has not written either — nothing to compare",
        )
    if last_dual is None:
        return DualLogAudit(
            enabled=True,
            rows=rows,
            settled=settled,
            last_dual_write=None,
            last_main_write=last_main,
            lag_days=None,
            healthy=False,
            reason="challenger enabled but the dual log has NEVER been written",
        )
    if last_main - last_dual > timedelta(days=STALE_AFTER_DAYS):
        return DualLogAudit(
            enabled=True,
            rows=rows,
            settled=settled,
            last_dual_write=last_dual,
            last_main_write=last_main,
            lag_days=lag_days,
            healthy=False,
            reason=(
                f"dual log is {lag_days:.1f}d behind the main prediction path "
                f"(limit {STALE_AFTER_DAYS}d)"
            ),
        )
    return DualLogAudit(
        enabled=True,
        rows=rows,
        settled=settled,
        last_dual_write=last_dual,
        last_main_write=last_main,
        lag_days=lag_days,
        healthy=True,
        reason="keeping up with the main prediction path",
    )


async def report_dual_log_health(
    settings, *, send_fn=None, now: datetime | None = None
) -> DualLogAudit:
    """Audit the ledger and tell the operator when it is not accumulating.

    Returns the audit so callers/tests can assert on it. Never raises: this is
    a daemon job, and a monitor that can take the daemon down is worse than the
    thing it monitors.
    """
    audit = audit_dual_log(now=now)

    if not audit.enabled:
        # Loud enough to be seen, bounded so it cannot become daily noise:
        # notify_operator's per-kind cooldown collapses repeats.
        log.warning(
            "challenger_dual_log_disabled",
            rows=audit.rows,
            settled=audit.settled,
            last_write=str(audit.last_dual_write),
            note="no challenger writes to model_predictions in this build",
        )
        await notify_operator(
            settings,
            "Challenger dual-log is NOT accumulating.\n\n"
            f"{audit.summary}\n\n"
            "No code in this build writes model_predictions — the dispersion "
            "and MOV challengers were removed with the World Cup engine "
            "(6abc132), so the counts above are FINAL, not in progress. Any "
            "plan that waits for this table to reach a sample size will wait "
            "forever until a challenger is rebuilt for the club engine.",
            kind="challenger_dual_log_stale",
            send_fn=send_fn,
        )
        return audit

    if not audit.healthy:
        log.error(
            "challenger_dual_log_stale",
            rows=audit.rows,
            settled=audit.settled,
            lag_days=audit.lag_days,
            reason=audit.reason,
        )
        await notify_operator(
            settings,
            "Challenger dual-log has gone STALE.\n\n"
            f"{audit.summary}\n\n"
            f"The main prediction path is writing but model_predictions is "
            f"not: {audit.reason}. Challenger comparison is dead until this "
            "is fixed — do not read the gate off this table meanwhile.",
            kind="challenger_dual_log_stale",
            send_fn=send_fn,
        )
        return audit

    log.info(
        "challenger_dual_log_ok",
        rows=audit.rows,
        settled=audit.settled,
        lag_days=audit.lag_days,
    )
    return audit


async def dual_log_audit_tick(settings) -> None:
    """Scheduled entry point. Best-effort; never raises into the scheduler."""
    try:
        await report_dual_log_health(settings)
    except Exception as e:  # noqa: BLE001 — a monitor must not kill the daemon
        log.error("dual_log_audit_failed", error=str(e))
