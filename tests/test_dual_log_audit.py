"""The challenger dual-log must never go cold in silence again.

``model_predictions`` stopped receiving rows on 2026-07-17 and nobody found
out until 2026-08-22 — five weeks in which the roadmap believed it was
accumulating challenger results and was in fact accumulating nothing. These
tests pin the alarm, and pin the honesty of the "not accumulating" state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import betbot.dual_log as dual_log
from betbot.dual_log import (
    CHALLENGER_DUAL_LOG_ENABLED,
    audit_dual_log,
    report_dual_log_health,
)
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import ModelPrediction, PredictionRow

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def db(tmp_path):
    init_engine(tmp_path / "dual.sqlite")
    yield


def _add_main_prediction(created_at: datetime) -> None:
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=int(created_at.timestamp()) % 100000,
                competition_code="PL",
                kickoff=created_at,
                run_date=created_at.date().isoformat(),
                home_team="Arsenal FC",
                away_team="Chelsea FC",
                p_home=0.5,
                p_draw=0.25,
                p_away=0.25,
                home_score=1.0,
                away_score=0.5,
                draw_score=0.0,
                created_at=created_at,
            )
        )


def _add_dual_row(created_at: datetime, *, settled: bool = False) -> None:
    with session_scope() as s:
        s.add(
            ModelPrediction(
                fixture_id=int(created_at.timestamp()) % 100000,
                home_team="Spain",
                away_team="Argentina",
                g_home=0.4, g_draw=0.3, g_away=0.3,
                e_home=0.4, e_draw=0.3, e_away=0.3,
                w_glicko=0.5, w_ensemble=0.5,
                outcome="HOME" if settled else None,
                created_at=created_at,
            )
        )


# ---------------------------------------------------------------------
# The build fact
# ---------------------------------------------------------------------
def test_club_dual_log_is_wired_and_the_flag_matches():
    """The flag must not drift from reality.

    6abc132 left this table dead; it was RE-ARMED for the club engine
    (betbot.main._score_and_log_one -> repos.upsert_model_prediction, scored at
    settlement), and the flag was flipped in that same commit — which is what
    arms the staleness alarm below.
    """
    assert CHALLENGER_DUAL_LOG_ENABLED is True


def test_the_removed_challenger_flags_are_genuinely_unread():
    """BETBOT_MOV_FIX / BETBOT_DISPERSION_FIX are absent from .env because
    nothing reads them — not because the operator dropped them."""
    from betbot.config import Settings

    s = Settings(_env_file=None)
    assert not hasattr(s, "mov_fix")
    assert not hasattr(s, "dispersion_fix")


# ---------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------
def test_frozen_dual_log_now_trips_the_stale_alarm(db):
    """The 2026-07-17 live shape, re-judged: now that the club dual-log is
    wired, a main path writing today against a five-week-old dual log is a
    STALE FAULT, not a declared gap."""
    _add_dual_row(datetime(2026, 7, 17, 8, 1, tzinfo=timezone.utc), settled=True)
    _add_main_prediction(datetime(2026, 8, 22, 8, 8, tzinfo=timezone.utc))

    a = audit_dual_log(now=NOW)
    assert a.enabled is True
    assert a.rows == 1
    assert a.settled == 1
    assert a.lag_days > 35
    assert a.healthy is False
    assert "behind the main prediction path" in a.reason


def test_audit_survives_a_completely_empty_ledger(db):
    a = audit_dual_log(now=NOW)
    assert a.rows == 0
    assert a.last_dual_write is None


@pytest.mark.asyncio
async def test_operator_is_paged_when_the_club_dual_log_goes_stale(db, settings):
    """With the club dual-log wired, a stale ledger pages the human as a STALE
    fault (not the old 'not accumulating' notice)."""
    _add_dual_row(datetime(2026, 7, 17, 8, 1, tzinfo=timezone.utc), settled=True)
    _add_main_prediction(datetime(2026, 8, 22, 8, 8, tzinfo=timezone.utc))
    sent: list[str] = []

    async def _send(_settings, _chat_id, text, **_kw):
        sent.append(text)
        return True

    settings.telegram_allowed_user_id = 123
    await report_dual_log_health(settings, send_fn=_send, now=NOW)

    assert sent, "a stale dual log must reach the human"
    assert "STALE" in sent[0]


# ---------------------------------------------------------------------
# Once a challenger IS wired, the same audit becomes a real alarm
# ---------------------------------------------------------------------
def test_enabled_and_stale_is_unhealthy(db, monkeypatch):
    monkeypatch.setattr(dual_log, "CHALLENGER_DUAL_LOG_ENABLED", True)
    _add_dual_row(NOW - timedelta(days=10))
    _add_main_prediction(NOW)

    a = audit_dual_log(now=NOW)
    assert a.enabled is True
    assert a.healthy is False
    assert a.lag_days == pytest.approx(10.0, abs=0.1)


def test_enabled_and_keeping_up_is_healthy(db, monkeypatch):
    monkeypatch.setattr(dual_log, "CHALLENGER_DUAL_LOG_ENABLED", True)
    _add_dual_row(NOW - timedelta(hours=2))
    _add_main_prediction(NOW)

    a = audit_dual_log(now=NOW)
    assert a.healthy is True


def test_enabled_but_never_written_is_unhealthy(db, monkeypatch):
    """A challenger that claims to log but has produced nothing is a fault."""
    monkeypatch.setattr(dual_log, "CHALLENGER_DUAL_LOG_ENABLED", True)
    _add_main_prediction(NOW)

    a = audit_dual_log(now=NOW)
    assert a.healthy is False
    assert "NEVER been written" in a.reason


def test_enabled_with_no_main_predictions_is_not_blamed(db, monkeypatch):
    """An idle bot is not a broken dual log — don't cry wolf in the off-season."""
    monkeypatch.setattr(dual_log, "CHALLENGER_DUAL_LOG_ENABLED", True)
    a = audit_dual_log(now=NOW)
    assert a.healthy is True


@pytest.mark.asyncio
async def test_stale_challenger_pages_the_operator(db, settings, monkeypatch):
    monkeypatch.setattr(dual_log, "CHALLENGER_DUAL_LOG_ENABLED", True)
    _add_dual_row(NOW - timedelta(days=10))
    _add_main_prediction(NOW)
    sent: list[str] = []

    async def _send(_settings, _chat_id, text, **_kw):
        sent.append(text)
        return True

    settings.telegram_allowed_user_id = 123
    a = await report_dual_log_health(settings, send_fn=_send, now=NOW)

    assert a.healthy is False
    assert sent and "STALE" in sent[0]


# ---------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_audit_tick_never_raises_into_the_scheduler(settings, monkeypatch):
    """A monitor that can kill the daemon is worse than the gap it watches."""
    def _boom(**_kw):
        raise RuntimeError("db gone")

    monkeypatch.setattr(dual_log, "audit_dual_log", _boom)
    await dual_log.dual_log_audit_tick(settings)  # must not raise


def test_audit_is_registered_as_an_awaitable_daily_job(settings):
    """Registered through add_async_job, so it cannot become a dropped
    coroutine like reschedule_kickoff_alerts did."""
    from betbot.daily_jobs import register_daily_jobs
    from betbot.scheduling import is_async_job

    registered = {}

    class _Rec:
        def add_job(self, func, trigger=None, *, id=None, args=None, **_kw):
            registered[id] = func

    async def _noop():
        return None

    register_daily_jobs(_Rec(), settings, matchday_notice=_noop)

    assert "challenger_dual_log_audit" in registered
    assert is_async_job(registered["challenger_dual_log_audit"])


@pytest.mark.asyncio
async def test_the_tick_redacts_a_token_shaped_error_before_logging(
    settings, monkeypatch
):
    """`dual_log_audit_tick` must log the error's SHAPE, not its raw text.

    The codebase's rule is "log shape, not text" (a Telegram bot token once
    leaked through httpx INFO logs on every send). If the audit fails with a
    message that happens to contain a token-shaped substring, the tick's
    catch-all handler must pass it through `notify._redact` so the token never
    reaches the logs verbatim.
    """
    token = "123456789:AAHkq9w6f2xVeryLongLookingBotToken12"

    def _boom(**_kw):
        raise RuntimeError(f"connect failed for bot {token}")

    monkeypatch.setattr(dual_log, "audit_dual_log", _boom)

    logged: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        dual_log.log, "error", lambda event, **kw: logged.append((event, kw))
    )

    await dual_log.dual_log_audit_tick(settings)  # must not raise

    assert logged and logged[0][0] == "dual_log_audit_failed"
    error_text = logged[0][1]["error"]
    assert token not in error_text
    assert "<REDACTED>" in error_text
