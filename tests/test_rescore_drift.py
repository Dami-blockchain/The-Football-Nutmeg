"""Rescore-drift instrumentation tests (TASK E — measurement only)."""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import inspect

from betbot.storage.db import init_engine
from betbot.storage.repos import (
    record_rescore_drift,
    rescore_drift_stats,
)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "drift.sqlite")
    yield


def test_init_engine_creates_table_on_fresh_db(tmp_path):
    engine = init_engine(tmp_path / "fresh.sqlite")
    assert "rescore_drift_log" in set(inspect(engine).get_table_names())


def test_table_added_to_preexisting_db(tmp_path):
    """A DB that predates the table gains it cleanly (create_all checkfirst).

    Simulates a deployed DB with the OLD schema — a throwaway file holding an
    unrelated table and NO ``rescore_drift_log`` — then boots the engine on it
    and confirms the table is added and usable. Never touches the live DB.
    """
    path = tmp_path / "old_schema.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE legacy (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()
    assert "rescore_drift_log" not in _tables(path)

    engine = init_engine(path)
    names = set(inspect(engine).get_table_names())
    assert "rescore_drift_log" in names
    assert "legacy" in names  # additive only — the old table survives
    # And it is writable.
    record_rescore_drift(111, "early_fire", 0.7, 0.2, 0.1)
    stats = rescore_drift_stats()
    assert stats["n_fixtures"] == 1


def _tables(path) -> set[str]:
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        con.close()


def test_record_persists_row_per_stage(db):
    record_rescore_drift(42, "early_fire", 0.70, 0.20, 0.10)
    record_rescore_drift(42, "kickoff_60", 0.72, 0.18, 0.10)
    record_rescore_drift(42, "result", 0.60, 0.25, 0.15)
    stats = rescore_drift_stats()
    assert stats["n_fixtures"] == 1
    # early_fire is the baseline (no delta of its own); two later stages.
    assert set(stats["stages"]) == {"kickoff_60", "result"}
    assert stats["stages"]["kickoff_60"]["n"] == 1
    assert stats["stages"]["result"]["n"] == 1


def test_record_is_best_effort_and_never_raises(tmp_path):
    """With no engine initialised the write fails but must be swallowed."""
    import betbot.storage.db as db_mod

    db_mod._engine = None
    db_mod._SessionLocal = None
    # Must NOT raise even though there is no session factory.
    record_rescore_drift(1, "early_fire", 0.5, 0.3, 0.2)


def test_drift_stats_math(db):
    # Fixture 1: HOME pick 0.70 -> kickoff_60 0.75 (+0.05 up), result 0.60 (-0.10 down)
    record_rescore_drift(1, "early_fire", 0.70, 0.20, 0.10)
    record_rescore_drift(1, "kickoff_60", 0.75, 0.15, 0.10)
    record_rescore_drift(1, "result", 0.60, 0.25, 0.15)
    # Fixture 2: AWAY pick 0.68 -> result 0.639 (-0.041 down); no kickoff_60
    record_rescore_drift(2, "early_fire", 0.20, 0.12, 0.68)
    record_rescore_drift(2, "result", 0.231, 0.130, 0.639)
    # Fixture 3: HOME pick 0.66 -> kickoff_60 0.66 (flat)
    record_rescore_drift(3, "early_fire", 0.66, 0.20, 0.14)
    record_rescore_drift(3, "kickoff_60", 0.66, 0.20, 0.14)

    stats = rescore_drift_stats()
    assert stats["n_fixtures"] == 3

    ko = stats["stages"]["kickoff_60"]
    assert ko["n"] == 2
    assert (ko["up"], ko["down"], ko["flat"]) == (1, 0, 1)
    assert ko["mean_delta"] == pytest.approx((0.05 + 0.0) / 2)

    res = stats["stages"]["result"]
    assert res["n"] == 2
    assert (res["up"], res["down"], res["flat"]) == (0, 2, 0)
    assert res["mean_delta"] == pytest.approx((-0.10 + -0.041) / 2)


def test_baseline_pick_tracked_not_argmax_per_stage(db):
    """Delta follows the ALERT-TIME pick even if the argmax flips later."""
    # Baseline pick HOME (0.55). At result HOME crashes to 0.30 (AWAY now top).
    record_rescore_drift(9, "early_fire", 0.55, 0.25, 0.20)
    record_rescore_drift(9, "result", 0.30, 0.20, 0.50)
    res = rescore_drift_stats()["stages"]["result"]
    # Tracks HOME (the sold pick): 0.30 - 0.55 = -0.25, a downward drift.
    assert res["down"] == 1
    assert res["mean_delta"] == pytest.approx(-0.25)


def test_empty_report(db):
    stats = rescore_drift_stats()
    assert stats == {"n_fixtures": 0, "stages": {}}


# ----------------------------------------------------------------------
# Wiring: the alert paths actually record a drift observation
# ----------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from sqlalchemy import select as _select  # noqa: E402

from betbot.storage.db import session_scope  # noqa: E402
from betbot.storage.models import RescoreDriftLog  # noqa: E402


def _drift_rows():
    with session_scope() as s:
        rows = list(s.execute(_select(RescoreDriftLog)).scalars())
        s.expunge_all()
        return rows


async def test_send_prediction_alert_records_early_fire(tmp_path):
    from tests.test_daily_jobs import (
        _Pred, _User, _ent, _lineup_fn_stub, _rescore_stub, _tg_settings,
    )
    from betbot.daily_jobs import send_prediction_alert

    init_engine(tmp_path / "fire.sqlite")
    s = _tg_settings(tmp_path)

    async def fake_send(_se, _cid, _txt):
        return True

    await send_prediction_alert(
        s, 1, send_fn=fake_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
        alert_tag="early",
    )
    rows = _drift_rows()
    assert len(rows) == 1
    r = rows[0]
    assert (r.fixture_id, r.stage) == (1, "early_fire")
    assert (r.p_home, r.p_draw, r.p_away) == (0.39, 0.31, 0.30)


async def test_send_prediction_alert_late_tag_maps_to_kickoff_60(tmp_path):
    from tests.test_daily_jobs import (
        _Pred, _User, _ent, _lineup_fn_stub, _rescore_stub, _tg_settings,
    )
    from betbot.daily_jobs import send_prediction_alert

    init_engine(tmp_path / "late.sqlite")
    s = _tg_settings(tmp_path)

    async def fake_send(_se, _cid, _txt):
        return True

    await send_prediction_alert(
        s, 1, send_fn=fake_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
        alert_tag="late",
    )
    rows = _drift_rows()
    assert [r.stage for r in rows] == ["kickoff_60"]


async def test_run_result_alerts_records_result_stage(tmp_path):
    from tests.test_daily_jobs import _Pred, _User, _tg_settings
    from betbot import daily_jobs

    init_engine(tmp_path / "result.sqlite")
    s = _tg_settings(tmp_path)
    object.__setattr__(s, "telegram_allowed_user_id", 999)

    outcome = SimpleNamespace(
        fixture_id=7,
        competition_code="PL",
        predicted_home=0.6, predicted_draw=0.25, predicted_away=0.15,
        predicted_pick="HOME", correct=True,
        home_goals=2, away_goals=0,
    )

    async def fake_send(_se, _cid, _txt):
        return True

    n = await daily_jobs.run_result_alerts(
        s,
        send_fn=fake_send,
        outcomes_fn=lambda: [outcome],
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        users_fn=lambda: [_User(111)],
        mark_notified_fn=lambda fid: None,
    )
    assert n >= 1  # operator at least
    rows = _drift_rows()
    assert len(rows) == 1
    r = rows[0]
    assert (r.fixture_id, r.stage) == (7, "result")
    assert (r.p_home, r.p_draw, r.p_away) == (0.6, 0.25, 0.15)
