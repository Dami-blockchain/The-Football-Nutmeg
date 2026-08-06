"""Storage-layer tests — KillSwitch CRUD (Phase 4)."""

from __future__ import annotations

import pytest

from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_kill_switch,
    is_kill_switch_tripped,
    reset_kill_switch,
    trip_kill_switch,
)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "ks.sqlite")
    yield


def test_get_kill_switch_creates_untripped(db):
    ks = get_kill_switch()
    assert ks.id == 1
    assert ks.tripped_at is None
    assert is_kill_switch_tripped() is False


def test_trip_and_reset(db):
    trip_kill_switch("test drawdown", -120.0, 120.0)
    assert is_kill_switch_tripped() is True
    ks = get_kill_switch()
    assert ks.tripped_at is not None
    assert ks.reason == "test drawdown"
    assert ks.realized_pnl_usd == -120.0
    assert ks.staked_usd == 120.0

    reset_kill_switch()
    assert is_kill_switch_tripped() is False
    assert get_kill_switch().tripped_at is None


def test_reason_is_truncated(db):
    trip_kill_switch("x" * 500, -1.0, 1.0)
    assert len(get_kill_switch().reason) <= 300


# ----------------------------------------------------------------------
# Tipster delivery queries — dedup guard (money path)
# ----------------------------------------------------------------------
def test_kickoff_range_returns_one_row_per_fixture_latest_run(db):
    """The 48h scoring window writes one row per (fixture_id, run_date), so a
    fixture kicking off today typically has 2-3 rows. The delivery query must
    dedupe to the LATEST run — a duplicate row would double-charge a paying
    user for the same fixture in one message."""
    from datetime import datetime, timedelta, timezone

    from betbot.storage.db import session_scope
    from betbot.storage.models import PredictionRow
    from betbot.storage.repos import predictions_for_kickoff_range

    ko1 = datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc)
    ko2 = datetime(2026, 8, 8, 16, 30, tzinfo=timezone.utc)

    def _row(fid, run_date, kickoff, p_home):
        return PredictionRow(
            fixture_id=fid, competition_code="PL", kickoff=kickoff,
            run_date=run_date, home_team="H", away_team="A",
            p_home=p_home, p_draw=0.3, p_away=round(0.7 - p_home, 2),
            home_score=1.0, away_score=1.0, draw_score=1.0,
        )

    with session_scope() as s:
        # Fixture 1 scored on three consecutive runs; fixture 2 on one.
        s.add_all([
            _row(1, "2026-08-06", ko1, 0.40),
            _row(1, "2026-08-07", ko1, 0.45),
            _row(1, "2026-08-08", ko1, 0.50),  # freshest — must win
            _row(2, "2026-08-08", ko2, 0.33),
        ])

    rows = predictions_for_kickoff_range(ko1 - timedelta(hours=1),
                                         ko2 + timedelta(hours=1))
    assert [r.fixture_id for r in rows] == [1, 2]  # deduped, kickoff order
    assert rows[0].run_date == "2026-08-08"
    assert rows[0].p_home == 0.50
