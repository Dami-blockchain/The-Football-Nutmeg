"""Ledger-side of the confidence filter: split accuracy + the epoch cutoff.

Pins two things the honesty rules depend on:

* ``track_record`` reports ALL-MATCH accuracy and CALLED-PICK hit rate as two
  separate blocks that are never merged into one number;
* outcomes settled before ``BETBOT_ACCURACY_LEDGER_EPOCH`` are excluded from
  every accuracy read (the pre-2026-08-17 rows are poisoned by the degenerate
  0/0/100-AWAY bug).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot.config import get_settings
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    prediction_outcomes_since,
    record_prediction_outcome,
    track_record,
)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "confidence.sqlite")
    yield


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _score(fixture_id, ph, pd, pa, actual, *, settled_at=None):
    return record_prediction_outcome(
        fixture_id=fixture_id,
        competition_code="PL",
        p_home=ph,
        p_draw=pd,
        p_away=pa,
        actual_outcome=actual,
        home_goals=1,
        away_goals=0,
        settled_at=settled_at or _now(),
        result_notified=True,
    )


@pytest.fixture
def filter_on(monkeypatch):
    monkeypatch.setenv("BETBOT_CONFIDENCE_FILTER", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---- two metrics, never merged ---------------------------------------

def test_track_record_reports_called_block_separately(db, filter_on):
    # 2 called picks (p>=0.60, draw far enough away): one hit, one miss.
    _score(1, 0.70, 0.18, 0.12, "HOME")   # called, hit
    _score(2, 0.66, 0.20, 0.14, "DRAW")   # called, miss
    # 2 uncalled fixtures that are BOTH all-match hits — so the two metrics
    # must come out different, which is exactly the point of splitting them.
    _score(3, 0.52, 0.26, 0.22, "HOME")   # below threshold
    _score(4, 0.20, 0.55, 0.25, "DRAW")   # draw favourite

    tr = track_record(365)
    assert tr["n"] == 4
    assert tr["hits"] == 3
    assert tr["hit_rate"] == pytest.approx(0.75)

    called = tr["called"]
    assert called["enabled"] is True
    assert called["n"] == 2
    assert called["hits"] == 1
    assert called["hit_rate"] == pytest.approx(0.5)
    assert called["call_rate"] == pytest.approx(0.5)
    assert called["ci_lo"] < called["hit_rate"] < called["ci_hi"]
    assert called["hit_rate"] != tr["hit_rate"]


def test_track_record_called_block_empty_when_flag_off(db):
    _score(1, 0.90, 0.06, 0.04, "HOME")
    tr = track_record(365)
    assert tr["n"] == 1 and tr["hits"] == 1
    assert tr["called"]["enabled"] is False
    assert tr["called"]["n"] == 0


def test_track_record_empty_ledger_still_carries_called_block(db):
    tr = track_record(365)
    assert tr["n"] == 0
    assert tr["called"]["n"] == 0


# ---- accuracy-ledger epoch -------------------------------------------

def test_outcomes_before_the_epoch_are_excluded(db, monkeypatch):
    now = _now()
    epoch = (now - timedelta(days=2)).date().isoformat()
    _score(1, 0.60, 0.25, 0.15, "HOME", settled_at=now - timedelta(days=10))
    _score(2, 0.60, 0.25, 0.15, "HOME", settled_at=now - timedelta(hours=1))

    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", epoch)
    get_settings.cache_clear()
    try:
        rows = prediction_outcomes_since(365)
        assert [r.fixture_id for r in rows] == [2]
        assert track_record(365)["n"] == 1
    finally:
        get_settings.cache_clear()


def test_epoch_never_widens_a_shorter_window(db, monkeypatch):
    """A far-past epoch must not resurrect rows outside the ``days`` window."""
    now = _now()
    _score(1, 0.60, 0.25, 0.15, "HOME", settled_at=now - timedelta(days=40))
    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", "2000-01-01")
    get_settings.cache_clear()
    try:
        assert prediction_outcomes_since(30) == []
    finally:
        get_settings.cache_clear()


def test_unparseable_epoch_disables_the_cutoff(db, monkeypatch):
    now = _now()
    _score(1, 0.60, 0.25, 0.15, "HOME", settled_at=now - timedelta(days=10))
    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", "not-a-date")
    get_settings.cache_clear()
    try:
        assert len(prediction_outcomes_since(365)) == 1
    finally:
        get_settings.cache_clear()
