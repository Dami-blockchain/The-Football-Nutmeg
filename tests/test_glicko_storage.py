"""Tests for Glicko rating storage + rating-period application (Phase 5.5)."""

from __future__ import annotations

import pytest

from betbot.storage.db import init_engine
from betbot.storage.repos import (
    all_ratings,
    apply_rating_period,
    get_rating,
    upsert_rating,
)
from betbot.strategy.glicko import Glicko2Rating


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "glicko.sqlite")
    yield


def test_get_default_for_unknown(db):
    r = get_rating("Nowhere", default_rating=1500, default_rd=200)
    assert r.rating == 1500 and r.rd == 200


def test_upsert_and_get_roundtrip(db):
    upsert_rating("Brazil", Glicko2Rating(1900, 55, 0.05, "2026-06-01"), team_id=10)
    r = get_rating("Brazil")
    assert r.rating == 1900 and r.rd == 55 and r.last_period == "2026-06-01"


def test_all_ratings_sorted_desc(db):
    upsert_rating("A", Glicko2Rating(1400, 60))
    upsert_rating("B", Glicko2Rating(1800, 60))
    names = [n for n, _ in all_ratings()]
    assert names == ["B", "A"]


def test_apply_rating_period_moves_ratings(db):
    upsert_rating("Strong", Glicko2Rating(1600, 80))
    upsert_rating("Weak", Glicko2Rating(1500, 80))
    # Weak beats Strong (an upset) -> Weak's rating rises, Strong's falls.
    n = apply_rating_period([("Strong", "Weak", "AWAY")], period="2026-06-11")
    assert n == 2
    assert get_rating("Weak").rating > 1500
    assert get_rating("Strong").rating < 1600
    # RD shrinks after a played match (more certainty).
    assert get_rating("Strong").rd < 80
