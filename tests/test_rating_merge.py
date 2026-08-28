"""The weekly re-seed must MERGE, not SET — it may never regress an in-season
Glicko nudge just because the football-data.co.uk CSV lags the live DB.

repos.upsert_rating_if_fresher is the guard: overwrite only when the incoming
last_period (a CSV match date, in the re-seed) is >= the stored one (a settle
date, from the nudge). This pins that behaviour.
"""

from __future__ import annotations

import pytest

from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_rating,
    rating_exists,
    upsert_rating,
    upsert_rating_if_fresher,
)
from betbot.strategy.glicko import Glicko2Rating


@pytest.fixture()
def db(tmp_path):
    init_engine(tmp_path / "ratings.sqlite")
    yield


def test_creates_when_absent(db):
    assert not rating_exists("Arsenal FC")
    wrote = upsert_rating_if_fresher(
        "Arsenal FC", Glicko2Rating(1600.0, 80.0, 0.06, "2026-08-20")
    )
    assert wrote is True
    assert rating_exists("Arsenal FC")
    assert get_rating("Arsenal FC").rating == pytest.approx(1600.0)


def test_stale_reseed_does_not_clobber_a_fresher_nudge(db):
    # Live nudge advanced the row to 2026-08-27.
    upsert_rating("Chelsea FC", Glicko2Rating(1700.0, 60.0, 0.06, "2026-08-27"))
    # Weekly re-seed replays a CSV that only knows up to 2026-08-20.
    wrote = upsert_rating_if_fresher(
        "Chelsea FC", Glicko2Rating(1500.0, 200.0, 0.06, "2026-08-20")
    )
    assert wrote is False
    kept = get_rating("Chelsea FC")
    assert kept.rating == pytest.approx(1700.0)  # in-season learning preserved
    assert kept.last_period == "2026-08-27"


def test_fresh_reseed_takes_over_cleanly(db):
    upsert_rating("Spurs FC", Glicko2Rating(1500.0, 200.0, 0.06, "2026-08-10"))
    # CSV has caught up past the stored period -> full replay is authoritative.
    wrote = upsert_rating_if_fresher(
        "Spurs FC", Glicko2Rating(1580.0, 70.0, 0.06, "2026-08-24")
    )
    assert wrote is True
    r = get_rating("Spurs FC")
    assert r.rating == pytest.approx(1580.0)
    assert r.last_period == "2026-08-24"


def test_equal_period_reseed_wins(db):
    upsert_rating("Everton FC", Glicko2Rating(1490.0, 90.0, 0.06, "2026-08-24"))
    wrote = upsert_rating_if_fresher(
        "Everton FC", Glicko2Rating(1495.0, 88.0, 0.06, "2026-08-24")
    )
    assert wrote is True
    assert get_rating("Everton FC").rating == pytest.approx(1495.0)


def test_stored_row_without_period_is_always_advanced(db):
    upsert_rating("Leeds FC", Glicko2Rating(1500.0, 200.0, 0.06, None))
    wrote = upsert_rating_if_fresher(
        "Leeds FC", Glicko2Rating(1520.0, 120.0, 0.06, "2026-08-01")
    )
    assert wrote is True
    assert get_rating("Leeds FC").last_period == "2026-08-01"


def test_team_id_is_refreshed_even_when_rating_is_kept(db):
    upsert_rating("Villa FC", Glicko2Rating(1650.0, 55.0, 0.06, "2026-08-27"))
    # Stale re-seed keeps the rating but should still stamp the football-data id.
    upsert_rating_if_fresher(
        "Villa FC", Glicko2Rating(1500.0, 200.0, 0.06, "2026-08-01"), team_id=58
    )
    assert get_rating("Villa FC").rating == pytest.approx(1650.0)
