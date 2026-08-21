"""The record covers this season's CLUB football, and nothing else.

The live fault: /record said "4 hits in 8 since 17 August", and two of those
eight were France-England and Spain-Argentina at the World Cup, played 18-19
JULY. They passed every filter the ledger had — settled 17 August (so inside
the epoch), non-degenerate probabilities — because the only date the ledger
stored was the SETTLEMENT time, not the kickoff.

Pins:
  * the outcome row now records the kickoff, and settlement supplies it;
  * international fixtures are excluded by an allowlist, so the next EURO or
    friendly is excluded too, not just the code "WC";
  * a fixture played before the season start is excluded even when it was
    settled inside the window;
  * a row that cannot be dated is excluded rather than guessed at;
  * the startup backfill dates historical rows from `predictions`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionOutcome, PredictionRow
from betbot.storage.repos import (
    prediction_outcomes_since,
    record_prediction_outcome,
    track_record,
)

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
THIS_SEASON = datetime(2026, 8, 17, 19, 0, tzinfo=timezone.utc)
WORLD_CUP = datetime(2026, 7, 18, 21, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "season.sqlite")
    yield


@pytest.fixture(autouse=True)
def _season(monkeypatch):
    """Switch the season scope back ON for this file.

    conftest's ``_no_season_scope`` disables it suite-wide (most tests pin a
    June "now"); these tests exist to exercise it, so they opt back in — the
    same shape as test_confidence_ledger.py re-enabling the epoch.
    """
    from betbot.config import get_settings

    monkeypatch.setenv("BETBOT_SEASON_START", "2026-08-01")
    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _record(fixture_id, code, kickoff, *, outcome="HOME", settled=NOW):
    return record_prediction_outcome(
        fixture_id=fixture_id, competition_code=code,
        p_home=0.5, p_draw=0.25, p_away=0.25,
        actual_outcome=outcome, home_goals=1, away_goals=0,
        settled_at=settled, kickoff=kickoff,
    )


# ----------------------------------------------------------------------
# The live fault
# ----------------------------------------------------------------------
def test_a_world_cup_tie_settled_this_season_is_still_excluded(db):
    """Settled 17 Aug, played 18 Jul — the exact shape that inflated /record."""
    _record(1, "PD", THIS_SEASON)
    _record(2, "WC", WORLD_CUP, settled=datetime(2026, 8, 17, 18, tzinfo=timezone.utc))

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [1]


def test_every_international_code_is_excluded_not_just_wc(db):
    """An allowlist, so the next EURO or friendly does not walk back in."""
    _record(1, "PD", THIS_SEASON)
    for i, code in enumerate(("WC", "EC", "FRIENDLY", "NL"), start=2):
        _record(i, code, THIS_SEASON)

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [1]


@pytest.mark.parametrize("code", ["PL", "PD", "BL1", "SA", "FL1", "CL"])
def test_every_club_competition_is_included(db, code):
    """CL included: it is club football, played by the same club engine."""
    _record(1, code, THIS_SEASON)
    assert len(prediction_outcomes_since(365)) == 1


def test_a_club_match_from_last_season_is_excluded(db):
    _record(1, "PD", datetime(2026, 5, 20, 19, 0, tzinfo=timezone.utc))
    _record(2, "PD", THIS_SEASON)

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [2]


def test_a_match_on_the_season_boundary_is_included(db):
    """The boundary is inclusive — 1 August IS the new season."""
    _record(1, "PD", datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))
    assert len(prediction_outcomes_since(365)) == 1


def test_an_undated_row_is_excluded_rather_than_guessed(db):
    """A record that guesses at a season is worse than one that abstains."""
    _record(1, "PD", None)
    _record(2, "PD", THIS_SEASON)

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [2]


def test_naive_kickoffs_are_treated_as_utc(db):
    """SQLite hands back naive datetimes; a tz slip must not drop a real row."""
    _record(1, "PD", THIS_SEASON)
    with session_scope() as s:
        row = s.query(PredictionOutcome).filter_by(fixture_id=1).one()
        row.kickoff = THIS_SEASON.replace(tzinfo=None)
    assert len(prediction_outcomes_since(365)) == 1


# ----------------------------------------------------------------------
# The figure the bot quotes
# ----------------------------------------------------------------------
def test_track_record_counts_only_club_season_matches(db):
    _record(1, "PD", THIS_SEASON, outcome="HOME")   # HIT (pick is HOME)
    _record(2, "PD", THIS_SEASON, outcome="AWAY")   # miss
    _record(3, "WC", WORLD_CUP, outcome="HOME")     # excluded
    _record(4, "PD", datetime(2026, 5, 1, tzinfo=timezone.utc), outcome="HOME")

    tr = track_record(365)
    assert tr["n"] == 2
    assert tr["hits"] == 1
    assert tr["hit_rate"] == 0.5


def test_track_record_is_zero_when_the_season_has_no_club_results(db):
    _record(1, "WC", WORLD_CUP)
    tr = track_record(365)
    assert tr["n"] == 0
    assert tr["hit_rate"] == 0.0


# ----------------------------------------------------------------------
# Storage + migration
# ----------------------------------------------------------------------
def test_settlement_stores_the_kickoff(db, settings):
    """Without this the row cannot be placed in a season at all."""
    from betbot.settlement import SettlementWatcher

    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=99, competition_code="PD", kickoff=THIS_SEASON,
                run_date="2026-08-17", home_team="A", away_team="B",
                p_home=0.5, p_draw=0.25, p_away=0.25,
                home_score=1.0, away_score=0.0, draw_score=2.4,
            )
        )

    class FakeFD:
        async def get_match(self, fixture_id):
            return {
                "status": "FINISHED",
                "score": {"winner": "HOME_TEAM", "fullTime": {"home": 2, "away": 0}},
            }

    import asyncio

    w = SettlementWatcher(FakeFD(), settings)
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        w.settle_due(now=THIS_SEASON + timedelta(hours=3))
    )

    with session_scope() as s:
        row = s.query(PredictionOutcome).filter_by(fixture_id=99).one()
        stored = row.kickoff
    assert stored is not None
    if stored.tzinfo is None:
        stored = stored.replace(tzinfo=timezone.utc)
    assert stored == THIS_SEASON


def test_startup_backfills_kickoffs_for_historical_rows(tmp_path):
    """Rows predating the column must be dated, or the record loses its history."""
    from sqlalchemy import text

    import betbot.storage.db as dbmod
    from betbot.storage.db import _backfill_outcome_kickoffs

    init_engine(tmp_path / "backfill.sqlite")
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=77, competition_code="PD", kickoff=THIS_SEASON,
                run_date="2026-08-17", home_team="A", away_team="B",
                p_home=0.5, p_draw=0.25, p_away=0.25,
                home_score=1.0, away_score=0.0, draw_score=2.4,
            )
        )
    _record(77, "PD", None)  # column exists but this row is undated

    _backfill_outcome_kickoffs(dbmod._engine)

    with session_scope() as s:
        ko = s.execute(
            text("SELECT kickoff FROM prediction_outcomes WHERE fixture_id = 77")
        ).scalar_one()
    assert ko is not None
    assert len(prediction_outcomes_since(365)) == 1
