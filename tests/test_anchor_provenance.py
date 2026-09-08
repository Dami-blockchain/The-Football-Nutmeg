"""Defect B — anchor provenance survives from the prediction row to the
scored outcome, so the pre-registered anchored-vs-unanchored validation gate
is computable straight from the DB.

Three facts must hold end to end:
* an odds-anchored prediction stores ``anchor_source='odds'`` AND its raw
  pre-anchor triple on the prediction row;
* an unanchored prediction stores ``anchor_source=NULL`` and a raw triple that
  equals its displayed triple (raw == shown when nothing anchored it);
* settlement carries ``anchor_source`` onto the outcome row it scores.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from betbot.settlement import SettlementWatcher
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionOutcome, PredictionRow
from betbot.storage.repos import upsert_prediction
from betbot.strategy.engine import Prediction

NOW = datetime.now(timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "anchor.sqlite")
    yield


def _raw_pred(fixture_id: int) -> Prediction:
    return Prediction(
        fixture_id=fixture_id, competition_code="PL", home_team="A", away_team="B",
        p_home=0.55, p_draw=0.25, p_away=0.20,
        home_score=1.0, away_score=0.0, draw_score=2.4,
    )


def _odds_anchored(fixture_id: int) -> Prediction:
    """Mimics odds_anchor.anchor_prediction: displayed triple is the anchored
    number, ``model_probs`` retains the raw pre-anchor triple."""
    raw = _raw_pred(fixture_id)
    return dataclasses.replace(
        raw,
        p_home=0.48, p_draw=0.29, p_away=0.23,  # shrunk toward the book
        model_probs=(raw.p_home, raw.p_draw, raw.p_away),
        anchor_source="odds",
    )


def _row(fixture_id: int) -> PredictionRow:
    with session_scope() as s:
        row = s.query(PredictionRow).filter_by(fixture_id=fixture_id).one()
        s.expunge(row)
        return row


def _outcome(fixture_id: int) -> PredictionOutcome:
    with session_scope() as s:
        row = s.query(PredictionOutcome).filter_by(fixture_id=fixture_id).one()
        s.expunge(row)
        return row


def test_anchored_prediction_stores_source_and_raw_triple(db):
    upsert_prediction(_odds_anchored(1), kickoff=NOW)
    row = _row(1)
    assert row.anchor_source == "odds"
    # displayed p_* are the anchored (shrunk) numbers...
    assert row.p_home == pytest.approx(0.48)
    # ...and the raw triple is the model's PRE-anchor probability.
    assert (row.raw_p_home, row.raw_p_draw, row.raw_p_away) == pytest.approx(
        (0.55, 0.25, 0.20)
    )


def test_unanchored_prediction_has_null_source_and_raw_equals_shown(db):
    upsert_prediction(_raw_pred(2), kickoff=NOW)
    row = _row(2)
    assert row.anchor_source is None
    # With nothing to anchor to, the raw triple IS the displayed triple.
    assert (row.raw_p_home, row.raw_p_draw, row.raw_p_away) == pytest.approx(
        (0.55, 0.25, 0.20)
    )


def test_rescore_updates_source_and_raw_triple_in_place(db):
    # First an unanchored row, then a rescore anchors it: the update branch of
    # upsert_prediction must refresh both the source and the raw triple, not
    # leave the pre-anchor NULLs behind (Defect A can leave a fixture unanchored
    # at rescore time — the row must faithfully record which happened).
    upsert_prediction(_raw_pred(3), kickoff=NOW)
    upsert_prediction(_odds_anchored(3), kickoff=NOW)
    row = _row(3)
    assert row.anchor_source == "odds"
    assert (row.raw_p_home, row.raw_p_draw, row.raw_p_away) == pytest.approx(
        (0.55, 0.25, 0.20)
    )


class _FakeFD:
    def __init__(self, results):
        self._results = results

    async def get_match(self, fixture_id):
        return self._results.get(fixture_id)


def _finished(winner):
    return {"status": "FINISHED", "score": {"winner": winner}}


async def test_settlement_carries_anchor_source_onto_the_outcome(db, settings):
    past = NOW - timedelta(minutes=200)  # beyond the grace window
    upsert_prediction(_odds_anchored(10), kickoff=past)
    upsert_prediction(_raw_pred(11), kickoff=past)
    w = SettlementWatcher(
        _FakeFD({10: _finished("HOME_TEAM"), 11: _finished("HOME_TEAM")}), settings
    )
    # ``summary.settled`` counts settled BETS; these fixtures carry no bet, but
    # settlement still scores an outcome row for every prediction. Assert on the
    # outcome ledger directly.
    await w.settle_due(now=NOW)
    assert _outcome(10).anchor_source == "odds"
    assert _outcome(11).anchor_source is None


async def test_legacy_prediction_row_settles_with_null_anchor_source(db, settings):
    """Every one of the 148 existing outcomes predates these columns: their
    prediction rows carry a NULL ``anchor_source`` (added by ALTER, never set).
    Settling such a row must not raise and must store SQL NULL — never the
    string "None". Insert a row the way a pre-migration path did (no anchor
    fields touched at all) and drive it through settlement."""
    past = NOW - timedelta(minutes=200)
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=99, competition_code="PL", kickoff=past,
                run_date=past.date().isoformat(), home_team="A", away_team="B",
                p_home=0.5, p_draw=0.3, p_away=0.2,
                home_score=1.0, away_score=0.0, draw_score=2.4,
                # anchor_source / raw_p_* deliberately left unset -> NULL
            )
        )
    w = SettlementWatcher(_FakeFD({99: _finished("HOME_TEAM")}), settings)
    await w.settle_due(now=NOW)  # must not raise
    row = _outcome(99)
    assert row.anchor_source is None
    assert row.anchor_source != "None"  # NULL, not the string
