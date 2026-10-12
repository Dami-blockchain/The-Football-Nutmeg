"""feat/cl-stage-logging — the CL ``stage`` is logged end to end.

Logging only: the column is additive & nullable, populated from the
football-data.org match payload for Champions League fixtures and left NULL for
domestic leagues. Facts under test:

* ``extract_stage`` returns the CL stage (upper-cased, clamped), and ``None``
  for non-CL / missing / blank / non-dict payloads;
* ``upsert_prediction`` stores the stage on a CL prediction row;
* the kickoff-60 rescore (which has no match payload, so ``stage=None``) does
  NOT clobber the stage written at the daily run;
* settlement carries the stage onto the scored outcome row for a CL match, and
  leaves it NULL for a domestic one;
* a legacy prediction row (no stage column value) and a CL payload that omits
  the ``stage`` key both settle without raising and store SQL NULL;
* the ``stage`` column is created idempotently on an already-migrated DB.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect

from betbot.data.football_data import extract_stage
from betbot.settlement import SettlementWatcher
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionOutcome, PredictionRow
from betbot.storage.repos import upsert_prediction
from betbot.strategy.engine import Prediction

NOW = datetime.now(timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "stage.sqlite")
    yield


def _cl_pred(fixture_id: int) -> Prediction:
    return Prediction(
        fixture_id=fixture_id, competition_code="CL", home_team="A", away_team="B",
        p_home=0.55, p_draw=0.25, p_away=0.20,
        home_score=1.0, away_score=0.0, draw_score=2.4,
    )


def _pl_pred(fixture_id: int) -> Prediction:
    return Prediction(
        fixture_id=fixture_id, competition_code="PL", home_team="A", away_team="B",
        p_home=0.55, p_draw=0.25, p_away=0.20,
        home_score=1.0, away_score=0.0, draw_score=2.4,
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


# --- extract_stage ---------------------------------------------------------
def test_extract_stage_cl_passthrough_upper_and_clamped():
    assert extract_stage({"stage": "last_16"}, "CL") == "LAST_16"
    assert extract_stage({"stage": "LEAGUE_STAGE"}, "cl") == "LEAGUE_STAGE"
    assert extract_stage({"stage": "QUARTER_FINALS"}, "CL") == "QUARTER_FINALS"
    # Clamp defensively to the column width (16).
    assert len(extract_stage({"stage": "X" * 40}, "CL")) == 16


def test_extract_stage_none_for_domestic_and_missing():
    assert extract_stage({"stage": "REGULAR_SEASON"}, "PL") is None
    assert extract_stage({}, "CL") is None
    assert extract_stage({"stage": None}, "CL") is None
    assert extract_stage({"stage": "   "}, "CL") is None
    assert extract_stage(None, "CL") is None
    assert extract_stage({"stage": "LAST_16"}, None) is None


# --- prediction write ------------------------------------------------------
def test_cl_prediction_stores_stage(db):
    upsert_prediction(_cl_pred(1), kickoff=NOW, stage="LEAGUE_STAGE")
    assert _row(1).stage == "LEAGUE_STAGE"


def test_domestic_prediction_stage_is_null(db):
    upsert_prediction(_pl_pred(2), kickoff=NOW)  # no stage passed
    assert _row(2).stage is None


def test_rescore_without_payload_preserves_stage(db):
    # Daily run writes the stage; the kickoff-60 rescore re-upserts the same
    # (fixture_id, run_date) row with stage=None and must NOT wipe it.
    upsert_prediction(_cl_pred(3), kickoff=NOW, stage="LAST_16")
    upsert_prediction(_cl_pred(3), kickoff=NOW)  # rescore, stage defaulted None
    assert _row(3).stage == "LAST_16"


def test_rescore_insert_on_new_day_keeps_stage(db):
    # The kickoff-60 rescore can land on a day with NO daily row for today, so
    # it INSERTS a fresh (fixture_id, run_date) row rather than updating. The
    # daily_jobs rescore call now forwards ``stage=baseline.stage``, so the new
    # row must carry the stage and not fall back to NULL.
    yesterday = (NOW - timedelta(days=1)).date().isoformat()
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=30, competition_code="CL", kickoff=NOW,
                run_date=yesterday, home_team="A", away_team="B",
                p_home=0.55, p_draw=0.25, p_away=0.20,
                home_score=1.0, away_score=0.0, draw_score=2.4,
                stage="SEMI_FINALS",
            )
        )
    # Rescore TODAY, forwarding the baseline stage exactly as daily_jobs does.
    upsert_prediction(_cl_pred(30), kickoff=NOW, stage="SEMI_FINALS")
    with session_scope() as s:
        rows = s.query(PredictionRow).filter_by(fixture_id=30).all()
        assert len(rows) == 2  # old daily row + new rescore row (an INSERT)
        today = date.today().isoformat()
        new = [r for r in rows if r.run_date == today]
        assert len(new) == 1 and new[0].stage == "SEMI_FINALS"


def test_missing_stage_does_not_break_storage(db):
    # A CL fixture whose payload omits stage -> extract_stage None -> stored NULL,
    # and nothing raises.
    stage = extract_stage({"id": 4}, "CL")
    upsert_prediction(_cl_pred(4), kickoff=NOW, stage=stage)
    assert _row(4).stage is None


# --- settlement carries stage onto the outcome ----------------------------
class _FakeFD:
    def __init__(self, results):
        self._results = results

    async def get_match(self, fixture_id):
        return self._results.get(fixture_id)


def _finished(winner, stage=None):
    m = {"status": "FINISHED", "score": {"winner": winner}}
    if stage is not None:
        m["stage"] = stage
    return m


async def test_settlement_carries_cl_stage_onto_outcome(db, settings):
    past = NOW - timedelta(minutes=200)
    upsert_prediction(_cl_pred(10), kickoff=past, stage="LAST_16")
    upsert_prediction(_pl_pred(11), kickoff=past)
    w = SettlementWatcher(
        _FakeFD({
            10: _finished("HOME_TEAM", stage="LAST_16"),
            11: _finished("HOME_TEAM", stage="REGULAR_SEASON"),
        }),
        settings,
    )
    await w.settle_due(now=NOW)
    assert _outcome(10).stage == "LAST_16"      # CL -> stage recorded
    assert _outcome(11).stage is None           # domestic -> NULL by design


async def test_settlement_tolerates_cl_payload_without_stage(db, settings):
    past = NOW - timedelta(minutes=200)
    upsert_prediction(_cl_pred(12), kickoff=past, stage="FINAL")
    # Provider payload omits the stage key entirely -> settlement falls back
    # to the stage already stored on the prediction row (FINAL).
    w = SettlementWatcher(_FakeFD({12: _finished("AWAY_TEAM")}), settings)
    await w.settle_due(now=NOW)  # must not raise
    assert _outcome(12).stage == "FINAL"


async def test_legacy_prediction_row_settles_with_null_stage(db, settings):
    """A pre-migration CL prediction row (stage never set) settles fine; with a
    payload that also omits stage the outcome stores SQL NULL, never "None"."""
    past = NOW - timedelta(minutes=200)
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=99, competition_code="CL", kickoff=past,
                run_date=past.date().isoformat(), home_team="A", away_team="B",
                p_home=0.5, p_draw=0.3, p_away=0.2,
                home_score=1.0, away_score=0.0, draw_score=2.4,
                # stage deliberately left unset -> NULL
            )
        )
    w = SettlementWatcher(_FakeFD({99: _finished("HOME_TEAM")}), settings)
    await w.settle_due(now=NOW)  # must not raise
    row = _outcome(99)
    assert row.stage is None
    assert row.stage != "None"


# --- schema migration idempotency -----------------------------------------
def test_stage_column_created_and_idempotent(tmp_path):
    path = tmp_path / "mig.sqlite"
    eng = init_engine(path)
    cols = {c["name"] for c in inspect(eng).get_columns("predictions")}
    assert "stage" in cols
    ocols = {c["name"] for c in inspect(eng).get_columns("prediction_outcomes")}
    assert "stage" in ocols
    # Re-running init on the same DB (column already present) must not raise.
    eng2 = init_engine(path)
    cols2 = {c["name"] for c in inspect(eng2).get_columns("predictions")}
    assert "stage" in cols2
