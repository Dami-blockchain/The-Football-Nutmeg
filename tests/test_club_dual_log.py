"""The club dual-log (model_predictions) write + settlement scoring, and the
calibration fit's raw-triple + minimum-sample discipline.

Re-arms the head-to-head ledger that froze on 2026-07-17, and provides the raw
pre-calibration triples the isotonic fit trains on (avoiding a train/serve
skew). This is a PASSIVE ledger: it changes nothing about what is served.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import ModelPrediction
from betbot.storage.repos import (
    score_model_prediction,
    upsert_model_prediction,
)
from betbot.strategy.ensemble import ranked_probability_score


@pytest.fixture()
def db(tmp_path):
    init_engine(tmp_path / "dual.sqlite")
    yield


def _row(fixture_id: int) -> ModelPrediction | None:
    with session_scope() as s:
        return s.execute(
            select(ModelPrediction).where(ModelPrediction.fixture_id == fixture_id)
        ).scalar_one_or_none()


def test_upsert_then_score_fills_rps_for_both_models(db):
    g = (0.5, 0.3, 0.2)
    e = (0.6, 0.25, 0.15)
    upsert_model_prediction(
        fixture_id=1, home_team="Arsenal FC", away_team="Chelsea FC",
        glicko=g, ensemble=e, w_glicko=0.0, w_ensemble=1.0,
    )
    assert score_model_prediction(fixture_id=1, actual_outcome="HOME") is True
    row = _row(1)
    assert row.outcome == "HOME"
    assert row.rps_glicko == pytest.approx(ranked_probability_score(g, 0))
    assert row.rps_ensemble == pytest.approx(ranked_probability_score(e, 0))


def test_upsert_refreshes_while_unsettled_then_freezes_after_settlement(db):
    upsert_model_prediction(
        fixture_id=2, home_team="A", away_team="B",
        glicko=(0.4, 0.3, 0.3), ensemble=(0.4, 0.3, 0.3),
        w_glicko=0.0, w_ensemble=1.0,
    )
    # Rescore before settlement: triples update.
    upsert_model_prediction(
        fixture_id=2, home_team="A", away_team="B",
        glicko=(0.5, 0.25, 0.25), ensemble=(0.55, 0.25, 0.20),
        w_glicko=0.0, w_ensemble=1.0,
    )
    assert _row(2).e_home == pytest.approx(0.55)
    assert score_model_prediction(fixture_id=2, actual_outcome="AWAY") is True
    # Post-settlement upsert must NOT move the frozen snapshot.
    upsert_model_prediction(
        fixture_id=2, home_team="A", away_team="B",
        glicko=(0.9, 0.05, 0.05), ensemble=(0.9, 0.05, 0.05),
        w_glicko=0.0, w_ensemble=1.0,
    )
    assert _row(2).e_home == pytest.approx(0.55)


def test_score_is_idempotent_and_safe_when_row_absent(db):
    assert score_model_prediction(fixture_id=999, actual_outcome="HOME") is False
    upsert_model_prediction(
        fixture_id=3, home_team="A", away_team="B",
        glicko=(0.4, 0.3, 0.3), ensemble=(0.4, 0.3, 0.3),
        w_glicko=0.0, w_ensemble=1.0,
    )
    assert score_model_prediction(fixture_id=3, actual_outcome="DRAW") is True
    assert score_model_prediction(fixture_id=3, actual_outcome="DRAW") is False


# ---------------------------------------------------------------------
# club_engine.dual_triples
# ---------------------------------------------------------------------
def test_dual_triples_returns_raw_ensemble_and_pure_glicko(settings):
    from betbot.strategy.club_engine import ClubStrategyEngine
    from betbot.strategy.glicko import Glicko2Rating

    ratings = {
        "Home": Glicko2Rating(1650.0, 40.0, 0.06, "2026-08-20"),
        "Away": Glicko2Rating(1500.0, 40.0, 0.06, "2026-08-20"),
    }
    eng = ClubStrategyEngine(
        settings, get_rating=lambda n: ratings.get(n, Glicko2Rating(1500.0, 999.0, 0.06)),
        dc_params=None, calibrators=None, name_map={},
    )
    out = eng.dual_triples("Home", "Away")
    assert out is not None
    glicko, raw = out
    assert sum(glicko) == pytest.approx(1.0, abs=1e-6)
    assert sum(raw) == pytest.approx(1.0, abs=1e-6)
    # With identity calibration (no artifact) the RAW ensemble equals what
    # probability_triple serves, so the ledger reflects the served forecast.
    served, _, _ = eng.probability_triple("Home", "Away")
    assert raw == pytest.approx(served, abs=1e-9)


def test_dual_triples_is_none_for_an_unrated_side(settings):
    from betbot.strategy.club_engine import ClubStrategyEngine
    from betbot.strategy.glicko import Glicko2Rating

    ratings = {"Home": Glicko2Rating(1650.0, 40.0, 0.06, "2026-08-20")}
    eng = ClubStrategyEngine(
        settings,
        get_rating=lambda n: ratings.get(n, Glicko2Rating(1500.0, 999.0, 0.06)),
        dc_params=None, calibrators=None, name_map={},
    )
    # "Away" falls back to a huge RD (>= default) -> unrated -> None.
    assert eng.dual_triples("Home", "Away") is None


# ---------------------------------------------------------------------
# Calibration fit CLI (item 2): raw-triple training + minimum-sample guard
# ---------------------------------------------------------------------
def _load_fit_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_ensemble_calibration_club.py"
    spec = importlib.util.spec_from_file_location("fit_cal_club", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_settled(n_home: int, n_away: int) -> None:
    fid = 1000
    for _ in range(n_home):
        upsert_model_prediction(
            fixture_id=fid, home_team="A", away_team="B",
            glicko=(0.5, 0.3, 0.2), ensemble=(0.55, 0.25, 0.20),
            w_glicko=0.0, w_ensemble=1.0,
        )
        score_model_prediction(fixture_id=fid, actual_outcome="HOME")
        fid += 1
    for _ in range(n_away):
        upsert_model_prediction(
            fixture_id=fid, home_team="A", away_team="B",
            glicko=(0.3, 0.3, 0.4), ensemble=(0.25, 0.25, 0.50),
            w_glicko=0.0, w_ensemble=1.0,
        )
        score_model_prediction(fixture_id=fid, actual_outcome="AWAY")
        fid += 1


def test_fit_refuses_below_the_min_n_guard(db, monkeypatch):
    mod = _load_fit_module()
    monkeypatch.setattr(mod, "init_engine", lambda *a, **k: None)
    _seed_settled(5, 5)  # 10 rows, far below the default 500
    with pytest.raises(SystemExit) as ei:
        mod.main([])
    assert "REFUSING to fit" in str(ei.value)
    assert mod.MIN_FIT_N == 500


def test_fit_writes_artifact_from_raw_triples_when_sample_suffices(db, tmp_path, monkeypatch):
    mod = _load_fit_module()
    monkeypatch.setattr(mod, "init_engine", lambda *a, **k: None)
    _seed_settled(8, 6)
    out = tmp_path / "cal.json"
    mod.main(["--out", str(out), "--min-n", "5"])
    import json

    d = json.loads(out.read_text())
    assert set(d) == {"home", "draw", "away"}
    for k in d:
        assert d[k]["xs"] and d[k]["ys"]
    # And the club engine can load exactly this artifact shape.
    from betbot.strategy.club_engine import _load_calibrators

    cals = _load_calibrators(out)
    assert cals is not None and len(cals) == 3
