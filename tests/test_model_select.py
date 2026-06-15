"""Tests for online model selection (Hedge) — dual-logging + live weighting."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    model_select_losses,
    model_select_summary,
    score_model_prediction,
    upsert_model_prediction,
)
from betbot.strategy.dixon_coles import DCParams, DCTeam
from betbot.strategy.glicko import Glicko2Rating
from betbot.strategy.international_engine import InternationalStrategyEngine
from betbot.strategy.model_select import hedge_weights

KICKOFF = datetime(2026, 6, 15, 19, 0, tzinfo=timezone.utc)


# ---- hedge math ----------------------------------------------------------

def test_equal_losses_give_even_weights():
    assert hedge_weights(3.0, 3.0) == pytest.approx((0.5, 0.5))


def test_lower_loss_gets_more_weight():
    w_a, w_b = hedge_weights(1.0, 2.0, eta=2.0)
    assert w_a > 0.5 > w_b
    assert w_a + w_b == pytest.approx(1.0)


def test_eta_zero_ignores_evidence():
    assert hedge_weights(0.0, 99.0, eta=0.0) == pytest.approx((0.5, 0.5))


def test_large_gap_converges_to_better_expert():
    w_a, _ = hedge_weights(1.0, 6.0, eta=2.0)
    assert w_a > 0.99


def test_one_fluky_match_barely_moves_weights():
    # A single bad match (RPS gap ~0.1) must not swing the bot to one model.
    w_a, _ = hedge_weights(0.15, 0.25, eta=2.0)
    assert 0.5 < w_a < 0.6


# ---- storage roundtrip ----------------------------------------------------

@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "ms.sqlite")
    yield


def test_record_score_and_losses(db):
    upsert_model_prediction(
        900, "Japan", "Senegal",
        (0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.5, 0.5),
    )
    assert model_select_losses() == (0.0, 0.0, 0)  # nothing scored yet
    assert score_model_prediction(900, "HOME") is True
    lg, le, n = model_select_losses()
    assert n == 1
    # Ensemble was more confident in the actual outcome -> lower RPS.
    assert le < lg


def test_scoring_is_idempotent_and_frozen(db):
    upsert_model_prediction(901, "A", "B", (0.4, 0.3, 0.3), (0.4, 0.3, 0.3), (0.5, 0.5))
    assert score_model_prediction(901, "DRAW") is True
    assert score_model_prediction(901, "AWAY") is False  # already scored
    # Re-upserting after settlement must not rewrite history.
    upsert_model_prediction(901, "A", "B", (0.9, 0.05, 0.05), (0.9, 0.05, 0.05), (1, 0))
    lg, le, n = model_select_losses()
    assert n == 1
    summary = model_select_summary()
    assert summary["scored_matches"] == 1


def test_score_unknown_fixture_or_outcome(db):
    assert score_model_prediction(999, "HOME") is False
    upsert_model_prediction(902, "A", "B", (0.4, 0.3, 0.3), (0.4, 0.3, 0.3), (0.5, 0.5))
    assert score_model_prediction(902, "VOID") is False


def test_summary_reports_leader(db):
    upsert_model_prediction(903, "A", "B", (0.2, 0.2, 0.6), (0.8, 0.1, 0.1), (0.5, 0.5))
    score_model_prediction(903, "HOME")  # ensemble called it, glicko didn't
    s = model_select_summary()
    assert s["leader"] == "ensemble"
    assert s["weight_ensemble"] > s["weight_glicko"]


# ---- engine wiring --------------------------------------------------------

def _fixture_form(home, away):
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=950, home_team=ht, away_team=at, kickoff=KICKOFF,
                 competition_code="WC")
    snap = FormSnapshot(team=ht, weighted_points=0.0, raw_points=0,
                        matches_considered=0)
    return FixtureForm(fixture=fx, home_form=snap, away_form=snap)


EQUAL = {"Japan": Glicko2Rating(1600, 60), "Senegal": Glicko2Rating(1600, 60)}
DC_FAVOURS_HOME = DCParams(teams={
    "japan": DCTeam(0.5, 0.5), "senegal": DCTeam(-0.5, -0.5),
})


def _engine(settings, losses, recorded):
    return InternationalStrategyEngine(
        settings,
        lambda n: EQUAL.get(n, Glicko2Rating(1500, 200)),
        dc_params=DC_FAVOURS_HOME,
        get_model_losses=lambda: losses,
        record_model_prediction=lambda *a: recorded.append(a),
    )


def test_weights_follow_the_winning_model(settings):
    recorded: list = []
    # Glicko losing badly -> prediction should sit near the ensemble view.
    p_ens = _engine(settings, (6.0, 1.0, 20), recorded).predict(
        _fixture_form("Japan", "Senegal"))
    # Ensemble losing badly -> prediction should sit near pure Glicko (50/50).
    p_gli = _engine(settings, (1.0, 6.0, 20), recorded).predict(
        _fixture_form("Japan", "Senegal"))
    assert p_ens.p_home > p_gli.p_home  # DC favours home; glicko is even
    assert p_gli.p_home == pytest.approx(p_gli.p_away, abs=0.05)


def test_predict_records_both_views(settings):
    recorded: list = []
    _engine(settings, (0.0, 0.0, 0), recorded).predict(
        _fixture_form("Japan", "Senegal"))
    assert len(recorded) == 1
    # The recorder now also receives the dispersion challenger triple (c_*),
    # dual-logged on every prediction regardless of the live flag.
    fid, home, away, glicko, ens, weights, challenger = recorded[0]
    assert fid == 950 and home == "Japan"
    assert sum(glicko) == pytest.approx(1.0)
    assert sum(ens) == pytest.approx(1.0)
    assert weights == pytest.approx((0.5, 0.5))  # no evidence yet
    assert sum(challenger) == pytest.approx(1.0)   # well-formed distribution
    assert all(0.0 <= x <= 1.0 for x in challenger)


def test_model_select_disabled_reverts_to_blend(settings):
    recorded: list = []
    s = settings.model_copy(update={"model_select_enabled": False})
    _engine(s, (6.0, 1.0, 20), recorded).predict(_fixture_form("Japan", "Senegal"))
    assert recorded == []  # no dual-logging when disabled
