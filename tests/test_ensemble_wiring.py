"""Tests for the ensemble wiring inside InternationalStrategyEngine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.exchanges.base import Outcome
from betbot.strategy.dixon_coles import DCParams, DCTeam
from betbot.strategy.ensemble import IsotonicCalibrator
from betbot.strategy.glicko import Glicko2Rating, match_probabilities
from betbot.strategy.international_engine import InternationalStrategyEngine

KICKOFF = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)


def _fixture_form(home, away):
    ht = Team(id=1, name=home)
    at = Team(id=2, name=away)
    fx = Fixture(id=900, home_team=ht, away_team=at, kickoff=KICKOFF, competition_code="WC")
    snap = FormSnapshot(team=ht, weighted_points=0.0, raw_points=0, matches_considered=0)
    return FixtureForm(fixture=fx, home_form=snap, away_form=snap)


def _ratings(table):
    return lambda name: table.get(name, Glicko2Rating(1500, 200, 0.06))


EQUAL = {"Japan": Glicko2Rating(1600, 60), "Senegal": Glicko2Rating(1600, 60)}


def test_no_artifacts_falls_back_to_pure_glicko(settings):
    eng = InternationalStrategyEngine(settings, _ratings(EQUAL))
    p = eng.predict(_fixture_form("Japan", "Senegal"))
    gh, gd, ga = match_probabilities(
        EQUAL["Japan"], EQUAL["Senegal"],
        home_field_mu=0.0, draw_rho=settings.glicko_draw_rho,
    )
    assert (p.p_home, p.p_draw, p.p_away) == pytest.approx((gh, gd, ga))


def test_dc_component_shifts_prediction(settings):
    # DC params that strongly favour Japan; Glicko sees the teams as equal.
    dc_params = DCParams(teams={
        "japan": DCTeam(attack=0.5, defence=0.5),
        "senegal": DCTeam(attack=-0.5, defence=-0.5),
    })
    eng = InternationalStrategyEngine(settings, _ratings(EQUAL), dc_params=dc_params)
    p = eng.predict(_fixture_form("Japan", "Senegal"))
    base = InternationalStrategyEngine(settings, _ratings(EQUAL)).predict(
        _fixture_form("Japan", "Senegal")
    )
    assert p.p_home > base.p_home
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0)


def test_dc_params_loaded_from_settings_path(settings, tmp_path):
    f = tmp_path / "dc.json"
    f.write_text(DCParams(teams={"japan": DCTeam(0.4, 0.4)}).to_json())
    s = settings.model_copy(update={"dc_params_path": f})
    eng = InternationalStrategyEngine(s, _ratings(EQUAL))
    assert eng._dc_params is not None
    assert eng._dc_params.team("japan").attack == pytest.approx(0.4)


def test_calibrators_applied(settings):
    # A calibrator that says "the model is over-confident on home wins".
    squash = IsotonicCalibrator(xs=[0.0, 1.0], ys=[0.25, 0.75])
    identity = IsotonicCalibrator()
    eng = InternationalStrategyEngine(
        settings,
        _ratings({"Brazil": Glicko2Rating(1900, 60), "Haiti": Glicko2Rating(1400, 60)}),
        calibrators=(squash, identity, identity),
    )
    raw = InternationalStrategyEngine(
        settings,
        _ratings({"Brazil": Glicko2Rating(1900, 60), "Haiti": Glicko2Rating(1400, 60)}),
    ).predict(_fixture_form("Brazil", "Haiti"))
    cal = eng.predict(_fixture_form("Brazil", "Haiti"))
    assert cal.p_home < raw.p_home


def test_decide_anchors_probability_toward_market(settings):
    eng = InternationalStrategyEngine(settings, _ratings(EQUAL))
    pred = eng.predict(_fixture_form("Japan", "Senegal"))
    # Force a model view well above the market price.
    import dataclasses

    pred = dataclasses.replace(pred, p_home=0.70)
    decision = eng.decide_with_market(pred, Outcome.HOME, 0.40, require_edge=False)
    assert decision is not None
    assert 0.40 < decision.our_probability < 0.70   # pulled toward the price
    assert decision.edge < 0.30                      # shrunk vs the raw 0.30


def test_decide_vetoes_marginal_edge_after_anchoring(settings):
    """Raw edge exactly at threshold must no longer survive the shrink —
    the ensemble bets only on genuine divergence."""
    eng = InternationalStrategyEngine(settings, _ratings(EQUAL))
    pred = eng.predict(_fixture_form("Japan", "Senegal"))
    import dataclasses

    pred = dataclasses.replace(pred, p_home=0.55)
    assert eng.decide_with_market(pred, Outcome.HOME, 0.50) is None


def test_decide_zero_market_weight_preserves_old_behaviour(settings):
    s = settings.model_copy(update={"ensemble_weight_market": 0.0})
    eng = InternationalStrategyEngine(s, _ratings(EQUAL))
    pred = eng.predict(_fixture_form("Japan", "Senegal"))
    import dataclasses

    pred = dataclasses.replace(pred, p_home=0.55)
    decision = eng.decide_with_market(pred, Outcome.HOME, 0.50)
    assert decision is not None
    assert decision.our_probability == pytest.approx(0.55)
    assert decision.edge == pytest.approx(0.05)
