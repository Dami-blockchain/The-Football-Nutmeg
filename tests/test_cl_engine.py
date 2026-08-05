"""Unit tests for the cross-league Elo engine (EuropeanStrategyEngine).

No DB / network: the ClubElo snapshot, DC params and name map are all injected,
and resolver=None means the engine loads config/team_aliases.yaml but resolves
against the injected snapshot's club list (the fixture team names below are used
verbatim as snapshot keys, so exact-normalised matching resolves them).
"""

from __future__ import annotations

from datetime import datetime, timezone

from betbot.config import get_settings
from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.strategy import dixon_coles as dc
from betbot.strategy.cl_engine import EuropeanStrategyEngine
from betbot.strategy.engine import Outcome


def _ff(home: str, away: str, hp: float = 1.5, ap: float = 1.5) -> FixtureForm:
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=7, home_team=ht, away_team=at,
                 kickoff=datetime.now(timezone.utc), competition_code="CL")
    return FixtureForm(
        fixture=fx,
        home_form=FormSnapshot(team=ht, weighted_points=hp, raw_points=0, matches_considered=5),
        away_form=FormSnapshot(team=at, weighted_points=ap, raw_points=0, matches_considered=5),
    )


def _engine(snapshot: dict[str, float], *, dc_params=None, name_map=None):
    s = get_settings()
    return EuropeanStrategyEngine(
        s,
        snapshot=snapshot,
        dc_params=dc_params,
        name_map=name_map if name_map is not None else {},
        resolver=None,
    )


def test_probs_sum_to_one_and_favour_stronger_home():
    snap = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    eng = _engine(snap)
    p = eng.predict(_ff("Strong FC", "Weak FC"))
    assert abs(p.p_home + p.p_draw + p.p_away - 1.0) < 1e-9
    assert p.p_home > p.p_away
    assert p.best_outcome is Outcome.HOME


def test_home_advantage_shifts_equal_elo_toward_home():
    snap = {"A FC": 1700.0, "B FC": 1700.0}
    eng = _engine(snap)
    # Equal Elo: the home side must still be favoured by the Elo home-advantage.
    p = eng.predict(_ff("A FC", "B FC"))
    assert p.p_home > p.p_away


def test_unresolved_team_falls_back_to_naive():
    # Away side not in the snapshot -> naive fallback, byte-identical to naive.
    s = get_settings()
    snap = {"Known FC": 1800.0}  # "Missing FC" absent from snapshot
    eng = _engine(snap)
    ff = _ff("Known FC", "Missing FC", hp=2.5, ap=0.4)
    from betbot.strategy.engine import StrategyEngine
    naive = StrategyEngine(s).predict(ff)
    got = eng.predict(ff)
    assert got.p_home == naive.p_home
    assert got.p_draw == naive.p_draw
    assert got.p_away == naive.p_away


def test_decide_with_market_no_edge_returns_none():
    snap = {"Strong FC": 1950.0, "Weak FC": 1450.0}
    eng = _engine(snap)
    pred = eng.predict(_ff("Strong FC", "Weak FC"))
    # Market price == model prob: after anchoring there is no edge -> veto.
    d = eng.decide_with_market(pred, Outcome.HOME, pred.p_home, require_edge=True)
    assert d is None


def test_dc_component_lifts_strong_team_when_enabled():
    # cl_weight_dc defaults to 1.0 (blend shipped), so a DC-strong team must
    # lift its home probability vs the Elo-only (no DC params) prediction.
    # Team names carry no noise tokens ("FC"/"CF"), so normalize() maps them
    # straight to the DC keys — the engine only blends DC when BOTH clubs are
    # actually in the goal model.
    snap = {"Ajax": 1700.0, "Porto": 1700.0}
    params = dc.DCParams(
        base_mu=0.1, home_adv=0.25, rho=0.0,
        teams={"ajax": dc.DCTeam(attack=0.9, defence=0.5),
               "porto": dc.DCTeam(attack=-0.4, defence=-0.3)},
    )
    with_dc = _engine(snap, dc_params=params).predict(_ff("Ajax", "Porto"))
    without_dc = _engine(snap, dc_params=None).predict(_ff("Ajax", "Porto"))
    assert with_dc.p_home > without_dc.p_home
