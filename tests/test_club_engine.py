"""Unit tests for the club ensemble engine (ClubStrategyEngine).

No DB / network: ratings, DC params and the name map are all injected.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.config import get_settings
from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.strategy import dixon_coles as dc
from betbot.strategy.club_engine import ClubStrategyEngine
from betbot.strategy.engine import Outcome
from betbot.strategy.glicko import Glicko2Rating


def _ff(home: str, away: str, hp: float = 1.5, ap: float = 1.5) -> FixtureForm:
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=7, home_team=ht, away_team=at,
                 kickoff=datetime.now(timezone.utc), competition_code="PL")
    return FixtureForm(
        fixture=fx,
        home_form=FormSnapshot(team=ht, weighted_points=hp, raw_points=0, matches_considered=5),
        away_form=FormSnapshot(team=at, weighted_points=ap, raw_points=0, matches_considered=5),
    )


def _engine(ratings: dict[str, Glicko2Rating], *, dc_params=None):
    s = get_settings()
    return ClubStrategyEngine(
        s,
        get_rating=lambda n: ratings.get(n, Glicko2Rating(
            s.glicko_default_rating, s.glicko_default_rd, s.glicko_default_vol)),
        dc_params=dc_params,
        calibrators=None,
        name_map={},  # team names are used directly as DC keys in these tests
    )


def test_probs_sum_to_one_and_favour_stronger_home():
    ratings = {
        "Strong FC": Glicko2Rating(1800.0, 60.0, 0.06),
        "Weak FC": Glicko2Rating(1400.0, 60.0, 0.06),
    }
    eng = _engine(ratings)
    p = eng.predict(_ff("Strong FC", "Weak FC"))
    assert abs(p.p_home + p.p_draw + p.p_away - 1.0) < 1e-9
    assert p.p_home > p.p_away  # much stronger side, at home, should be favoured
    assert p.best_outcome is Outcome.HOME


def test_unknown_team_falls_back_to_naive():
    # Away side has no rating history (RD == default) -> naive fallback.
    s = get_settings()
    ratings = {"Known FC": Glicko2Rating(1700.0, 60.0, 0.06)}
    eng = _engine(ratings)
    ff = _ff("Known FC", "Promoted FC", hp=2.5, ap=0.4)
    from betbot.strategy.engine import StrategyEngine
    naive = StrategyEngine(s).predict(ff)
    got = eng.predict(ff)
    assert got.p_home == pytest.approx(naive.p_home)
    assert got.p_away == pytest.approx(naive.p_away)


def test_home_advantage_shifts_probability():
    ratings = {
        "A FC": Glicko2Rating(1600.0, 60.0, 0.06),
        "B FC": Glicko2Rating(1600.0, 60.0, 0.06),
    }
    eng = _engine(ratings)
    # Equal ratings + equal form: the home side must still be favoured over the
    # away side because of the always-on club home advantage.
    p = eng.predict(_ff("A FC", "B FC", hp=1.5, ap=1.5))
    assert p.p_home > p.p_away


def test_dc_component_is_used_when_present():
    ratings = {
        "A FC": Glicko2Rating(1600.0, 60.0, 0.06),
        "B FC": Glicko2Rating(1600.0, 60.0, 0.06),
    }
    params = dc.DCParams(
        base_mu=0.1, home_adv=0.25, rho=0.0,
        teams={"a fc": dc.DCTeam(attack=0.9, defence=0.5),
               "b fc": dc.DCTeam(attack=-0.4, defence=-0.3)},
    )
    with_dc = _engine(ratings, dc_params=params).predict(_ff("A FC", "B FC"))
    without_dc = _engine(ratings, dc_params=None).predict(_ff("A FC", "B FC"))
    # DC says A is much stronger; the ensemble with DC must lift A's home prob.
    assert with_dc.p_home > without_dc.p_home


def test_decide_anchors_towards_market():
    ratings = {
        "Strong FC": Glicko2Rating(1850.0, 60.0, 0.06),
        "Weak FC": Glicko2Rating(1350.0, 60.0, 0.06),
    }
    eng = _engine(ratings)
    pred = eng.predict(_ff("Strong FC", "Weak FC"))
    # With no edge over a market that agrees, require_edge should veto.
    d = eng.decide_with_market(pred, Outcome.HOME, pred.p_home, require_edge=True)
    assert d is None
