"""Engine wiring: predict() applies the availability rating adjustment.

Verifies (a) adj=0 reproduces current output EXACTLY, and (b) a negative home
adjustment drops the home win prob and lifts draw + away.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.strategy.glicko import Glicko2Rating
from betbot.strategy.international_engine import InternationalStrategyEngine

KICKOFF = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)


def _fixture_form(home, away):
    ht = Team(id=1, name=home)
    at = Team(id=2, name=away)
    fx = Fixture(id=901, home_team=ht, away_team=at, kickoff=KICKOFF,
                 competition_code="WC")
    snap = FormSnapshot(team=ht, weighted_points=0.0, raw_points=0,
                        matches_considered=0)
    return FixtureForm(fixture=fx, home_form=snap, away_form=snap)


def _ratings(table):
    return lambda name: table.get(name, Glicko2Rating(1500, 200, 0.06))


def _engine(settings):
    return InternationalStrategyEngine(settings, _ratings({
        "Spain": Glicko2Rating(1700, 60),
        "Croatia": Glicko2Rating(1650, 60),
    }))


def test_zero_adjustment_is_identical(settings):
    eng = _engine(settings)
    base = eng.predict(_fixture_form("Spain", "Croatia"))
    same = eng.predict(
        _fixture_form("Spain", "Croatia"), home_rating_adj=0.0, away_rating_adj=0.0
    )
    assert (same.p_home, same.p_draw, same.p_away) == (
        base.p_home, base.p_draw, base.p_away
    )
    assert same.home_score == base.home_score
    assert same.away_score == base.away_score


def test_negative_home_adjustment_drops_home_prob(settings):
    eng = _engine(settings)
    base = eng.predict(_fixture_form("Spain", "Croatia"))
    # Spain heavily depleted -> -40 Glicko points.
    adj = eng.predict(_fixture_form("Spain", "Croatia"), home_rating_adj=-40.0)
    assert adj.p_home < base.p_home          # home weaker -> wins less
    assert adj.p_away > base.p_away          # opponent benefits
    assert adj.p_draw >= base.p_draw         # closer game -> draw rises (or equal)
    assert adj.p_home + adj.p_draw + adj.p_away == pytest.approx(1.0)
    # Stored score reflects the ADJUSTED rating for transparency.
    assert adj.home_score == pytest.approx(base.home_score - 40.0)


def test_more_injuries_for_away_helps_home(settings):
    eng = _engine(settings)
    base = eng.predict(_fixture_form("Spain", "Croatia"))
    adj = eng.predict(_fixture_form("Spain", "Croatia"), away_rating_adj=-40.0)
    assert adj.p_home > base.p_home
    assert adj.away_score == pytest.approx(base.away_score - 40.0)
