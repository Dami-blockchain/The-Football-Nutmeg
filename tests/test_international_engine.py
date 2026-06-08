"""Tests for the Glicko-based international engine (Phase 5.5)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.exchanges.base import Outcome
from betbot.strategy.glicko import Glicko2Rating
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


def test_predict_probs_sum_to_one(settings):
    eng = InternationalStrategyEngine(settings, _ratings({
        "Brazil": Glicko2Rating(1900, 60), "Bolivia": Glicko2Rating(1450, 60),
    }))
    p = eng.predict(_fixture_form("Brazil", "Bolivia"))
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0)
    assert p.p_home > p.p_away          # stronger team favoured
    assert p.competition_code == "WC"


def test_host_bump_only_for_hosts(settings):
    table = {"Mexico": Glicko2Rating(1500, 60), "Germany": Glicko2Rating(1500, 60)}
    eng = InternationalStrategyEngine(settings, _ratings(table))
    # Mexico (host) at home vs equal-rated Germany -> p_home gets the host bump.
    p_host = eng.predict(_fixture_form("Mexico", "Germany"))
    # Germany (non-host) at home vs equal Mexico -> no bump.
    p_nonhost = eng.predict(_fixture_form("Germany", "Mexico"))
    assert p_host.p_home > p_nonhost.p_home


def test_unknown_team_uses_default_rating(settings):
    eng = InternationalStrategyEngine(settings, _ratings({}))
    p = eng.predict(_fixture_form("Atlantis", "Wakanda"))
    # Two unknowns => default ratings => near-symmetric.
    assert p.p_home == pytest.approx(p.p_away, abs=0.05)


def test_decide_with_market_uses_edge_filter(settings):
    eng = InternationalStrategyEngine(settings, _ratings({
        "Brazil": Glicko2Rating(1950, 50), "Bolivia": Glicko2Rating(1400, 50),
    }))
    p = eng.predict(_fixture_form("Brazil", "Bolivia"))
    # cheap market price on a strong favourite -> edge -> a decision
    assert eng.decide_with_market(p, Outcome.HOME, 0.40) is not None
    # market price above our prob -> no edge -> None
    assert eng.decide_with_market(p, Outcome.HOME, 0.99) is None
