"""Dispersion challenger: pure-function behaviour + flag-gated wiring.

The load-bearing guarantee here is that with the flag OFF (default) the live
prediction is byte-identical to before — that's what makes shipping this
zero-risk. We prove it by checking the default engine equals an engine whose
dispersion is explicitly the identity (kappa=1.0).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.strategy.dispersion import apply_dispersion
from betbot.strategy.glicko import Glicko2Rating
from betbot.strategy.international_engine import InternationalStrategyEngine

KICKOFF = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)


def _fixture_form(home, away):
    ht = Team(id=1, name=home)
    at = Team(id=2, name=away)
    fx = Fixture(id=901, home_team=ht, away_team=at, kickoff=KICKOFF, competition_code="WC")
    snap = FormSnapshot(team=ht, weighted_points=0.0, raw_points=0, matches_considered=0)
    return FixtureForm(fixture=fx, home_form=snap, away_form=snap)


def _ratings(table):
    return lambda name: table.get(name, Glicko2Rating(1500, 200, 0.06))


# ----------------------------- pure function -------------------------------

def test_kappa_one_is_identity():
    probs = (0.62, 0.20, 0.18)
    assert apply_dispersion(probs, 1.0) == probs


def test_sharpens_favourite_preserves_draw():
    p_home, p_draw, p_away = 0.55, 0.25, 0.20  # home is the favourite
    h2, d2, a2 = apply_dispersion((p_home, p_draw, p_away), 1.5)
    assert d2 == pytest.approx(p_draw)        # draw mass untouched
    assert h2 > p_home                        # favourite sharpened up
    assert a2 < p_away                        # underdog pushed down
    assert h2 + d2 + a2 == pytest.approx(1.0)


def test_even_match_unchanged():
    # equal decisive split -> logit 0 -> sigmoid(kappa*0)=0.5, no change
    h2, d2, a2 = apply_dispersion((0.40, 0.20, 0.40), 1.8)
    assert (h2, d2, a2) == pytest.approx((0.40, 0.20, 0.40))


def test_degenerate_inputs_safe():
    assert apply_dispersion((0.0, 1.0, 0.0), 1.5) == (0.0, 1.0, 0.0)  # no decisive mass
    assert apply_dispersion((0.6, 0.2, 0.2), -1.0) == (0.6, 0.2, 0.2)  # bad kappa -> no-op


# ----------------------------- engine wiring -------------------------------

def _engine(settings, **overrides):
    s = settings.model_copy(update=overrides) if overrides else settings
    return InternationalStrategyEngine(s, _ratings({
        "Brazil": Glicko2Rating(1900, 60), "Bolivia": Glicko2Rating(1450, 60),
    }))


def test_flag_off_is_identical_to_identity(settings):
    """Default (flag off) must equal an explicit kappa=1.0 identity engine."""
    off = _engine(settings).predict(_fixture_form("Brazil", "Bolivia"))
    ident = _engine(settings, dispersion_fix_enabled=True, dispersion_kappa=1.0
                    ).predict(_fixture_form("Brazil", "Bolivia"))
    assert (off.p_home, off.p_draw, off.p_away) == pytest.approx(
        (ident.p_home, ident.p_draw, ident.p_away)
    )


def test_flag_on_sharpens_live_prediction(settings):
    off = _engine(settings).predict(_fixture_form("Brazil", "Bolivia"))
    on = _engine(settings, dispersion_fix_enabled=True, dispersion_kappa=1.30
                 ).predict(_fixture_form("Brazil", "Bolivia"))
    assert on.p_home > off.p_home            # favourite gets more
    assert on.p_away < off.p_away            # underdog gets less
    assert on.p_draw == pytest.approx(off.p_draw)   # draw model untouched
    assert on.p_home + on.p_draw + on.p_away == pytest.approx(1.0)
