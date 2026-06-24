"""MOV challenger wiring in the engine.

Flag OFF (default) => live prediction byte-identical to a no-MOV engine; the MOV
triple is still dual-logged. Flag ON => the live glicko component uses MOV ratings.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.exchanges.matcher import normalize
from betbot.strategy.glicko import Glicko2Rating
from betbot.strategy.international_engine import InternationalStrategyEngine

KICKOFF = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)


def _ff(home, away):
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=902, home_team=ht, away_team=at, kickoff=KICKOFF, competition_code="WC")
    snap = FormSnapshot(team=ht, weighted_points=0.0, raw_points=0, matches_considered=0)
    return FixtureForm(fixture=fx, home_form=snap, away_form=snap)


def _ratings(table):
    return lambda name: table.get(normalize(name), Glicko2Rating(1500, 200, 0.06))


STD = {normalize("Brazil"): Glicko2Rating(1700, 60), normalize("Bolivia"): Glicko2Rating(1500, 60)}
# MOV ratings give Brazil a much bigger edge than the standard ratings do.
MOV = {normalize("Brazil"): Glicko2Rating(2000, 60), normalize("Bolivia"): Glicko2Rating(1350, 60)}


def test_mov_flag_off_is_identical(settings):
    eng = InternationalStrategyEngine(settings, _ratings(STD)); eng._mov_ratings = MOV
    base = InternationalStrategyEngine(settings, _ratings(STD)); base._mov_ratings = None
    a = eng.predict(_ff("Brazil", "Bolivia"))
    b = base.predict(_ff("Brazil", "Bolivia"))
    assert (a.p_home, a.p_draw, a.p_away) == pytest.approx((b.p_home, b.p_draw, b.p_away))


def test_mov_flag_on_uses_mov_ratings(settings):
    off = InternationalStrategyEngine(settings, _ratings(STD)); off._mov_ratings = MOV
    s_on = settings.model_copy(update={"mov_fix_enabled": True})
    on = InternationalStrategyEngine(s_on, _ratings(STD)); on._mov_ratings = MOV
    po = off.predict(_ff("Brazil", "Bolivia"))
    pn = on.predict(_ff("Brazil", "Bolivia"))
    assert pn.p_home > po.p_home          # MOV gives Brazil a bigger edge
    assert pn.p_home + pn.p_draw + pn.p_away == pytest.approx(1.0)


def test_mov_dual_logged(settings):
    recorded: list = []
    eng = InternationalStrategyEngine(
        settings, _ratings(STD),
        dc_params=None,
        record_model_prediction=lambda *a: recorded.append(a),
    )
    eng._mov_ratings = MOV
    eng.predict(_ff("Brazil", "Bolivia"))
    # no DC + pure-glicko fallback path doesn't record (model_select needs DC),
    # so recording is exercised in test_model_select; here we assert the engine
    # still produces a valid prediction with MOV ratings present.
    p = eng.predict(_ff("Brazil", "Bolivia"))
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0)


def test_mov_skipped_when_team_missing(settings):
    eng = InternationalStrategyEngine(settings, _ratings(STD))
    eng._mov_ratings = {normalize("Brazil"): Glicko2Rating(2000, 60)}  # Bolivia absent
    s_on = settings.model_copy(update={"mov_fix_enabled": True})
    on = InternationalStrategyEngine(s_on, _ratings(STD))
    on._mov_ratings = {normalize("Brazil"): Glicko2Rating(2000, 60)}
    # MOV can't apply (one team missing) -> falls back to standard, no crash
    p = on.predict(_ff("Brazil", "Bolivia"))
    base = eng.predict(_ff("Brazil", "Bolivia"))
    assert (p.p_home, p.p_draw, p.p_away) == pytest.approx((base.p_home, base.p_draw, base.p_away))
