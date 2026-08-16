"""Unit tests for the naive StrategyEngine calibration (per-game form fix).

The naive engine is the unrated-team fallback for the club ensemble, so it must
never emit degenerate H0/D0/A100 triples and must produce sensible draws on the
live per-game form scale (weighted_points is a recency-weighted SUM over the last
N matches; ~11-12 for N=5, so it is divided by matches_considered here).
"""

from __future__ import annotations

from datetime import datetime, timezone

from betbot.config import get_settings
from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.strategy.club_engine import ClubStrategyEngine
from betbot.strategy.engine import StrategyEngine
from betbot.strategy.glicko import Glicko2Rating


def _ff(
    home: str = "Home FC",
    away: str = "Away FC",
    *,
    home_pts: float = 0.0,
    away_pts: float = 0.0,
    home_n: int = 5,
    away_n: int = 5,
) -> FixtureForm:
    """Build a FixtureForm with SUMMED weighted_points + matches_considered.

    ``home_pts`` / ``away_pts`` are the SUMMED weighted_points (what FormService
    actually returns); the engine divides by matches_considered internally.
    """
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=7, home_team=ht, away_team=at,
                 kickoff=datetime.now(timezone.utc), competition_code="PL")
    return FixtureForm(
        fixture=fx,
        home_form=FormSnapshot(
            team=ht, weighted_points=home_pts, raw_points=0, matches_considered=home_n
        ),
        away_form=FormSnapshot(
            team=at, weighted_points=away_pts, raw_points=0, matches_considered=away_n
        ),
    )


def _probs_sane(p) -> None:
    assert abs(p.p_home + p.p_draw + p.p_away - 1.0) < 1e-9
    for v in (p.p_home, p.p_draw, p.p_away):
        assert 0.0 < v < 1.0, f"degenerate prob {v} in {p}"


def test_equal_per_game_form_reasonable_draw_and_balanced():
    # Two average teams: 7.0 summed over 5 games => 1.4 per game each.
    s = get_settings()
    p = StrategyEngine(s).predict(
        _ff(home_pts=7.0, away_pts=7.0, home_n=5, away_n=5)
    )
    _probs_sane(p)
    assert 0.18 <= p.p_draw <= 0.35, p.p_draw
    # "home ~ away": only the home-advantage tilt separates them.
    assert abs(p.p_home - p.p_away) < 0.15, (p.p_home, p.p_away)
    assert p.p_home >= p.p_away  # home advantage


def test_season_start_no_form_gives_prior_not_degenerate():
    # matches_considered == 0 for both sides (season start). MUST never be
    # 0/0/100 — the old bug. All three in a near-uniform band, home >= away.
    s = get_settings()
    p = StrategyEngine(s).predict(
        _ff(home_pts=0.0, away_pts=0.0, home_n=0, away_n=0)
    )
    _probs_sane(p)
    for v in (p.p_home, p.p_draw, p.p_away):
        assert 0.15 <= v <= 0.55, (v, p)
    assert p.p_home >= p.p_away  # home advantage nudges home up


def test_strong_home_weak_away_favours_home_but_others_alive():
    # Strong home (2.6 pg) vs weak away (0.6 pg): home favoured, but draw and
    # away must both stay above a few percent (never collapsed).
    s = get_settings()
    p = StrategyEngine(s).predict(
        _ff(home_pts=13.0, away_pts=3.0, home_n=5, away_n=5)
    )
    _probs_sane(p)
    assert p.p_home > p.p_away
    assert p.p_home > p.p_draw
    assert p.p_away > 0.03, p.p_away
    assert p.p_draw > 0.03, p.p_draw


def test_single_match_form_uses_recency_normaliser_not_match_count():
    # REAL FormService scale: one win = 1.5 (recency weight) * 3 pts = 4.5
    # weighted_points with matches_considered=1. Dividing by the match count
    # (the original hotfix) yields 4.5 "per game" -> H95/D4/A<1 on ONE match of
    # evidence. Normalised by the weight sum it is 3.0 pg, and the away/draw
    # tails must stay above ~3%.
    s = get_settings()
    p = StrategyEngine(s).predict(
        _ff(home_pts=4.5, away_pts=0.0, home_n=1, away_n=1)
    )
    _probs_sane(p)
    assert p.p_away >= 0.03, p.p_away
    assert p.p_draw >= 0.05, p.p_draw
    assert p.p_home <= 0.90, p.p_home


def test_opponent_factor_inflated_form_is_clamped_to_points_scale():
    # weighted_points folds in an opponent-strength factor up to 1.5x: a 5-win
    # streak vs top sides can reach 5.8 * 3 * 1.5 = 26.1 summed. The per-game
    # value must clamp to 3.0 so the softmax spread stays bounded and no outcome
    # collapses below ~3%.
    s = get_settings()
    p = StrategyEngine(s).predict(
        _ff(home_pts=26.1, away_pts=3.0, home_n=5, away_n=5)
    )
    _probs_sane(p)
    assert p.p_away >= 0.03, p.p_away
    assert p.p_draw >= 0.05, p.p_draw


def test_shrinkage_makes_low_n_more_conservative_than_high_n():
    # Same per-game form (3.0 pg home, 0.0 pg away) but n=1 vs n=5. With
    # sample-size shrinkage the favourite's probability must be STRICTLY LOWER
    # at n=1 (one game of evidence is trusted less), and the draw stays alive.
    s = get_settings()
    eng = StrategyEngine(s)
    # n=1: one win = 1.5 recency-weight * 3 pts = 4.5 weighted => 3.0 pg raw.
    p_low = eng.predict(_ff(home_pts=4.5, away_pts=0.0, home_n=1, away_n=1))
    # n=5: same 3.0 pg raw => weighted_points = 3.0 * recency_weight_sum(5).
    from betbot.data.form import recency_weight_sum

    hi_pts = 3.0 * recency_weight_sum(5)
    p_high = eng.predict(_ff(home_pts=hi_pts, away_pts=0.0, home_n=5, away_n=5))
    _probs_sane(p_low)
    _probs_sane(p_high)
    assert p_low.p_home < p_high.p_home, (p_low.p_home, p_high.p_home)
    assert p_low.p_draw >= 0.10, p_low.p_draw


def test_shrinkage_n0_unchanged_near_uniform_prior():
    # n=0 both sides: shrinkage must NOT change the season-start prior behaviour
    # (already fully-prior at 0.0 pg). All three near-uniform, home >= away.
    s = get_settings()
    p = StrategyEngine(s).predict(_ff(home_pts=0.0, away_pts=0.0, home_n=0, away_n=0))
    _probs_sane(p)
    for v in (p.p_home, p.p_draw, p.p_away):
        assert 0.15 <= v <= 0.55, (v, p)
    assert p.p_home >= p.p_away


def test_shrinkage_large_n_approaches_unshrunk():
    # As n grows the data weight n/(n+K) -> 1, so a large-n prediction must
    # approach the K=0 (un-shrunk) value for the SAME per-game form.
    from betbot.data.form import recency_weight_sum

    big_n = 200
    pg = 2.5
    pts = pg * recency_weight_sum(big_n)
    ff = _ff(home_pts=pts, away_pts=0.0, home_n=big_n, away_n=big_n)

    # model_copy => independent instances (get_settings is lru_cache singleton).
    base = get_settings()
    shrunk = StrategyEngine(base.model_copy(update={"form_shrinkage_k": 4.0})).predict(ff)
    unshrunk = StrategyEngine(base.model_copy(update={"form_shrinkage_k": 0.0})).predict(ff)
    _probs_sane(shrunk)
    assert abs(shrunk.p_home - unshrunk.p_home) < 0.01, (
        shrunk.p_home,
        unshrunk.p_home,
    )


def test_shrinkage_never_degenerate_across_n():
    # No NaN / 0 / 1 at any small n for a strong-vs-weak matchup; draws alive.
    import math

    s = get_settings()
    eng = StrategyEngine(s)
    for n in (1, 2, 3, 4, 5):
        p = eng.predict(_ff(home_pts=4.5 * n, away_pts=0.0, home_n=n, away_n=n))
        _probs_sane(p)
        for v in (p.p_home, p.p_draw, p.p_away):
            assert not math.isnan(v)
        assert p.p_draw >= 0.10, (n, p.p_draw)


def test_club_engine_unrated_fallback_not_degenerate():
    # The Racing-Santander case: away side has default RD (no rating history),
    # so the club engine defers to the naive engine. Must NOT be 0/0/100.
    s = get_settings()
    ratings = {"Known FC": Glicko2Rating(1700.0, 60.0, 0.06)}
    eng = ClubStrategyEngine(
        s,
        get_rating=lambda n: ratings.get(
            n,
            Glicko2Rating(
                s.glicko_default_rating, s.glicko_default_rd, s.glicko_default_vol
            ),
        ),
        dc_params=None,
        calibrators=None,
        name_map={},
    )
    # Newly-promoted, season start: zero form on both sides.
    p = eng.predict(_ff("Known FC", "Real Racing Club de Santander",
                        home_pts=0.0, away_pts=0.0, home_n=0, away_n=0))
    assert not (p.p_home == 0.0 and p.p_draw == 0.0), p
    assert p.p_draw > 0.05, p.p_draw
    _probs_sane(p)
