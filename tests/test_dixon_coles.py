"""Tests for the Dixon-Coles goal model (ensemble Layer 2)."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from betbot.strategy.dixon_coles import (
    DCMatch,
    DCParams,
    DCTeam,
    decay_weight,
    expected_goals,
    fit,
    match_probabilities,
    priors_from_scores,
    score_matrix,
)


def test_score_matrix_sums_to_one():
    grid = score_matrix(1.4, 1.1, -0.1)
    assert sum(sum(r) for r in grid) == pytest.approx(1.0)
    assert all(p >= 0 for row in grid for p in row)


def test_negative_rho_boosts_low_score_corner():
    """DC's empirical finding: independent Poisson underprices 0-0 and 1-1."""
    indep = score_matrix(1.3, 1.1, 0.0)
    dc = score_matrix(1.3, 1.1, -0.1)
    assert dc[0][0] > indep[0][0]
    assert dc[1][1] > indep[1][1]
    assert dc[1][0] < indep[1][0]


def test_equal_teams_neutral_venue_symmetric():
    params = DCParams(teams={"a": DCTeam(0.1, 0.1), "b": DCTeam(0.1, 0.1)})
    ph, pd, pa = match_probabilities(params, "a", "b", home_field=False)
    assert ph == pytest.approx(pa, abs=1e-9)
    assert ph + pd + pa == pytest.approx(1.0)


def test_home_field_raises_expected_goals():
    params = DCParams(teams={"a": DCTeam(), "b": DCTeam()}, home_adv=0.3)
    lam_neutral, _ = expected_goals(params, "a", "b", home_field=False)
    lam_home, mu_home = expected_goals(params, "a", "b", home_field=True)
    assert lam_home > lam_neutral
    assert mu_home == pytest.approx(expected_goals(params, "a", "b")[1])


def test_unknown_team_gets_default():
    params = DCParams(teams={})
    ph, pd, pa = match_probabilities(params, "x", "y")
    assert ph == pytest.approx(pa, abs=1e-9)


def test_decay_weight_halves_at_half_life():
    ref = date(2026, 6, 1)
    assert decay_weight(ref, ref, 540) == 1.0
    assert decay_weight(ref - timedelta(days=540), ref, 540) == pytest.approx(0.5)


def test_priors_from_scores_scaling():
    priors = priors_from_scores({"strong": 1.0, "weak": -1.0}, attack_scale=0.18)
    assert priors["strong"].attack == pytest.approx(0.18)
    assert priors["weak"].attack == pytest.approx(-0.18)


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth sampler — fine for the small lambdas used here."""
    l_exp, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l_exp:
            return k
        k += 1


def test_fit_recovers_strength_ordering():
    """Synthetic league: 'strong' generated with higher attack/defence must
    come out of the fit ahead of 'weak', and be favoured on neutral ground."""
    rng = random.Random(42)
    truth = DCParams(
        teams={"strong": DCTeam(0.35, 0.35), "mid": DCTeam(0.0, 0.0),
               "weak": DCTeam(-0.35, -0.35)},
        base_mu=0.15, home_adv=0.0, rho=0.0,
    )
    matches = []
    d = date(2024, 1, 1)
    pairs = [("strong", "mid"), ("strong", "weak"), ("mid", "weak"),
             ("mid", "strong"), ("weak", "strong"), ("weak", "mid")]
    for i in range(360):
        h, a = pairs[i % len(pairs)]
        lam, mu = expected_goals(truth, h, a)
        matches.append(DCMatch(
            date=d + timedelta(days=i), home=h, away=a,
            home_goals=_poisson(rng, lam), away_goals=_poisson(rng, mu),
        ))
    params = fit(matches, iterations=150, half_life_days=10_000)
    assert params.teams["strong"].attack > params.teams["weak"].attack
    assert params.teams["strong"].defence > params.teams["weak"].defence
    ph, _, pa = match_probabilities(params, "strong", "weak")
    assert ph > 0.5 > pa


def test_prior_anchors_unseen_team():
    """A team with no match data must keep its fundamentals prior exactly."""
    priors = {"ghost": DCTeam(0.2, 0.1), "a": DCTeam(), "b": DCTeam()}
    matches = [DCMatch(date(2026, 1, 1), "a", "b", 1, 1)]
    params = fit(matches, priors=priors, iterations=50)
    assert params.teams["ghost"].attack == pytest.approx(0.2)
    assert params.teams["ghost"].defence == pytest.approx(0.1)


def test_params_json_roundtrip():
    params = DCParams(
        teams={"a": DCTeam(0.1, -0.2), "b": DCTeam(-0.05, 0.3)},
        base_mu=0.12, home_adv=0.28, rho=-0.07,
    )
    back = DCParams.from_json(params.to_json())
    assert back == params
