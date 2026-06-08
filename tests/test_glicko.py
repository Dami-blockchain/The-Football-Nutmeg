"""Tests for the Glicko-2 engine (Phase 5.5). The worked example is the anchor."""

from __future__ import annotations

import pytest

from betbot.strategy.glicko import (
    Glicko2Rating,
    match_probabilities,
    update_rating,
)


def test_glickman_worked_example():
    """Glickman (2012) worked example — the single most important test.

    Subject 1500/RD200/vol0.06/tau0.5 vs opponents
    (1400,RD30,win), (1550,RD100,loss), (1700,RD300,loss).
    Expected: rating≈1464.05, RD≈151.52, vol≈0.05999.
    """
    subject = Glicko2Rating(rating=1500, rd=200, volatility=0.06)
    results = [(1400, 30, 1.0), (1550, 100, 0.0), (1700, 300, 0.0)]
    out = update_rating(subject, results, tau=0.5)
    assert out.rating == pytest.approx(1464.05, abs=0.1)
    assert out.rd == pytest.approx(151.52, abs=0.1)
    assert out.volatility == pytest.approx(0.05999, abs=0.0001)


def test_rd_grows_when_not_playing():
    r = Glicko2Rating(rating=1500, rd=200, volatility=0.06)
    out = update_rating(r, [], tau=0.5)
    assert out.rd > r.rd                  # uncertainty increases
    assert out.rating == r.rating         # rating unchanged
    assert out.volatility == r.volatility


def test_probabilities_sum_to_one_and_order():
    strong = Glicko2Rating(rating=1800, rd=60)
    weak = Glicko2Rating(rating=1400, rd=60)
    p_home, p_draw, p_away = match_probabilities(strong, weak)
    assert p_home + p_draw + p_away == pytest.approx(1.0)
    assert p_home > p_away                # stronger home team favoured


def test_draw_probability_clamped():
    even = Glicko2Rating(rating=1500, rd=50)
    _, p_draw, _ = match_probabilities(even, even, draw_rho=0.9)
    assert 0.05 <= p_draw <= 0.40         # clamp respected even with high rho


def test_home_field_helps_home():
    a = Glicko2Rating(rating=1500, rd=60)
    b = Glicko2Rating(rating=1500, rd=60)
    ph_neutral, _, _ = match_probabilities(a, b, home_field_mu=0.0)
    ph_host, _, _ = match_probabilities(a, b, home_field_mu=0.3)
    assert ph_host > ph_neutral
