"""Margin-of-victory Glicko-2 update.

Load-bearing guarantee: with every margin == 1 goal (or a draw), the MOV update
is IDENTICAL to standard Glicko-2 — so MOV only ever *adds* information from
bigger scorelines, it never silently changes the baseline.
"""

from __future__ import annotations

import math

import pytest

from betbot.strategy.glicko import Glicko2Rating, update_rating
from betbot.strategy.glicko_mov import mov_multiplier, update_rating_mov


def test_mov_multiplier_shape():
    assert mov_multiplier(0) == 1.0          # draw
    assert mov_multiplier(1) == 1.0          # 1-goal win == standard
    assert mov_multiplier(2) == pytest.approx(1.0 + math.log(2))
    assert mov_multiplier(5) == pytest.approx(1.0 + math.log(5))
    # diminishing returns: each extra goal adds less
    assert mov_multiplier(3) - mov_multiplier(2) > mov_multiplier(5) - mov_multiplier(4)


def test_reduces_to_standard_at_unit_margin():
    r = Glicko2Rating(1500, 200, 0.06)
    opp = (1600.0, 80.0)
    std = update_rating(r, [(opp[0], opp[1], 1.0)])
    mov = update_rating_mov(r, [(opp[0], opp[1], 1.0, 1)])   # gd=1 -> multiplier 1
    assert mov.rating == pytest.approx(std.rating)
    assert mov.rd == pytest.approx(std.rd)
    assert mov.volatility == pytest.approx(std.volatility)


def test_bigger_margin_moves_rating_more():
    r = Glicko2Rating(1500, 200, 0.06)
    opp = (1600.0, 80.0)
    narrow = update_rating_mov(r, [(opp[0], opp[1], 1.0, 1)])   # 1-0 win
    blowout = update_rating_mov(r, [(opp[0], opp[1], 1.0, 5)])  # 5-0 win
    # winning by more lifts the rating further
    assert blowout.rating > narrow.rating


def test_no_results_only_inflates_rd():
    r = Glicko2Rating(1700, 60, 0.06)
    out = update_rating_mov(r, [])
    assert out.rating == r.rating
    assert out.rd > r.rd
