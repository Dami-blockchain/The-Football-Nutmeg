"""Tests for the pure availability -> Glicko-adjustment math."""

from __future__ import annotations

import pytest

from betbot.strategy.availability import assess, availability_penalty


def test_zero_unavailable_is_zero():
    assert availability_penalty(0, per_player=8.0, cap=40.0) == 0.0


def test_linear_below_cap():
    assert availability_penalty(3, per_player=8.0, cap=40.0) == pytest.approx(-24.0)


def test_caps_the_penalty():
    # 10 * 8 = 80, capped at 40 -> -40
    assert availability_penalty(10, per_player=8.0, cap=40.0) == pytest.approx(-40.0)
    # exactly at the cap boundary
    assert availability_penalty(5, per_player=8.0, cap=40.0) == pytest.approx(-40.0)


def test_negative_inputs_are_floored_to_zero():
    assert availability_penalty(-3, per_player=8.0, cap=40.0) == 0.0


def test_always_non_positive():
    for n in range(0, 20):
        assert availability_penalty(n, per_player=8.0, cap=40.0) <= 0.0


def test_assess_dataclass():
    a = assess(2, per_player=8.0, cap=40.0)
    assert a.n_unavailable == 2
    assert a.rating_delta == pytest.approx(-16.0)
