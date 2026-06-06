"""Tests for the pure math helpers."""

from __future__ import annotations

import math

import pytest

from betbot.strategy.probabilities import (
    edge,
    implied_probability,
    opponent_strength_factor,
    softmax,
)


class TestSoftmax:
    def test_uniform_input(self) -> None:
        p = softmax([1.0, 1.0, 1.0])
        assert all(math.isclose(x, 1 / 3, abs_tol=1e-9) for x in p)

    def test_sums_to_one(self) -> None:
        p = softmax([1.0, 2.0, 3.0])
        assert math.isclose(sum(p), 1.0, abs_tol=1e-9)

    def test_monotonic(self) -> None:
        p = softmax([1.0, 2.0, 3.0])
        assert p[0] < p[1] < p[2]

    def test_handles_large_scores(self) -> None:
        # The max-subtraction trick should prevent overflow.
        p = softmax([1000.0, 1001.0, 1002.0])
        assert math.isclose(sum(p), 1.0, abs_tol=1e-9)

    def test_empty(self) -> None:
        assert softmax([]) == []


class TestOpponentStrength:
    def test_top_team(self) -> None:
        assert opponent_strength_factor(1.0, 20, 0.5) == pytest.approx(1.5)

    def test_bottom_team(self) -> None:
        assert opponent_strength_factor(20.0, 20, 0.5) == pytest.approx(0.5)

    def test_unknown_position(self) -> None:
        assert opponent_strength_factor(None, 20, 0.5) == 1.0


class TestEdgeAndImplied:
    def test_edge(self) -> None:
        assert edge(0.6, 0.5) == pytest.approx(0.1)
        assert edge(0.4, 0.5) == pytest.approx(-0.1)

    def test_implied_clipping(self) -> None:
        assert implied_probability(-0.1) == 0.0
        assert implied_probability(1.5) == 1.0
        assert implied_probability(0.4) == 0.4
