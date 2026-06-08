"""Tests for the backtest harness (Phase 5)."""

from __future__ import annotations

import pytest

from betbot.backtest import backtest_mock, compute_stats


class _Bet:
    """Minimal stand-in exposing the fields compute_stats reads."""

    def __init__(self, outcome, settled, p, stake, pnl):
        self.outcome = outcome
        self.settled_outcome = settled
        self.our_probability = p
        self.stake_usd = stake
        self.pnl_usd = pnl


def test_compute_stats_basic():
    bets = [
        _Bet("HOME", "HOME", 0.5, 10.0, 10.0),   # win
        _Bet("HOME", "AWAY", 0.5, 10.0, -10.0),  # loss
        _Bet("DRAW", "DRAW", 0.3, 10.0, 23.33),  # win
    ]
    r = compute_stats(bets)
    assert r.n == 3
    assert r.wins == 2
    assert r.hit_rate == pytest.approx(2 / 3)
    assert r.staked_usd == 30.0
    assert r.pnl_usd == pytest.approx(23.33)
    assert r.roi == pytest.approx(23.33 / 30.0)
    # Brier: (0.5-1)^2 + (0.5-0)^2 + (0.3-1)^2 = 0.25+0.25+0.49 = 0.99 over 3
    assert r.brier == pytest.approx(0.99 / 3)
    assert r.per_outcome["HOME"].n == 2
    assert r.per_outcome["HOME"].wins == 1
    assert r.per_outcome["DRAW"].hit_rate == 1.0


def test_compute_stats_empty():
    r = compute_stats([])
    assert r.n == 0 and r.hit_rate == 0.0 and r.roi == 0.0 and r.brier == 0.0


def test_backtest_mock_fair_market_has_no_large_edge():
    r = backtest_mock(n=2000, seed=7)
    assert r.n > 0
    assert 0.0 <= r.hit_rate <= 1.0
    # Fair market => the "edge" is pure noise => no large systematic ROI.
    assert abs(r.roi) < 0.5


def test_backtest_mock_is_deterministic():
    assert backtest_mock(n=500, seed=42).pnl_usd == backtest_mock(n=500, seed=42).pnl_usd
