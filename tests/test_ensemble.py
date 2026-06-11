"""Tests for the ensemble blend + isotonic calibration (Layer 3)."""

from __future__ import annotations

import pytest

from betbot.strategy.ensemble import (
    EnsembleWeights,
    IsotonicCalibrator,
    blend,
    calibrate,
    de_vig,
    log_pool,
)


def test_pool_of_identical_components_is_identity():
    p = (0.5, 0.3, 0.2)
    assert log_pool([(1.0, p), (2.0, p)]) == pytest.approx(p)


def test_pool_weights_pull_toward_heavier_component():
    a, b = (0.7, 0.2, 0.1), (0.3, 0.3, 0.4)
    toward_a = log_pool([(5.0, a), (1.0, b)])
    toward_b = log_pool([(1.0, a), (5.0, b)])
    assert toward_a[0] > toward_b[0]
    assert sum(toward_a) == pytest.approx(1.0)


def test_pool_skips_nonpositive_weights():
    a = (0.6, 0.25, 0.15)
    assert log_pool([(1.0, a), (0.0, (0.1, 0.1, 0.8))]) == pytest.approx(a)


def test_pool_handles_zero_probability():
    # A degenerate component must not blow up the log.
    pooled = log_pool([(1.0, (1.0, 0.0, 0.0)), (1.0, (0.4, 0.3, 0.3))])
    assert sum(pooled) == pytest.approx(1.0)
    assert all(p > 0 for p in pooled)


def test_blend_without_market():
    g, dc = (0.5, 0.3, 0.2), (0.6, 0.25, 0.15)
    out = blend(g, dc, market=None)
    assert sum(out) == pytest.approx(1.0)
    assert g[0] < out[0] < dc[0] or dc[0] < out[0] < g[0]


def test_blend_market_dominates_with_default_weights():
    g = dc = (0.5, 0.3, 0.2)
    market = (0.8, 0.15, 0.05)
    out = blend(g, dc, market, EnsembleWeights(glicko=1, dixon_coles=1, market=100))
    assert out[0] == pytest.approx(market[0], abs=0.01)


def test_de_vig_strips_overround():
    # Bookmaker triple summing to 1.06 -> proper probabilities.
    probs = de_vig([0.55, 0.30, 0.21])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.55 / 1.06)


def test_isotonic_recovers_monotone_map():
    # Model that is systematically over-confident: true freq = 0.5 * p + 0.25.
    preds = [i / 20 for i in range(1, 20)]
    obs = [0.5 * p + 0.25 for p in preds]
    cal = IsotonicCalibrator().fit(preds, obs)
    assert cal.transform(0.2) < cal.transform(0.8)          # monotone
    assert cal.transform(0.8) == pytest.approx(0.65, abs=0.06)  # pulled down


def test_isotonic_identity_when_unfitted():
    assert IsotonicCalibrator().transform(0.42) == 0.42


def test_isotonic_clamps_at_extremes():
    cal = IsotonicCalibrator().fit([0.3, 0.5, 0.7], [0.0, 1.0, 1.0])
    assert cal.transform(0.01) == cal.transform(0.3)
    assert cal.transform(0.99) == cal.transform(0.7)


def test_isotonic_json_roundtrip():
    cal = IsotonicCalibrator().fit([0.2, 0.4, 0.6, 0.8], [0.0, 0.5, 0.5, 1.0])
    back = IsotonicCalibrator.from_json(cal.to_json())
    for p in (0.1, 0.35, 0.7, 0.95):
        assert back.transform(p) == pytest.approx(cal.transform(p))


def test_calibrate_renormalises():
    cal = IsotonicCalibrator().fit([0.2, 0.5, 0.8], [0.1, 0.6, 0.9])
    out = calibrate((0.5, 0.3, 0.2), (cal, cal, cal))
    assert sum(out) == pytest.approx(1.0)


def test_calibrate_none_is_identity():
    assert calibrate((0.5, 0.3, 0.2), None) == (0.5, 0.3, 0.2)
