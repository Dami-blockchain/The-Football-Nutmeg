"""Tests for the WC model calibration report."""

from __future__ import annotations

import pytest

from betbot.calibration import build_report, calibrate_model
from betbot.reports import format_calibration_report


def test_empty_is_safe():
    m = calibrate_model("ensemble", [])
    assert m.n == 0 and m.rps == 0.0 and m.bins == ()
    rep = build_report([])
    assert "No settled WC matches yet" in format_calibration_report(rep)


def test_perfect_model_scores_zero():
    # Always 100% on the actual outcome → RPS, Brier, log-loss all ~0.
    preds = [((1.0, 0.0, 0.0), 0), ((0.0, 0.0, 1.0), 2)]
    m = calibrate_model("ensemble", preds)
    assert m.rps == pytest.approx(0.0)
    assert m.brier == pytest.approx(0.0)
    assert m.favourite_hit_rate == pytest.approx(1.0)


def test_hit_rate_and_brier():
    # Two favourites at HOME; one wins, one loses.
    preds = [((0.7, 0.2, 0.1), 0), ((0.7, 0.2, 0.1), 2)]
    m = calibrate_model("ensemble", preds)
    assert m.favourite_hit_rate == pytest.approx(0.5)
    # Brier per match: hit 0.09+0.04+0.01=0.14 ; miss 0.49+0.04+0.81=1.34 → mean 0.74
    assert m.brier == pytest.approx(0.74, abs=1e-6)


def test_reliability_detects_overconfidence():
    # Favourite predicted ~80% but only wins half the time → positive gap.
    preds = [((0.8, 0.1, 0.1), 0)] * 2 + [((0.8, 0.1, 0.1), 2)] * 2
    m = calibrate_model("ensemble", preds)
    band = [b for b in m.bins if b.lo == 0.80][0]
    assert band.n == 4
    assert band.mean_predicted == pytest.approx(0.8)
    assert band.hit_rate == pytest.approx(0.5)
    assert band.gap == pytest.approx(0.3)  # over-confident by 30 pts


def test_well_calibrated_has_small_gap():
    # 70% favourites that win ~70% of the time.
    preds = [((0.7, 0.2, 0.1), 0)] * 7 + [((0.7, 0.2, 0.1), 2)] * 3
    m = calibrate_model("ensemble", preds)
    band = [b for b in m.bins if b.lo == 0.70][0]
    assert abs(band.gap) < 0.05


def test_report_renders_both_models():
    rows = [
        ((0.5, 0.3, 0.2), (0.6, 0.25, 0.15), 0),
        ((0.4, 0.3, 0.3), (0.5, 0.25, 0.25), 2),
    ]
    rep = build_report(rows)
    assert rep.glicko.n == 2 and rep.ensemble.n == 2
    text = format_calibration_report(rep)
    assert "calibration" in text.lower()
    assert "ensemble" in text and "glicko" in text
    assert "Reliability" in text


def test_build_report_splits_models():
    # Glicko favours AWAY, ensemble favours HOME; outcome HOME.
    rows = [((0.2, 0.2, 0.6), (0.7, 0.2, 0.1), 0)]
    rep = build_report(rows)
    assert rep.ensemble.favourite_hit_rate == 1.0   # ensemble called it
    assert rep.glicko.favourite_hit_rate == 0.0     # glicko didn't
