"""Tests for the pre-registered confidence filter on the BET / NO BET call.

Covers the two pre-registered rules (threshold + draw abstention), the
flag-gate (default OFF => nothing is ever called), the two-metric split, and
the fact that the filter NEVER changes a probability or a prediction — only
whether that prediction is put forward as a bet.
"""

from __future__ import annotations

import pytest

from betbot.config import get_settings
from betbot.strategy.confidence import (
    ConfidenceCall,
    call_stats,
    evaluate,
    evaluate_settings,
    favourite,
    wilson_interval,
)

ON = {"enabled": True, "threshold": 0.60, "draw_margin": 0.05}


# ---- favourite -------------------------------------------------------

def test_favourite_picks_argmax():
    assert favourite((0.62, 0.22, 0.16)) == ("HOME", 0.62)
    assert favourite((0.16, 0.22, 0.62)) == ("AWAY", 0.62)
    assert favourite((0.25, 0.40, 0.35))[0] == "DRAW"


# ---- flag gate -------------------------------------------------------

def test_flag_defaults_off_in_settings():
    """Shipped default is OFF — live behaviour must be unchanged."""
    assert get_settings().club_confidence_filter is False


def test_disabled_never_calls_even_on_a_stone_cold_favourite():
    c = evaluate((0.90, 0.06, 0.04), enabled=False, threshold=0.60, draw_margin=0.05)
    assert c.called is False
    assert c.reason == "disabled"
    assert c.is_no_bet is True
    # The prediction itself is untouched.
    assert c.pick == "HOME" and c.p_pick == pytest.approx(0.90)


def test_evaluate_settings_reads_the_flag(monkeypatch):
    s = get_settings()
    assert evaluate_settings((0.70, 0.15, 0.15), s).called is False  # default OFF
    monkeypatch.setattr(s, "club_confidence_filter", True, raising=False)
    assert evaluate_settings((0.70, 0.15, 0.15), s).called is True


# ---- rule 1: threshold ----------------------------------------------

def test_above_threshold_is_called():
    c = evaluate((0.62, 0.22, 0.16), **ON)
    assert c.called is True and c.reason == "called" and c.pick == "HOME"


def test_below_threshold_is_no_bet():
    c = evaluate((0.58, 0.24, 0.18), **ON)
    assert c.called is False and c.reason == "below_threshold"


def test_threshold_boundary_is_inclusive():
    """p == threshold clears it; a hair under does not. Boundary pinned so a
    later refactor cannot silently move the pre-registered cut."""
    assert evaluate((0.60, 0.22, 0.18), **ON).called is True
    assert evaluate((0.5999, 0.22, 0.1801), **ON).called is False


def test_threshold_is_configurable():
    probs = (0.58, 0.22, 0.20)
    assert evaluate(probs, enabled=True, threshold=0.60, draw_margin=0.05).called is False
    assert evaluate(probs, enabled=True, threshold=0.55, draw_margin=0.05).called is True


# ---- rule 2: draw abstention ----------------------------------------

def test_draw_too_close_forces_no_bet_even_above_threshold():
    """Favourite clears 0.60 but the draw is within 0.05 — abstain."""
    c = evaluate((0.62, 0.60, 0.30), **ON)  # not normalised on purpose
    assert c.called is False and c.reason == "draw_too_close"


def test_draw_margin_boundary():
    # gap exactly 0.05 -> called (margin is the minimum acceptable gap)
    assert evaluate((0.65, 0.60, 0.20), **ON).called is True
    # gap 0.049 -> abstain
    assert evaluate((0.649, 0.60, 0.20), **ON).called is False


def test_draw_favourite_is_never_called():
    c = evaluate((0.20, 0.62, 0.18), **ON)
    assert c.called is False and c.reason == "draw_favourite" and c.pick == "DRAW"


def test_draw_margin_is_configurable():
    probs = (0.66, 0.62, 0.20)
    assert evaluate(probs, enabled=True, threshold=0.60, draw_margin=0.05).called is False
    assert evaluate(probs, enabled=True, threshold=0.60, draw_margin=0.02).called is True


# ---- the filter is a SELECTION rule, not a model change --------------

def test_filter_never_alters_the_prediction():
    probs = (0.58, 0.24, 0.18)
    for enabled in (False, True):
        c = evaluate(probs, enabled=enabled, threshold=0.60, draw_margin=0.05)
        assert (c.pick, c.p_pick, c.p_draw) == ("HOME", 0.58, 0.24)


def test_call_is_immutable():
    c = evaluate((0.7, 0.2, 0.1), **ON)
    assert isinstance(c, ConfidenceCall)
    with pytest.raises(Exception):
        c.called = False  # frozen dataclass


# ---- reporting: two metrics, never merged ---------------------------

def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(70, 100)
    assert lo < 0.70 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_interval_is_not_degenerate_at_the_extremes():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.0 < lo < 1.0
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0


def test_wilson_interval_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_call_stats_splits_all_match_from_called():
    records = [
        ((0.70, 0.18, 0.12), "HOME"),   # called, hit
        ((0.68, 0.20, 0.12), "DRAW"),   # called, miss
        ((0.52, 0.26, 0.22), "HOME"),   # below threshold -> not called, all-match hit
        ((0.20, 0.55, 0.25), "DRAW"),   # draw favourite -> not called, all-match hit
    ]
    st = call_stats(records, **ON)
    assert st["all"] == {
        "n": 4, "hits": 3, "hit_rate": 0.75,
        "ci_lo": st["all"]["ci_lo"], "ci_hi": st["all"]["ci_hi"],
    }
    assert st["called"]["n"] == 2
    assert st["called"]["hits"] == 1
    assert st["called"]["hit_rate"] == pytest.approx(0.5)
    assert st["called"]["call_rate"] == pytest.approx(0.5)
    # The two metrics are genuinely different numbers here — that is the point.
    assert st["all"]["hit_rate"] != st["called"]["hit_rate"]


def test_call_stats_with_filter_off_calls_nothing():
    records = [((0.90, 0.06, 0.04), "HOME")]
    st = call_stats(records, enabled=False, threshold=0.60, draw_margin=0.05)
    assert st["all"]["n"] == 1 and st["all"]["hits"] == 1
    assert st["called"]["n"] == 0 and st["called"]["hit_rate"] == 0.0
