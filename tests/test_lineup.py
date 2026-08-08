"""Unit tests for the pure lineup rating adjustment (no network, no DB)."""

from __future__ import annotations

import pytest

from betbot.strategy.lineup import lineup_rating_adjustment

MAX = 120.0


def _minutes_11() -> dict[str, int]:
    """A team of 11 with a clear minutes hierarchy; total = 22000."""
    return {
        "star": 3000, "b": 2500, "c": 2400, "d": 2300, "e": 2200,
        "f": 2100, "g": 2000, "h": 1900, "i": 1800, "j": 1700, "k": 1600,
    }


def _full_xi() -> set[str]:
    return {"star", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"}


def test_full_xi_present_zero_adjustment():
    adj = lineup_rating_adjustment(_full_xi(), _minutes_11(), max_penalty=MAX)
    assert adj == pytest.approx(0.0)


def test_missing_top_player_is_negative_and_scales_with_its_share():
    mins = _minutes_11()
    total = sum(mins.values())
    xi = _full_xi() - {"star"}  # drop the 3000-minute regular
    adj = lineup_rating_adjustment(xi, mins, max_penalty=MAX)
    expected = -(3000 / total) * MAX
    assert adj == pytest.approx(expected)
    assert adj < 0.0


def test_missing_everyone_is_full_penalty():
    adj = lineup_rating_adjustment({"nobody"}, _minutes_11(), max_penalty=MAX)
    # Confirmed XI has no expected regular in it -> full -MAX.
    assert adj == pytest.approx(-MAX)


def test_empty_confirmed_xi_returns_zero():
    # Lineup not posted yet -> no data -> no adjustment (baseline prediction).
    assert lineup_rating_adjustment(set(), _minutes_11(), max_penalty=MAX) == 0.0


def test_empty_minutes_returns_zero():
    assert lineup_rating_adjustment(_full_xi(), {}, max_penalty=MAX) == 0.0


def test_name_normalization_full_name_match():
    # api-football startXI and /players both give full names, so normalize
    # matches them directly.
    mins = {"Kevin De Bruyne": 3000, "Erling Haaland": 2500, "Rodri": 2000}
    xi = {"Kevin De Bruyne", "Erling Haaland", "Rodri"}
    assert lineup_rating_adjustment(xi, mins, max_penalty=MAX) == pytest.approx(0.0)


def test_surname_only_startxi_still_matches_regular():
    # Documented last-name fallback: an abbreviated startXI surname
    # ("De Bruyne") still counts the full-name regular ("Kevin De Bruyne") as
    # present, so the team is not falsely penalised.
    mins = {"Kevin De Bruyne": 3000, "Erling Haaland": 2500}
    xi = {"De Bruyne", "Haaland"}
    assert lineup_rating_adjustment(xi, mins, max_penalty=MAX) == pytest.approx(0.0)


def test_zero_minute_players_are_not_expected_regulars():
    # A player with 0 minutes is not an expected regular, so its absence from
    # the XI must not add any penalty.
    mins = {"a": 2000, "b": 1000, "benchwarmer": 0}
    xi = {"a", "b"}  # benchwarmer absent, but it has 0 minutes
    assert lineup_rating_adjustment(xi, mins, max_penalty=MAX) == pytest.approx(0.0)


def test_adjustment_clamped_to_penalty_band():
    mins = _minutes_11()
    for xi in (_full_xi(), _full_xi() - {"star"}, {"none"}):
        adj = lineup_rating_adjustment(xi, mins, max_penalty=MAX)
        assert -MAX <= adj <= 0.0


def test_top_n_limits_expected_regulars():
    # With 13 players and top_n=11, the two lowest-minutes players are NOT
    # expected regulars: their absence must not be penalised.
    mins = {f"p{i}": (2000 - i * 100) for i in range(13)}  # p0..p12 descending
    top11 = {f"p{i}" for i in range(11)}
    adj = lineup_rating_adjustment(top11, mins, max_penalty=MAX, top_n=11)
    assert adj == pytest.approx(0.0)
