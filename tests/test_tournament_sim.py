"""Tests for the pure single-elimination bracket Monte Carlo."""

from __future__ import annotations

from betbot.strategy.tournament_sim import simulate_knockout


def _deterministic(a: str, b: str) -> float:
    """A always beats everyone; otherwise the alphabetically-first team wins."""
    if a == "A":
        return 1.0
    if b == "A":
        return 0.0
    return 1.0 if a < b else 0.0


def _elo_like(ratings: dict[str, float]):
    def fn(a: str, b: str) -> float:
        da = ratings[a] - ratings[b]
        return 1.0 / (1.0 + 10.0 ** (-da / 400.0))

    return fn


def test_two_entrant_a_always_wins():
    res = simulate_knockout(
        entrants=["A", "B"], advance_prob_fn=_deterministic, n_sims=2000, seed=1
    )
    assert res["A"] == 1.0
    assert res["B"] == 0.0


def test_p_win_sums_to_one():
    ratings = {"A": 2000, "B": 1900, "C": 1800, "D": 1700}
    res = simulate_knockout(
        entrants=list(ratings), advance_prob_fn=_elo_like(ratings),
        n_sims=5000, seed=7,
    )
    assert abs(sum(res.values()) - 1.0) < 1e-9
    assert set(res) == set(ratings)


def test_favourite_has_highest_p_win():
    ratings = {"Fav": 2100, "B": 1800, "C": 1780, "D": 1760,
               "E": 1750, "F": 1740, "G": 1730, "H": 1720}
    res = simulate_knockout(
        entrants=sorted(ratings, key=lambda t: -ratings[t]),
        advance_prob_fn=_elo_like(ratings), n_sims=8000, seed=3,
    )
    assert max(res, key=res.get) == "Fav"
    assert res["Fav"] > 0.5


def test_determinism():
    ratings = {"A": 2000, "B": 1950, "C": 1900, "D": 1850, "E": 1800}
    fn = _elo_like(ratings)
    r1 = simulate_knockout(entrants=list(ratings), advance_prob_fn=fn,
                           n_sims=3000, seed=42)
    r2 = simulate_knockout(entrants=list(ratings), advance_prob_fn=fn,
                           n_sims=3000, seed=42)
    assert r1 == r2
    r3 = simulate_knockout(entrants=list(ratings), advance_prob_fn=fn,
                           n_sims=3000, seed=43)
    assert r1 != r3


def test_byes_odd_entrant_count_no_crash():
    # 5 entrants -> pads to 8 with 3 byes to the top seeds; must not crash and
    # must return a proper distribution over exactly the real entrants.
    ratings = {"A": 2000, "B": 1900, "C": 1800, "D": 1700, "E": 1600}
    res = simulate_knockout(
        entrants=sorted(ratings, key=lambda t: -ratings[t]),
        advance_prob_fn=_elo_like(ratings), n_sims=4000, seed=11,
    )
    assert set(res) == set(ratings)
    assert abs(sum(res.values()) - 1.0) < 1e-9
    # top seed benefits from a bye AND strength -> clear favourite
    assert max(res, key=res.get) == "A"


def test_single_entrant():
    res = simulate_knockout(entrants=["Solo"], advance_prob_fn=_deterministic,
                            n_sims=10, seed=1)
    assert res == {"Solo": 1.0}


def test_empty_entrants():
    res = simulate_knockout(entrants=[], advance_prob_fn=_deterministic,
                            n_sims=10, seed=1)
    assert res == {}
