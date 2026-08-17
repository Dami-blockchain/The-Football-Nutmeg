"""Tests for the pure season-title Monte Carlo (no I/O)."""

from __future__ import annotations

from betbot.strategy.season_sim import simulate_season


def _home_heavy(home: str, away: str):
    # A always at home vs B -> A wins ~certainly; used to pin P(title)~1.
    return (0.99, 0.005, 0.005)


def _even(home: str, away: str):
    return (1 / 3, 1 / 3, 1 / 3)


def test_dominant_team_wins_title():
    # A beats B in every remaining fixture -> A takes the title with prob ~1.
    teams = {"A", "B"}
    remaining = [("A", "B")] * 10
    res = simulate_season(
        teams=teams, played=[], remaining=remaining,
        match_prob_fn=_home_heavy, n_sims=2000, seed=1,
    )
    assert res["A"]["p_title"] > 0.98
    assert res["B"]["p_title"] < 0.02


def test_title_probs_sum_to_one():
    teams = {"A", "B", "C", "D"}
    # a small round-robin of remaining fixtures
    remaining = [
        ("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"),
        ("A", "D"), ("B", "C"),
    ]
    res = simulate_season(
        teams=teams, played=[], remaining=remaining,
        match_prob_fn=_even, n_sims=4000, seed=7,
    )
    total = sum(res[t]["p_title"] for t in teams)
    assert abs(total - 1.0) < 1e-9  # exactly one team is 1st each sim


def test_played_points_carry():
    # Two equally-rated teams; A already 6 pts ahead with few games left must
    # have a strictly higher title probability than B.
    teams = {"A", "B"}
    # A won two matches vs a since-departed side; B lost both -> A +6.
    played = [
        ("A", "X", 1, 0), ("A", "X", 1, 0),
        ("X", "B", 1, 0), ("X", "B", 1, 0),
    ]
    # X isn't in the title race set; only A and B ranked.
    remaining = [("A", "B"), ("B", "A")]  # one each way, coin-flippy
    res = simulate_season(
        teams=teams, played=played, remaining=remaining,
        match_prob_fn=_even, n_sims=4000, seed=3,
    )
    assert res["A"]["p_title"] > res["B"]["p_title"]
    # And the head start shows up in expected points too.
    assert res["A"]["exp_points"] > res["B"]["exp_points"] + 4.0


def test_determinism_same_seed():
    teams = {"A", "B", "C"}
    remaining = [("A", "B"), ("B", "C"), ("C", "A")] * 3
    kw = dict(
        teams=teams, played=[], remaining=remaining,
        match_prob_fn=_even, n_sims=1500,
    )
    r1 = simulate_season(seed=42, **kw)
    r2 = simulate_season(seed=42, **kw)
    r3 = simulate_season(seed=99, **kw)
    assert r1 == r2
    # A different seed should (almost surely) give a different tally.
    assert r1 != r3


def test_relegation_bottom_three():
    # 5-team league, D and E get thrashed every game -> they anchor relegation.
    teams = {"A", "B", "C", "D", "E"}

    def fn(home, away):
        # Strong = {A,B,C}. When a strong side plays a weak one, strong wins.
        strong = {"A", "B", "C"}
        if home in strong and away not in strong:
            return (0.9, 0.05, 0.05)
        if away in strong and home not in strong:
            return (0.05, 0.05, 0.9)
        return (1 / 3, 1 / 3, 1 / 3)

    remaining = []
    ts = sorted(teams)
    for i in range(len(ts)):
        for j in range(len(ts)):
            if i != j:
                remaining.append((ts[i], ts[j]))
    res = simulate_season(
        teams=teams, played=[], remaining=remaining,
        match_prob_fn=fn, n_sims=3000, seed=5,
    )
    # D and E (the weak sides) should carry high relegation probability.
    assert res["D"]["p_relegation"] > 0.6
    assert res["E"]["p_relegation"] > 0.6
    assert res["A"]["p_relegation"] < 0.3
