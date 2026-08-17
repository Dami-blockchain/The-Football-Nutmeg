"""Season-title Monte Carlo — pure, engine-agnostic, no I/O.

Given the results already played, the fixtures still to come, and a callable
that prices any single fixture as a 1X2 probability triple, this rolls the rest
of the season ``n_sims`` times: each remaining fixture's outcome is sampled from
its (p_home, p_draw, p_away), points are awarded 3/1/0, and every team's final
league position is read off the completed table. Across the runs we tally each
team's P(finishing 1st) = P(title), P(top-4), P(relegation = bottom-3), and its
mean final points.

Deliberately dependency-free and deterministic given a ``seed`` (a seeded
``random.Random``), so the same inputs always yield the same numbers — the
tests pin this. It is *engine-agnostic*: the caller injects ``match_prob_fn``,
so the same simulator serves the live club ensemble, a backtest, or a unit test
with hand-made probabilities.

Efficiency: the inner loop is one ``Random.random()`` draw and a dict bump per
remaining fixture, and per-fixture probabilities are cached once up-front (they
don't change across sims), so 10k sims x ~380 fixtures is a few seconds of pure
Python. Tie-breaking for ranking is points only (v1), with a deterministic
alphabetical fallback so a given points table always ranks the same way.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence

Triple = tuple[float, float, float]
PlayedResult = tuple[str, str, int, int]  # home, away, home_goals, away_goals
Fixture = tuple[str, str]                  # home, away


def _seed_points(
    teams: set[str], played: Iterable[PlayedResult]
) -> dict[str, float]:
    """Current points from already-played results (3 win / 1 draw / 0 loss)."""
    pts = {t: 0.0 for t in teams}
    for home, away, hg, ag in played:
        # A team appearing only in results (e.g. a since-relegated side) is
        # ignored — the standings we care about are over ``teams``.
        if home in pts:
            pts[home] += 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
        if away in pts:
            pts[away] += 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
    return pts


def simulate_season(
    *,
    teams: set[str],
    played: Sequence[PlayedResult],
    remaining: Sequence[Fixture],
    match_prob_fn: Callable[[str, str], Triple],
    n_sims: int = 10000,
    seed: int = 20260817,
) -> dict[str, dict[str, float]]:
    """Monte-Carlo the rest of the season.

    Returns ``{team: {"p_title", "p_top4", "p_relegation", "exp_points"}}``.
    ``match_prob_fn(home, away) -> (p_home, p_draw, p_away)`` prices each
    remaining fixture; it is called once per distinct fixture and cached.
    """
    team_list = sorted(teams)
    n_teams = len(team_list)
    base_points = _seed_points(teams, played)

    # Price every remaining fixture ONCE (probabilities are static across sims)
    # and pre-compute the home/draw cumulative thresholds for a single-draw
    # sample: r < c_home -> home win, r < c_draw -> draw, else away win.
    priced: list[tuple[str, str, float, float]] = []
    for home, away in remaining:
        p_home, p_draw, _p_away = match_prob_fn(home, away)
        priced.append((home, away, p_home, p_home + p_draw))

    rng = random.Random(seed)
    rand = rng.random

    title_counts = {t: 0 for t in team_list}
    top4_counts = {t: 0 for t in team_list}
    releg_counts = {t: 0 for t in team_list}
    points_sum = {t: 0.0 for t in team_list}

    releg_cut = n_teams - 3  # bottom-3 = relegation (indices >= this after sort)

    for _ in range(n_sims):
        pts = dict(base_points)
        for home, away, c_home, c_draw in priced:
            r = rand()
            if r < c_home:
                pts[home] += 3.0
            elif r < c_draw:
                pts[home] += 1.0
                pts[away] += 1.0
            else:
                pts[away] += 3.0

        # Final table: points desc, then a deterministic alphabetical fallback
        # so an exact points tie always ranks the same way (points-only v1).
        table = sorted(team_list, key=lambda t: (-pts[t], t))
        title_counts[table[0]] += 1
        for t in table[:4]:
            top4_counts[t] += 1
        for t in table[releg_cut:]:
            releg_counts[t] += 1
        for t in team_list:
            points_sum[t] += pts[t]

    return {
        t: {
            "p_title": title_counts[t] / n_sims,
            "p_top4": top4_counts[t] / n_sims,
            "p_relegation": releg_counts[t] / n_sims,
            "exp_points": points_sum[t] / n_sims,
        }
        for t in team_list
    }
