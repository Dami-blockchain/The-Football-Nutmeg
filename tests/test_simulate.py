"""Tournament simulator: determinism, probability mass, bracket validation.

Sim counts are deliberately tiny (<=200) — these are correctness checks, not
precision estimates; the stacked-team test uses a gap so large that even 100
sims can't plausibly invert it.
"""

from __future__ import annotations

import random

import pytest

from betbot.strategy.dixon_coles import DCParams, DCTeam
from betbot.strategy.simulate import (
    WC2026_BRACKET,
    Bracket,
    KnockoutMatch,
    best_third,
    group_runner_up,
    group_winner,
    simulate_tournament,
    validate_bracket,
    winner_of,
)

WC_GROUPS = {g: [f"{g}{i}" for i in range(1, 5)] for g in "ABCDEFGHIJKL"}

# 2 groups of 4, winners cross runners-up in the semis — the minimal bracket
# exercising group play, advancement, and knockout draws.
MINI_BRACKET = Bracket(matches=(
    KnockoutMatch("S1", "SF", group_winner("A"), group_runner_up("B")),
    KnockoutMatch("S2", "SF", group_winner("B"), group_runner_up("A")),
    KnockoutMatch("FF", "F", winner_of("S1"), winner_of("S2")),
))
MINI_GROUPS = {"A": ["a1", "a2", "a3", "a4"], "B": ["b1", "b2", "b3", "b4"]}


def _run_wc(seed: int, n_sims: int = 40):
    return simulate_tournament(
        WC_GROUPS, DCParams(), random.Random(seed), n_sims=n_sims
    )


def test_seeded_determinism():
    assert _run_wc(123) == _run_wc(123)


def test_different_seeds_differ():
    assert _run_wc(123) != _run_wc(124)


def test_probability_mass():
    probs = _run_wc(7, n_sims=50)
    assert len(probs) == 48
    # Exactly one champion, two finalists, four semifinalists, 32 advancers
    # per simulated tournament.
    assert sum(p.win for p in probs.values()) == pytest.approx(1.0)
    assert sum(p.final for p in probs.values()) == pytest.approx(2.0)
    assert sum(p.semi for p in probs.values()) == pytest.approx(4.0)
    assert sum(p.r32 for p in probs.values()) == pytest.approx(32.0)
    for p in probs.values():
        # Reaching a later stage is never more likely than an earlier one.
        assert p.r32 >= p.r16 >= p.quarter >= p.semi >= p.final >= p.win


def test_stacked_team_wins_most():
    params = DCParams(teams={"a1": DCTeam(attack=1.3, defence=1.3)})
    probs = simulate_tournament(
        MINI_GROUPS, params, random.Random(5),
        bracket=MINI_BRACKET, n_sims=150,
    )
    stacked = probs["a1"]
    assert stacked.win == max(p.win for p in probs.values())
    assert stacked.win > 0.5


def test_host_home_field_boost_in_group_stage():
    # Equal-strength teams, but a1 is a host with a huge home_adv: it should
    # almost always advance from its group, while group B stays ~symmetric.
    params = DCParams(home_adv=2.5)
    probs = simulate_tournament(
        MINI_GROUPS, params, random.Random(11),
        bracket=MINI_BRACKET, n_sims=150, hosts=frozenset({"a1"}),
    )
    assert probs["a1"].semi > 0.85


def test_penalty_randomness_validated():
    with pytest.raises(ValueError):
        simulate_tournament(
            MINI_GROUPS, DCParams(), random.Random(1),
            bracket=MINI_BRACKET, n_sims=5, penalty_randomness=1.5,
        )


def test_wc2026_bracket_validates():
    validate_bracket(WC2026_BRACKET, frozenset("ABCDEFGHIJKL"))


def test_wc2026_third_never_meets_own_group_winner():
    # FIFA rule encoded in the slot data: a third-placed team's admissible
    # groups never include the group whose winner it would face.
    for m in WC2026_BRACKET.matches:
        for one, other in ((m.a, m.b), (m.b, m.a)):
            if one.kind == "third" and other.kind == "winner":
                assert other.ref not in one.groups, m.id


def test_bracket_rejects_double_consumed_winner():
    broken = Bracket(matches=(
        KnockoutMatch("S1", "SF", group_winner("A"), group_winner("A")),
        KnockoutMatch("S2", "SF", group_winner("B"), group_runner_up("A")),
        KnockoutMatch("FF", "F", winner_of("S1"), winner_of("S2")),
    ))
    with pytest.raises(ValueError, match="winner"):
        validate_bracket(broken, frozenset("AB"))


def test_bracket_rejects_unconsumed_match():
    # S2's winner never feeds anything; the "final" replays S1's winner twice.
    broken = Bracket(matches=(
        KnockoutMatch("S1", "SF", group_winner("A"), group_runner_up("B")),
        KnockoutMatch("S2", "SF", group_winner("B"), group_runner_up("A")),
        KnockoutMatch("FF", "F", winner_of("S1"), winner_of("S1")),
    ))
    with pytest.raises(ValueError, match="consumed"):
        validate_bracket(broken, frozenset("AB"))


def test_bracket_rejects_forward_reference():
    broken = Bracket(matches=(
        KnockoutMatch("FF", "F", winner_of("S1"), winner_of("S2")),
        KnockoutMatch("S1", "SF", group_winner("A"), group_runner_up("B")),
        KnockoutMatch("S2", "SF", group_winner("B"), group_runner_up("A")),
    ))
    with pytest.raises(ValueError, match="forward"):
        validate_bracket(broken, frozenset("AB"))


def test_bracket_rejects_unknown_third_group():
    broken = Bracket(matches=(
        KnockoutMatch("S1", "SF", group_winner("A"), best_third("Z")),
        KnockoutMatch("S2", "SF", group_winner("B"), group_runner_up("A")),
        KnockoutMatch("X1", "SF", group_runner_up("B"), best_third("A")),
        KnockoutMatch("FF", "F", winner_of("S1"), winner_of("S2")),
    ))
    with pytest.raises(ValueError, match="unknown groups"):
        validate_bracket(broken, frozenset("AB"))
