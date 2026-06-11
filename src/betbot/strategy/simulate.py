"""Monte Carlo World Cup tournament simulator — outright/futures probabilities.

Why simulate instead of solving analytically: the 2026 format (12 groups of
four, top two plus the eight best third-placed teams into a round of 32, with
third-place slots filled from constraint tables) makes exact propagation of
the Dixon-Coles match distributions intractable. Sampling whole tournaments
is simple, exact in the limit, and goal-difference tiebreaks fall out
naturally because every group match has a sampled scoreline.

Pure module — no IO, no clock, no global RNG. The caller injects the fitted
``DCParams``, the group assignments, the bracket (a plain data structure so
the operator can correct it without touching logic), and a seeded
``random.Random``; identical inputs give identical outputs.

Documented modelling approximations:

- Goals come from the DC score matrix (:func:`betbot.strategy.dixon_coles.
  score_matrix`), so points, goal difference and goals scored are all real
  sampled quantities for group tiebreaks; ties remaining after points/GD/GF
  are broken by the injected rng (a stand-in for FIFA's conduct-score and
  ranking criteria, which we do not model).
- Knockout draws after "90 minutes" are resolved by a coin-weighted
  extra-time/penalty model: ``P(A advances | draw) = penalty_randomness * 0.5
  + (1 - penalty_randomness) * skill`` where ``skill`` is A's share of the
  decisive-result mass in the score matrix. ``penalty_randomness=1`` is a
  pure coin flip (shootouts as lotteries); ``0`` trusts the DC strengths
  fully. Default 0.5 splits the difference.
- Host advantage: the caller-supplied ``hosts`` (USA/Mexico/Canada in 2026)
  get the DC ``home_field`` boost in the GROUP STAGE ONLY. This approximates
  the real schedule — hosts are guaranteed home venues in their groups,
  while their knockout venues depend on the draw path — so knockout matches
  are treated as neutral for everyone.
- Third-place allocation: FIFA's published chart maps each of the 495
  combinations of qualified third-placed teams to specific R32 slots. We
  implement the verified per-slot CONSTRAINTS (each third slot lists its
  admissible source groups, exactly as printed in the official match
  schedule — e.g. match 74 takes the third from group A/B/C/D/F) and use a
  documented deterministic fallback for the choice WITHIN those constraints:
  slots are filled in match order, preferring the best-ranked qualified
  third, with backtracking so a complete assignment always exists. This is
  NOT the literal 495-row FIFA table; if FIFA's realised allocation differs,
  edit ``WC2026_BRACKET`` (it is data, not code).
"""

from __future__ import annotations

import itertools
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Literal

from betbot.strategy.dixon_coles import DCParams, expected_goals, score_matrix

SlotKind = Literal["winner", "runner_up", "third", "match"]

# Round labels with special meaning to the stats accumulator. Custom brackets
# (e.g. test mini-tournaments) may use any subset; unknown labels are ignored.
ROUND_LABELS: tuple[str, ...] = ("R32", "R16", "QF", "SF", "F")


@dataclass(frozen=True)
class Slot:
    """Where one side of a knockout match comes from.

    ``kind="winner"``/``"runner_up"``: ``ref`` is a group name.
    ``kind="third"``: ``groups`` lists the admissible source groups.
    ``kind="match"``: ``ref`` is the id of an earlier knockout match.
    """

    kind: SlotKind
    ref: str = ""
    groups: tuple[str, ...] = ()


def group_winner(group: str) -> Slot:
    return Slot("winner", group)


def group_runner_up(group: str) -> Slot:
    return Slot("runner_up", group)


def best_third(*groups: str) -> Slot:
    return Slot("third", groups=tuple(groups))


def winner_of(match_id: str) -> Slot:
    return Slot("match", match_id)


@dataclass(frozen=True)
class KnockoutMatch:
    id: str
    round: str  # one of ROUND_LABELS (anything else still plays, just untallied)
    a: Slot
    b: Slot


@dataclass(frozen=True)
class Bracket:
    """The knockout structure, in play order; the LAST match is the final.

    Kept as data so the operator can correct slot constraints (e.g. the
    third-place chart) without touching simulation logic.
    """

    matches: tuple[KnockoutMatch, ...]


# Local aliases keep the 31-match literal below readable.
_W, _RU, _T, _M = group_winner, group_runner_up, best_third, winner_of

# 2026 bracket per the official FIFA match schedule (matches 73-104), as
# published on the FIFA site and Wikipedia's "2026 FIFA World Cup knockout
# stage" page (verified 2026-06-11). Match 103 (third-place playoff) is
# omitted — it never affects outright/futures probabilities. Third slots
# carry the admissible-groups constraint from the schedule; see the module
# docstring for how a concrete third is chosen within that constraint.
WC2026_BRACKET = Bracket(matches=(
    # Round of 32
    KnockoutMatch("M73", "R32", _RU("A"), _RU("B")),
    KnockoutMatch("M74", "R32", _W("E"), _T("A", "B", "C", "D", "F")),
    KnockoutMatch("M75", "R32", _W("F"), _RU("C")),
    KnockoutMatch("M76", "R32", _W("C"), _RU("F")),
    KnockoutMatch("M77", "R32", _W("I"), _T("C", "D", "F", "G", "H")),
    KnockoutMatch("M78", "R32", _RU("E"), _RU("I")),
    KnockoutMatch("M79", "R32", _W("A"), _T("C", "E", "F", "H", "I")),
    KnockoutMatch("M80", "R32", _W("L"), _T("E", "H", "I", "J", "K")),
    KnockoutMatch("M81", "R32", _W("D"), _T("B", "E", "F", "I", "J")),
    KnockoutMatch("M82", "R32", _W("G"), _T("A", "E", "H", "I", "J")),
    KnockoutMatch("M83", "R32", _RU("K"), _RU("L")),
    KnockoutMatch("M84", "R32", _W("H"), _RU("J")),
    KnockoutMatch("M85", "R32", _W("B"), _T("E", "F", "G", "I", "J")),
    KnockoutMatch("M86", "R32", _W("J"), _RU("H")),
    KnockoutMatch("M87", "R32", _W("K"), _T("D", "E", "I", "J", "L")),
    KnockoutMatch("M88", "R32", _RU("D"), _RU("G")),
    # Round of 16
    KnockoutMatch("M89", "R16", _M("M74"), _M("M77")),
    KnockoutMatch("M90", "R16", _M("M73"), _M("M75")),
    KnockoutMatch("M91", "R16", _M("M76"), _M("M78")),
    KnockoutMatch("M92", "R16", _M("M79"), _M("M80")),
    KnockoutMatch("M93", "R16", _M("M83"), _M("M84")),
    KnockoutMatch("M94", "R16", _M("M81"), _M("M82")),
    KnockoutMatch("M95", "R16", _M("M86"), _M("M88")),
    KnockoutMatch("M96", "R16", _M("M85"), _M("M87")),
    # Quarter-finals
    KnockoutMatch("M97", "QF", _M("M89"), _M("M90")),
    KnockoutMatch("M98", "QF", _M("M93"), _M("M94")),
    KnockoutMatch("M99", "QF", _M("M91"), _M("M92")),
    KnockoutMatch("M100", "QF", _M("M95"), _M("M96")),
    # Semi-finals
    KnockoutMatch("M101", "SF", _M("M97"), _M("M98")),
    KnockoutMatch("M102", "SF", _M("M99"), _M("M100")),
    # Final
    KnockoutMatch("M104", "F", _M("M101"), _M("M102")),
))


@dataclass(frozen=True)
class TeamProbs:
    """Per-team futures probabilities. Each field is P(reaching AT LEAST that
    stage); exit probabilities are differences, e.g. P(out in the R32) =
    ``r32 - r16``. ``win`` is the outright (champion) probability."""

    team: str
    win: float
    final: float
    semi: float
    quarter: float
    r16: float
    r32: float


def validate_bracket(bracket: Bracket, group_names: Set[str]) -> None:
    """Structural validation; raises ``ValueError`` on the first defect.

    Guarantees: unique match ids; every group's winner and runner-up consumed
    exactly once; third slots constrained to known groups and not more
    numerous than the groups; match references point only backwards; and
    every match's winner feeds exactly one later match except the final,
    which feeds none. Together: every advancing slot is consumed exactly
    once, so no simulated team can vanish or play twice in a round.
    """
    if not bracket.matches:
        raise ValueError("bracket has no matches")
    ids = [m.id for m in bracket.matches]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate knockout match ids")

    winner_refs: list[str] = []
    runner_refs: list[str] = []
    match_refs: list[str] = []
    n_third_slots = 0
    seen: set[str] = set()
    for m in bracket.matches:
        for slot in (m.a, m.b):
            if slot.kind in ("winner", "runner_up"):
                if slot.ref not in group_names:
                    raise ValueError(f"{m.id}: unknown group {slot.ref!r}")
                (winner_refs if slot.kind == "winner" else runner_refs).append(slot.ref)
            elif slot.kind == "third":
                if not slot.groups:
                    raise ValueError(f"{m.id}: third slot with no admissible groups")
                unknown = set(slot.groups) - set(group_names)
                if unknown:
                    raise ValueError(f"{m.id}: third slot names unknown groups {sorted(unknown)}")
                n_third_slots += 1
            elif slot.kind == "match":
                if slot.ref not in seen:
                    raise ValueError(f"{m.id}: forward/unknown match reference {slot.ref!r}")
                match_refs.append(slot.ref)
            else:
                raise ValueError(f"{m.id}: unknown slot kind {slot.kind!r}")
        seen.add(m.id)

    for label, refs in (("winner", winner_refs), ("runner-up", runner_refs)):
        counts = Counter(refs)
        bad = sorted(g for g in group_names if counts[g] != 1)
        if bad:
            raise ValueError(f"group {label} slots not consumed exactly once: {bad}")
    if n_third_slots > len(group_names):
        raise ValueError("more third-place slots than groups")

    feed = Counter(match_refs)
    final_id = ids[-1]
    for mid in ids:
        want = 0 if mid == final_id else 1
        if feed[mid] != want:
            raise ValueError(
                f"match {mid} winner consumed {feed[mid]} times (expected {want})"
            )


class _MatchSampler:
    """Caches the per-pairing cumulative scoreline distribution.

    Teams and params are fixed for a whole simulation run, so each distinct
    (home, away, home_field) pairing builds its DC score matrix exactly once;
    sampling is then a single rng draw + bisect. This is what makes 20k full
    tournaments tractable in pure Python.
    """

    def __init__(self, params: DCParams) -> None:
        self._params = params
        self._cache: dict[
            tuple[str, str, bool],
            tuple[list[float], list[tuple[int, int]], float, float],
        ] = {}

    def _entry(
        self, home: str, away: str, home_field: bool
    ) -> tuple[list[float], list[tuple[int, int]], float, float]:
        key = (home, away, home_field)
        entry = self._cache.get(key)
        if entry is None:
            lam, mu = expected_goals(self._params, home, away, home_field=home_field)
            grid = score_matrix(lam, mu, self._params.rho)
            cum: list[float] = []
            cells: list[tuple[int, int]] = []
            total = p_home = p_away = 0.0
            for x, row in enumerate(grid):
                for y, p in enumerate(row):
                    total += p
                    cum.append(total)
                    cells.append((x, y))
                    if x > y:
                        p_home += p
                    elif y > x:
                        p_away += p
            entry = (cum, cells, p_home, p_away)
            self._cache[key] = entry
        return entry

    def sample(
        self, home: str, away: str, rng: random.Random, *, home_field: bool = False
    ) -> tuple[int, int]:
        cum, cells, _, _ = self._entry(home, away, home_field)
        i = bisect_right(cum, rng.random() * cum[-1])
        return cells[min(i, len(cells) - 1)]

    def decisive_share(self, home: str, away: str) -> float:
        """P(home wins | the 90 minutes are decisive), on neutral ground —
        the 'skill' input to the extra-time/penalty model."""
        _, _, p_home, p_away = self._entry(home, away, False)
        decisive = p_home + p_away
        return 0.5 if decisive <= 0.0 else p_home / decisive


def _play_group(
    teams: Sequence[str],
    sampler: _MatchSampler,
    rng: random.Random,
    hosts: Set[str],
) -> tuple[list[str], dict[str, tuple[int, int, int]]]:
    """Round-robin; returns (ranked table, per-team (points, GD, GF)).

    When exactly one side is a host it plays 'at home' with the DC
    home-field boost (group-stage-only approximation — see module
    docstring); all other pairings are neutral and orientation-symmetric.
    """
    pts = dict.fromkeys(teams, 0)
    gd = dict.fromkeys(teams, 0)
    gf = dict.fromkeys(teams, 0)
    for a, b in itertools.combinations(teams, 2):
        if b in hosts and a not in hosts:
            a, b = b, a
        home_field = a in hosts and b not in hosts
        ga, gb = sampler.sample(a, b, rng, home_field=home_field)
        gf[a] += ga
        gf[b] += gb
        gd[a] += ga - gb
        gd[b] += gb - ga
        if ga > gb:
            pts[a] += 3
        elif gb > ga:
            pts[b] += 3
        else:
            pts[a] += 1
            pts[b] += 1
    table = sorted(
        teams, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True
    )
    return table, {t: (pts[t], gd[t], gf[t]) for t in teams}


def _assign_thirds(
    slots: Sequence[tuple[tuple[str, str], tuple[str, ...]]],
    ranked_groups: Sequence[str],
) -> dict[tuple[str, str], str]:
    """Deterministic constrained assignment of qualified thirds to slots.

    ``slots`` is ``[(slot_key, admissible_groups), ...]`` in bracket order;
    ``ranked_groups`` lists the qualified thirds' groups best-first. Each
    slot greedily takes the best-ranked unused admissible group, with
    backtracking so a complete assignment is found whenever one exists
    (FIFA's constraint chart admits one for every qualifying combination).
    """
    assignment: dict[tuple[str, str], str] = {}
    used: set[str] = set()

    def dfs(i: int) -> bool:
        if i == len(slots):
            return True
        key, allowed = slots[i]
        for g in ranked_groups:
            if g in used or g not in allowed:
                continue
            assignment[key] = g
            used.add(g)
            if dfs(i + 1):
                return True
            used.discard(g)
            del assignment[key]
        return False

    if not dfs(0):
        raise ValueError(
            f"no feasible third-place assignment for groups {sorted(ranked_groups)}"
        )
    return assignment


def simulate_tournament(
    groups: Mapping[str, Sequence[str]],
    params: DCParams,
    rng: random.Random,
    *,
    bracket: Bracket = WC2026_BRACKET,
    n_sims: int = 20_000,
    hosts: Set[str] = frozenset(),
    penalty_randomness: float = 0.5,
) -> dict[str, TeamProbs]:
    """Simulate the whole tournament ``n_sims`` times; per-team futures probs.

    ``groups`` maps group name -> team names (names must match
    ``params.teams`` keys; unknown teams get the neutral ``DCTeam()``
    prior, which the caller should warn about). ``hosts`` are team names
    that receive the group-stage home-field boost.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if not 0.0 <= penalty_randomness <= 1.0:
        raise ValueError("penalty_randomness must be in [0, 1]")
    validate_bracket(bracket, frozenset(groups))

    third_slots = [
        ((m.id, side), slot.groups)
        for m in bracket.matches
        for side, slot in (("a", m.a), ("b", m.b))
        if slot.kind == "third"
    ]
    if third_slots and any(len(ts) < 3 for ts in groups.values()):
        raise ValueError("bracket uses third-place slots but a group has <3 teams")

    sampler = _MatchSampler(params)
    reach: dict[str, Counter[str]] = {r: Counter() for r in ROUND_LABELS}
    champions: Counter[str] = Counter()
    final_id = bracket.matches[-1].id
    group_names = sorted(groups)

    for _ in range(n_sims):
        winners: dict[str, str] = {}
        runners: dict[str, str] = {}
        thirds: list[tuple[tuple[int, int, int], float, str, str]] = []
        for g in group_names:
            table, stats = _play_group(tuple(groups[g]), sampler, rng, hosts)
            winners[g] = table[0]
            runners[g] = table[1]
            if third_slots:
                t3 = table[2]
                thirds.append((stats[t3], rng.random(), g, t3))

        third_team: dict[tuple[str, str], str] = {}
        if third_slots:
            # Rank ALL thirds by points/GD/GF (rng breaks residual ties —
            # FIFA uses conduct + ranking there); top-N qualify.
            thirds.sort(reverse=True)
            ranked = [g for _, _, g, _ in thirds[: len(third_slots)]]
            team_of = {g: t for _, _, g, t in thirds}
            assignment = _assign_thirds(third_slots, ranked)
            third_team = {key: team_of[g] for key, g in assignment.items()}

        match_winner: dict[str, str] = {}
        for m in bracket.matches:
            sides = []
            for side, slot in (("a", m.a), ("b", m.b)):
                if slot.kind == "winner":
                    sides.append(winners[slot.ref])
                elif slot.kind == "runner_up":
                    sides.append(runners[slot.ref])
                elif slot.kind == "third":
                    sides.append(third_team[(m.id, side)])
                else:  # "match" — validated to point backwards
                    sides.append(match_winner[slot.ref])
            a, b = sides
            tally = reach.get(m.round)
            if tally is not None:
                tally[a] += 1
                tally[b] += 1
            ga, gb = sampler.sample(a, b, rng)  # knockout = neutral venue
            if ga > gb:
                won = a
            elif gb > ga:
                won = b
            else:
                p_a = (
                    penalty_randomness * 0.5
                    + (1.0 - penalty_randomness) * sampler.decisive_share(a, b)
                )
                won = a if rng.random() < p_a else b
            match_winner[m.id] = won
        champions[match_winner[final_id]] += 1

    n = float(n_sims)
    return {
        t: TeamProbs(
            team=t,
            win=champions[t] / n,
            final=reach["F"][t] / n,
            semi=reach["SF"][t] / n,
            quarter=reach["QF"][t] / n,
            r16=reach["R16"][t] / n,
            r32=reach["R32"][t] / n,
        )
        for ts in groups.values()
        for t in ts
    }
