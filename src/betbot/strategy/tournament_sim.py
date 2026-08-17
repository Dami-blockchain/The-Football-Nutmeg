"""Tournament (single-elimination bracket) Monte Carlo — pure, no I/O.

The season-title sim (:mod:`betbot.strategy.season_sim`) answers "who wins a
round-robin league?"; a knockout cup like the Champions League final stages is a
*bracket*, not a points table, so it needs a different simulator. This one is
that: given a seeded list of entrants and an injected ``advance_prob_fn(a, b) ->
P(a advances past b)``, it plays the bracket ``n_sims`` times and tallies each
entrant's P(winning the whole thing).

Design mirrors ``season_sim``:

* **engine-agnostic** — the caller injects ``advance_prob_fn`` (built from
  ClubElo in :mod:`betbot.cl_service`), so the same code serves the live CL
  projection, a backtest, or a unit test with hand-made win probs;
* **pure + deterministic** given a ``seed`` (a single :class:`random.Random`),
  so identical inputs always yield identical numbers — the tests pin this;
* **fast** — pairwise advance probabilities are memoised across sims (they do
  not change), so each sim is just ``log2(N)`` rounds of one ``random()`` draw
  per surviving tie. 10k sims of a 32-team bracket is well under a second.

Bracket shape: a standard single-elimination bracket needs a power-of-two field.
For a non-power-of-two entrant count we pad up to the next power of two with
``None`` **byes**: a real team drawn against a bye advances for free (no draw
consumed). Byes are handed to the *top seeds* (entrants earliest in the list),
which is how real seeded competitions grant them. The initial pairing is the
classic seed bracket (1 vs N, 2 vs N-1, ... within each half) so a stronger seed
meets weaker opposition early — this only shapes the *path*, not the pairwise
model, and keeps runs reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

AdvanceProbFn = Callable[[str, str], float]


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _seed_order(size: int) -> list[int]:
    """Standard single-elimination seed slots for a bracket of ``size`` (a power
    of two). Returns 0-based seed indices so that slot 0 holds seed 0 (top),
    the final would be seed 0 vs seed 1, and strong seeds are kept apart.
    """
    order = [0]
    while len(order) < size:
        nxt = []
        pair_sum = len(order) * 2 - 1
        for s in order:
            nxt.append(s)
            nxt.append(pair_sum - s)
        order = nxt
    return order


def _build_bracket(entrants: Sequence[str], byes: int) -> list[str | None]:
    """Lay entrants into power-of-two slots by seed, padding with ``None`` byes.

    ``entrants`` is assumed pre-sorted best-first (index 0 = top seed). The top
    ``byes`` seeds are guaranteed a first-round bye by placing the ``None`` pads
    opposite them in the classic seed bracket.
    """
    n = len(entrants)
    size = _next_pow2(n)
    # Slots indexed by seed rank; ranks >= n are byes (None).
    ranked: list[str | None] = list(entrants) + [None] * (size - n)
    slots = _seed_order(size)
    return [ranked[s] for s in slots]


def simulate_knockout(
    *,
    entrants: Sequence[str],
    advance_prob_fn: AdvanceProbFn,
    n_sims: int = 10000,
    seed: int = 20260817,
    byes: int | None = None,
) -> dict[str, float]:
    """Monte-Carlo a single-elimination bracket to each entrant's P(win).

    ``advance_prob_fn(a, b)`` returns P(a beats b and advances); it is called
    at most once per unordered pair (memoised) so a heavy pricer is cheap.
    ``byes`` (default: exactly enough to fill the next power of two) go to the
    top seeds. ``entrants`` should be ordered best-first for the seeding to
    grant byes and initial pairings to the stronger sides.

    Returns ``{team: p_win}`` for every real entrant; probabilities sum to ~1.
    """
    real = [e for e in entrants if e]
    if not real:
        return {}
    if len(real) == 1:
        return {real[0]: 1.0}

    size = _next_pow2(len(real))
    needed_byes = size - len(real)
    if byes is not None and byes != needed_byes:
        # Caller-supplied byes only affects documentation of intent; the actual
        # pad count is fixed by the field size. We honour the field size.
        pass

    bracket = _build_bracket(real, needed_byes)

    # Memoise pairwise advance probabilities (symmetric key), computed lazily.
    cache: dict[tuple[str, str], float] = {}

    def p_adv(a: str, b: str) -> float:
        key = (a, b)
        val = cache.get(key)
        if val is None:
            val = advance_prob_fn(a, b)
            cache[key] = val
            # store the complementary orientation too
            cache[(b, a)] = 1.0 - val
        return val

    rng = random.Random(seed)
    rand = rng.random
    wins = {t: 0 for t in real}

    for _ in range(n_sims):
        alive: list[str | None] = list(bracket)
        while len(alive) > 1:
            nxt: list[str | None] = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                if a is None:
                    nxt.append(b)
                elif b is None:
                    nxt.append(a)
                else:
                    nxt.append(a if rand() < p_adv(a, b) else b)
            alive = nxt
        champ = alive[0]
        if champ is not None:
            wins[champ] += 1

    return {t: wins[t] / n_sims for t in real}
