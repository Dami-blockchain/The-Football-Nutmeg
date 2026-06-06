"""Pure-math helpers — no IO, no settings dependency.

Importable in any environment, even without the rest of the bot's deps.
"""

from __future__ import annotations

import math


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically-stable softmax.

    The max-subtraction trick keeps overflow at bay even when scores are
    in the thousands — handy when callers pass already-scaled values.
    """
    if not scores:
        return []
    t = max(temperature, 1e-9)
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    if total <= 0:
        # Degenerate case — return uniform.
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


def opponent_strength_factor(
    opponent_position: float | None,
    league_size: int,
    weight: float,
) -> float:
    """Scale an opponent's threat by their current league position.

    Higher league position (smaller number) = stronger opponent = higher
    factor. Position 1 in a 20-team league with weight 0.5 yields 1.5;
    last place yields 0.5; missing data yields 1.0.
    """
    if opponent_position is None or league_size <= 1:
        return 1.0
    # Normalise to [0, 1] where 0 = top, 1 = bottom.
    norm = (opponent_position - 1) / (league_size - 1)
    norm = max(0.0, min(1.0, norm))
    # Convert to a scaling factor centred at 1.0:
    #   top (norm=0) -> 1 + weight
    #   bottom (norm=1) -> 1 - weight
    return 1.0 + weight * (1.0 - 2.0 * norm)


def edge(our_probability: float, market_price: float) -> float:
    """Edge in probability units. Positive = we think it's underpriced."""
    return our_probability - market_price


def implied_probability(market_price: float) -> float:
    """Clip a market price into [0, 1] before treating it as a probability."""
    return max(0.0, min(1.0, market_price))
