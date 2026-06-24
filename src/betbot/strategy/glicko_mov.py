"""Margin-of-victory Glicko-2 update (flag-gated challenger).

Standard ``update_rating`` uses only W/D/L (score 1/0.5/0) and throws away the
goal margin. Opta's Power Rankings instead scale each rating move by goal
difference with *diminishing returns* — a 5-0 counts more than a 1-0, but not
five times more. This adds the same idea to Glicko-2: each match's mean-update
residual is multiplied by a margin multiplier, while the variance/volatility (RD)
machinery is left untouched, so only the *magnitude* of the rating change
responds to the scoreline.

``mov_multiplier(gd)``: 1.0 for ``gd <= 1`` (so a 1-goal win or a draw reduces
EXACTLY to standard Glicko-2), then ``1 + ln(gd)`` with diminishing returns
(gd=2 -> 1.69, 3 -> 2.10, 5 -> 2.61). With every multiplier == 1 the result is
identical to ``update_rating`` — pinned by a test.
"""

from __future__ import annotations

import math

from betbot.strategy.glicko import (
    DEFAULT_TAU,
    SCALE,
    Glicko2Rating,
    _E,
    _g,
    _new_volatility,
)


def mov_multiplier(goal_diff: int) -> float:
    """Diminishing-returns margin weight; 1.0 at gd<=1, 1+ln(gd) above."""
    gd = abs(int(goal_diff))
    return 1.0 if gd <= 1 else 1.0 + math.log(gd)


def update_rating_mov(
    rating: Glicko2Rating,
    results: list[tuple[float, float, float, int]],
    *,
    tau: float = DEFAULT_TAU,
    period: str | None = None,
) -> Glicko2Rating:
    """One MOV-aware rating period.

    ``results`` is ``(opponent_rating, opponent_rd, score, goal_diff)`` where
    score is 1.0/0.5/0.0 and goal_diff is the absolute margin of that match. The
    margin multiplier scales only the mean-update residual; the variance ``v``
    and volatility update keep standard Glicko-2 semantics.
    """
    mu = (rating.rating - 1500.0) / SCALE
    phi = rating.rd / SCALE
    sigma = rating.volatility

    if not results:
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        return Glicko2Rating(rating.rating, phi_star * SCALE, sigma, period or rating.last_period)

    mus = [(r - 1500.0) / SCALE for r, _, _, _ in results]
    phis = [rd / SCALE for _, rd, _, _ in results]
    scores = [s for _, _, s, _ in results]
    movs = [mov_multiplier(gd) for _, _, _, gd in results]

    v_inv = sum(_g(pj) ** 2 * _E(mu, mj, pj) * (1.0 - _E(mu, mj, pj))
                for mj, pj in zip(mus, phis))
    v = 1.0 / v_inv
    # MOV-weighted residual drives both the volatility delta and the mean update.
    resid = sum(_g(pj) * mv * (s - _E(mu, mj, pj))
                for mj, pj, s, mv in zip(mus, phis, scores, movs))
    delta = v * resid

    sigma_p = _new_volatility(sigma, delta, phi, v, tau)
    phi_star = math.sqrt(phi * phi + sigma_p * sigma_p)
    phi_p = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_p = mu + phi_p * phi_p * resid

    return Glicko2Rating(
        rating=mu_p * SCALE + 1500.0,
        rd=phi_p * SCALE,
        volatility=sigma_p,
        last_period=period or rating.last_period,
    )
