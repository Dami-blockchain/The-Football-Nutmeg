"""Glicko-2 rating engine (Appendix A / Phase 5.5).

Pure math — no DB, no network. Produces a continuous strength estimate
(rating) plus an uncertainty (rating deviation, RD) and a volatility per team,
which is the right tool for international football where teams play rarely and
"recent form" is meaningless. The implementation follows Glickman (2012); the
volatility update uses the Illinois variant exactly as in §5.1 step 5 — that's
where naive implementations diverge, so it's pinned by a worked-example test.

Used only for INTERNATIONAL_COMPETITIONS (World Cup), paper-mode only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SCALE = 173.7178  # Glicko-2 internal-scale factor
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
DEFAULT_TAU = 0.5


@dataclass(frozen=True)
class Glicko2Rating:
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    volatility: float = DEFAULT_VOL
    last_period: str | None = None


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _new_volatility(sigma: float, delta: float, phi: float, v: float,
                    tau: float, eps: float = 1e-6) -> float:
    """Illinois-algorithm solution for the new volatility (Glickman §5.1.5)."""
    a = math.log(sigma * sigma)
    d2 = delta * delta
    phi2 = phi * phi

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (d2 - phi2 - v - ex)
        den = 2.0 * (phi2 + v + ex) ** 2
        return num / den - (x - a) / (tau * tau)

    A = a
    if d2 > phi2 + v:
        B = math.log(d2 - phi2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA, fB = f(A), f(B)
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC
    return math.exp(A / 2.0)


def update_rating(
    rating: Glicko2Rating,
    results: list[tuple[float, float, float]],
    *,
    tau: float = DEFAULT_TAU,
    period: str | None = None,
) -> Glicko2Rating:
    """One rating-period update.

    ``results`` is a list of ``(opponent_rating, opponent_rd, score)`` where
    score is 1.0 win / 0.5 draw / 0.0 loss. With no results, only RD grows.
    """
    mu = (rating.rating - 1500.0) / SCALE
    phi = rating.rd / SCALE
    sigma = rating.volatility

    if not results:
        phi_star = math.sqrt(phi * phi + sigma * sigma)
        return Glicko2Rating(rating.rating, phi_star * SCALE, sigma, period or rating.last_period)

    mus = [(r - 1500.0) / SCALE for r, _, _ in results]
    phis = [rd / SCALE for _, rd, _ in results]
    scores = [s for _, _, s in results]

    v_inv = sum(_g(pj) ** 2 * _E(mu, mj, pj) * (1.0 - _E(mu, mj, pj))
                for mj, pj in zip(mus, phis))
    v = 1.0 / v_inv
    delta = v * sum(_g(pj) * (s - _E(mu, mj, pj))
                    for mj, pj, s in zip(mus, phis, scores))

    sigma_p = _new_volatility(sigma, delta, phi, v, tau)
    phi_star = math.sqrt(phi * phi + sigma_p * sigma_p)
    phi_p = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_p = mu + phi_p * phi_p * sum(_g(pj) * (s - _E(mu, mj, pj))
                                    for mj, pj, s in zip(mus, phis, scores))

    return Glicko2Rating(
        rating=mu_p * SCALE + 1500.0,
        rd=phi_p * SCALE,
        volatility=sigma_p,
        last_period=period or rating.last_period,
    )


# Draw model floor/cap. The floor is deliberately high (12%, not 5%):
# empirically even lopsided international matches draw far more than 5% — the
# underdog parks the bus and nicks a point — so a big favourite must never be
# assigned an unrealistically tiny draw chance (the Canada-host-draws-minnow
# miss). draw_rho ~0.30 targets the ~25% draw rate of major-tournament FINALS
# (World Cup group games are cagey), above the 22% all-international average.
DRAW_FLOOR = 0.12
DRAW_CAP = 0.40


def match_probabilities(
    home: Glicko2Rating,
    away: Glicko2Rating,
    *,
    home_field_mu: float = 0.0,
    draw_rho: float = 0.30,
) -> tuple[float, float, float]:
    """(p_home, p_draw, p_away) from two ratings.

    The draw split is a heuristic (not derived from Glicko) but calibrated to
    the empirical international draw rate (~22% all matches, ~25% major-finals;
    measured over 8k matches 2018+). ``home_field_mu`` is applied only for
    genuine host nations (caller's responsibility); default 0 (neutral venue).
    """
    mu_h = (home.rating - 1500.0) / SCALE
    mu_a = (away.rating - 1500.0) / SCALE
    phi_h = home.rd / SCALE
    phi_a = away.rd / SCALE

    g_comb = 1.0 / math.sqrt(1.0 + 3.0 * (phi_h * phi_h + phi_a * phi_a) / (math.pi * math.pi))
    p_home_raw = 1.0 / (1.0 + math.exp(-g_comb * (mu_h - mu_a + home_field_mu)))

    p_draw = draw_rho * (1.0 - abs(p_home_raw - 0.5) * 2.0)
    p_draw = min(DRAW_CAP, max(DRAW_FLOOR, p_draw))
    p_home = (1.0 - p_draw) * p_home_raw
    p_away = (1.0 - p_draw) * (1.0 - p_home_raw)
    return p_home, p_draw, p_away
