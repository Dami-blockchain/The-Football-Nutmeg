"""Dixon-Coles goal model — ensemble Layer 2 (scoreline distributions).

Dixon & Coles (1997): each team gets an attack and a defence strength; the
two sides' goal counts are Poisson with a small correction (``rho``) for the
dependence the independent-Poisson model gets wrong in low-scoring games
(0-0, 1-0, 0-1, 1-1 — i.e. most football). Compared with Glicko, which only
sees win/draw/loss, this extracts goal-level signal from every match and
natively prices draws and scorelines.

Fitting is weighted maximum likelihood: exponential time decay (old matches
fade), friendlies downweighted, and L2 shrinkage of each team's parameters
toward priors derived from the Klement fundamentals layer — that anchoring is
what keeps sparse-data national teams sane. The Poisson parameters are fit by
full-batch gradient ascent (gradients are analytic and trivial); ``rho`` is
then a 1-D grid search on the full DC likelihood, the standard two-stage
approach.

Pure math — no DB, no network. The data pull + fit orchestration lives in
scripts/fit_dixon_coles.py, which persists a JSON params artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date

MAX_GOALS = 10  # score-matrix truncation; P(>10 goals) is negligible


@dataclass(frozen=True)
class DCTeam:
    attack: float = 0.0
    defence: float = 0.0


@dataclass(frozen=True)
class DCMatch:
    """One observed result. ``neutral`` kills the home-advantage term
    (almost all WC matches); ``friendly`` downweights the observation."""

    date: date
    home: str
    away: str
    home_goals: int
    away_goals: int
    neutral: bool = True
    friendly: bool = False


@dataclass
class DCParams:
    teams: dict[str, DCTeam] = field(default_factory=dict)
    base_mu: float = 0.1      # log of baseline goals per side
    home_adv: float = 0.25    # log-scale boost for a genuine home side
    rho: float = -0.10        # low-score dependence correction

    def team(self, name: str) -> DCTeam:
        return self.teams.get(name, DCTeam())

    def to_json(self) -> str:
        return json.dumps({
            "base_mu": self.base_mu, "home_adv": self.home_adv, "rho": self.rho,
            "teams": {k: [t.attack, t.defence] for k, t in sorted(self.teams.items())},
        }, indent=1)

    @classmethod
    def from_json(cls, raw: str) -> DCParams:
        d = json.loads(raw)
        return cls(
            teams={k: DCTeam(a, df) for k, (a, df) in d["teams"].items()},
            base_mu=d["base_mu"], home_adv=d["home_adv"], rho=d["rho"],
        )


def expected_goals(
    params: DCParams, home: str, away: str, *, home_field: bool = False
) -> tuple[float, float]:
    """(lambda_home, mu_away) — the two Poisson means."""
    h, a = params.team(home), params.team(away)
    lam = math.exp(params.base_mu + h.attack - a.defence
                   + (params.home_adv if home_field else 0.0))
    mu = math.exp(params.base_mu + a.attack - h.defence)
    return lam, mu


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correction; 1.0 outside the 2x2 corner."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS
) -> list[list[float]]:
    """P(home=x, away=y) grid, renormalised over the truncation."""
    px = [math.exp(-lam) * lam**x / math.factorial(x) for x in range(max_goals + 1)]
    py = [math.exp(-mu) * mu**y / math.factorial(y) for y in range(max_goals + 1)]
    grid = [
        [max(px[x] * py[y] * _tau(x, y, lam, mu, rho), 0.0)
         for y in range(max_goals + 1)]
        for x in range(max_goals + 1)
    ]
    total = sum(sum(row) for row in grid)
    return [[p / total for p in row] for row in grid]


def match_probabilities(
    params: DCParams, home: str, away: str, *, home_field: bool = False
) -> tuple[float, float, float]:
    """(p_home, p_draw, p_away) — same shape as glicko.match_probabilities."""
    lam, mu = expected_goals(params, home, away, home_field=home_field)
    grid = score_matrix(lam, mu, params.rho)
    p_home = sum(grid[x][y] for x in range(len(grid)) for y in range(x))
    p_draw = sum(grid[x][x] for x in range(len(grid)))
    return p_home, p_draw, 1.0 - p_home - p_draw


def decay_weight(match_date: date, ref_date: date, half_life_days: float) -> float:
    """Exponential time decay: 1.0 today, 0.5 one half-life ago."""
    days = max((ref_date - match_date).days, 0)
    return 0.5 ** (days / half_life_days)


def fit(
    matches: list[DCMatch],
    *,
    priors: dict[str, DCTeam] | None = None,
    prior_weight: float = 5.0,
    half_life_days: float = 540.0,
    friendly_weight: float = 0.5,
    iterations: int = 200,
    learning_rate: float = 0.1,
) -> DCParams:
    """Weighted MLE. ``priors`` (e.g. from the fundamentals layer via
    :func:`priors_from_scores`) anchor each team via L2 shrinkage with
    strength ``prior_weight`` — in effective-match units, roughly: a team
    with far fewer weighted matches than ``prior_weight`` stays near its
    prior; one with far more lets the data speak.
    """
    if not matches:
        return DCParams(teams=dict(priors or {}))

    priors = priors or {}
    ref = max(m.date for m in matches)
    weighted = [
        (m, decay_weight(m.date, ref, half_life_days)
            * (friendly_weight if m.friendly else 1.0))
        for m in matches
    ]
    names = sorted({t for m in matches for t in (m.home, m.away)} | set(priors))
    atk = {n: priors.get(n, DCTeam()).attack for n in names}
    dfn = {n: priors.get(n, DCTeam()).defence for n in names}
    base, home_adv = 0.1, 0.25

    # Per-team observation weight (+prior) preconditions the gradient steps:
    # a step is then roughly an average residual, stable for any match count.
    w_tot = {n: prior_weight for n in names}
    for m, w in weighted:
        w_tot[m.home] += w
        w_tot[m.away] += w
    w_all = sum(w for _, w in weighted)
    w_home = sum(w for m, w in weighted if not m.neutral) or 1.0

    for _ in range(iterations):
        g_atk = {n: 0.0 for n in names}
        g_dfn = {n: 0.0 for n in names}
        g_base = g_home = 0.0
        for m, w in weighted:
            ha = 0.0 if m.neutral else home_adv
            lam = math.exp(base + atk[m.home] - dfn[m.away] + ha)
            mu = math.exp(base + atk[m.away] - dfn[m.home])
            rh = w * (m.home_goals - lam)   # d(loglik)/d(log lam)
            ra = w * (m.away_goals - mu)
            g_atk[m.home] += rh
            g_dfn[m.away] -= rh
            g_atk[m.away] += ra
            g_dfn[m.home] -= ra
            g_base += rh + ra
            if not m.neutral:
                g_home += rh
        # L2 shrinkage toward the fundamentals priors.
        for n in names:
            p = priors.get(n, DCTeam())
            g_atk[n] -= prior_weight * (atk[n] - p.attack)
            g_dfn[n] -= prior_weight * (dfn[n] - p.defence)
        for n in names:
            atk[n] += learning_rate * g_atk[n] / w_tot[n]
            dfn[n] += learning_rate * g_dfn[n] / w_tot[n]
        base += learning_rate * g_base / (2.0 * w_all)
        home_adv += learning_rate * g_home / w_home

    params = DCParams(
        teams={n: DCTeam(atk[n], dfn[n]) for n in names},
        base_mu=base, home_adv=home_adv, rho=0.0,
    )
    params.rho = _fit_rho(params, weighted)
    return params


def _fit_rho(params: DCParams, weighted: list[tuple[DCMatch, float]]) -> float:
    """Grid-search rho on the full DC log-likelihood, Poisson params fixed.
    Only the low-score corner contributes, so this is cheap and stable."""
    best_rho, best_ll = 0.0, -math.inf
    for step in range(-40, 41):
        rho = step / 100.0
        ll = 0.0
        ok = True
        for m, w in weighted:
            if m.home_goals > 1 or m.away_goals > 1:
                continue
            ha = 0.0 if m.neutral else params.home_adv
            lam = math.exp(params.base_mu + params.team(m.home).attack
                           - params.team(m.away).defence + ha)
            mu = math.exp(params.base_mu + params.team(m.away).attack
                          - params.team(m.home).defence)
            t = _tau(m.home_goals, m.away_goals, lam, mu, rho)
            if t <= 0.0:
                ok = False
                break
            ll += w * math.log(t)
        if ok and ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho


def priors_from_scores(
    composite_scores: dict[str, float], *, attack_scale: float = 0.18,
    defence_scale: float = 0.18,
) -> dict[str, DCTeam]:
    """Map fundamentals composite z-scores to DC priors: a one-sigma-better
    team scores ~exp(0.18) ≈ 20% more and concedes ~20% fewer goals."""
    return {
        name: DCTeam(attack=z * attack_scale, defence=z * defence_scale)
        for name, z in composite_scores.items()
    }
