"""Ensemble blend + calibration — Layer 3 (the model that actually trades).

Combines per-match 1X2 probabilities from the Glicko engine, the Dixon-Coles
goal model, and (when available) the de-vigged market line via logarithmic
pooling — a weighted geometric mean, the standard way to combine calibrated
forecasters (an over-confident component is damped rather than averaged in).
The Klement fundamentals layer enters indirectly: it seeds Glicko ratings and
anchors the DC priors, so it is NOT pooled again here.

Market weight deserves emphasis: closing lines are the strongest single
football forecaster in the literature, which is why the default weights lean
on the market and the model components act as a disagreement detector —
exactly the framing in CLAUDE.md (the edge, if any, is where the blend and
the venue price diverge, not in out-predicting Pinnacle).

Calibration is pool-adjacent-violators isotonic regression per outcome
(then renormalised), fit offline by the backtest script and shipped as a
JSON artifact next to the DC params.

Pure math — no DB, no network, no new dependencies.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass

_EPS = 1e-6

Probs = tuple[float, float, float]  # (p_home, p_draw, p_away)


@dataclass(frozen=True)
class EnsembleWeights:
    """Relative log-pool weights (normalised at use). The market lean is
    deliberate — see module docstring. Tuned by scripts/backtest_ensemble.py."""

    glicko: float = 1.0
    dixon_coles: float = 1.5
    market: float = 2.5


def _clip(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def log_pool(components: list[tuple[float, Probs]]) -> Probs:
    """Weighted geometric pool of probability triples, renormalised.

    ``components`` is ``[(weight, (ph, pd, pa)), ...]``; zero/negative
    weights are skipped, so callers can pass a 0-weight market when no
    price exists yet.
    """
    live = [(w, p) for w, p in components if w > 0.0]
    if not live:
        return (1 / 3, 1 / 3, 1 / 3)
    total_w = sum(w for w, _ in live)
    pooled = []
    for i in range(3):
        log_p = sum(w * math.log(_clip(p[i])) for w, p in live) / total_w
        pooled.append(math.exp(log_p))
    s = sum(pooled)
    return (pooled[0] / s, pooled[1] / s, pooled[2] / s)


def de_vig(prices: list[float]) -> list[float]:
    """Strip the overround: clip prices to (0,1), renormalise to sum 1."""
    clipped = [_clip(p) for p in prices]
    s = sum(clipped)
    return [p / s for p in clipped]


def blend(
    glicko: Probs,
    dixon_coles: Probs,
    market: Probs | None = None,
    weights: EnsembleWeights | None = None,
) -> Probs:
    """The ensemble's 1X2 probability for one match."""
    w = weights or EnsembleWeights()
    components = [(w.glicko, glicko), (w.dixon_coles, dixon_coles)]
    if market is not None:
        components.append((w.market, market))
    return log_pool(components)


def ranked_probability_score(probs: Probs, outcome_index: int) -> float:
    """RPS over the ordered 1X2 outcomes (0=home, 1=draw, 2=away). Lower is
    better; unlike accuracy it rewards putting mass NEAR the truth, which is
    the right yardstick for three-way football forecasts."""
    cum_p = cum_o = total = 0.0
    for i in range(2):
        cum_p += probs[i]
        cum_o += 1.0 if i == outcome_index else 0.0
        total += (cum_p - cum_o) ** 2
    return total / 2.0


def anchor_to_market(
    p_model: float, p_market: float, w_model: float, w_market: float
) -> float:
    """Blend a single-outcome model probability toward the market price in
    logit space (the binary-marginal version of :func:`log_pool`).

    This deliberately SHRINKS edges: with equal weights, half the logit
    distance to the market disappears, so only disagreements large enough to
    survive the shrink clear the edge threshold — the bot bets selectively
    where the model and the venue genuinely diverge.
    """
    total = w_model + w_market
    if total <= 0 or not 0.0 < p_market < 1.0:
        return p_model
    pm, pk = _clip(p_model), _clip(p_market)
    z = (
        w_model * math.log(pm / (1.0 - pm)) + w_market * math.log(pk / (1.0 - pk))
    ) / total
    return 1.0 / (1.0 + math.exp(-z))


class IsotonicCalibrator:
    """Pool-adjacent-violators isotonic regression: monotone map from
    predicted probability to observed frequency. Step function with
    interpolation at block midpoints; identity when never fitted."""

    def __init__(self, xs: list[float] | None = None, ys: list[float] | None = None):
        self._xs = xs or []
        self._ys = ys or []

    def fit(self, predicted: list[float], observed: list[float]) -> "IsotonicCalibrator":
        """``observed`` is 1.0/0.0 outcome indicators (or frequencies)."""
        if len(predicted) != len(observed) or not predicted:
            raise ValueError("predicted and observed must be equal non-zero length")
        pairs = sorted(zip(predicted, observed))
        # PAV: merge adjacent blocks until block means are non-decreasing.
        blocks: list[list[float]] = []  # [sum_y, count, sum_x]
        for x, y in pairs:
            blocks.append([y, 1.0, x])
            while len(blocks) > 1 and (
                blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]
            ):
                b = blocks.pop()
                blocks[-1][0] += b[0]
                blocks[-1][1] += b[1]
                blocks[-1][2] += b[2]
        self._xs = [b[2] / b[1] for b in blocks]  # block mean prediction
        self._ys = [b[0] / b[1] for b in blocks]  # block mean outcome
        return self

    def transform(self, p: float) -> float:
        if not self._xs:
            return p
        if p <= self._xs[0]:
            return self._ys[0]
        if p >= self._xs[-1]:
            return self._ys[-1]
        i = bisect.bisect_right(self._xs, p)
        x0, x1 = self._xs[i - 1], self._xs[i]
        y0, y1 = self._ys[i - 1], self._ys[i]
        t = (p - x0) / (x1 - x0) if x1 > x0 else 0.0
        return y0 + t * (y1 - y0)

    def to_json(self) -> str:
        return json.dumps({"xs": self._xs, "ys": self._ys})

    @classmethod
    def from_json(cls, raw: str) -> "IsotonicCalibrator":
        d = json.loads(raw)
        return cls(xs=d["xs"], ys=d["ys"])


def calibrate(probs: Probs, calibrators: tuple[IsotonicCalibrator, ...] | None) -> Probs:
    """Apply per-outcome calibrators and renormalise. ``None`` = identity."""
    if not calibrators:
        return probs
    mapped = [_clip(c.transform(p)) for c, p in zip(calibrators, probs)]
    s = sum(mapped)
    return (mapped[0] / s, mapped[1] / s, mapped[2] / s)
