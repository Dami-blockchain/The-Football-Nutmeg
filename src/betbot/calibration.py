"""Calibration scoring for the WC model race — pure, no IO.

Judges the model the way it should be judged (per CLAUDE.md A.8): not by how
many favourites won, but by whether its probabilities are *honest* — when it
says 70%, does the outcome happen ~70% of the time? That's calibration, and it
needs many matches and a reliability breakdown, not a single result.

Consumes the scored rows from the ``model_predictions`` table (each carries
both the pure-Glicko and the ensemble probability triple plus the realised
outcome) and produces, per model: RPS, Brier, log-loss, favourite hit-rate,
and a reliability table bucketing the favourite's confidence against how often
that favourite actually won.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from betbot.strategy.ensemble import ranked_probability_score

# Reliability buckets over the FAVOURITE's predicted probability. A 3-way
# favourite is always >= 1/3, so the lowest meaningful band starts at 0.40.
_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01),
)


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_predicted: float   # avg predicted prob of the favourite in this band
    hit_rate: float         # how often that favourite actually won

    @property
    def gap(self) -> float:
        """Predicted minus observed. Positive = over-confident."""
        return self.mean_predicted - self.hit_rate


@dataclass(frozen=True)
class ModelCalibration:
    name: str
    n: int
    rps: float
    brier: float
    log_loss: float
    favourite_hit_rate: float
    bins: tuple[ReliabilityBin, ...]


def _brier(probs: tuple[float, float, float], oi: int) -> float:
    return sum((probs[i] - (1.0 if i == oi else 0.0)) ** 2 for i in range(3))


def _log_loss(probs: tuple[float, float, float], oi: int) -> float:
    return -math.log(max(probs[oi], 1e-12))


def _reliability(
    preds: list[tuple[tuple[float, float, float], int]],
) -> tuple[ReliabilityBin, ...]:
    bins: list[ReliabilityBin] = []
    for lo, hi in _BANDS:
        members = []
        for probs, oi in preds:
            fav = max(range(3), key=lambda i: probs[i])
            if lo <= probs[fav] < hi:
                members.append((probs[fav], 1.0 if fav == oi else 0.0))
        if not members:
            continue
        n = len(members)
        bins.append(ReliabilityBin(
            lo=lo, hi=hi, n=n,
            mean_predicted=sum(p for p, _ in members) / n,
            hit_rate=sum(h for _, h in members) / n,
        ))
    return tuple(bins)


def calibrate_model(
    name: str, preds: list[tuple[tuple[float, float, float], int]]
) -> ModelCalibration:
    """One model's calibration over its scored predictions.

    ``preds`` is ``[(probs_triple, outcome_index), ...]`` where outcome_index
    is 0=home, 1=draw, 2=away.
    """
    n = len(preds)
    if n == 0:
        return ModelCalibration(name, 0, 0.0, 0.0, 0.0, 0.0, ())
    rps = sum(ranked_probability_score(p, oi) for p, oi in preds) / n
    brier = sum(_brier(p, oi) for p, oi in preds) / n
    ll = sum(_log_loss(p, oi) for p, oi in preds) / n
    hit = sum(
        1.0 for p, oi in preds if max(range(3), key=lambda i: p[i]) == oi
    ) / n
    return ModelCalibration(
        name=name, n=n, rps=rps, brier=brier, log_loss=ll,
        favourite_hit_rate=hit, bins=_reliability(preds),
    )


@dataclass(frozen=True)
class CalibrationReport:
    glicko: ModelCalibration
    ensemble: ModelCalibration


def build_report(
    rows: list[tuple[tuple[float, float, float], tuple[float, float, float], int]],
) -> CalibrationReport:
    """From scored model_predictions rows ``[(glicko, ensemble, oi), ...]``."""
    return CalibrationReport(
        glicko=calibrate_model("glicko", [(g, oi) for g, _e, oi in rows]),
        ensemble=calibrate_model("ensemble", [(e, oi) for _g, e, oi in rows]),
    )
