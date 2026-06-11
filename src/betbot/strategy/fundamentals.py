"""Klement 5-factor fundamentals — ensemble Layer 1 (structural priors).

Joachim Klement's World Cup model (Panmure Liberum; correct outright winner
2014/2018/2022) explains ~55% of tournament success with five slow-moving
covariates: GDP per capita (diminishing returns past ~$60k), population,
distance of the country's average temperature from a ~14C football optimum,
FIFA ranking points, and host advantage. His fitted coefficients are not
public, so this module reproduces the *structure*: transform each covariate,
z-score it within the tournament cohort, and combine with documented weights
into a composite that maps onto the Glicko rating scale as a prior.

Its job is regularisation for sparse-data teams — a structurally sensible
starting strength where match history is thin — NOT match-level edge
(Klement himself: anyone betting on his prediction alone is "beyond help").

Pure math + CSV loading. No DB, no network — the data pull lives in
scripts/build_fundamentals.py, which writes data/fundamentals_2026.csv.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from betbot.exchanges.matcher import normalize

# Klement's documented non-linearities.
GDP_SATURATION_USD = 60_000.0  # past this, more wealth stops helping
TEMP_OPTIMUM_C = 14.0          # ideal climate for football development

# How far one cohort standard deviation moves the Glicko prior. 120 points
# keeps the whole cohort within roughly +-300 of the base — informative for
# seeding, but easily overridden by actual results (RD starts high).
DEFAULT_PRIOR_SPREAD = 120.0
DEFAULT_PRIOR_BASE = 1500.0


@dataclass(frozen=True)
class TeamFundamentals:
    team: str
    iso3: str
    gdp_per_capita_usd: float
    population: int
    avg_temp_c: float
    fifa_points: float
    host: bool = False


@dataclass(frozen=True)
class FactorWeights:
    """Relative factor weights (normalised at use). FIFA points dominate —
    they are the only sporting input; the socioeconomic factors refine.

    ``host`` defaults to 0 because host advantage is already applied
    per-match by the engine (``glicko_host_home_mu``); weighting it here too
    would double-count. Raise it only if that per-match bump is disabled.
    """

    fifa: float = 0.55
    gdp: float = 0.15
    population: float = 0.15
    temperature: float = 0.15
    host: float = 0.0


def gdp_factor(gdp_per_capita_usd: float) -> float:
    """Log wealth, capped at the saturation point (rich-country kids defect
    to other pastimes — Klement). Monotone up to $60k, flat beyond."""
    capped = min(max(gdp_per_capita_usd, 1.0), GDP_SATURATION_USD)
    return math.log(capped)


def population_factor(population: int) -> float:
    """Log talent pool. A 10x population is one unit, not ten."""
    return math.log10(max(population, 1))


def temperature_factor(avg_temp_c: float) -> float:
    """Penalty for distance from the ~14C optimum (0 at the optimum)."""
    return -abs(avg_temp_c - TEMP_OPTIMUM_C)


def _zscores(values: list[float]) -> list[float]:
    """Z-score within the cohort; all-equal cohorts score 0 (no signal)."""
    if len(values) < 2:
        return [0.0] * len(values)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd <= 1e-12:
        return [0.0] * len(values)
    return [(v - mean) / sd for v in values]


def composite_scores(
    fundamentals: dict[str, TeamFundamentals],
    weights: FactorWeights | None = None,
) -> dict[str, float]:
    """Weighted z-score composite per team, itself z-scored across the cohort
    so the output is in 'cohort standard deviations' regardless of weights."""
    w = weights or FactorWeights()
    keys = list(fundamentals)
    teams = [fundamentals[k] for k in keys]

    factors = [
        (w.fifa, _zscores([t.fifa_points for t in teams])),
        (w.gdp, _zscores([gdp_factor(t.gdp_per_capita_usd) for t in teams])),
        (w.population, _zscores([population_factor(t.population) for t in teams])),
        (w.temperature, _zscores([temperature_factor(t.avg_temp_c) for t in teams])),
        (w.host, _zscores([1.0 if t.host else 0.0 for t in teams])),
    ]
    total_w = sum(fw for fw, _ in factors)
    if total_w <= 0:
        return {k: 0.0 for k in keys}

    raw = [
        sum(fw * zs[i] for fw, zs in factors) / total_w
        for i in range(len(keys))
    ]
    return dict(zip(keys, _zscores(raw)))


def prior_ratings(
    fundamentals: dict[str, TeamFundamentals],
    *,
    base: float = DEFAULT_PRIOR_BASE,
    spread: float = DEFAULT_PRIOR_SPREAD,
    weights: FactorWeights | None = None,
) -> dict[str, float]:
    """Glicko-scale prior rating per team: base + spread * composite z."""
    scores = composite_scores(fundamentals, weights)
    return {k: base + spread * z for k, z in scores.items()}


def load_fundamentals(path: str | Path) -> dict[str, TeamFundamentals]:
    """Load the CSV written by scripts/build_fundamentals.py.

    Keys are matcher-normalised team names so lookups line up with how the
    engine and exchanges refer to teams.
    """
    out: dict[str, TeamFundamentals] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tf = TeamFundamentals(
                team=row["team"],
                iso3=row["iso3"],
                gdp_per_capita_usd=float(row["gdp_per_capita_usd"]),
                population=int(float(row["population"])),
                avg_temp_c=float(row["avg_temp_c"]),
                fifa_points=float(row["fifa_points"]),
                host=row.get("host", "").strip().lower() in {"1", "true", "yes"},
            )
            out[normalize(tf.team)] = tf
    return out
