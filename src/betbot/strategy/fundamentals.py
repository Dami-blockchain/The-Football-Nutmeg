"""Klement fundamentals + squad value — ensemble Layer 1 (structural priors).

Joachim Klement's World Cup model (Panmure Liberum; correct outright winner
2014/2018/2022) explains ~55% of tournament success with five slow-moving
covariates: GDP per capita (diminishing returns past ~$60k), population,
distance of the country's average temperature from a ~14C football optimum,
FIFA ranking points, and host advantage. His fitted coefficients are not
public, so this module reproduces the *structure*: transform each covariate,
z-score it within the tournament cohort, and combine with documented weights
into a composite that maps onto the Glicko rating scale as a prior.

We add a sixth factor Klement does not use: total squad market value (EUR,
Transfermarkt-style). This is the bottom-up "EA-style" squad-strength signal —
a golden generation shows up in player valuations before it shows up in
results — and is especially useful as a cold-start prior for sparse-data
national teams whose FIFA points lag their actual talent. Squad value is a
real sporting signal (unlike the socioeconomic proxies, which only correlate),
so it carries meaningful weight in the default mix.

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
    # Total squad market value in EUR (Transfermarkt-style). 0.0 means
    # "unknown" and is treated as no-signal in z-scoring (see composite_scores),
    # NOT as a genuine zero — so a missing value can't drag a team to the floor.
    squad_value_eur: float = 0.0


@dataclass(frozen=True)
class FactorWeights:
    """Relative factor weights (normalised at use). FIFA points stay dominant —
    they are a direct sporting input. Squad value is the *second* genuine
    sporting signal (player valuations encode squad strength, often ahead of
    the FIFA ranking), so it carries meaningful weight; the socioeconomic
    proxies (gdp/population/temperature) only correlate with footballing
    strength and merely refine, so each is trimmed to make room.

    Defaults: fifa 0.45, squad_value 0.20, gdp/population/temperature 0.10 each,
    host 0.0. These are relative weights — they are sum-normalised at use, so a
    factor that is skipped (e.g. squad value when >50% of the cohort is unknown,
    see composite_scores) simply drops out and the rest re-normalise.

    ``host`` defaults to 0 because host advantage is already applied
    per-match by the engine (``glicko_host_home_mu``); weighting it here too
    would double-count. Raise it only if that per-match bump is disabled.
    """

    fifa: float = 0.45
    gdp: float = 0.10
    population: float = 0.10
    temperature: float = 0.10
    squad_value: float = 0.20
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


def squad_value_factor(value_eur: float) -> float:
    """Log total squad market value — diminishing returns, like population:
    a 10x more valuable squad is not 10x stronger. Guards log(0); a missing
    (0.0) value maps to the same floor as the cheapest possible squad here,
    but composite_scores imputes/skips missing values *before* z-scoring so a
    0 never actually reaches this on the unknown-value path."""
    return math.log(max(value_eur, 1.0))


def _squad_value_zscores(values: list[float]) -> list[float] | None:
    """Z-score squad values with missing-data handling.

    A value of 0.0 means "unknown". If more than half the cohort is unknown the
    factor is too sparse to trust, so we return ``None`` (caller drops it). If
    some are unknown but most are known, impute each missing value as the
    cohort median of the *known* values before transforming + z-scoring, so an
    unknown squad sits at the cohort centre (no signal) rather than the floor.
    """
    known = [v for v in values if v > 0.0]
    if len(known) * 2 <= len(values):  # >=50% unknown -> skip the factor
        return None
    median = statistics.median(known)
    imputed = [v if v > 0.0 else median for v in values]
    return _zscores([squad_value_factor(v) for v in imputed])


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
    # Squad value handles missing data specially: it imputes the cohort median
    # for unknowns, or drops out entirely when >50% of the cohort is unknown.
    squad_zs = _squad_value_zscores([t.squad_value_eur for t in teams])
    if squad_zs is not None:
        factors.append((w.squad_value, squad_zs))
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
                # Optional/additive: older CSVs (written before this column
                # existed) lack it; default 0.0 = unknown so they still load.
                squad_value_eur=float(row.get("squad_value_eur") or 0.0),
            )
            out[normalize(tf.team)] = tf
    return out
