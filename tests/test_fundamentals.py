"""Tests for the Klement 5-factor fundamentals layer (ensemble Layer 1)."""

from __future__ import annotations

import pytest

from betbot.strategy.fundamentals import (
    GDP_SATURATION_USD,
    FactorWeights,
    TeamFundamentals,
    composite_scores,
    gdp_factor,
    load_fundamentals,
    population_factor,
    prior_ratings,
    temperature_factor,
)


def _tf(team, *, gdp=20_000.0, pop=50_000_000, temp=14.0, fifa=1500.0, host=False):
    return TeamFundamentals(
        team=team, iso3="XXX", gdp_per_capita_usd=gdp, population=pop,
        avg_temp_c=temp, fifa_points=fifa, host=host,
    )


# ---- factor transforms -------------------------------------------------

def test_gdp_monotone_then_saturates():
    assert gdp_factor(30_000) > gdp_factor(5_000)
    # Klement's diminishing returns: beyond ~$60k more wealth stops helping.
    assert gdp_factor(90_000) == pytest.approx(gdp_factor(GDP_SATURATION_USD))


def test_temperature_optimum_at_14c():
    assert temperature_factor(14.0) == 0.0
    assert temperature_factor(14.0) > temperature_factor(25.0)
    # Symmetric: equally far above or below the optimum is equally bad.
    assert temperature_factor(4.0) == pytest.approx(temperature_factor(24.0))


def test_population_is_log_scaled():
    tenfold = population_factor(100_000_000) - population_factor(10_000_000)
    assert tenfold == pytest.approx(1.0)


# ---- composite + priors ------------------------------------------------

def test_composite_orders_strong_above_weak():
    cohort = {
        "brazil": _tf("Brazil", gdp=10_000, pop=215_000_000, temp=25, fifa=1800),
        "france": _tf("France", gdp=45_000, pop=68_000_000, temp=12, fifa=1850),
        "minnow": _tf("Minnow", gdp=2_000, pop=400_000, temp=28, fifa=1100),
    }
    scores = composite_scores(cohort)
    assert scores["france"] > scores["minnow"]
    assert scores["brazil"] > scores["minnow"]


def test_composite_is_cohort_zscored():
    cohort = {f"t{i}": _tf(f"T{i}", fifa=1200.0 + 100 * i) for i in range(5)}
    scores = composite_scores(cohort)
    assert sum(scores.values()) == pytest.approx(0.0, abs=1e-9)


def test_identical_cohort_has_no_signal():
    cohort = {"a": _tf("A"), "b": _tf("B")}
    assert composite_scores(cohort) == {"a": 0.0, "b": 0.0}
    priors = prior_ratings(cohort)
    assert priors["a"] == priors["b"] == 1500.0


def test_prior_ratings_centre_on_base():
    cohort = {f"t{i}": _tf(f"T{i}", fifa=1200.0 + 100 * i) for i in range(5)}
    priors = prior_ratings(cohort, base=1500.0, spread=120.0)
    assert sum(priors.values()) / len(priors) == pytest.approx(1500.0)
    assert max(priors.values()) > 1500.0 > min(priors.values())


def test_host_flag_inert_by_default():
    # Host advantage is applied per-match by the engine; the default weights
    # must not double-count it in the prior.
    cohort_a = {"usa": _tf("USA", host=True), "ger": _tf("Germany")}
    cohort_b = {"usa": _tf("USA", host=False), "ger": _tf("Germany")}
    assert composite_scores(cohort_a) == composite_scores(cohort_b)


def test_host_weight_opt_in():
    cohort = {"usa": _tf("USA", host=True), "ger": _tf("Germany")}
    w = FactorWeights(host=0.2)
    scores = composite_scores(cohort, w)
    assert scores["usa"] > scores["ger"]


# ---- CSV loader ---------------------------------------------------------

def test_load_fundamentals_roundtrip(tmp_path):
    p = tmp_path / "fundamentals.csv"
    p.write_text(
        "team,iso3,gdp_per_capita_usd,population,avg_temp_c,fifa_points,host\n"
        "Mexico,MEX,13800.5,128000000,21.0,1660,true\n"
        "Japan,JPN,33800,124000000,11.2,1652,false\n",
        encoding="utf-8",
    )
    funds = load_fundamentals(p)
    # Keys are matcher-normalised names.
    assert "mexico" in funds and "japan" in funds
    mx = funds["mexico"]
    assert mx.host is True
    assert mx.gdp_per_capita_usd == pytest.approx(13800.5)
    assert funds["japan"].host is False
