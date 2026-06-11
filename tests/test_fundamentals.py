"""Tests for the Klement 5-factor fundamentals layer (ensemble Layer 1)."""

from __future__ import annotations

import math

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
    squad_value_factor,
    temperature_factor,
)


def _tf(
    team, *, gdp=20_000.0, pop=50_000_000, temp=14.0, fifa=1500.0, host=False,
    squad_value_eur=300_000_000.0,
):
    return TeamFundamentals(
        team=team, iso3="XXX", gdp_per_capita_usd=gdp, population=pop,
        avg_temp_c=temp, fifa_points=fifa, host=host,
        squad_value_eur=squad_value_eur,
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


def test_squad_value_monotone_and_log():
    # Monotone increasing in value.
    assert squad_value_factor(1_000_000_000) > squad_value_factor(20_000_000)
    # Diminishing returns: natural log, so a 10x value is +ln(10), not +10.
    step = squad_value_factor(2_000_000_000) - squad_value_factor(200_000_000)
    assert step == pytest.approx(math.log(10))


def test_squad_value_guards_zero():
    # log(0) would explode; the guard maps it to log(1) = 0.0, no error.
    assert squad_value_factor(0.0) == 0.0
    assert squad_value_factor(-5.0) == 0.0


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


# ---- squad value in the composite --------------------------------------

def test_high_squad_value_lifts_a_team():
    # Two teams identical on every factor; give one a much higher squad value
    # and it must rank above the other (squad value carries real weight).
    base = dict(gdp=20_000, pop=50_000_000, temp=14, fifa=1500)
    cohort = {
        "rich": _tf("Rich", **base, squad_value_eur=1_000_000_000),
        "poor": _tf("Poor", **base, squad_value_eur=20_000_000),
    }
    scores = composite_scores(cohort)
    assert scores["rich"] > scores["poor"]


def test_squad_factor_skipped_when_majority_unknown():
    # >50% of the cohort has unknown (0.0) squad value -> factor drops out, so
    # the one team that DOES have a value gets no squad-driven lift, and the
    # composite is identical to one where squad value is uniformly unknown.
    base = dict(gdp=20_000, pop=50_000_000, temp=14)
    cohort_sparse = {
        "a": _tf("A", fifa=1500, **base, squad_value_eur=1_000_000_000),
        "b": _tf("B", fifa=1400, **base, squad_value_eur=0.0),
        "c": _tf("C", fifa=1300, **base, squad_value_eur=0.0),
    }
    cohort_none = {
        "a": _tf("A", fifa=1500, **base, squad_value_eur=0.0),
        "b": _tf("B", fifa=1400, **base, squad_value_eur=0.0),
        "c": _tf("C", fifa=1300, **base, squad_value_eur=0.0),
    }
    assert composite_scores(cohort_sparse) == composite_scores(cohort_none)


def test_missing_squad_value_imputed_to_median_not_floor():
    # Minority unknown -> the unknown team is imputed to the cohort median, so
    # it sits at the centre on the squad factor (not dragged to the bottom).
    # Build a cohort where everything but squad value is equal; the team with a
    # missing value must NOT score below the cheapest known team.
    base = dict(gdp=20_000, pop=50_000_000, temp=14, fifa=1500)
    cohort = {
        "top": _tf("Top", **base, squad_value_eur=1_000_000_000),
        "mid": _tf("Mid", **base, squad_value_eur=300_000_000),
        "cheap": _tf("Cheap", **base, squad_value_eur=20_000_000),
        "unknown": _tf("Unknown", **base, squad_value_eur=0.0),
    }
    scores = composite_scores(cohort)
    # Imputed to the median of {1000M, 300M, 20M} -> 300M, so "unknown" lands
    # near "mid" and strictly above the cheapest, never at the floor.
    assert scores["unknown"] > scores["cheap"]


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


def test_load_tolerates_missing_squad_value_column(tmp_path):
    # Older CSVs (written before squad_value_eur existed) must still load,
    # defaulting the new column to 0.0 (unknown).
    p = tmp_path / "old.csv"
    p.write_text(
        "team,iso3,gdp_per_capita_usd,population,avg_temp_c,fifa_points,host\n"
        "Japan,JPN,33800,124000000,11.2,1652,false\n",
        encoding="utf-8",
    )
    funds = load_fundamentals(p)
    assert funds["japan"].squad_value_eur == 0.0


def test_load_reads_squad_value_when_present(tmp_path):
    p = tmp_path / "new.csv"
    p.write_text(
        "team,iso3,gdp_per_capita_usd,population,avg_temp_c,fifa_points,host,"
        "squad_value_eur\n"
        "Spain,ESP,35326,48848840,13.3,1880,false,1400000000.0\n",
        encoding="utf-8",
    )
    funds = load_fundamentals(p)
    assert funds["spain"].squad_value_eur == pytest.approx(1_400_000_000.0)
