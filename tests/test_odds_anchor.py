"""ensemble.anchor_triple + strategy.odds_anchor.anchor_prediction.

The two properties that matter:

* the anchored triple sits BETWEEN the model and the de-vigged market — it can
  never overshoot the price, because anchoring is shrinkage toward the market,
  not an edge over it;
* every failure path returns the prediction UNCHANGED, so an alert can never
  break because a free CSV was slow.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import pytest

from betbot.data.odds import MatchOdds, OddsQuote, OddsService
from betbot.strategy.engine import Prediction
from betbot.strategy.ensemble import anchor_triple, de_vig
from betbot.strategy.odds_anchor import anchor_prediction, engine_model_weight


class _Settings:
    odds_anchor_enabled = True
    odds_anchor_market_weight = 1.0
    odds_cache_ttl_seconds = 3600.0
    odds_min_request_interval_seconds = 0.0
    odds_max_date_slack_days = 3
    leagues = ("PL", "PD")


def _prediction(ph=0.50, pd=0.28, pa=0.22) -> Prediction:
    return Prediction(
        fixture_id=1,
        competition_code="PD",
        home_team="Rayo Vallecano",
        away_team="Alaves",
        p_home=ph,
        p_draw=pd,
        p_away=pa,
        home_score=1500.0,
        away_score=1500.0,
        draw_score=0.0,
    )


def _odds(ph=2.25, pdw=3.0, pa=3.6) -> MatchOdds:
    return MatchOdds(
        league="PD", match_date=date(2026, 8, 20), home="vallecano", away="alaves",
        price_home=ph, price_draw=pdw, price_away=pa, source="test", book="B365H",
    )


class _StubService:
    """Minimal OddsService stand-in: returns a fixed quote (or None)."""

    def __init__(self, quote):
        self._quote = quote
        self.primed = 0

    async def prime(self, leagues):
        self.primed += 1
        return 1

    def quote(self, league, match_date, home, away):
        return self._quote


# ---------------------------------------------------------------------------
# anchor_triple
# ---------------------------------------------------------------------------
def test_anchored_triple_is_a_distribution():
    out = anchor_triple((0.5, 0.28, 0.22), (0.42, 0.32, 0.26), 2.0, 1.0)
    assert sum(out) == pytest.approx(1.0)
    assert all(0.0 < p < 1.0 for p in out)


def test_anchored_lies_between_model_and_market():
    """The defining property. Anchoring moves us TOWARD the price; it must
    never land outside the interval, which is what an 'edge' would look like."""
    model = (0.60, 0.25, 0.15)
    market = (0.40, 0.30, 0.30)
    out = anchor_triple(model, market, 1.0, 1.0)
    for i in range(3):
        lo, hi = sorted((model[i], market[i]))
        assert lo <= out[i] <= hi, f"outcome {i} escaped [{lo}, {hi}]: {out[i]}"


def test_heavier_market_weight_lands_closer_to_the_market():
    model = (0.60, 0.25, 0.15)
    market = (0.40, 0.30, 0.30)
    light = anchor_triple(model, market, 2.0, 0.5)
    heavy = anchor_triple(model, market, 2.0, 8.0)
    assert abs(heavy[0] - market[0]) < abs(light[0] - market[0])


def test_zero_market_weight_is_a_no_op():
    model = (0.60, 0.25, 0.15)
    assert anchor_triple(model, (0.4, 0.3, 0.3), 2.0, 0.0) == model


def test_degenerate_market_triple_is_ignored():
    model = (0.60, 0.25, 0.15)
    assert anchor_triple(model, (0.0, 0.5, 0.5), 2.0, 1.0) == model
    assert anchor_triple(model, (1.0, 0.0, 0.0), 2.0, 1.0) == model


def test_anchoring_to_itself_changes_nothing():
    model = (0.45, 0.30, 0.25)
    out = anchor_triple(model, model, 2.0, 1.0)
    for a, b in zip(out, model):
        assert a == pytest.approx(b, abs=1e-9)


def test_anchor_input_must_be_de_vigged():
    """Sanity guard on the contract: raw implied prices sum above 1, and
    de_vig is what brings them back to a distribution."""
    raw = [1 / 2.25, 1 / 3.0, 1 / 3.6]
    assert sum(raw) > 1.0
    assert sum(de_vig(raw)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# anchor_prediction — the wiring
# ---------------------------------------------------------------------------
def test_anchor_prediction_moves_the_prediction_toward_the_price():
    pred = _prediction(0.60, 0.25, 0.15)
    svc = _StubService(OddsQuote(odds=_odds()))
    out = asyncio.run(
        anchor_prediction(
            pred,
            league="PD",
            kickoff=datetime(2026, 8, 20, 19, 0, tzinfo=timezone.utc),
            settings=_Settings(),
            odds_service=svc,
        )
    )
    market = _odds().probabilities()
    assert out is not pred
    assert abs(out.p_home - market[0]) < abs(pred.p_home - market[0])
    assert out.p_home + out.p_draw + out.p_away == pytest.approx(1.0)
    # Identity fields must survive untouched.
    assert (out.home_team, out.away_team, out.fixture_id) == (
        pred.home_team, pred.away_team, pred.fixture_id,
    )


def test_flag_off_is_a_no_op():
    class Off(_Settings):
        odds_anchor_enabled = False

    pred = _prediction()
    svc = _StubService(OddsQuote(odds=_odds()))
    out = asyncio.run(
        anchor_prediction(
            pred, league="PD", kickoff=datetime(2026, 8, 20, tzinfo=timezone.utc),
            settings=Off(), odds_service=svc,
        )
    )
    assert out is pred
    assert svc.primed == 0, "flag off must not even touch the feed"


def test_no_service_is_a_no_op():
    pred = _prediction()
    out = asyncio.run(
        anchor_prediction(
            pred, league="PD", kickoff=datetime(2026, 8, 20, tzinfo=timezone.utc),
            settings=_Settings(), odds_service=None,
        )
    )
    assert out is pred


def test_missing_quote_degrades_to_the_unanchored_prediction():
    pred = _prediction()
    out = asyncio.run(
        anchor_prediction(
            pred, league="PD", kickoff=datetime(2026, 8, 20, tzinfo=timezone.utc),
            settings=_Settings(), odds_service=_StubService(None),
        )
    )
    assert out is pred


def test_service_blowing_up_degrades_instead_of_raising():
    class Boom:
        async def prime(self, leagues):
            raise RuntimeError("everything is on fire")

        def quote(self, *a):
            raise RuntimeError("nope")

    pred = _prediction()
    out = asyncio.run(
        anchor_prediction(
            pred, league="PD", kickoff=datetime(2026, 8, 20, tzinfo=timezone.utc),
            settings=_Settings(), odds_service=Boom(),
        )
    )
    assert out is pred


def test_missing_kickoff_degrades():
    pred = _prediction()
    out = asyncio.run(
        anchor_prediction(
            pred, league="PD", kickoff=None,
            settings=_Settings(), odds_service=_StubService(OddsQuote(odds=_odds())),
        )
    )
    assert out is pred


# ---------------------------------------------------------------------------
# model weight
# ---------------------------------------------------------------------------
def test_engine_without_model_weight_falls_back():
    assert engine_model_weight(object()) == 1.0


def test_engine_model_weight_is_used():
    class Eng:
        def model_weight(self):
            return 3.0

    assert engine_model_weight(Eng()) == 3.0


def test_broken_model_weight_falls_back():
    class Eng:
        def model_weight(self):
            raise RuntimeError

    assert engine_model_weight(Eng()) == 1.0


def test_club_engine_exposes_its_summed_component_weight():
    """The club engine's model_weight must equal the sum decide_with_market
    computes inline, so the anchor and the decision agree on the model's
    evidence weight."""
    from betbot.config import Settings
    from betbot.strategy.club_engine import ClubStrategyEngine

    s = Settings()
    eng = ClubStrategyEngine(s, get_rating=lambda n: None, name_map={})
    expected = s.club_weight_glicko + s.club_weight_form + (
        s.club_weight_dc if eng._dc_params is not None else 0.0
    )
    assert eng.model_weight() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# End-to-end through the real service object (no network)
# ---------------------------------------------------------------------------
def test_end_to_end_through_a_loaded_service():
    svc = OddsService(_Settings(), providers=[])
    svc.load_rows([_odds()])
    pred = _prediction(0.60, 0.25, 0.15)
    out = asyncio.run(
        anchor_prediction(
            pred, league="PD",
            kickoff=datetime(2026, 8, 20, 19, 0, tzinfo=timezone.utc),
            settings=_Settings(), odds_service=svc,
        )
    )
    assert out.p_home < pred.p_home  # the price is shorter on the home side
