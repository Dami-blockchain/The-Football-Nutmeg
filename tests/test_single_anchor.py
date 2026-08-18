"""The single-anchor invariant.

INVARIANT UNDER TEST: a probability is anchored to at most one market source,
exactly once, on any path.

Before this fix, with ``BETBOT_ODDS_ANCHOR=true`` the prediction reaching
``decide_with_market`` had already been pulled toward a bookmaker line, and
``decide_with_market`` anchored it a SECOND time toward the exchange price.
Beyond over-shrinking the model, that manufactured apparent edge wherever the
book and the exchange disagreed — a cross-market divergence signal nobody
designed or backtested, sitting in the real-money decision path.

The four-cell flag matrix below (odds anchor OFF/ON x exchange price
absent/present) pins the fix down. The cell that matters is BOTH ON.

Honesty note: anchoring shrinks the model toward the price. It moves us toward
market-level accuracy and cannot exceed it. It is not an edge.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import pytest

from betbot.config import get_settings
from betbot.data.odds import MatchOdds, OddsQuote
from betbot.strategy.cl_engine import EuropeanStrategyEngine
from betbot.strategy.club_engine import ClubStrategyEngine
from betbot.strategy.engine import Outcome, Prediction, StrategyEngine
from betbot.strategy.ensemble import anchor_to_market, anchor_triple
from betbot.strategy.glicko import Glicko2Rating
from betbot.strategy.odds_anchor import anchor_prediction

KICKOFF = datetime(2026, 8, 20, 19, 0, tzinfo=timezone.utc)

# The scenario, chosen so the defect is unmistakable: the MODEL AGREES EXACTLY
# with the exchange (both 0.50 on the home side), so the true edge is zero by
# construction. The bookmaker disagrees hard (~0.85 home). Any edge that shows
# up can only have come from the book/exchange disagreement.
MODEL_TRIPLE = (0.50, 0.28, 0.22)
EXCHANGE_HOME_PRICE = 0.50


class _OddsSettings:
    """Odds-layer settings with the anchor flag ON."""

    odds_anchor_enabled = True
    odds_anchor_market_weight = 1.0
    odds_cache_ttl_seconds = 3600.0
    odds_min_request_interval_seconds = 0.0
    odds_max_date_slack_days = 3
    leagues = ("PL", "PD", "CL")


class _OddsSettingsOff(_OddsSettings):
    odds_anchor_enabled = False


class _StubService:
    def __init__(self, quote):
        self._quote = quote
        self.primed = 0

    async def prime(self, leagues):
        self.primed += 1
        return 1

    def quote(self, league, match_date, home, away):
        return self._quote


def _book_disagreeing() -> MatchOdds:
    """A bookmaker line that de-vigs to roughly (0.85, 0.09, 0.06) — i.e. it
    disagrees strongly with both the model and the exchange."""
    return MatchOdds(
        league="PD", match_date=date(2026, 8, 20),
        home="rayo vallecano", away="alaves",
        price_home=1.18, price_draw=11.0, price_away=16.5,
        source="test", book="B365H",
    )


def _prediction(triple=MODEL_TRIPLE, competition="PD") -> Prediction:
    return Prediction(
        fixture_id=1,
        competition_code=competition,
        home_team="Rayo Vallecano",
        away_team="Alaves",
        p_home=triple[0], p_draw=triple[1], p_away=triple[2],
        home_score=1500.0, away_score=1500.0, draw_score=0.0,
    )


def _club_engine() -> ClubStrategyEngine:
    s = get_settings()
    ratings = {
        "Rayo Vallecano": Glicko2Rating(1520.0, 60.0, 0.06),
        "Alaves": Glicko2Rating(1480.0, 60.0, 0.06),
    }
    return ClubStrategyEngine(
        s,
        get_rating=lambda n: ratings.get(n, Glicko2Rating(
            s.glicko_default_rating, s.glicko_default_rd, s.glicko_default_vol)),
        dc_params=None, calibrators=None, name_map={},
    )


def _cl_engine() -> EuropeanStrategyEngine:
    return EuropeanStrategyEngine(
        get_settings(),
        snapshot={"Rayo Vallecano": 1700.0, "Alaves": 1650.0},
        name_map={},
    )


def _odds_anchored(pred: Prediction, engine=None, settings=None) -> Prediction:
    """Run the odds-anchor layer with the flag ON."""
    return asyncio.run(
        anchor_prediction(
            pred,
            league=pred.competition_code,
            kickoff=KICKOFF,
            settings=settings or _OddsSettings(),
            odds_service=_StubService(OddsQuote(odds=_book_disagreeing())),
            engine=engine,
        )
    )


# ---------------------------------------------------------------------------
# Cell 1 of 4 — BOTH FLAGS OFF. The default. Nothing anchors anything.
# ---------------------------------------------------------------------------
def test_both_flags_off_leaves_the_prediction_completely_unanchored():
    pred = _prediction()
    out = asyncio.run(
        anchor_prediction(
            pred, league="PD", kickoff=KICKOFF,
            settings=_OddsSettingsOff(),
            odds_service=_StubService(OddsQuote(odds=_book_disagreeing())),
        )
    )
    assert out is pred
    assert out.anchor_source is None
    assert out.is_anchored is False
    assert out.model_probs is None
    # With nothing recorded, model_probability just reads the live fields, so
    # every consumer sees byte-identical numbers to before this change.
    assert out.model_probability(Outcome.HOME) == pred.p_home
    assert out.model_probability(Outcome.DRAW) == pred.p_draw
    assert out.model_probability(Outcome.AWAY) == pred.p_away


# ---------------------------------------------------------------------------
# Cell 2 of 4 — ODDS ANCHOR ONLY (no exchange quote for this fixture).
# ---------------------------------------------------------------------------
def test_odds_anchor_only_anchors_exactly_once_and_records_its_source():
    pred = _prediction()
    out = _odds_anchored(pred)

    assert out is not pred
    assert out.anchor_source == "odds"
    assert out.is_anchored is True
    # Anchored exactly once: the displayed triple is EXACTLY one application of
    # anchor_triple to the raw model — not two, not a partial re-blend.
    market = _book_disagreeing().probabilities()
    expected = anchor_triple(MODEL_TRIPLE, market, 1.0, 1.0)
    assert (out.p_home, out.p_draw, out.p_away) == pytest.approx(expected)
    # It moved toward the book but never past it: shrinkage, not edge.
    assert pred.p_home < out.p_home < market[0]
    # The raw model survives, untouched, for the decision path.
    assert out.model_probs == pytest.approx(MODEL_TRIPLE)
    assert out.model_probability(Outcome.HOME) == pytest.approx(MODEL_TRIPLE[0])


def test_the_odds_layer_refuses_to_anchor_an_already_anchored_prediction():
    once = _odds_anchored(_prediction())
    twice = _odds_anchored(once)
    assert twice is once, "a second odds anchor must be a no-op"
    assert twice.model_probs == pytest.approx(MODEL_TRIPLE)


# ---------------------------------------------------------------------------
# Cell 3 of 4 — EXCHANGE PRICE ONLY (odds anchor off). The pre-existing path.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine_factory", [_club_engine, _cl_engine])
def test_exchange_only_anchors_exactly_once(engine_factory):
    s = get_settings()
    eng = engine_factory()
    pred = _prediction()
    dec = eng.decide_with_market(
        pred, Outcome.HOME, EXCHANGE_HOME_PRICE, require_edge=False
    )
    w_model = eng.model_weight()
    w_market = (
        s.club_weight_market if isinstance(eng, ClubStrategyEngine)
        else s.cl_weight_market
    )
    expected = anchor_to_market(
        MODEL_TRIPLE[0], EXCHANGE_HOME_PRICE, w_model, w_market
    )
    assert dec.our_probability == pytest.approx(expected)
    # Model and exchange agree here, so a single anchor leaves the number put
    # and the edge is exactly zero.
    assert dec.our_probability == pytest.approx(EXCHANGE_HOME_PRICE)
    assert dec.edge == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Cell 4 of 4 — BOTH ON. The cell the defect lived in.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine_factory", [_club_engine, _cl_engine])
def test_both_on_produces_the_identical_decision_to_exchange_only(engine_factory):
    """The bet decision is INVARIANT to BETBOT_ODDS_ANCHOR.

    The odds anchor moves what we display and store; it must not move what we
    bet, because the exchange-priced decision already anchors to the exchange.
    """
    eng = engine_factory()
    pred = _prediction()
    anchored = _odds_anchored(pred, engine=eng)
    assert anchored.p_home != pytest.approx(pred.p_home), "setup: the book moved us"

    flag_off = eng.decide_with_market(
        pred, Outcome.HOME, EXCHANGE_HOME_PRICE, require_edge=False
    )
    flag_on = eng.decide_with_market(
        anchored, Outcome.HOME, EXCHANGE_HOME_PRICE, require_edge=False
    )
    assert flag_on.our_probability == flag_off.our_probability
    assert flag_on.edge == flag_off.edge
    assert flag_on.stake_usd == flag_off.stake_usd


@pytest.mark.parametrize("engine_factory", [_club_engine, _cl_engine])
def test_both_on_manufactures_no_edge_against_either_venue(engine_factory):
    """No synthetic edge versus the exchange, and none versus the book either.

    The model sits exactly on the exchange price, so the honest edge is 0.000.
    The old double-anchored path would have reported a large positive edge —
    purely the book/exchange disagreement leaking through — and BET on it.
    """
    s = get_settings()
    eng = engine_factory()
    anchored = _odds_anchored(_prediction(), engine=eng)
    w_model = eng.model_weight()
    w_market = (
        s.club_weight_market if isinstance(eng, ClubStrategyEngine)
        else s.cl_weight_market
    )

    # What the DEFECT used to do: anchor the already-book-anchored number a
    # second time toward the exchange.
    double_anchored = anchor_to_market(
        anchored.p_home, EXCHANGE_HOME_PRICE, w_model, w_market
    )
    manufactured = double_anchored - EXCHANGE_HOME_PRICE
    assert manufactured > s.edge_threshold, (
        "setup check: the old behaviour really did manufacture a bettable edge "
        f"({manufactured:+.4f} vs threshold {s.edge_threshold:.3f})"
    )

    # What the FIXED path does: no edge at all, so the edge gate says NO BET.
    forced = eng.decide_with_market(
        anchored, Outcome.HOME, EXCHANGE_HOME_PRICE, require_edge=False
    )
    assert forced.edge == pytest.approx(0.0, abs=1e-12)
    assert eng.decide_with_market(
        anchored, Outcome.HOME, EXCHANGE_HOME_PRICE, require_edge=True
    ) is None, "a fixture the model prices at the exchange must not be a bet"


@pytest.mark.parametrize("engine_factory", [_club_engine, _cl_engine])
def test_the_decision_output_is_stamped_so_nothing_anchors_it_a_third_time(
    engine_factory,
):
    eng = engine_factory()
    anchored = _odds_anchored(_prediction(), engine=eng)
    p_final = anchor_to_market(
        anchored.model_probability(Outcome.HOME), EXCHANGE_HOME_PRICE,
        eng.model_weight(), 1.0,
    )
    out = anchored.anchored_to_market(Outcome.HOME, p_final)
    assert out.p_home == p_final
    assert out.anchor_source == "market"
    assert out.model_probs is None
    # With model_probs cleared, model_probability returns the anchored number
    # itself — there is no un-anchored value left to re-derive, so a repeat
    # trip through the decision path cannot stack another anchor on top.
    assert out.model_probability(Outcome.HOME) == p_final
    # And the odds layer will not touch it either.
    assert _odds_anchored(out) is out


# ---------------------------------------------------------------------------
# The naive fallback engine (unrated teams) shares the invariant.
# ---------------------------------------------------------------------------
def test_the_naive_engine_prices_off_the_raw_model_too():
    eng = StrategyEngine(get_settings())
    pred = _prediction()
    anchored = _odds_anchored(pred)
    off = eng.decide_with_market(
        pred, Outcome.HOME, EXCHANGE_HOME_PRICE, require_edge=False
    )
    on = eng.decide_with_market(
        anchored, Outcome.HOME, EXCHANGE_HOME_PRICE, require_edge=False
    )
    assert on.our_probability == off.our_probability == pytest.approx(MODEL_TRIPLE[0])
    assert on.edge == off.edge == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The invariant stated directly, over a grid of disagreements.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("book_home_price", [1.18, 1.60, 2.50, 4.00, 8.00])
@pytest.mark.parametrize("exchange_price", [0.20, 0.35, 0.50, 0.65, 0.80])
def test_anchored_exactly_once_across_a_grid_of_venue_disagreements(
    book_home_price, exchange_price
):
    """However far apart the two venues sit, the decided probability is one
    single anchor of the RAW model toward the exchange — the book never enters
    it, so no amount of divergence can conjure edge."""
    s = get_settings()
    eng = _club_engine()
    odds = MatchOdds(
        league="PD", match_date=date(2026, 8, 20),
        home="rayo vallecano", away="alaves",
        price_home=book_home_price, price_draw=3.6, price_away=4.2,
        source="test", book="B365H",
    )
    anchored = asyncio.run(
        anchor_prediction(
            _prediction(), league="PD", kickoff=KICKOFF,
            settings=_OddsSettings(),
            odds_service=_StubService(OddsQuote(odds=odds)), engine=eng,
        )
    )
    dec = eng.decide_with_market(
        anchored, Outcome.HOME, exchange_price, require_edge=False
    )
    expected = anchor_to_market(
        MODEL_TRIPLE[0], exchange_price, eng.model_weight(), s.club_weight_market
    )
    assert dec.our_probability == pytest.approx(expected)
    # Shrinkage property: the decided probability lies between the model and
    # the exchange price, never outside them.
    lo, hi = sorted((MODEL_TRIPLE[0], exchange_price))
    assert lo - 1e-12 <= dec.our_probability <= hi + 1e-12
