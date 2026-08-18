"""Anchor a scored Prediction to a free pre-match odds line.

The gap this closes: ``anchor_to_market`` only ran inside
``decide_with_market``, which only ran when Polymarket listed the fixture. Most
big-5 matches therefore shipped raw, unanchored model probabilities. This
module applies the same logit-space anchor to the FULL 1X2 triple of every
scored fixture, using a free bookmaker feed.

Scope discipline: this changes PROBABILITIES only. It does not touch the
bet/no-bet decision path (``decide_with_market``), the edge threshold, or any
staking logic.

Single-anchor invariant: the anchored prediction carries its PRE-anchor triple
in ``Prediction.model_probs`` and records ``anchor_source="odds"``. The
exchange-priced decision path reads ``Prediction.model_probability`` and so
anchors the RAW model toward the exchange, never this already-anchored number.
A probability is therefore anchored to at most one market source, exactly once,
on every path — and the bet decision is invariant to ``BETBOT_ODDS_ANCHOR``.
An already-anchored prediction is returned unchanged.

Honesty: anchoring shrinks the model toward the price. It moves us toward
market-level accuracy and cannot exceed it. It is not an edge.

Every failure mode returns the prediction unchanged.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from betbot.logging import get_logger
from betbot.strategy.ensemble import anchor_triple

log = get_logger(__name__)

# Fallback when an engine exposes no model_weight() (e.g. the naive
# StrategyEngine): 1.0 keeps the anchor at an even model/market split.
_DEFAULT_MODEL_WEIGHT = 1.0


def engine_model_weight(engine) -> float:
    fn = getattr(engine, "model_weight", None)
    if callable(fn):
        try:
            w = float(fn())
        except Exception:  # noqa: BLE001
            return _DEFAULT_MODEL_WEIGHT
        return w if w > 0.0 else _DEFAULT_MODEL_WEIGHT
    return _DEFAULT_MODEL_WEIGHT


async def anchor_prediction(
    prediction,
    *,
    league: str,
    kickoff: datetime | None,
    settings,
    odds_service,
    engine=None,
):
    """Return ``prediction`` anchored to the free odds line, or unchanged.

    Unchanged (and logged) when: the flag is off, no service is wired, the
    fixture's team names do not resolve confidently, the feed has no row for
    the fixture, or anything at all goes wrong. Graceful degradation is the
    point — an alert must never fail because a free CSV was slow.
    """
    if odds_service is None or not getattr(settings, "odds_anchor_enabled", False):
        return prediction
    if getattr(prediction, "is_anchored", False):
        # Already anchored to a market (a re-scored fixture, or a caller that
        # ran this twice). Anchoring again would double-count the price.
        log.info(
            "odds_anchor_skipped",
            league=league,
            home=prediction.home_team,
            away=prediction.away_team,
            reason="already_anchored",
            anchor_source=prediction.anchor_source,
        )
        return prediction
    try:
        await odds_service.prime(getattr(settings, "leagues", (league,)))
        match_date = (kickoff.date() if kickoff is not None else None)
        if match_date is None:
            return prediction
        quote = odds_service.quote(
            league, match_date, prediction.home_team, prediction.away_team
        )
        if quote is None:
            log.info(
                "odds_anchor_skipped",
                league=league,
                home=prediction.home_team,
                away=prediction.away_team,
                reason="no_quote",
            )
            return prediction
        w_model = engine_model_weight(engine)
        w_market = float(getattr(settings, "odds_anchor_market_weight", 1.0))
        market = quote.probabilities
        anchored = anchor_triple(
            (prediction.p_home, prediction.p_draw, prediction.p_away),
            market,
            w_model,
            w_market,
        )
        log.info(
            "odds_anchor_applied",
            league=league,
            home=prediction.home_team,
            away=prediction.away_team,
            source=quote.source,
            book=quote.odds.book,
            prices=[
                round(quote.odds.price_home, 2),
                round(quote.odds.price_draw, 2),
                round(quote.odds.price_away, 2),
            ],
            overround=round(quote.odds.overround, 4),
            market=[round(p, 4) for p in market],
            model=[
                round(prediction.p_home, 4),
                round(prediction.p_draw, 4),
                round(prediction.p_away, 4),
            ],
            anchored=[round(p, 4) for p in anchored],
            w_model=w_model,
            w_market=w_market,
        )
        return dataclasses.replace(
            prediction,
            p_home=anchored[0],
            p_draw=anchored[1],
            p_away=anchored[2],
            # Keep the raw model triple so the exchange-priced decision path
            # can anchor from it instead of anchoring this number a 2nd time.
            model_probs=(prediction.p_home, prediction.p_draw, prediction.p_away),
            anchor_source="odds",
        )
    except Exception as e:  # noqa: BLE001 — anchoring must never break an alert
        log.warning(
            "odds_anchor_failed",
            league=league,
            home=getattr(prediction, "home_team", None),
            away=getattr(prediction, "away_team", None),
            error=str(e),
        )
        return prediction
