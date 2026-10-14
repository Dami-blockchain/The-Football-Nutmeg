"""Anchor a scored Champions League prediction to a Polymarket per-match line.

The domestic free-odds anchor (:mod:`betbot.strategy.odds_anchor`) covers the
big-5 leagues only; CL fixtures fell through it unanchored. This module applies
the SAME logit-space ``anchor_triple`` to a CL prediction, using the de-vigged
orderbook-midpoint triple from
:class:`betbot.exchanges.polymarket_cl_odds.PolymarketCLOddsProvider`.

Parallel to :func:`betbot.strategy.odds_anchor.anchor_prediction` in every
respect that matters:

* flag-gated (``BETBOT_CL_PM_ANCHOR``), default OFF;
* blends with ``w_model = engine.model_weight()`` (CL engine returns 2.0) and
  ``w_market = settings.odds_anchor_market_weight`` (1.0) — the same weights the
  research backtest priced;
* marks the row ``anchor_source="pm"`` so the existing anchor badge renders
  "📊 Market-anchored";
* keeps the pre-anchor triple on ``model_probs`` so the bet-decision path still
  prices off the raw model (single-anchor invariant);
* records ``market_p_draw`` (the de-vigged market draw probability) so the
  draw-risk tag is reproducible and auditable;
* returns the prediction UNCHANGED (and logs ``odds_anchor_skipped``) on any
  failure — no listing, a one-sided book, a sum out of band, a dead feed.

Honesty: like every anchor, this SHRINKS the model toward the price. It moves
us toward market-level accuracy and cannot exceed it. It is not an edge.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from betbot.logging import get_logger
from betbot.strategy.ensemble import anchor_triple
from betbot.strategy.odds_anchor import engine_model_weight

log = get_logger(__name__)


async def anchor_prediction_cl(
    prediction,
    *,
    league: str,
    kickoff: datetime | None,
    settings,
    provider,
    engine=None,
):
    """Return a CL ``prediction`` anchored to Polymarket, or unchanged.

    Only acts when the flag is on, the league is CL, a provider is wired, the
    kickoff date is known, and the prediction is not already anchored. Every
    other path returns the prediction untouched.
    """
    if provider is None or not getattr(settings, "cl_pm_anchor_enabled", False):
        return prediction
    if league != "CL":
        return prediction
    if getattr(prediction, "is_anchored", False):
        log.info(
            "odds_anchor_skipped",
            league=league,
            home=getattr(prediction, "home_team", None),
            away=getattr(prediction, "away_team", None),
            reason="already_anchored",
            anchor_source=getattr(prediction, "anchor_source", None),
        )
        return prediction
    if kickoff is None:
        return prediction
    try:
        quote = await provider.quote(
            prediction.home_team, prediction.away_team, kickoff.date().isoformat()
        )
        if quote is None:
            # provider.quote already logged the specific skip reason
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
            source="pm",
            raw_sum=round(quote.raw_sum, 4),
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
            model_probs=(prediction.p_home, prediction.p_draw, prediction.p_away),
            anchor_source="pm",
            market_p_draw=market[1],
        )
    except Exception as e:  # noqa: BLE001 — anchoring must never break an alert
        log.warning(
            "cl_pm_anchor_failed",
            league=league,
            home=getattr(prediction, "home_team", None),
            away=getattr(prediction, "away_team", None),
            error=str(e),
        )
        return prediction
