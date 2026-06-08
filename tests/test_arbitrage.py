"""Tests for cross-venue arbitrage detection (Phase: arb)."""

from __future__ import annotations

from datetime import datetime, timezone

from betbot.exchanges.arbitrage import ArbScanner, size_legs
from betbot.exchanges.base import ExchangeName, MarketRef, OrderbookQuote, Outcome

KICKOFF = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)


class FakeAdapter:
    def __init__(self, name, prices):
        self.name = name
        self._prices = prices  # {Outcome: price} or None

    async def find_market(self, h, a, k):
        return None if not self._prices else MarketRef(self.name, "m", "", {})

    async def get_orderbook(self, market, outcome):
        p = self._prices.get(outcome)
        if p is None:
            return None
        return OrderbookQuote(self.name, "m", outcome, p, 100.0, KICKOFF)


async def test_picks_cheapest_per_outcome():
    poly = FakeAdapter(ExchangeName.POLYMARKET, {Outcome.HOME: 0.55, Outcome.DRAW: 0.30, Outcome.AWAY: 0.30})
    lim = FakeAdapter(ExchangeName.LIMITLESS, {Outcome.HOME: 0.50, Outcome.AWAY: 0.28})  # no draw
    opp = await ArbScanner([poly, lim]).scan("A", "B", KICKOFF)
    assert opp.legs["HOME"].exchange is ExchangeName.LIMITLESS  # 0.50 < 0.55
    assert opp.legs["DRAW"].exchange is ExchangeName.POLYMARKET  # only venue
    assert opp.legs["AWAY"].exchange is ExchangeName.LIMITLESS  # 0.28 < 0.30
    assert opp.complete
    assert opp.price_sum == 0.50 + 0.30 + 0.28
    assert opp.margin < 0  # sum 1.08 -> no arb


async def test_detects_real_arb():
    poly = FakeAdapter(ExchangeName.POLYMARKET, {Outcome.HOME: 0.45, Outcome.DRAW: 0.28, Outcome.AWAY: 0.30})
    lim = FakeAdapter(ExchangeName.LIMITLESS, {Outcome.HOME: 0.40, Outcome.AWAY: 0.25})
    opp = await ArbScanner([poly, lim]).scan("A", "B", KICKOFF)
    # best: 0.40 + 0.28 + 0.25 = 0.93 -> 7% locked
    assert opp.complete
    assert opp.margin > 0.06
    legs = size_legs(opp, 100.0)
    assert sum(legs.values()) == 100.0  # spends the budget
    # equal payout: stake/price equal across legs
    payouts = [legs[o] / opp.legs[o].price for o in legs]
    assert max(payouts) - min(payouts) < 1e-6


async def test_fees_erode_margin():
    poly = FakeAdapter(ExchangeName.POLYMARKET, {Outcome.HOME: 0.33, Outcome.DRAW: 0.32, Outcome.AWAY: 0.33})
    lim = FakeAdapter(ExchangeName.LIMITLESS, {})
    opp = await ArbScanner([poly, lim], fee_per_leg=0.02).scan("A", "B", KICKOFF)
    # sum 0.98, locked 0.02 gross, but 3 legs * 0.02 fee = 0.06 -> net negative
    assert opp.margin < 0


async def test_none_when_no_markets():
    a = FakeAdapter(ExchangeName.POLYMARKET, {})
    b = FakeAdapter(ExchangeName.LIMITLESS, {})
    assert await ArbScanner([a, b]).scan("A", "B", KICKOFF) is None
