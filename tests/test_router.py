"""Tests for ExchangeRouter (Phase 2). Offline: fake adapters."""

from __future__ import annotations

from datetime import datetime, timezone

from betbot.exchanges.base import ExchangeName, MarketRef, OrderbookQuote, Outcome
from betbot.exchanges.router import ExchangeRouter

KICKOFF = datetime(2026, 1, 20, 15, 0, tzinfo=timezone.utc)


def _quote(exchange, price, size, outcome=Outcome.HOME):
    return OrderbookQuote(
        exchange=exchange,
        market_id="m",
        outcome=outcome,
        yes_price=price,
        yes_size=size,
        timestamp=KICKOFF,
    )


class FakeAdapter:
    def __init__(self, name, *, market=True, quote=None, raises=False):
        self.name = name
        self._market = market
        self._quote = quote
        self._raises = raises

    async def find_market(self, home, away, kickoff):
        if self._raises:
            raise RuntimeError("boom")
        if not self._market:
            return None
        return MarketRef(self.name, "m", f"{home} vs {away}", {})

    async def get_orderbook(self, market, outcome):
        return self._quote


async def test_picks_lowest_price():
    poly = FakeAdapter(ExchangeName.POLYMARKET, quote=_quote(ExchangeName.POLYMARKET, 0.55, 100))
    limit = FakeAdapter(ExchangeName.LIMITLESS, quote=_quote(ExchangeName.LIMITLESS, 0.52, 80))
    r = ExchangeRouter([poly, limit])
    best = await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME)
    assert best.exchange is ExchangeName.LIMITLESS  # 0.52 < 0.55


async def test_tie_break_prefers_larger_size():
    a = FakeAdapter(ExchangeName.POLYMARKET, quote=_quote(ExchangeName.POLYMARKET, 0.50, 50))
    b = FakeAdapter(ExchangeName.LIMITLESS, quote=_quote(ExchangeName.LIMITLESS, 0.50, 200))
    r = ExchangeRouter([a, b])
    best = await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME)
    assert best.exchange is ExchangeName.LIMITLESS  # same price, larger size


async def test_none_when_no_markets():
    a = FakeAdapter(ExchangeName.POLYMARKET, market=False)
    r = ExchangeRouter([a])
    assert await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME) is None


async def test_failing_adapter_is_isolated():
    bad = FakeAdapter(ExchangeName.LIMITLESS, raises=True)
    good = FakeAdapter(ExchangeName.POLYMARKET, quote=_quote(ExchangeName.POLYMARKET, 0.6, 10))
    r = ExchangeRouter([bad, good])
    best = await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME)
    assert best is not None and best.exchange is ExchangeName.POLYMARKET


async def test_market_without_quote_skipped():
    a = FakeAdapter(ExchangeName.POLYMARKET, market=True, quote=None)
    r = ExchangeRouter([a])
    assert await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME) is None
