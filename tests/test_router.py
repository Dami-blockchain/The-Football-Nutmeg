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


# ----------------------------------------------------------------------
# price-sanity guard (matcher hardening) — reject implausible 1X2 prices
# ----------------------------------------------------------------------
async def test_rejects_implausibly_low_price(capsys):
    # The Mexico/SA 0.014 bug: a 1X2 outcome priced at 0.014 is a wrong-market
    # match and must be rejected, not logged as a 38-point edge.
    a = FakeAdapter(ExchangeName.POLYMARKET, quote=_quote(ExchangeName.POLYMARKET, 0.014, 100))
    r = ExchangeRouter([a])
    assert await r.find_best_quote("Mexico", "South Africa", KICKOFF, Outcome.HOME) is None
    assert "router_implausible_price_rejected" in capsys.readouterr().out


async def test_rejects_implausibly_high_price(capsys):
    a = FakeAdapter(ExchangeName.POLYMARKET, quote=_quote(ExchangeName.POLYMARKET, 0.99, 100))
    r = ExchangeRouter([a])
    assert await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME) is None
    assert "router_implausible_price_rejected" in capsys.readouterr().out


async def test_accepts_plausible_price():
    a = FakeAdapter(ExchangeName.POLYMARKET, quote=_quote(ExchangeName.POLYMARKET, 0.45, 100))
    r = ExchangeRouter([a])
    best = await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME)
    assert best is not None and abs(best.yes_price - 0.45) < 1e-9


async def test_implausible_leg_dropped_plausible_leg_kept():
    # One venue mis-matched (0.014), the other priced sanely (0.50): the bad leg
    # is dropped and the good one wins, instead of the bad leg producing a route.
    bad = FakeAdapter(ExchangeName.LIMITLESS, quote=_quote(ExchangeName.LIMITLESS, 0.014, 100))
    good = FakeAdapter(ExchangeName.POLYMARKET, quote=_quote(ExchangeName.POLYMARKET, 0.50, 100))
    r = ExchangeRouter([bad, good])
    best = await r.find_best_quote("A", "B", KICKOFF, Outcome.HOME)
    assert best is not None and best.exchange is ExchangeName.POLYMARKET
