"""ExchangeRouter — choose the best venue/price across exchanges.

The scoring loop hands the router a fixture + the outcome it wants to back; the
router polls every adapter, collects orderbook quotes, and returns the best one
(lowest ``yes_price`` — i.e. cheapest to buy the outcome — tie-broken by larger
available size).

Each adapter is isolated behind try/except so one exchange being down or
geo-blocked (see Phase 3 / Limitless) degrades gracefully instead of sinking
the whole route. ``find_best_quote`` returning ``None`` means *no usable quote
anywhere* — the caller treats that as "no market" and falls through to a
favourite-only paper bet. A quote that comes back but fails the edge filter is
the caller's concern (and must NOT fall back to favourite logging).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from betbot.exchanges.base import (
    ExchangeAdapter,
    MarketRef,
    OrderbookQuote,
    Outcome,
)
from betbot.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RoutedQuote:
    """Best quote plus the adapter + market that produced it, so the caller can
    both price the edge AND place an order on the winning venue."""

    adapter: ExchangeAdapter
    market: MarketRef
    quote: OrderbookQuote


class ExchangeRouter:
    def __init__(self, adapters: Iterable[ExchangeAdapter]) -> None:
        self._adapters = list(adapters)

    @property
    def adapters(self) -> list[ExchangeAdapter]:
        return self._adapters

    async def find_best_route(
        self,
        home_team: str,
        away_team: str,
        kickoff: datetime,
        outcome: Outcome,
    ) -> RoutedQuote | None:
        """Best (adapter, market, quote) for ``outcome``, or ``None``."""
        routes: list[RoutedQuote] = []
        for adapter in self._adapters:
            name = getattr(adapter, "name", "?")
            try:
                market = await adapter.find_market(home_team, away_team, kickoff)
            except Exception as e:  # noqa: BLE001 — isolate per-exchange failures
                log.warning("router_find_market_failed", exchange=str(name), error=str(e))
                continue
            if market is None:
                continue
            try:
                quote = await adapter.get_orderbook(market, outcome)
            except Exception as e:  # noqa: BLE001
                log.warning("router_orderbook_failed", exchange=str(name), error=str(e))
                continue
            if quote is not None:
                routes.append(RoutedQuote(adapter=adapter, market=market, quote=quote))

        if not routes:
            return None
        # Lowest price wins; tie-break on larger size (negate for ascending min).
        best = min(routes, key=lambda r: (r.quote.yes_price, -r.quote.yes_size))
        log.info(
            "router_best_quote",
            exchange=str(best.quote.exchange.value),
            outcome=best.quote.outcome.value,
            yes_price=round(best.quote.yes_price, 4),
            yes_size=round(best.quote.yes_size, 2),
            considered=len(routes),
        )
        return best

    async def find_best_quote(
        self,
        home_team: str,
        away_team: str,
        kickoff: datetime,
        outcome: Outcome,
    ) -> OrderbookQuote | None:
        """Best quote only (thin wrapper over :meth:`find_best_route`)."""
        route = await self.find_best_route(home_team, away_team, kickoff, outcome)
        return route.quote if route is not None else None
