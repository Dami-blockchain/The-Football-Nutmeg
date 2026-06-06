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
from datetime import datetime

from betbot.exchanges.base import ExchangeAdapter, OrderbookQuote, Outcome
from betbot.logging import get_logger

log = get_logger(__name__)


class ExchangeRouter:
    def __init__(self, adapters: Iterable[ExchangeAdapter]) -> None:
        self._adapters = list(adapters)

    @property
    def adapters(self) -> list[ExchangeAdapter]:
        return self._adapters

    async def find_best_quote(
        self,
        home_team: str,
        away_team: str,
        kickoff: datetime,
        outcome: Outcome,
    ) -> OrderbookQuote | None:
        """Best quote for ``outcome`` across all exchanges, or ``None``."""
        quotes: list[OrderbookQuote] = []
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
                quotes.append(quote)

        if not quotes:
            return None
        # Lowest price wins; tie-break on larger size (negate for ascending min).
        best = min(quotes, key=lambda q: (q.yes_price, -q.yes_size))
        log.info(
            "router_best_quote",
            exchange=str(best.exchange.value),
            outcome=best.outcome.value,
            yes_price=round(best.yes_price, 4),
            yes_size=round(best.yes_size, 2),
            considered=len(quotes),
        )
        return best
