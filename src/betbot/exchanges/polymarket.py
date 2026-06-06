"""PolymarketAdapter — discovery via Gamma, orderbook + orders via the CLOB v2.

Implements the :class:`~betbot.exchanges.base.ExchangeAdapter` protocol for
Polymarket. Two market layouts are handled:

* **Layout B (common for football):** one event per match with three binary
  YES/NO markets — "Will <home> win?", "Will it end in a draw?", "Will <away>
  win?". The YES token of each is the thing you buy to back that outcome.
* **Layout A:** a single market carrying three outcomes + three token ids.

Order placement is **double-gated**: it requires BOTH ``enable_orders=True`` at
construction AND ``mode == "live"``. Either missing → :class:`OrdersDisabled`
is raised before any CLOB call. (Polymarket migrated to CTF Exchange **V2** on
2026-04-22; this uses ``py-clob-client-v2``. Real signed-order wiring lands in
Phase 5; the gate and order construction live here.)
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from betbot.exchanges.base import (
    ExchangeName,
    MarketRef,
    OrderbookQuote,
    OrderResult,
    Outcome,
)
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket_gamma import GammaClient
from betbot.logging import get_logger

log = get_logger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137

# "Will Arsenal win on 2026-01-19?" -> "Arsenal"
_WIN_RE = re.compile(r"will\s+(?P<team>.+?)\s+win\b", re.IGNORECASE)
_DRAW_RE = re.compile(r"\bdraw\b", re.IGNORECASE)


class PolymarketError(RuntimeError):
    """A Polymarket operation failed."""


class OrdersDisabled(PolymarketError):
    """place_order was called while the live double-gate was not satisfied."""


def _extract_win_team(question: str) -> str | None:
    m = _WIN_RE.search(question or "")
    return m.group("team").strip() if m else None


class PolymarketAdapter:
    """ExchangeAdapter for Polymarket. Discovery + orderbook now; orders gated."""

    name = ExchangeName.POLYMARKET

    def __init__(
        self,
        gamma: GammaClient,
        resolver: TeamAliasResolver,
        *,
        enable_orders: bool = False,
        mode: str = "paper",
        clob_host: str = CLOB_HOST,
        chain_id: int = POLYGON_CHAIN_ID,
        private_key: str | None = None,
        funder: str | None = None,
        clob_client: Any | None = None,
    ) -> None:
        self._gamma = gamma
        self._resolver = resolver
        self._enable_orders = enable_orders
        self._mode = mode
        self._clob_host = clob_host
        self._chain_id = chain_id
        self._private_key = private_key
        self._funder = funder
        self._clob = clob_client  # injectable for tests; otherwise lazy-built
        self._events_cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    @property
    def orders_live(self) -> bool:
        """The double-gate: both flags must be set for a real order to post."""
        return self._enable_orders and self._mode == "live"

    def _get_clob(self) -> Any:
        if self._clob is None:
            # Imported lazily so the package import doesn't require the heavy
            # CLOB/web3 stack unless an adapter actually talks to the CLOB.
            from py_clob_client_v2.client import ClobClient

            self._clob = ClobClient(
                host=self._clob_host,
                chain_id=self._chain_id,
                key=self._private_key,
                funder=self._funder,
            )
        return self._clob

    # ------------------------------------------------------------------
    # Discovery / classification
    # ------------------------------------------------------------------
    def _classify_event(
        self, event: dict[str, Any], home_team: str, away_team: str
    ) -> dict[Outcome, str] | None:
        """Map an event's markets to {Outcome: yes_token_id}, or None.

        Returns a mapping only when at least HOME and AWAY can be matched to our
        two teams (DRAW is optional — a few events omit it). Handles Layout B
        (3 binary markets) and Layout A (1 market, 3 outcomes).
        """
        markets = event.get("markets") or []

        # ---- Layout A: a single market with three outcomes/token ids ----
        if len(markets) == 1:
            m = markets[0]
            outcomes = m.get("outcomes") or []
            tokens = m.get("clobTokenIds") or []
            if len(outcomes) == 3 and len(tokens) == 3:
                mapping: dict[Outcome, str] = {}
                for label, token in zip(outcomes, tokens):
                    if _DRAW_RE.search(str(label)):
                        mapping[Outcome.DRAW] = token
                    elif self._resolver.same_team(str(label), home_team):
                        mapping[Outcome.HOME] = token
                    elif self._resolver.same_team(str(label), away_team):
                        mapping[Outcome.AWAY] = token
                if Outcome.HOME in mapping and Outcome.AWAY in mapping:
                    return mapping
            return None

        # ---- Layout B: three binary YES/NO markets ----------------------
        mapping = {}
        for m in markets:
            tokens = m.get("clobTokenIds") or []
            if not tokens:
                continue
            yes_token = tokens[0]  # outcomes are ["Yes","No"]; YES is index 0
            question = m.get("question") or ""
            if _DRAW_RE.search(question):
                mapping[Outcome.DRAW] = yes_token
                continue
            team = _extract_win_team(question)
            if team is None:
                continue
            if self._resolver.same_team(team, home_team):
                mapping[Outcome.HOME] = yes_token
            elif self._resolver.same_team(team, away_team):
                mapping[Outcome.AWAY] = yes_token

        if Outcome.HOME in mapping and Outcome.AWAY in mapping:
            return mapping
        return None

    async def _soccer_events(self) -> list[dict[str, Any]]:
        if self._events_cache is None:
            self._events_cache = await self._gamma.list_soccer_events()
        return self._events_cache

    async def find_market(
        self, home_team: str, away_team: str, kickoff: datetime
    ) -> MarketRef | None:
        events = await self._soccer_events()
        for e in events:
            mapping = self._classify_event(e, home_team, away_team)
            if mapping is None:
                continue
            market_id = e.get("slug") or e.get("id") or ""
            return MarketRef(
                exchange=ExchangeName.POLYMARKET,
                market_id=str(market_id),
                title=e.get("title") or "",
                metadata={
                    "outcome_tokens": {o.value: t for o, t in mapping.items()},
                    "layout": "A" if len(e.get("markets") or []) == 1 else "B",
                },
            )
        return None

    # ------------------------------------------------------------------
    # Orderbook
    # ------------------------------------------------------------------
    @staticmethod
    def _best_ask(book: Any) -> tuple[float, float] | None:
        """Lowest-priced ask = cost to BUY a YES share. Returns (price, size)."""
        asks = getattr(book, "asks", None)
        if asks is None and isinstance(book, dict):
            asks = book.get("asks")
        if not asks:
            return None

        def _pf(level: Any, key: str) -> float:
            if isinstance(level, dict):
                return float(level.get(key))
            return float(getattr(level, key))

        best = min(asks, key=lambda lv: _pf(lv, "price"))
        return _pf(best, "price"), _pf(best, "size")

    async def get_orderbook(
        self, market: MarketRef, outcome: Outcome
    ) -> OrderbookQuote | None:
        token_id = market.metadata.get("outcome_tokens", {}).get(outcome.value)
        if not token_id:
            return None
        clob = self._get_clob()
        try:
            book = await asyncio.to_thread(clob.get_order_book, token_id)
        except Exception as e:  # noqa: BLE001 — SDK raises varied errors
            log.warning("polymarket_orderbook_failed", token=token_id, error=str(e))
            return None
        ask = self._best_ask(book)
        if ask is None:
            return None
        price, size = ask
        return OrderbookQuote(
            exchange=ExchangeName.POLYMARKET,
            market_id=market.market_id,
            outcome=outcome,
            yes_price=price,
            yes_size=size,
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Orders (double-gated)
    # ------------------------------------------------------------------
    async def place_order(
        self,
        market: MarketRef,
        outcome: Outcome,
        size_usd: float,
        max_price: float,
    ) -> OrderResult:
        if not self.orders_live:
            raise OrdersDisabled(
                "place_order blocked: requires enable_orders=True AND mode=live "
                f"(have enable_orders={self._enable_orders}, mode={self._mode!r})"
            )
        token_id = market.metadata.get("outcome_tokens", {}).get(outcome.value)
        if not token_id:
            raise PolymarketError(f"no token id for {outcome} on {market.market_id}")

        from py_clob_client_v2.clob_types import MarketOrderArgsV2

        order_args = MarketOrderArgsV2(
            token_id=token_id,
            amount=size_usd,
            side="BUY",
            price=max_price,
        )
        clob = self._get_clob()
        resp = await asyncio.to_thread(clob.create_and_post_market_order, order_args)
        resp = resp if isinstance(resp, dict) else {"raw": resp}
        return OrderResult(
            exchange=ExchangeName.POLYMARKET,
            order_id=str(resp.get("orderID") or resp.get("orderId") or ""),
            market_id=market.market_id,
            outcome=outcome,
            filled_size=float(resp.get("filled_size") or 0.0),
            avg_price=float(resp.get("avg_price") or max_price),
            status=str(resp.get("status") or "submitted"),
            raw_response=resp,
        )

    async def get_position(self, market: MarketRef) -> float:
        # Positions are tracked in our own DB for paper mode; real on-chain
        # position reads arrive with live trading (Phase 5).
        return 0.0

    async def claim_winnings(self, market: MarketRef) -> bool:
        # Redemptions are a live-trading concern (Phase 5).
        return False
