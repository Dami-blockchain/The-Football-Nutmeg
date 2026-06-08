"""SX Bet adapter — read-only price discovery for cross-venue arbitrage.

SX Bet (https://api.sx.bet) is a crypto sports-betting ORDER BOOK. Soccer is
``sportId=2``; the 1X2 market (type 1) carries ``teamOneName`` (home),
``teamTwoName`` (away), ``gameTime``, and a ``marketHash``. Orders expose
``percentageOdds`` in sportx format (÷1e20 = implied probability) and
``isMakerBettingOutcomeOne`` (the side the maker backs). A taker backing the
*other* side pays ``1 - makerOdds``.

⚠️ UNVERIFIED: this is built to the docs but cannot be tested until SX lists
actual soccer *matches* (right now it only has WC outrights). SX soccer may be
effectively 2-way (draw = void/refund), in which case DRAW pricing returns None
and a true 3-way arb with Polymarket won't complete. Treat its prices as
indicative until verified against a live match. Read-only — no order placement.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from betbot.exchanges.base import (
    ExchangeName,
    MarketRef,
    OrderbookQuote,
    OrderResult,
    Outcome,
)
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.logging import get_logger

log = get_logger(__name__)

SXBET_BASE = "https://api.sx.bet"
SOCCER_SPORT_ID = 2
MARKET_TYPE_1X2 = 1
_ODDS_SCALE = 10**20


class SXBetError(RuntimeError):
    """An SX Bet request failed."""


class SXBetClient:
    def __init__(self, base_url: str = SXBET_BASE, *, timeout_seconds: float = 20.0,
                 client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def __aenter__(self) -> "SXBetClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns and not self._client.is_closed:
            await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            resp = await self._client.get(f"{self._base_url}{path}", params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise SXBetError(f"GET {path} failed: {e}") from e

    async def soccer_markets(self) -> list[dict[str, Any]]:
        data = await self._get("/markets/active",
                               params={"sportId": SOCCER_SPORT_ID, "type": MARKET_TYPE_1X2})
        markets = (data.get("data") or {}).get("markets") if isinstance(data, dict) else None
        return markets or []

    async def orders(self, market_hash: str) -> list[dict[str, Any]]:
        data = await self._get("/orders", params={"marketHashes": market_hash})
        if isinstance(data, dict):
            return data.get("data") or []
        return data or []


class SXBetAdapter:
    name = ExchangeName.SXBET

    def __init__(self, client: SXBetClient, resolver: TeamAliasResolver) -> None:
        self._client = client
        self._resolver = resolver
        self._cache: list[dict[str, Any]] | None = None

    async def _markets(self) -> list[dict[str, Any]]:
        if self._cache is None:
            self._cache = await self._client.soccer_markets()
        return self._cache

    async def find_market(self, home_team: str, away_team: str, kickoff: datetime) -> MarketRef | None:
        for m in await self._markets():
            h, a = m.get("teamOneName"), m.get("teamTwoName")
            if not h or not a:
                continue
            if self._resolver.same_team(h, home_team) and self._resolver.same_team(a, away_team):
                return MarketRef(
                    exchange=ExchangeName.SXBET,
                    market_id=str(m.get("marketHash") or ""),
                    title=f"{h} vs {a}",
                    metadata={"market_hash": m.get("marketHash"), "home": h, "away": a},
                )
        return None

    async def get_orderbook(self, market: MarketRef, outcome: Outcome) -> OrderbookQuote | None:
        # DRAW often unsupported (SX soccer may be 2-way) -> no quote, scan skips.
        if outcome is Outcome.DRAW:
            return None
        market_hash = market.metadata.get("market_hash")
        if not market_hash:
            return None
        try:
            orders = await self._client.orders(market_hash)
        except SXBetError as e:
            log.warning("sxbet_orders_failed", market=market_hash, error=str(e))
            return None
        # To BACK outcome1 (home) we take makers on outcome2 (isMakerBettingOutcomeOne=false);
        # taker price = 1 - makerOdds. Best = lowest taker price.
        want_maker_outcome_one = outcome is Outcome.AWAY  # to back AWAY, take makers on outcome1
        best_price = None
        best_size = 0.0
        for o in orders:
            try:
                if bool(o.get("isMakerBettingOutcomeOne")) != want_maker_outcome_one:
                    continue
                taker_price = 1.0 - int(o["percentageOdds"]) / _ODDS_SCALE
            except (KeyError, ValueError, TypeError):
                continue
            if 0.0 < taker_price < 1.0 and (best_price is None or taker_price < best_price):
                best_price = taker_price
                # remaining size if present (best-effort)
                try:
                    best_size = float(o.get("sizeRemaining") or o.get("size") or 0) / 1e6
                except (ValueError, TypeError):
                    best_size = 0.0
        if best_price is None:
            return None
        return OrderbookQuote(
            exchange=ExchangeName.SXBET, market_id=market.market_id, outcome=outcome,
            yes_price=best_price, yes_size=best_size, timestamp=datetime.now(timezone.utc),
        )

    async def place_order(self, market: MarketRef, outcome: Outcome,
                          size_usd: float, max_price: float) -> OrderResult:
        raise NotImplementedError("SX Bet is read-only (scan/arb detection only) for now")

    async def get_position(self, market: MarketRef) -> float:
        return 0.0

    async def claim_winnings(self, market: MarketRef) -> bool:
        return False
