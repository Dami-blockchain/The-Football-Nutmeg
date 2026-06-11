"""LimitlessAdapter — discovery + orderbook on Limitless (Base mainnet).

Implements the :class:`~betbot.exchanges.base.ExchangeAdapter` protocol. A
football fixture is a Limitless ``group`` market carrying
``metadata.{homeTeam, awayTeam, sportType}`` and child ``single`` binary
markets titled by outcome ("Liechtenstein", "Cyprus", "Draw"). We match the
fixture on the structured metadata (orientation-agnostic) and classify the
children with the shared :func:`classify_binary_outcome`, so Polymarket and
Limitless can't drift apart.

DRAW is effectively Polymarket-only: most Limitless matches ship no draw child,
so ``get_orderbook(..., DRAW)`` returns ``None`` and the router simply skips
Limitless for draw favourites — no special routing logic required.

Order placement is **double-gated** (``enable_orders=True`` AND ``mode=live``);
the live signer + ``POST /orders`` path itself lands in Phase 5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from betbot.exchanges.base import (
    ExchangeName,
    MarketRef,
    OrderbookQuote,
    OrderResult,
    Outcome,
)
from betbot.exchanges.limitless_client import (
    LimitlessClient,
    LimitlessError,
    LimitlessGeoBlockedError,
)
from betbot.exchanges.matcher import TeamAliasResolver, classify_binary_outcome
from betbot.logging import get_logger

log = get_logger(__name__)

# Orderbook level sizes are in 6-decimal collateral (USDC) units.
_SIZE_DECIMALS = 1_000_000


class LimitlessOrdersDisabled(LimitlessError):
    """place_order called while the live double-gate was not satisfied."""


class LimitlessAdapter:
    name = ExchangeName.LIMITLESS

    def __init__(
        self,
        client: LimitlessClient,
        resolver: TeamAliasResolver,
        *,
        enable_orders: bool = False,
        mode: str = "paper",
        private_key: str | None = None,
        fee_rate_bps: int = 0,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._enable_orders = enable_orders
        self._mode = mode
        self._private_key = private_key
        # feeRateBps must fall in the exchange's per-user "band" — a zero fee is
        # rejected ("feeRateBps[0] is out of user's band"). The exact required
        # value is not exposed by the public API; resolve it from the Limitless
        # docs/support before live orders, then pass it through here. Also note
        # the per-market minSize floor (100 USDC on WC group markets), which
        # rules out sub-$100 test orders.
        self._fee_rate_bps = fee_rate_bps
        self._detail_cache: dict[str, dict[str, Any]] = {}

    @property
    def orders_live(self) -> bool:
        return self._enable_orders and self._mode == "live"

    # ------------------------------------------------------------------
    async def _search_candidates(
        self, home_team: str, away_team: str
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for query in (home_team, away_team):
            try:
                results = await self._client.search_markets(query)
            except LimitlessGeoBlockedError:
                raise
            except LimitlessError as e:
                log.warning("limitless_search_failed", query=query, error=str(e))
                results = []
            for r in results:
                slug = r.get("slug")
                if slug and slug not in seen:
                    seen.add(slug)
                    out.append(r)
        return out

    async def _detail(self, slug: str) -> dict[str, Any]:
        if slug not in self._detail_cache:
            self._detail_cache[slug] = await self._client.get_market(slug)
        return self._detail_cache[slug]

    @staticmethod
    def _teams_match(
        resolver: TeamAliasResolver, m_home: str, m_away: str, home: str, away: str
    ) -> bool:
        """Orientation-agnostic: the two markets name the same pair of teams."""
        straight = resolver.same_team(m_home, home) and resolver.same_team(m_away, away)
        swapped = resolver.same_team(m_home, away) and resolver.same_team(m_away, home)
        return straight or swapped

    def _classify_children(
        self, detail: dict[str, Any], home_team: str, away_team: str
    ) -> dict[Outcome, dict[str, Any]]:
        mapping: dict[Outcome, dict[str, Any]] = {}
        for child in detail.get("markets") or []:
            label = child.get("title") or child.get("proxyTitle") or ""
            outcome = classify_binary_outcome(label, home_team, away_team, self._resolver)
            if outcome is None:
                continue
            tokens = child.get("tokens") or {}
            mapping[outcome] = {
                "slug": child.get("slug"),
                "yes_token": tokens.get("yes"),
                "prices": child.get("prices"),
            }
        return mapping

    async def find_market(
        self, home_team: str, away_team: str, kickoff: datetime
    ) -> MarketRef | None:
        for summary in await self._search_candidates(home_team, away_team):
            slug = summary.get("slug")
            if not slug:
                continue
            try:
                detail = await self._detail(slug)
            except LimitlessGeoBlockedError:
                raise
            except LimitlessError:
                continue
            md = detail.get("metadata") or {}
            if (md.get("sportType") or "").lower() != "football":
                continue
            m_home, m_away = md.get("homeTeam"), md.get("awayTeam")
            if not m_home or not m_away:
                continue
            if not self._teams_match(self._resolver, m_home, m_away, home_team, away_team):
                continue
            mapping = self._classify_children(detail, home_team, away_team)
            if Outcome.HOME not in mapping or Outcome.AWAY not in mapping:
                continue
            venue = detail.get("venue") or {}
            return MarketRef(
                exchange=ExchangeName.LIMITLESS,
                market_id=str(slug),
                title=detail.get("title") or f"{m_home} vs {m_away}",
                metadata={
                    "outcome_markets": {o.value: c for o, c in mapping.items()},
                    "venue_exchange": venue.get("exchange"),
                    "layout": "binary-group",
                },
            )
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _best_ask(book: dict[str, Any]) -> tuple[float, float] | None:
        asks = book.get("asks") if isinstance(book, dict) else None
        if not asks:
            return None
        best = min(asks, key=lambda lv: float(lv.get("price")))
        return float(best.get("price")), float(best.get("size")) / _SIZE_DECIMALS

    async def get_orderbook(
        self, market: MarketRef, outcome: Outcome
    ) -> OrderbookQuote | None:
        child = market.metadata.get("outcome_markets", {}).get(outcome.value)
        if not child:
            # e.g. DRAW with no draw child — the router then skips Limitless here.
            return None
        slug = child.get("slug")
        if not slug:
            return None
        try:
            book = await self._client.get_orderbook(slug)
        except LimitlessGeoBlockedError:
            raise
        except LimitlessError as e:
            log.warning("limitless_orderbook_failed", slug=slug, error=str(e))
            book = {}

        ask = self._best_ask(book)
        if ask is None:
            # Fall back to the market's quoted YES price (size unknown) if the
            # live book is empty — better a price than no quote on a thin venue.
            prices = child.get("prices") or []
            if prices:
                return self._quote(market, outcome, float(prices[0]), 0.0)
            return None
        price, size = ask
        return self._quote(market, outcome, price, size)

    @staticmethod
    def _quote(
        market: MarketRef, outcome: Outcome, price: float, size: float
    ) -> OrderbookQuote:
        return OrderbookQuote(
            exchange=ExchangeName.LIMITLESS,
            market_id=market.market_id,
            outcome=outcome,
            yes_price=price,
            yes_size=size,
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    async def place_order(
        self,
        market: MarketRef,
        outcome: Outcome,
        size_usd: float,
        max_price: float,
    ) -> OrderResult:
        if not self.orders_live:
            raise LimitlessOrdersDisabled(
                "place_order blocked: requires enable_orders=True AND mode=live "
                f"(have enable_orders={self._enable_orders}, mode={self._mode!r})"
            )
        if not self._private_key:
            raise LimitlessError("LIMITLESS_PRIVATE_KEY not configured")

        child = market.metadata.get("outcome_markets", {}).get(outcome.value)
        token_id = (child or {}).get("yes_token")
        market_slug = (child or {}).get("slug")
        venue_exchange = market.metadata.get("venue_exchange")
        if not token_id or not venue_exchange or not market_slug:
            raise LimitlessError("missing yes_token / venue_exchange / slug for live order")

        from eth_account import Account

        from betbot.exchanges import limitless_signing as sg

        # ownerId (numeric profile id) is required in the order request.
        try:
            owner_id = (await self._client.get_profile()).get("id")
        except Exception as e:  # noqa: BLE001
            raise LimitlessError(f"could not fetch Limitless profile/ownerId: {e}") from e

        maker = Account.from_key(self._private_key).address
        maker_units = sg.to_usdc_units(size_usd)
        # Limitless FOK semantics (validated 2026-06-11 against the live API):
        # takerAmount MUST be exactly 1 (a market-order sentinel — "spend
        # makerAmount of collateral, accept the best available fill"); max_price
        # bounds slippage on our side via the chosen market, not the order.
        order = sg.build_order(
            token_id=token_id, maker=maker,
            maker_amount_units=maker_units, taker_amount_units=1, side=sg.BUY,
            fee_rate_bps=self._fee_rate_bps,
        )
        signed = sg.sign_order(order, self._private_key, verifying_contract=venue_exchange)
        # Wire schema (validated against the live API): the signature is nested
        # INSIDE `order`; salt/tokenId/expiration are decimal strings (uint256s
        # exceed JS safe-int range); makerAmount/takerAmount/nonce/feeRateBps/
        # side/signatureType stay numeric.
        _str_fields = {"salt", "tokenId", "expiration"}
        order_payload = {k: (str(v) if k in _str_fields else v) for k, v in order.items()}
        order_payload["signature"] = signed["signature"]
        payload = {
            "order": order_payload,
            "orderType": "FOK",
            "marketSlug": market_slug,
            "ownerId": owner_id,
        }
        resp = await self._client.post_order(payload)
        return OrderResult(
            exchange=ExchangeName.LIMITLESS,
            order_id=str(resp.get("id") or resp.get("orderId") or ""),
            market_id=market.market_id,
            outcome=outcome,
            filled_size=float(resp.get("filledSize") or resp.get("filled") or 0.0),
            avg_price=float(resp.get("price") or max_price),
            status=str(resp.get("status") or "submitted"),
            raw_response=resp,
        )

    async def get_position(self, market: MarketRef) -> float:
        return 0.0

    async def claim_winnings(self, market: MarketRef) -> bool:
        return False
