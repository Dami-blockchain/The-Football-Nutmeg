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
    LimitlessAuthError,
    LimitlessClient,
    LimitlessError,
    LimitlessGeoBlockedError,
)
from betbot.exchanges.matcher import (
    TeamAliasResolver,
    classify_binary_outcome,
    is_match_result_market,
)
from betbot.logging import get_logger

log = get_logger(__name__)

# Orderbook level sizes and settings.minSize are in OUTCOME-TOKEN SHARES with
# 6 decimals ("100000000" = 100 shares), NOT USDC. The dollar cost of an order
# of N shares at price p is N * p. (Verified 2026-06-11 against the live API.)
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
        max_order_usd: float | None = None,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._enable_orders = enable_orders
        self._mode = mode
        self._private_key = private_key
        # feeRateBps must fall in the exchange's per-user "band" — a zero fee is
        # rejected ("feeRateBps[0] is out of user's band"). The authoritative
        # value is GET /profiles/me -> rank.feeRateBps ("use this value when
        # constructing signed orders"); we fetch it at order-build time. A
        # non-zero LIMITLESS_FEE_RATE_BPS config override takes precedence.
        # Per-market settings.minSize is a floor in OUTCOME-TOKEN SHARES (6
        # decimals) — e.g. "100000000" = 100 shares, costing 100 * price USDC.
        # (An older comment here misread it as a 100 USDC floor.)
        self._fee_rate_bps = fee_rate_bps
        self._max_order_usd = max_order_usd
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

    @staticmethod
    def _min_shares_of(child: dict[str, Any], detail: dict[str, Any]) -> float:
        """Per-market minimum order size in OUTCOME-TOKEN SHARES.

        ``settings.minSize`` is a 6-decimal share count ("100000000" = 100
        shares); the minimum order's dollar cost is ``min_shares * price``.
        Falls back from the child market to the group, 0.0 if absent.
        """
        raw = (child.get("settings") or {}).get("minSize")
        if raw is None:
            raw = (detail.get("settings") or {}).get("minSize")
        try:
            return float(raw) / _SIZE_DECIMALS if raw is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

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
                "min_shares": self._min_shares_of(child, detail),
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
            # Market-identity guard: only a real 1X2 match-result market may be
            # routed as a fixture. Exclude props (spreads/totals/cards/scoreline
            # …) even when they carry homeTeam/awayTeam metadata.
            title = detail.get("title") or f"{m_home} vs {m_away}"
            if not is_match_result_market(md, title):
                log.info(
                    "limitless_market_not_match_result",
                    slug=slug, title=title,
                    market_type=md.get("marketType"),
                )
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
                title=title,
                metadata={
                    "outcome_markets": {o.value: c for o, c in mapping.items()},
                    "venue_exchange": venue.get("exchange"),
                    "layout": "binary-group",
                    # Identity fields for the cross-venue arb compat check.
                    "home_team": m_home,
                    "away_team": m_away,
                    "fixture_id": md.get("fixtureId"),
                    "market_type": "match_result",
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
    # Per-market sizing + per-user fee rate (live orders)
    # ------------------------------------------------------------------
    @staticmethod
    def min_shares_for(market: MarketRef, outcome: Outcome) -> float:
        """Market minimum order size in outcome-token shares (0.0 if unknown).

        Min order dollar cost = ``min_shares_for(...) * price``.
        """
        child = market.metadata.get("outcome_markets", {}).get(outcome.value) or {}
        return float(child.get("min_shares") or 0.0)

    async def fetch_user_fee_rate_bps(self) -> int:
        """The user's fee rate from ``GET /profiles/me -> rank.feeRateBps``.

        This is the value to embed in signed orders. Raises
        :class:`LimitlessAuthError` when the API key is invalid/revoked (401),
        so callers can degrade gracefully with a clear operator message.
        """
        profile = await self._client.get_profile()
        return self._fee_from_profile(profile)

    @staticmethod
    def _fee_from_profile(profile: dict[str, Any]) -> int:
        fee = ((profile or {}).get("rank") or {}).get("feeRateBps")
        if fee is None:
            raise LimitlessError(
                "Limitless profile has no rank.feeRateBps — cannot build a signed "
                "order (set LIMITLESS_FEE_RATE_BPS to override)"
            )
        return int(fee)

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
        # EIP-712 verifyingContract (and approval spender) is PER-MARKET — read
        # from the market payload (venue.exchange), never hardcoded.
        venue_exchange = market.metadata.get("venue_exchange")
        if not token_id or not venue_exchange or not market_slug:
            raise LimitlessError("missing yes_token / venue_exchange / slug for live order")

        from eth_account import Account

        from betbot.exchanges import limitless_signing as sg

        # ownerId (numeric profile id) is required in the order request; the
        # same response carries rank.feeRateBps. A 401 here means the API key
        # is invalid/revoked — surfaced as LimitlessAuthError, not swallowed.
        try:
            profile = await self._client.get_profile()
        except (LimitlessAuthError, LimitlessGeoBlockedError):
            raise
        except Exception as e:  # noqa: BLE001
            raise LimitlessError(f"could not fetch Limitless profile/ownerId: {e}") from e
        owner_id = profile.get("id")

        # feeRateBps: config override (non-zero LIMITLESS_FEE_RATE_BPS) wins,
        # else the per-user value the API tells us to use.
        fee_rate_bps = self._fee_rate_bps or self._fee_from_profile(profile)

        # minSize: the market floor is in shares; its dollar cost at our price
        # cap is min_shares * max_price. Bump the spend up to that floor, but
        # never beyond the configured cap — reject loudly instead.
        spend_usd = size_usd
        min_shares = self.min_shares_for(market, outcome)
        min_cost_usd = min_shares * max_price
        if spend_usd < min_cost_usd:
            log.info(
                "limitless_order_bumped_to_min",
                slug=market_slug,
                min_shares=min_shares,
                min_cost_usd=round(min_cost_usd, 2),
                requested_usd=round(size_usd, 2),
            )
            spend_usd = min_cost_usd
        if self._max_order_usd is not None and spend_usd > self._max_order_usd:
            log.error(
                "limitless_order_rejected_min_size_over_cap",
                slug=market_slug,
                min_shares=min_shares,
                min_cost_usd=round(min_cost_usd, 2),
                cap_usd=self._max_order_usd,
                note="market minimum order costs more than the per-order cap",
            )
            raise LimitlessError(
                f"min order cost ${min_cost_usd:.2f} ({min_shares:g} shares @ "
                f"{max_price:.3f}) exceeds cap ${self._max_order_usd:.2f}"
            )

        maker = Account.from_key(self._private_key).address
        maker_units = sg.to_usdc_units(spend_usd)
        # Limitless FOK semantics (validated 2026-06-11 against the live API):
        # takerAmount MUST be exactly 1 (a market-order sentinel — "spend
        # makerAmount of collateral, accept the best available fill"); max_price
        # bounds slippage on our side via the chosen market, not the order.
        order = sg.build_order(
            token_id=token_id, maker=maker,
            maker_amount_units=maker_units, taker_amount_units=1, side=sg.BUY,
            fee_rate_bps=fee_rate_bps,
        )
        signed = sg.sign_order(order, self._private_key, verifying_contract=venue_exchange)
        # Wire schema (validated against the live API): the signature is nested
        # INSIDE `order`; salt/tokenId/expiration are decimal strings (uint256s
        # exceed JS safe-int range); makerAmount/takerAmount/nonce/feeRateBps/
        # side/signatureType stay numeric.
        _str_fields = {"salt", "tokenId", "expiration"}
        order_payload = {k: (str(v) if k in _str_fields else v) for k, v in order.items()}
        order_payload["signature"] = signed["signature"]
        import uuid

        payload = {
            "order": order_payload,
            "orderType": "FOK",
            "marketSlug": market_slug,
            "ownerId": owner_id,
            # Idempotency key: a resubmit of the same attempt can't double-fill.
            "clientOrderId": f"tfsm-{uuid.uuid4().hex}",
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
