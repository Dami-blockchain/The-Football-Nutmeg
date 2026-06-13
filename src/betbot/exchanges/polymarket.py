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
from betbot.exchanges.matcher import (
    TeamAliasResolver,
    classify_binary_outcome,
    is_match_result_market,
    normalize,
)
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
        signature_type: int = 0,
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
        # CLOB signature type: 0=EOA (default), 1=POLY_PROXY (email/Magic
        # deposit wallet), 2=POLY_GNOSIS_SAFE (browser proxy). A bare EOA on
        # Polymarket v2 is rejected with "maker address not allowed, please
        # use the deposit wallet flow"; the working route for an automated
        # wallet is a Magic/proxy wallet with signature_type=1 and funder set
        # to that proxy address (the EOA-deposit-wallet SDK path is bugged
        # upstream — Polymarket/py-clob-client-v2#70).
        self._signature_type = signature_type
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

            clob = ClobClient(
                host=self._clob_host,
                chain_id=self._chain_id,
                key=self._private_key,
                funder=self._funder,
                signature_type=self._signature_type,
            )
            # CLOB v2 needs L2 API credentials derived from the wallet
            # signature before any order endpoint will accept a request —
            # without this every order fails with "API Credentials are needed
            # to interact with this endpoint!". Derive (or create) them once
            # per client and attach. create_or_derive returns the existing
            # creds for this key if they exist, else mints new ones — safe and
            # idempotent per wallet.
            creds = clob.create_or_derive_api_key()
            clob.set_api_creds(creds)
            self._clob = clob
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
                    outcome = classify_binary_outcome(
                        str(label), home_team, away_team, self._resolver
                    )
                    if outcome is not None:
                        mapping[outcome] = token
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
            outcome = classify_binary_outcome(team, home_team, away_team, self._resolver)
            if outcome in (Outcome.HOME, Outcome.AWAY):
                mapping[outcome] = yes_token

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
        nh, na = normalize(home_team), normalize(away_team)
        candidates: list[tuple[int, MarketRef]] = []
        for e in events:
            mapping = self._classify_event(e, home_team, away_team)
            if mapping is None:
                continue
            # Market-identity guard: reject prop events (exact scoreline, spread,
            # totals, BTTS, cards, "to win by N", first goal, …) so a longshot
            # prop can't be routed as a fixture's HOME/AWAY/DRAW. Layout A carries
            # its own market question; check both the event title and that.
            title = e.get("title") or ""
            markets = e.get("markets") or []
            prop_titles = [title] + [
                str((m.get("question") or m.get("groupItemTitle") or "")) for m in markets
            ]
            # The classifier sees no structured threshold metadata from Gamma,
            # so it leans on the title patterns; pass each candidate string.
            if not all(is_match_result_market(None, t) for t in prop_titles):
                log.info(
                    "polymarket_market_not_match_result",
                    slug=e.get("slug") or e.get("id"), title=title,
                )
                continue
            ref = MarketRef(
                exchange=ExchangeName.POLYMARKET,
                market_id=str(e.get("slug") or e.get("id") or ""),
                title=title,
                metadata={
                    "outcome_tokens": {o.value: t for o, t in mapping.items()},
                    "layout": "A" if len(markets) == 1 else "B",
                    # Identity fields for the cross-venue arb compat check.
                    "home_team": home_team,
                    "away_team": away_team,
                    "fixture_id": e.get("fixtureId") or e.get("fixture_id"),
                    "market_type": "match_result",
                },
            )
            # Prefer the genuine head-to-head over the tournament-winner
            # outright (which carries a per-team winner market for BOTH teams,
            # so it classifies even though it isn't this fixture). A real
            # fixture names both teams in its title ("Brazil vs. Morocco") and
            # carries a DRAW outcome; the outright does neither.
            ntitle = normalize(title)
            score = 0
            if nh and nh in ntitle and na and na in ntitle:
                score += 10                      # both teams in the title = H2H
            if Outcome.DRAW in mapping:
                score += 1                       # a real 1X2 has a draw
            candidates.append((score, ref))

        if not candidates:
            return None
        # Highest score wins; ties keep discovery order (stable).
        best_score = max(s for s, _ in candidates)
        if best_score == 0:
            log.info(
                "polymarket_no_h2h_match",
                home=home_team, away=away_team,
                note="only non-H2H (likely outright) candidates — not routing",
            )
            return None
        return next(ref for s, ref in candidates if s == best_score)

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
