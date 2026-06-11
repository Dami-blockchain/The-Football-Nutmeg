"""Tests for LimitlessAdapter (Phase 3 + live-order fixes). Offline: fake REST client."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.exchanges.base import ExchangeName, Outcome
from betbot.exchanges.limitless import LimitlessAdapter, LimitlessOrdersDisabled
from betbot.exchanges.limitless_client import (
    LimitlessAuthError,
    LimitlessError,
    LimitlessGeoBlockedError,
)
from betbot.exchanges.matcher import TeamAliasResolver

KICKOFF = datetime(2026, 6, 7, 18, 0, tzinfo=timezone.utc)

# settings.minSize is in OUTCOME-TOKEN SHARES with 6 decimals: 100 shares.
MIN_SIZE_100_SHARES = "100000000"
# Per-market venue.exchange (EIP-712 verifyingContract) — must be a real address
# shape because order signing ABI-encodes it.
VENUE_EXCH = "0x05c748E2f4DcDe0ec9Fa8DDc40DE6b867f923fa5"


def _group(with_draw=True, sport="football", min_size=None):
    # token ids are uint256s on the wire — numeric strings, like the live API.
    markets = [
        {"slug": "lie", "title": "Liechtenstein",
         "tokens": {"yes": "1111", "no": "1112"}, "prices": [0.49, 0.51]},
        {"slug": "cyp", "title": "Cyprus",
         "tokens": {"yes": "2221", "no": "2222"}, "prices": [0.49, 0.51]},
    ]
    if min_size is not None:
        for m in markets:
            m["settings"] = {"minSize": min_size}
    if with_draw:
        markets.append({"slug": "draw", "title": "Draw",
                        "tokens": {"yes": "3331", "no": "3332"}, "prices": [0.55, 0.45]})
    return {
        "slug": "frnd-lie-cyp",
        "title": "Liechtenstein vs Cyprus",
        "metadata": {"homeTeam": "Liechtenstein", "awayTeam": "Cyprus", "sportType": sport},
        "venue": {"exchange": VENUE_EXCH, "adapter": "0xADAP"},
        "markets": markets,
    }


class FakeClient:
    def __init__(self, *, details=None, books=None, geo=False, profile=None,
                 profile_error=None):
        self._details = details or {}
        self._books = books or {}
        self._geo = geo
        self._profile = profile if profile is not None else {
            "id": 777, "rank": {"feeRateBps": 100},
        }
        self._profile_error = profile_error
        self.posted: list[dict] = []

    async def search_markets(self, query):
        if self._geo:
            raise LimitlessGeoBlockedError("geo")
        return [{"slug": s} for s in self._details]

    async def get_market(self, slug):
        if self._geo:
            raise LimitlessGeoBlockedError("geo")
        return self._details.get(slug, {})

    async def get_orderbook(self, slug):
        if self._geo:
            raise LimitlessGeoBlockedError("geo")
        return self._books.get(slug, {})

    async def get_profile(self):
        if self._profile_error is not None:
            raise self._profile_error
        return self._profile

    async def post_order(self, payload):
        self.posted.append(payload)
        return {"id": "ord-1", "status": "matched", "price": 0.5}


def _adapter(details=None, books=None, geo=False, *, enable_orders=False,
             mode="paper", client=None, **kwargs):
    return LimitlessAdapter(
        client or FakeClient(details=details, books=books, geo=geo),
        TeamAliasResolver(),
        enable_orders=enable_orders,
        mode=mode,
        **kwargs,
    )


# ---- discovery -------------------------------------------------------
async def test_find_market_classifies_three_outcomes():
    a = _adapter({"frnd-lie-cyp": _group()})
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    assert ref is not None
    assert ref.exchange is ExchangeName.LIMITLESS
    om = ref.metadata["outcome_markets"]
    assert om["HOME"]["slug"] == "lie"
    assert om["AWAY"]["slug"] == "cyp"
    assert om["DRAW"]["slug"] == "draw"
    assert ref.metadata["venue_exchange"] == VENUE_EXCH


async def test_find_market_orientation_agnostic():
    # We pass teams swapped vs the market's home/away; children still map right.
    a = _adapter({"frnd-lie-cyp": _group()})
    ref = await a.find_market("Cyprus", "Liechtenstein", KICKOFF)
    assert ref is not None
    om = ref.metadata["outcome_markets"]
    assert om["HOME"]["slug"] == "cyp"   # our home is Cyprus now
    assert om["AWAY"]["slug"] == "lie"


async def test_find_market_non_football_skipped():
    a = _adapter({"frnd-lie-cyp": _group(sport="esports")})
    assert await a.find_market("Liechtenstein", "Cyprus", KICKOFF) is None


async def test_find_market_no_match():
    a = _adapter({"frnd-lie-cyp": _group()})
    assert await a.find_market("Arsenal", "Chelsea", KICKOFF) is None


# ---- DRAW is effectively Polymarket-only -----------------------------
async def test_draw_absent_orderbook_none():
    a = _adapter({"frnd-lie-cyp": _group(with_draw=False)},
                 books={"lie": {"asks": [{"price": 0.5, "size": 1_000_000}]}})
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    assert "DRAW" not in ref.metadata["outcome_markets"]
    # router asks for DRAW -> no child -> None -> router skips Limitless for draw
    assert await a.get_orderbook(ref, Outcome.DRAW) is None


# ---- orderbook -------------------------------------------------------
async def test_get_orderbook_best_ask_and_size_scaling():
    books = {"lie": {"asks": [
        {"price": 0.60, "size": 20_000_000, "side": "SELL"},
        {"price": 0.55, "size": 10_000_000, "side": "SELL"},  # best ask
    ]}}
    a = _adapter({"frnd-lie-cyp": _group()}, books=books)
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    q = await a.get_orderbook(ref, Outcome.HOME)
    assert q.yes_price == 0.55
    assert q.yes_size == 10.0  # 10_000_000 / 1e6


async def test_get_orderbook_falls_back_to_quoted_price_when_book_empty():
    a = _adapter({"frnd-lie-cyp": _group()}, books={"lie": {"asks": []}})
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    q = await a.get_orderbook(ref, Outcome.HOME)
    assert q is not None and q.yes_price == 0.49 and q.yes_size == 0.0


# ---- geo-block + order gate ------------------------------------------
async def test_geo_block_propagates():
    a = _adapter({"frnd-lie-cyp": _group()}, geo=True)
    with pytest.raises(LimitlessGeoBlockedError):
        await a.find_market("Liechtenstein", "Cyprus", KICKOFF)


async def test_place_order_double_gated():
    a = _adapter({"frnd-lie-cyp": _group()}, enable_orders=True, mode="paper")
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    with pytest.raises(LimitlessOrdersDisabled):
        await a.place_order(ref, Outcome.HOME, 10.0, 0.6)


# ---- per-user fee rate (GET /profiles/me -> rank.feeRateBps) ----------
async def test_fetch_user_fee_rate_bps():
    a = _adapter(client=FakeClient(profile={"id": 1, "rank": {"feeRateBps": 175}}))
    assert await a.fetch_user_fee_rate_bps() == 175


async def test_fetch_user_fee_rate_auth_error_propagates():
    a = _adapter(client=FakeClient(profile_error=LimitlessAuthError("401")))
    with pytest.raises(LimitlessAuthError):
        await a.fetch_user_fee_rate_bps()


async def test_fetch_user_fee_rate_missing_is_clear_error():
    a = _adapter(client=FakeClient(profile={"id": 1, "rank": {}}))
    with pytest.raises(LimitlessError, match="feeRateBps"):
        await a.fetch_user_fee_rate_bps()


# ---- minSize is in shares, not USDC ----------------------------------
async def test_min_shares_exposed_from_market_settings():
    a = _adapter({"frnd-lie-cyp": _group(min_size=MIN_SIZE_100_SHARES)})
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    assert a.min_shares_for(ref, Outcome.HOME) == 100.0
    assert a.min_shares_for(ref, Outcome.DRAW) == 0.0  # draw child has no settings


# ---- live order construction (signed; fake client captures payload) ---
def _live_adapter(client, **kwargs):
    from eth_account import Account

    acct = Account.create()
    a = _adapter(client=client, enable_orders=True, mode="live",
                 private_key=acct.key.hex(), **kwargs)
    return a, acct


async def test_place_order_uses_profile_fee_and_per_market_exchange():
    client = FakeClient(details={"frnd-lie-cyp": _group()},
                        profile={"id": 42, "rank": {"feeRateBps": 100}})
    a, acct = _live_adapter(client)  # fee_rate_bps default 0 -> fetch at runtime
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    res = await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert res.order_id == "ord-1"
    [payload] = client.posted
    assert payload["ownerId"] == 42
    assert payload["orderType"] == "FOK"
    assert payload["marketSlug"] == "lie"
    assert payload["clientOrderId"].startswith("tfsm-")
    order = payload["order"]
    assert order["feeRateBps"] == 100          # from rank.feeRateBps
    assert order["makerAmount"] == 10_000_000  # $10 in 6-decimal USDC
    assert order["takerAmount"] == 1           # FOK market-buy sentinel
    assert order["side"] == 0                  # BUY
    # Signed against the PER-MARKET venue.exchange.
    from betbot.exchanges import limitless_signing as sg

    unsigned = {k: v for k, v in order.items() if k != "signature"}
    for f in ("salt", "tokenId", "expiration"):
        unsigned[f] = int(unsigned[f])
    recovered = sg.recover_signer(unsigned, order["signature"], verifying_contract=VENUE_EXCH)
    assert recovered.lower() == acct.address.lower()


async def test_place_order_config_fee_override_wins():
    client = FakeClient(details={"frnd-lie-cyp": _group()},
                        profile={"id": 42, "rank": {"feeRateBps": 100}})
    a, _ = _live_adapter(client, fee_rate_bps=250)
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert client.posted[0]["order"]["feeRateBps"] == 250


async def test_place_order_auth_error_surfaces():
    client = FakeClient(details={"frnd-lie-cyp": _group()},
                        profile_error=LimitlessAuthError("key revoked"))
    a, _ = _live_adapter(client)
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    with pytest.raises(LimitlessAuthError):
        await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert client.posted == []  # never reached POST /orders


async def test_place_order_bumps_spend_to_min_shares_cost():
    # 100-share floor @ max_price 0.6 -> min cost $60; we asked for $10.
    client = FakeClient(details={"frnd-lie-cyp": _group(min_size=MIN_SIZE_100_SHARES)})
    a, _ = _live_adapter(client, max_order_usd=100.0)
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert client.posted[0]["order"]["makerAmount"] == 60_000_000  # $60


async def test_place_order_rejects_when_min_cost_exceeds_cap():
    # min cost $60 > $50 cap -> clear rejection, nothing posted.
    client = FakeClient(details={"frnd-lie-cyp": _group(min_size=MIN_SIZE_100_SHARES)})
    a, _ = _live_adapter(client, max_order_usd=50.0)
    ref = await a.find_market("Liechtenstein", "Cyprus", KICKOFF)
    with pytest.raises(LimitlessError, match="exceeds cap"):
        await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert client.posted == []


# ---- client-level auth behaviour (httpx mocked) ------------------------
def _http_client(handler, **kwargs):
    import httpx

    from betbot.exchanges.limitless_client import LimitlessClient

    transport = httpx.MockTransport(handler)
    return LimitlessClient(
        client=httpx.AsyncClient(transport=transport), **kwargs
    )


async def test_client_get_profile_401_raises_auth_error():
    import httpx

    def handler(request):
        return httpx.Response(401, json={"message": "invalid api key"})

    c = _http_client(handler, api_key="dead-key")
    with pytest.raises(LimitlessAuthError, match="invalid/revoked"):
        await c.get_profile()
    await c.close()


async def test_client_sends_x_api_key_when_no_secret():
    import httpx

    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"id": 1, "rank": {"feeRateBps": 100}})

    c = _http_client(handler, api_key="k123")
    profile = await c.get_profile()
    assert profile["rank"]["feeRateBps"] == 100
    assert seen.get("x-api-key") == "k123"
    await c.close()


async def test_client_sends_hmac_headers_when_secret_set():
    import httpx

    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"id": 1})

    c = _http_client(handler, api_key="k123", api_secret="c2VjcmV0")
    await c.get_profile()
    assert seen.get("lmts-api-key") == "k123"
    assert "lmts-timestamp" in seen and "lmts-signature" in seen
    await c.close()
