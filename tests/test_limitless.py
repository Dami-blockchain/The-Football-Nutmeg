"""Tests for LimitlessAdapter (Phase 3). Offline: fake REST client."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.exchanges.base import ExchangeName, Outcome
from betbot.exchanges.limitless import LimitlessAdapter, LimitlessOrdersDisabled
from betbot.exchanges.limitless_client import LimitlessGeoBlockedError
from betbot.exchanges.matcher import TeamAliasResolver

KICKOFF = datetime(2026, 6, 7, 18, 0, tzinfo=timezone.utc)


def _group(with_draw=True, sport="football"):
    markets = [
        {"slug": "lie", "title": "Liechtenstein",
         "tokens": {"yes": "LIE_YES", "no": "LIE_NO"}, "prices": [0.49, 0.51]},
        {"slug": "cyp", "title": "Cyprus",
         "tokens": {"yes": "CYP_YES", "no": "CYP_NO"}, "prices": [0.49, 0.51]},
    ]
    if with_draw:
        markets.append({"slug": "draw", "title": "Draw",
                        "tokens": {"yes": "DRAW_YES", "no": "DRAW_NO"}, "prices": [0.55, 0.45]})
    return {
        "slug": "frnd-lie-cyp",
        "title": "Liechtenstein vs Cyprus",
        "metadata": {"homeTeam": "Liechtenstein", "awayTeam": "Cyprus", "sportType": sport},
        "venue": {"exchange": "0xEXCH", "adapter": "0xADAP"},
        "markets": markets,
    }


class FakeClient:
    def __init__(self, *, details=None, books=None, geo=False):
        self._details = details or {}
        self._books = books or {}
        self._geo = geo

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


def _adapter(details=None, books=None, geo=False, *, enable_orders=False, mode="paper"):
    return LimitlessAdapter(
        FakeClient(details=details, books=books, geo=geo),
        TeamAliasResolver(),
        enable_orders=enable_orders,
        mode=mode,
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
    assert ref.metadata["venue_exchange"] == "0xEXCH"


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
