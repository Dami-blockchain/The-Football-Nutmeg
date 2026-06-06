"""Tests for PolymarketAdapter (Phase 2). Offline: fake Gamma + fake CLOB."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.exchanges.base import ExchangeName, Outcome
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket import OrdersDisabled, PolymarketAdapter

KICKOFF = datetime(2026, 1, 20, 15, 0, tzinfo=timezone.utc)


# ---- fakes -----------------------------------------------------------
class FakeGamma:
    def __init__(self, events):
        self._events = events

    async def list_soccer_events(self, **kwargs):
        return self._events


class FakeClob:
    def __init__(self, book=None, order_resp=None):
        self._book = book
        self._order_resp = order_resp or {}
        self.order_calls = []

    def get_order_book(self, token_id):
        return self._book

    def create_and_post_market_order(self, order_args, *a, **k):
        self.order_calls.append(order_args)
        return self._order_resp


def _layout_b_event():
    return {
        "slug": "arsenal-vs-chelsea-2026-01-20",
        "id": "evt1",
        "title": "Arsenal vs. Chelsea",
        "markets": [
            {"question": "Will Arsenal win on 2026-01-20?",
             "outcomes": ["Yes", "No"], "clobTokenIds": ["HOME_YES", "HOME_NO"]},
            {"question": "Will Arsenal vs. Chelsea end in a draw?",
             "outcomes": ["Yes", "No"], "clobTokenIds": ["DRAW_YES", "DRAW_NO"]},
            {"question": "Will Chelsea win on 2026-01-20?",
             "outcomes": ["Yes", "No"], "clobTokenIds": ["AWAY_YES", "AWAY_NO"]},
        ],
    }


def _layout_a_event():
    return {
        "slug": "real-vs-barca",
        "id": "evt2",
        "title": "Real Madrid vs. Barcelona",
        "markets": [
            {"question": "Real Madrid vs. Barcelona",
             "outcomes": ["Real Madrid", "Draw", "Barcelona"],
             "clobTokenIds": ["T_HOME", "T_DRAW", "T_AWAY"]},
        ],
    }


def _adapter(events, clob=None, *, enable_orders=False, mode="paper"):
    return PolymarketAdapter(
        FakeGamma(events),
        TeamAliasResolver(),
        clob_client=clob,
        enable_orders=enable_orders,
        mode=mode,
    )


# ---- discovery / classification --------------------------------------
async def test_find_market_layout_b():
    a = _adapter([_layout_b_event()])
    ref = await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF)
    assert ref is not None
    assert ref.exchange is ExchangeName.POLYMARKET
    assert ref.metadata["layout"] == "B"
    toks = ref.metadata["outcome_tokens"]
    assert toks["HOME"] == "HOME_YES"
    assert toks["DRAW"] == "DRAW_YES"
    assert toks["AWAY"] == "AWAY_YES"


async def test_find_market_layout_a():
    a = _adapter([_layout_a_event()])
    ref = await a.find_market("Real Madrid CF", "FC Barcelona", KICKOFF)
    assert ref is not None
    assert ref.metadata["layout"] == "A"
    toks = ref.metadata["outcome_tokens"]
    assert toks == {"HOME": "T_HOME", "DRAW": "T_DRAW", "AWAY": "T_AWAY"}


async def test_find_market_no_match_returns_none():
    a = _adapter([_layout_b_event()])
    assert await a.find_market("Liverpool FC", "Everton FC", KICKOFF) is None


# ---- orderbook -------------------------------------------------------
async def test_get_orderbook_returns_best_ask_dict_shape():
    # The live CLOB get_order_book returns a DICT with string price/size.
    book = {"asks": [
        {"price": "0.58", "size": "50"},
        {"price": "0.55", "size": "120"},  # best (lowest) ask
    ], "bids": []}
    a = _adapter([_layout_b_event()], clob=FakeClob(book=book))
    ref = await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF)
    q = await a.get_orderbook(ref, Outcome.HOME)
    assert q is not None
    assert q.yes_price == 0.55
    assert q.yes_size == 120.0
    assert q.outcome is Outcome.HOME


async def test_best_ask_also_handles_object_shape():
    # Defensive: _best_ask also reads an object with an .asks attribute.
    book = type("B", (), {"asks": [{"price": 0.4, "size": 10},
                                   {"price": 0.3, "size": 5}]})()
    assert PolymarketAdapter._best_ask(book) == (0.3, 5.0)


async def test_best_ask_empty_book_none():
    assert PolymarketAdapter._best_ask({"asks": []}) is None


async def test_get_orderbook_unknown_outcome_token_none():
    a = _adapter([_layout_a_event()], clob=FakeClob(book=None))
    ref = await a.find_market("Real Madrid CF", "FC Barcelona", KICKOFF)
    # tamper: drop the token so the outcome has no id
    ref.metadata["outcome_tokens"].pop("HOME")
    assert await a.get_orderbook(ref, Outcome.HOME) is None


# ---- place_order double-gate (the safety-critical test) --------------
async def test_place_order_blocked_in_paper_mode():
    clob = FakeClob(order_resp={"orderID": "x"})
    a = _adapter([_layout_b_event()], clob=clob, enable_orders=True, mode="paper")
    ref = await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF)
    with pytest.raises(OrdersDisabled):
        await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert clob.order_calls == []  # CLOB never touched


async def test_place_order_blocked_without_enable_flag():
    clob = FakeClob(order_resp={"orderID": "x"})
    a = _adapter([_layout_b_event()], clob=clob, enable_orders=False, mode="live")
    ref = await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF)
    with pytest.raises(OrdersDisabled):
        await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert clob.order_calls == []


async def test_place_order_posts_when_double_gated_open():
    clob = FakeClob(order_resp={
        "orderID": "0xabc", "status": "matched",
        "filled_size": 10.0, "avg_price": 0.55,
    })
    a = _adapter([_layout_b_event()], clob=clob, enable_orders=True, mode="live")
    ref = await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF)
    result = await a.place_order(ref, Outcome.HOME, 10.0, 0.6)
    assert len(clob.order_calls) == 1
    assert result.order_id == "0xabc"
    assert result.status == "matched"
    assert result.exchange is ExchangeName.POLYMARKET
