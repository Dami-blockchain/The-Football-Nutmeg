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

    def create_or_derive_api_key(self):
        self.derived = True
        return {"api_key": "k", "secret": "s", "passphrase": "p"}

    def set_api_creds(self, creds):
        self.creds_set = creds


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


async def test_find_market_one_team_only_rejected():
    # Only HOME matches (Arsenal); AWAY is a stranger -> no HOME+AWAY mapping.
    a = _adapter([_layout_b_event()])
    assert await a.find_market("Arsenal FC", "Liverpool FC", KICKOFF) is None


def _prop_event():
    # A scoreline/win-by prop event that still names the two teams — must be
    # excluded from 1X2 routing.
    return {
        "slug": "arsenal-chelsea-winby2",
        "id": "evtprop",
        "title": "Arsenal to win by 2+ goals vs. Chelsea",
        "markets": [
            {"question": "Will Arsenal win by 2+ goals?",
             "outcomes": ["Yes", "No"], "clobTokenIds": ["P_YES", "P_NO"]},
            {"question": "Will Chelsea win by 2+ goals?",
             "outcomes": ["Yes", "No"], "clobTokenIds": ["Q_YES", "Q_NO"]},
        ],
    }


async def test_find_market_prop_event_rejected():
    a = _adapter([_prop_event()])
    assert await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF) is None


async def test_find_market_attaches_identity_metadata():
    a = _adapter([_layout_b_event()])
    ref = await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF)
    assert ref.metadata["home_team"] == "Arsenal FC"
    assert ref.metadata["away_team"] == "Chelsea FC"
    assert ref.metadata["market_type"] == "match_result"


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


async def test_get_clob_derives_and_sets_api_creds(monkeypatch):
    """CLOB v2 rejects every order endpoint until L2 API creds are derived
    from the wallet and attached. _get_clob must do that on construction —
    otherwise live orders fail with 'API Credentials are needed'."""
    import sys
    import types

    calls: dict = {}

    class _FakeClobClient:
        def __init__(self, **kw):
            calls["init_kwargs"] = kw

        def create_or_derive_api_key(self):
            calls["derived"] = True
            return {"api_key": "k", "secret": "s", "passphrase": "p"}

        def set_api_creds(self, creds):
            calls["creds_set"] = creds

    fake_mod = types.ModuleType("py_clob_client_v2.client")
    fake_mod.ClobClient = _FakeClobClient
    monkeypatch.setitem(sys.modules, "py_clob_client_v2.client", fake_mod)

    # No clob injected → _get_clob constructs a real (here faked) client.
    a = _adapter([], enable_orders=True, mode="live")
    clob = a._get_clob()

    assert calls.get("derived") is True, "must derive API creds on construction"
    assert calls.get("creds_set") == {"api_key": "k", "secret": "s", "passphrase": "p"}
    # Cached — a second call must not re-derive.
    calls.clear()
    assert a._get_clob() is clob
    assert "derived" not in calls


async def test_get_clob_passes_signature_type_and_funder(monkeypatch):
    """The Magic/proxy deposit-wallet route needs signature_type + funder
    threaded into the CLOB client; a bare EOA (type 0) is rejected by
    Polymarket v2 with 'maker address not allowed'."""
    import sys
    import types

    calls: dict = {}

    class _FakeClobClient:
        def __init__(self, **kw):
            calls["init_kwargs"] = kw

        def create_or_derive_api_key(self):
            return {"api_key": "k"}

        def set_api_creds(self, creds):
            pass

    fake_mod = types.ModuleType("py_clob_client_v2.client")
    fake_mod.ClobClient = _FakeClobClient
    monkeypatch.setitem(sys.modules, "py_clob_client_v2.client", fake_mod)

    a = PolymarketAdapter(
        FakeGamma([]), TeamAliasResolver(),
        enable_orders=True, mode="live",
        private_key="0xabc", funder="0xPROXY", signature_type=1,
    )
    a._get_clob()
    kw = calls["init_kwargs"]
    assert kw["signature_type"] == 1
    assert kw["funder"] == "0xPROXY"
    assert kw["key"] == "0xabc"
