"""Tests for PolymarketAdapter (Phase 2). Offline: fake Gamma + fake CLOB."""

from __future__ import annotations

from datetime import datetime, timezone

from betbot.exchanges.base import ExchangeName, Outcome
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket import PolymarketAdapter

KICKOFF = datetime(2026, 1, 20, 15, 0, tzinfo=timezone.utc)


# ---- fakes -----------------------------------------------------------
class FakeGamma:
    def __init__(self, events):
        self._events = events

    async def list_soccer_events(self, **kwargs):
        return self._events


class FakeClob:
    def __init__(self, book=None):
        self._book = book

    def get_order_book(self, token_id):
        return self._book


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


def _adapter(events, clob=None):
    return PolymarketAdapter(
        FakeGamma(events),
        TeamAliasResolver(),
        clob_client=clob,
    )


def _outright_winner_event():
    """The tournament-winner outright: a per-team winner market for many
    teams. Both queried teams appear, so it classifies — but it is NOT the
    fixture, and its title names neither team and it has no draw."""
    return {
        "slug": "world-cup-winner",
        "id": "wcw",
        "title": "World Cup Winner",
        "markets": [
            {"question": "Will Arsenal win the World Cup?",
             "outcomes": ["Yes", "No"], "clobTokenIds": ["ARS_OUT_Y", "ARS_OUT_N"]},
            {"question": "Will Chelsea win the World Cup?",
             "outcomes": ["Yes", "No"], "clobTokenIds": ["CHE_OUT_Y", "CHE_OUT_N"]},
        ],
    }


# ---- discovery / classification --------------------------------------
async def test_find_market_prefers_h2h_over_outright():
    # Outright listed FIRST (as in the live feed), real fixture second.
    a = _adapter([_outright_winner_event(), _layout_b_event()])
    ref = await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF)
    assert ref is not None
    assert ref.market_id == "arsenal-vs-chelsea-2026-01-20"  # the fixture, not outright
    assert ref.metadata["outcome_tokens"]["DRAW"] == "DRAW_YES"


async def test_find_market_rejects_outright_only():
    # No real fixture available — must NOT route to the outright.
    a = _adapter([_outright_winner_event()])
    assert await a.find_market("Arsenal FC", "Chelsea FC", KICKOFF) is None


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


# ---- read-only: no order/sign capability (the safety-critical test) --
def test_adapter_has_no_order_or_sign_capability():
    """Predictions-only build: the former double-gated place_order machinery
    was deleted so a careless edit can never re-arm live trading. The adapter
    exposes NO order-placement or signing surface — only reads.
    """
    a = _adapter([])
    # No order path at all.
    assert not hasattr(PolymarketAdapter, "place_order")
    assert not hasattr(PolymarketAdapter, "orders_live")
    # No signing key / order-gate state carried on the instance.
    for attr in ("_private_key", "_funder", "_signature_type",
                 "_enable_orders", "_mode"):
        assert not hasattr(a, attr), f"unexpected signing/order state: {attr}"


async def test_get_clob_is_read_only(monkeypatch):
    """_get_clob builds a bare read-only CLOB client — no wallet key, no funder,
    no signature type, and it never derives/sets L2 API creds (get_order_book is
    a public endpoint). There is nothing here that could sign an order."""
    import sys
    import types

    calls: dict = {}

    class _FakeClobClient:
        def __init__(self, **kw):
            calls["init_kwargs"] = kw

        # If _get_clob ever tried to sign/derive, these would blow up.
        def create_or_derive_api_key(self):  # pragma: no cover
            raise AssertionError("read-only client must not derive API creds")

    fake_mod = types.ModuleType("py_clob_client_v2.client")
    fake_mod.ClobClient = _FakeClobClient
    monkeypatch.setitem(sys.modules, "py_clob_client_v2.client", fake_mod)

    a = _adapter([])
    clob = a._get_clob()
    kw = calls["init_kwargs"]
    # No signing surface passed to the client.
    assert "key" not in kw
    assert "funder" not in kw
    assert "signature_type" not in kw
    # Cached — a second call reuses the same client.
    assert a._get_clob() is clob
