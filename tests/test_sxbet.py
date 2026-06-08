"""Tests for the SX Bet read-only adapter (offline; fake client)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.exchanges.base import ExchangeName, Outcome
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.sxbet import SXBetAdapter

KICKOFF = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)
SCALE = 10**20


class FakeSX:
    def __init__(self, markets, orders):
        self._m, self._o = markets, orders

    async def soccer_markets(self):
        return self._m

    async def orders(self, market_hash):
        return self._o


def _market():
    return {"marketHash": "0xabc", "teamOneName": "Brazil", "teamTwoName": "Serbia"}


def _order(outcome_one: bool, implied: float, size: int = 100_000_000):
    return {"isMakerBettingOutcomeOne": outcome_one,
            "percentageOdds": str(int(implied * SCALE)), "sizeRemaining": str(size)}


def _adapter(markets, orders):
    return SXBetAdapter(FakeSX(markets, orders), TeamAliasResolver())


async def test_find_market_matches_teams():
    a = _adapter([_market()], [])
    ref = await a.find_market("Brazil", "Serbia", KICKOFF)
    assert ref is not None and ref.exchange is ExchangeName.SXBET
    assert ref.metadata["market_hash"] == "0xabc"


async def test_no_match():
    a = _adapter([_market()], [])
    assert await a.find_market("France", "Mexico", KICKOFF) is None


async def test_home_price_from_outcome2_makers():
    # to back HOME (outcome1) we take makers on outcome2; price = 1 - makerOdds
    orders = [_order(False, 0.60), _order(False, 0.55), _order(True, 0.70)]
    a = _adapter([_market()], orders)
    ref = await a.find_market("Brazil", "Serbia", KICKOFF)
    q = await a.get_orderbook(ref, Outcome.HOME)
    assert q.yes_price == pytest.approx(0.40)  # best = 1 - max(0.60, 0.55)


async def test_away_price_from_outcome1_makers():
    orders = [_order(True, 0.70), _order(False, 0.60)]
    a = _adapter([_market()], orders)
    ref = await a.find_market("Brazil", "Serbia", KICKOFF)
    q = await a.get_orderbook(ref, Outcome.AWAY)
    assert q.yes_price == pytest.approx(0.30)  # 1 - 0.70


async def test_draw_returns_none():
    a = _adapter([_market()], [_order(False, 0.6)])
    ref = await a.find_market("Brazil", "Serbia", KICKOFF)
    assert await a.get_orderbook(ref, Outcome.DRAW) is None
