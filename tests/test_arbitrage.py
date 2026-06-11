"""Tests for cross-venue arbitrage detection (Phase: arb)."""

from __future__ import annotations

from datetime import datetime, timezone

from betbot.exchanges.arbitrage import ArbScanner, size_legs
from betbot.exchanges.base import ExchangeName, MarketRef, OrderbookQuote, Outcome

KICKOFF = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)


def _match_meta(home="A", away="B", fixture_id=None):
    """Identity metadata a real adapter now attaches to a 1X2 MarketRef."""
    md = {"home_team": home, "away_team": away, "market_type": "match_result"}
    if fixture_id is not None:
        md["fixture_id"] = fixture_id
    return md


class FakeAdapter:
    def __init__(self, name, prices, *, home="A", away="B", fixture_id=None,
                 metadata=None, title="A vs B"):
        self.name = name
        self._prices = prices  # {Outcome: price} or None
        self._home = home
        self._away = away
        self._fixture_id = fixture_id
        self._metadata = metadata
        self._title = title

    async def find_market(self, h, a, k):
        if not self._prices:
            return None
        md = self._metadata if self._metadata is not None else _match_meta(
            self._home, self._away, self._fixture_id
        )
        return MarketRef(self.name, "m", self._title, md)

    async def get_orderbook(self, market, outcome):
        p = self._prices.get(outcome)
        if p is None:
            return None
        return OrderbookQuote(self.name, "m", outcome, p, 100.0, KICKOFF)


async def test_picks_cheapest_per_outcome():
    poly = FakeAdapter(ExchangeName.POLYMARKET, {Outcome.HOME: 0.55, Outcome.DRAW: 0.30, Outcome.AWAY: 0.30})
    lim = FakeAdapter(ExchangeName.LIMITLESS, {Outcome.HOME: 0.50, Outcome.AWAY: 0.28})  # no draw
    opp = await ArbScanner([poly, lim]).scan("A", "B", KICKOFF)
    assert opp.legs["HOME"].exchange is ExchangeName.LIMITLESS  # 0.50 < 0.55
    assert opp.legs["DRAW"].exchange is ExchangeName.POLYMARKET  # only venue
    assert opp.legs["AWAY"].exchange is ExchangeName.LIMITLESS  # 0.28 < 0.30
    assert opp.complete
    assert opp.price_sum == 0.50 + 0.30 + 0.28
    assert opp.margin < 0  # sum 1.08 -> no arb


async def test_detects_real_arb():
    poly = FakeAdapter(ExchangeName.POLYMARKET, {Outcome.HOME: 0.45, Outcome.DRAW: 0.28, Outcome.AWAY: 0.30})
    lim = FakeAdapter(ExchangeName.LIMITLESS, {Outcome.HOME: 0.40, Outcome.AWAY: 0.25})
    opp = await ArbScanner([poly, lim]).scan("A", "B", KICKOFF)
    # best: 0.40 + 0.28 + 0.25 = 0.93 -> 7% locked
    assert opp.complete
    assert opp.margin > 0.06
    legs = size_legs(opp, 100.0)
    assert sum(legs.values()) == 100.0  # spends the budget
    # equal payout: stake/price equal across legs
    payouts = [legs[o] / opp.legs[o].price for o in legs]
    assert max(payouts) - min(payouts) < 1e-6


async def test_fees_erode_margin():
    poly = FakeAdapter(ExchangeName.POLYMARKET, {Outcome.HOME: 0.33, Outcome.DRAW: 0.32, Outcome.AWAY: 0.33})
    lim = FakeAdapter(ExchangeName.LIMITLESS, {})
    opp = await ArbScanner([poly, lim], fee_per_leg=0.02).scan("A", "B", KICKOFF)
    # sum 0.98, locked 0.02 gross, but 3 legs * 0.02 fee = 0.06 -> net negative
    assert opp.margin < 0


async def test_none_when_no_markets():
    a = FakeAdapter(ExchangeName.POLYMARKET, {})
    b = FakeAdapter(ExchangeName.LIMITLESS, {})
    assert await ArbScanner([a, b]).scan("A", "B", KICKOFF) is None


# ----------------------------------------------------------------------
# cross-venue identity guard (matcher hardening) — only the SAME market pairs
# ----------------------------------------------------------------------
async def test_same_fixture_legs_pair():
    # Both venues quote the SAME fixture (matching fixture_id) — they pair.
    poly = FakeAdapter(
        ExchangeName.POLYMARKET,
        {Outcome.HOME: 0.45, Outcome.DRAW: 0.28, Outcome.AWAY: 0.30},
        fixture_id=99,
    )
    lim = FakeAdapter(
        ExchangeName.LIMITLESS,
        {Outcome.HOME: 0.40, Outcome.AWAY: 0.25},
        fixture_id=99,
    )
    opp = await ArbScanner([poly, lim]).scan("A", "B", KICKOFF)
    assert opp is not None and opp.complete
    assert opp.legs["HOME"].exchange is ExchangeName.LIMITLESS


async def test_different_fixture_legs_do_not_pair(capsys):
    # Same teams quoted, but the venues report DIFFERENT fixture ids — these are
    # NOT the same market and must not be locked together. Only the reference
    # venue's legs survive; the mismatched venue's legs are dropped, so the
    # cheaper away leg from the wrong fixture is NOT used.
    poly = FakeAdapter(
        ExchangeName.POLYMARKET,
        {Outcome.HOME: 0.45, Outcome.DRAW: 0.28, Outcome.AWAY: 0.30},
        fixture_id=99,
    )
    lim = FakeAdapter(
        ExchangeName.LIMITLESS,
        {Outcome.HOME: 0.40, Outcome.AWAY: 0.25},
        fixture_id=12345,  # different fixture
    )
    opp = await ArbScanner([poly, lim]).scan("A", "B", KICKOFF)
    assert "arb_leg_incompatible_dropped" in capsys.readouterr().out
    # The mismatched-fixture legs are dropped; any survivors are the reference
    # (Polymarket) fixture's, never a blend of two different fixtures.
    if opp is not None:
        for leg in opp.legs.values():
            assert leg.exchange is ExchangeName.POLYMARKET


async def test_different_team_pair_legs_do_not_pair(capsys):
    # No fixture ids; team pairs disagree -> incompatible, mismatched legs dropped.
    poly = FakeAdapter(
        ExchangeName.POLYMARKET,
        {Outcome.HOME: 0.45, Outcome.DRAW: 0.28, Outcome.AWAY: 0.30},
        home="Mexico", away="South Africa",
    )
    lim = FakeAdapter(
        ExchangeName.LIMITLESS,
        {Outcome.HOME: 0.40, Outcome.AWAY: 0.25},
        home="Mexico", away="Czech Republic",  # different opponent
    )
    opp = await ArbScanner([poly, lim]).scan("Mexico", "South Africa", KICKOFF)
    assert "arb_leg_incompatible_dropped" in capsys.readouterr().out
    if opp is not None:
        for leg in opp.legs.values():
            assert leg.exchange is ExchangeName.POLYMARKET


async def test_prop_vs_1x2_legs_do_not_pair(capsys):
    # The Limitless leg is a PROP (goalsThreshold set) — it must never be paired
    # with the Polymarket 1X2 legs into a "lock".
    poly = FakeAdapter(
        ExchangeName.POLYMARKET,
        {Outcome.HOME: 0.45, Outcome.DRAW: 0.28, Outcome.AWAY: 0.30},
        home="Mexico", away="South Africa",
    )
    lim = FakeAdapter(
        ExchangeName.LIMITLESS,
        {Outcome.HOME: 0.10, Outcome.AWAY: 0.08},  # tempting cheap prop legs
        metadata={
            "home_team": "Mexico", "away_team": "South Africa",
            "goalsThreshold": 2.5,
        },
        title="3+ total goals",
    )
    opp = await ArbScanner([poly, lim]).scan("Mexico", "South Africa", KICKOFF)
    assert "arb_leg_incompatible_dropped" in capsys.readouterr().out
    # The valid 1X2 (Polymarket) legs still pair into a clean opportunity, but
    # the cheap prop legs are NEVER blended into the lock.
    assert opp is not None and opp.complete
    for leg in opp.legs.values():
        assert leg.exchange is ExchangeName.POLYMARKET
    # If the prop legs had leaked in, HOME would be 0.10 not 0.45.
    assert opp.legs["HOME"].price == 0.45
