"""Tests for the cross-venue arb executor — all venue calls mocked."""

from __future__ import annotations

import json

import pytest

from betbot.arb_executor import (
    STATUS_ABORTED,
    STATUS_FILLED,
    STATUS_PARTIAL,
    ArbExecutor,
)
from betbot.config import Settings
from betbot.exchanges.arbitrage import ArbOpportunity, Leg
from betbot.exchanges.base import ExchangeName, MarketRef, OrderResult, Outcome
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    arb_staked_today_usd,
    insert_arb_execution,
    list_recent_arb_executions,
    trip_kill_switch,
)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "arb.sqlite")
    yield


def make_settings(**over) -> Settings:
    kwargs = {
        "BETBOT_MODE": "live",
        "BETBOT_ARB_EXECUTE": True,
        "BETBOT_ARB_EXECUTE_MIN_MARGIN": 0.02,
        "BETBOT_ARB_MAX_STAKE_USD": 25.0,
        "BETBOT_ARB_DAILY_CAP_USD": 100.0,
        "FOOTBALL_DATA_API_KEY": "fake",
    }
    kwargs.update(over)
    return Settings(**kwargs)


def _market(venue: ExchangeName, outcome: str, min_shares: float = 0.0) -> MarketRef:
    if venue == ExchangeName.LIMITLESS:
        meta = {
            "outcome_markets": {outcome: {"slug": f"lm-{outcome}", "yes_token": "1",
                                          "min_shares": min_shares}},
            "venue_exchange": "0x05c748E2f4DcDe0ec9Fa8DDc40DE6b867f923fa5",
        }
    else:
        meta = {"outcome_tokens": {outcome: "tok"}}
    return MarketRef(exchange=venue, market_id=f"{venue.value}-mkt", title="t", metadata=meta)


def make_opp(
    prices: dict[str, tuple[str, float]] | None = None,
    *,
    lm_min_shares: float = 0.0,
) -> ArbOpportunity:
    """Default: HOME on Limitless @0.40, DRAW/AWAY on Polymarket @0.25/0.30.

    price_sum 0.95 -> margin +0.05.
    """
    prices = prices or {
        "HOME": ("LIMITLESS", 0.40),
        "DRAW": ("POLYMARKET", 0.25),
        "AWAY": ("POLYMARKET", 0.30),
    }
    legs = {}
    for ov, (venue_name, price) in prices.items():
        venue = ExchangeName[venue_name]
        legs[ov] = Leg(
            outcome=Outcome(ov), exchange=venue, price=price, size=500.0,
            market=_market(venue, ov,
                           lm_min_shares if venue == ExchangeName.LIMITLESS else 0.0),
        )
    price_sum = sum(lg.price for lg in legs.values())
    return ArbOpportunity("Home FC", "Away FC", legs, price_sum, fees=0.0)


class FakeAdapter:
    def __init__(self, name: ExchangeName, *, fail: bool = False,
                 status: str = "matched", call_log: list | None = None):
        self.name = name
        self.fail = fail
        self.status = status
        self.orders: list[tuple] = []
        self.call_log = call_log if call_log is not None else []

    async def place_order(self, market, outcome, size_usd, max_price):
        self.call_log.append((self.name, outcome.value, size_usd))
        if self.fail:
            raise RuntimeError(f"{self.name.value} order failed")
        self.orders.append((market, outcome, size_usd, max_price))
        return OrderResult(
            exchange=self.name, order_id=f"{self.name.value}-1",
            market_id=market.market_id, outcome=outcome,
            filled_size=size_usd / max_price, avg_price=max_price,
            status=self.status, raw_response={},
        )


class FakeNotify:
    def __init__(self):
        self.messages: list[str] = []

    async def __call__(self, settings, text: str) -> bool:
        self.messages.append(text)
        return True


def make_executor(settings, *, lm_fail=False, pm_fail=False, balances=None):
    calls: list = []
    lm = FakeAdapter(ExchangeName.LIMITLESS, fail=lm_fail, call_log=calls)
    pm = FakeAdapter(ExchangeName.POLYMARKET, fail=pm_fail, call_log=calls)
    notify = FakeNotify()
    balances = balances if balances is not None else {"polygon": 1000.0, "base": 1000.0}
    ex = ArbExecutor(
        settings,
        {ExchangeName.LIMITLESS: lm, ExchangeName.POLYMARKET: pm},
        notify=notify,
        balance_usd=lambda chain: balances.get(chain, 0.0),
    )
    return ex, lm, pm, notify, calls


# ----------------------------------------------------------------------
# Gating matrix
# ----------------------------------------------------------------------
async def test_flag_off_no_orders_no_rows(db):
    ex, lm, pm, notify, calls = make_executor(make_settings(BETBOT_ARB_EXECUTE=False))
    r = await ex.execute(make_opp())
    assert r.status == "skipped_disabled"
    assert calls == [] and notify.messages == []
    assert list_recent_arb_executions() == []


async def test_paper_mode_no_orders(db):
    ex, *_, calls = make_executor(make_settings(BETBOT_MODE="paper"))
    r = await ex.execute(make_opp())
    assert r.status == "skipped_paper"
    assert calls == []
    assert list_recent_arb_executions() == []


async def test_kill_switch_blocks(db):
    trip_kill_switch("drawdown", -50.0, 200.0)
    ex, *_, calls = make_executor(make_settings())
    r = await ex.execute(make_opp())
    assert r.status == "skipped_kill_switch"
    assert calls == []


async def test_thin_margin_rejected_and_persisted(db):
    # price_sum 0.99 -> margin 0.01: above the 0.01 alert bar, below 0.02 exec bar.
    opp = make_opp({
        "HOME": ("LIMITLESS", 0.44),
        "DRAW": ("POLYMARKET", 0.25),
        "AWAY": ("POLYMARKET", 0.30),
    })
    assert 0.0 < opp.margin < 0.02
    ex, *_, calls = make_executor(make_settings())
    r = await ex.execute(opp)
    assert r.status == "rejected_thin_margin"
    assert calls == []
    [row] = list_recent_arb_executions()
    assert row.status == "rejected_thin_margin"
    assert arb_staked_today_usd() == 0.0  # rejections don't count toward the cap


async def test_incomplete_opportunity_rejected(db):
    opp = make_opp()
    incomplete = ArbOpportunity(
        opp.home_team, opp.away_team,
        {k: v for k, v in opp.legs.items() if k != "DRAW"},
        0.70, 0.0,
    )
    ex, *_, calls = make_executor(make_settings())
    r = await ex.execute(incomplete)
    assert r.status == "rejected_incomplete"
    assert calls == []


async def test_unsupported_venue_rejected(db):
    opp = make_opp({
        "HOME": ("SXBET", 0.40),
        "DRAW": ("POLYMARKET", 0.25),
        "AWAY": ("POLYMARKET", 0.30),
    })
    ex, *_, calls = make_executor(make_settings())
    r = await ex.execute(opp)
    assert r.status == "rejected_unsupported_venue"
    assert calls == []


async def test_insufficient_balance_rejected(db):
    # Base (Limitless leg) is broke; Polygon is fine.
    ex, *_, calls = make_executor(
        make_settings(), balances={"polygon": 1000.0, "base": 1.0}
    )
    r = await ex.execute(make_opp())
    assert r.status == "rejected_insufficient_balance"
    assert "base" in r.detail
    assert calls == []
    [row] = list_recent_arb_executions()
    assert row.status == "rejected_insufficient_balance"


async def test_daily_cap_rejected(db):
    insert_arb_execution(
        home_team="X", away_team="Y", margin=0.03, price_sum=0.97,
        stake_usd=90.0, status="filled", legs_json="[]",
    )
    ex, *_, calls = make_executor(make_settings())  # next ~25 would exceed 100
    r = await ex.execute(make_opp())
    assert r.status == "rejected_daily_cap"
    assert calls == []


# ----------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------
async def test_sizing_proportional_and_capped(db):
    ex, lm, pm, *_ = make_executor(make_settings())
    r = await ex.execute(make_opp())
    assert r.status == STATUS_FILLED
    stakes = {o.value: s for (_, o, s, _) in lm.orders + pm.orders}
    # P = 25 / 0.95 = 26.3157...; stake_o = floor_cents(P * price_o)
    assert stakes == {"HOME": 10.52, "DRAW": 6.57, "AWAY": 7.89}
    assert sum(stakes.values()) <= 25.0


async def test_limitless_min_shares_within_budget_ok(db):
    # P = 26.31 shares >= 20 min shares -> fine.
    ex, lm, *_ = make_executor(make_settings())
    r = await ex.execute(make_opp(lm_min_shares=20.0))
    assert r.status == STATUS_FILLED
    assert len(lm.orders) == 1


async def test_limitless_min_shares_over_budget_rejected(db):
    # Min 100 shares would cost 100*0.95 = $95 total payout-equivalent > $25 cap.
    ex, *_, calls = make_executor(make_settings())
    r = await ex.execute(make_opp(lm_min_shares=100.0))
    assert r.status == "rejected_min_size"
    assert "LIMITLESS" in r.detail
    assert calls == []
    [row] = list_recent_arb_executions()
    assert row.status == "rejected_min_size"


async def test_polymarket_minimum_rejected(db):
    # Tiny budget -> PM legs under the $1 floor.
    ex, *_, calls = make_executor(make_settings(BETBOT_ARB_MAX_STAKE_USD=2.0))
    r = await ex.execute(make_opp())
    assert r.status == "rejected_min_size"
    assert "POLYMARKET" in r.detail
    assert calls == []


# ----------------------------------------------------------------------
# Execution paths
# ----------------------------------------------------------------------
async def test_happy_path_limitless_first_then_polymarket(db):
    ex, lm, pm, notify, calls = make_executor(make_settings())
    r = await ex.execute(make_opp())
    assert r.status == STATUS_FILLED
    # Less liquid venue (Limitless) strictly before any Polymarket leg.
    venues = [v for (v, _, _) in calls]
    assert venues[0] == ExchangeName.LIMITLESS
    assert venues.count(ExchangeName.POLYMARKET) == 2
    [row] = list_recent_arb_executions()
    assert row.status == STATUS_FILLED
    assert row.net_expected_usd is not None and row.net_expected_usd > 0
    legs = json.loads(row.legs_json)
    assert all(lg["status"] == "filled" for lg in legs)
    assert len(notify.messages) == 1 and "Arb executed" in notify.messages[0]


async def test_leg1_failure_aborts_without_exposure(db):
    ex, lm, pm, notify, calls = make_executor(make_settings(), lm_fail=True)
    r = await ex.execute(make_opp())
    assert r.status == STATUS_ABORTED
    # Polymarket legs never attempted after the Limitless leg failed.
    assert [v for (v, _, _) in calls] == [ExchangeName.LIMITLESS]
    assert pm.orders == []
    [row] = list_recent_arb_executions()
    assert row.status == STATUS_ABORTED
    assert len(notify.messages) == 1 and "no exposure" in notify.messages[0]


async def test_leg2_failure_persists_partial_state_and_notifies(db):
    ex, lm, pm, notify, calls = make_executor(make_settings(), pm_fail=True)
    r = await ex.execute(make_opp())
    assert r.status == STATUS_PARTIAL
    assert len(lm.orders) == 1  # Limitless leg filled -> exposure open

    [row] = list_recent_arb_executions()
    assert row.status == STATUS_PARTIAL
    legs = {lg["outcome"]: lg for lg in json.loads(row.legs_json)}
    assert legs["HOME"]["status"] == "filled"
    assert legs["HOME"]["fill_price"] == pytest.approx(0.40)
    failed = [lg for lg in legs.values() if lg["status"] == "failed"]
    not_attempted = [lg for lg in legs.values() if lg["status"] == "not_attempted"]
    assert len(failed) == 1 and len(not_attempted) == 1

    [msg] = notify.messages
    assert "exposure open" in msg
    assert "$10.52" in msg            # the exact open Limitless stake
    assert "NOT auto-chasing" in msg

    # The attempt still counts toward the daily cap (money moved).
    assert arb_staked_today_usd() == pytest.approx(24.98)


async def test_execute_all_caps_across_opportunities(db):
    # Cap 30: the first opportunity stakes ~24.98, the second must be refused.
    ex, *_ = make_executor(make_settings(BETBOT_ARB_DAILY_CAP_USD=30.0))
    results = await ex.execute_all([make_opp(), make_opp()])
    assert [r.status for r in results] == [STATUS_FILLED, "rejected_daily_cap"]
