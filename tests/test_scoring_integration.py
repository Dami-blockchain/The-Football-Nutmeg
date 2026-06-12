"""End-to-end integration test for the scoring-loop tri-state routing.

Unit tests cover the adapters/router/matcher in isolation; this drives the real
`_score_and_log_one` (predict -> route -> edge filter -> DB) against a real
temp SQLite, exercising all three branches that decide whether/what we log:

  * no market quote        -> favourite-only paper bet (market_price NULL)
  * quote WITH edge        -> market-priced paper bet (market_price + edge set)
  * quote WITHOUT edge     -> market veto, NOTHING logged (no favourite fallback)

This is the money-logging path, so it gets a real integration test rather than
trusting the components compose correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.data.models import Fixture, FixtureForm, FormSnapshot
from betbot.exchanges.base import (
    ExchangeName,
    MarketRef,
    OrderbookQuote,
    OrderResult,
    Outcome,
)
from betbot.exchanges.router import RoutedQuote
from betbot.main import _score_and_log_one
from betbot.storage.db import init_engine
from betbot.storage.repos import list_recent_paper_bets
from betbot.strategy.engine import StrategyEngine

KICKOFF_ISO = "2026-06-20T15:00:00Z"
MATCH = {
    "id": 9001,
    "utcDate": KICKOFF_ISO,
    "homeTeam": {"id": 1, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS"},
    "awayTeam": {"id": 2, "name": "Chelsea FC", "shortName": "Chelsea", "tla": "CHE"},
}


class FakeForm:
    """Deterministic form: home strong, away weak -> HOME is the clear favourite."""

    def __init__(self, home_w=6.0, away_w=1.0):
        self.home_w, self.away_w = home_w, away_w

    async def fixture_form(self, fixture_id, competition_code, kickoff, home_team, away_team):
        fixture = Fixture(
            id=fixture_id, home_team=home_team, away_team=away_team,
            kickoff=kickoff, competition_code=competition_code,
        )
        return FixtureForm(
            fixture=fixture,
            home_form=FormSnapshot(team=home_team, weighted_points=self.home_w,
                                   raw_points=int(self.home_w), matches_considered=5),
            away_form=FormSnapshot(team=away_team, weighted_points=self.away_w,
                                   raw_points=int(self.away_w), matches_considered=5),
        )


class FakeAdapter:
    def __init__(self):
        self.placed = []

    async def place_order(self, market, outcome, size_usd, max_price):
        self.placed.append((outcome, size_usd, max_price))
        return OrderResult(ExchangeName.POLYMARKET, "ord1", market.market_id,
                           outcome, size_usd, max_price, "matched", {})


class FakeRouter:
    def __init__(self, quote, adapter=None):
        self._quote = quote
        self.adapter = adapter or FakeAdapter()

    async def find_best_route(self, home, away, kickoff, outcome):
        if self._quote is None:
            return None
        market = MarketRef(ExchangeName.POLYMARKET, "m", "A vs B", {})
        return RoutedQuote(adapter=self.adapter, market=market, quote=self._quote)

    async def find_best_quote(self, home, away, kickoff, outcome):
        return self._quote


def _home_quote(price: float) -> OrderbookQuote:
    return OrderbookQuote(
        exchange=ExchangeName.POLYMARKET, market_id="m", outcome=Outcome.HOME,
        yes_price=price, yes_size=100.0, timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def fresh_db(tmp_path):
    init_engine(tmp_path / "score.sqlite")
    yield


async def test_no_market_logs_favourite_only(fresh_db, settings):
    n = await _score_and_log_one(MATCH, "PL", FakeForm(), StrategyEngine(settings),
                                 FakeRouter(None), settings)
    assert n == 1
    rows = list_recent_paper_bets(days=7)
    assert len(rows) == 1
    assert rows[0].outcome == "HOME"
    assert rows[0].market_price is None  # favourite-only -> no market price


async def test_market_edge_logs_market_bet(fresh_db, settings):
    # HOME p≈0.98; market price 0.50 -> big positive edge -> bet logged WITH price.
    n = await _score_and_log_one(MATCH, "PL", FakeForm(), StrategyEngine(settings),
                                 FakeRouter(_home_quote(0.50)), settings)
    assert n == 1
    rows = list_recent_paper_bets(days=7)
    assert len(rows) == 1
    assert rows[0].outcome == "HOME"
    assert rows[0].market_price == 0.50
    assert rows[0].edge is not None and rows[0].edge >= settings.edge_threshold


async def test_no_edge_logs_nothing(fresh_db, settings):
    # HOME p≈0.98; market price 0.97 -> edge ≈0.01 < 0.05 -> market vetoes, no log.
    n = await _score_and_log_one(MATCH, "PL", FakeForm(), StrategyEngine(settings),
                                 FakeRouter(_home_quote(0.97)), settings)
    assert n == 0
    assert list_recent_paper_bets(days=7) == []


async def test_live_order_placed_when_enabled(fresh_db, settings):
    router = FakeRouter(_home_quote(0.50))
    n = await _score_and_log_one(MATCH, "PL", FakeForm(), StrategyEngine(settings),
                                 router, settings, False, True)  # live_orders=True
    assert n == 1
    assert len(router.adapter.placed) == 1  # real order placed on the venue


async def test_wc_live_order_skipped(fresh_db, settings):
    # World Cup (INTERNATIONAL_COMPETITIONS): paper bet logs, but NEVER a live order.
    router = FakeRouter(_home_quote(0.50))
    n = await _score_and_log_one(MATCH, "WC", FakeForm(), StrategyEngine(settings),
                                 router, settings, False, True)  # live_orders=True
    assert n == 1
    assert router.adapter.placed == []  # guard held — no live order on WC


async def test_wc_live_order_allowed_when_flag_set(fresh_db, settings):
    settings.allow_international_live = True  # operator opt-in
    router = FakeRouter(_home_quote(0.50))
    n = await _score_and_log_one(MATCH, "WC", FakeForm(), StrategyEngine(settings),
                                 router, settings, False, True)
    assert n == 1
    assert len(router.adapter.placed) == 1  # now WC can place live


async def test_paper_mode_places_no_live_order(fresh_db, settings):
    router = FakeRouter(_home_quote(0.50))
    await _score_and_log_one(MATCH, "PL", FakeForm(), StrategyEngine(settings),
                             router, settings, False, False)  # live_orders=False
    assert router.adapter.placed == []


async def test_bet_every_match_wc_forces_bet_without_edge(fresh_db, settings):
    settings.international_bet_every_match = True
    router = FakeRouter(_home_quote(0.97))  # high price -> normally no edge
    n = await _score_and_log_one(MATCH, "WC", FakeForm(), StrategyEngine(settings),
                                 router, settings, False, False)
    assert n == 1  # bet placed despite no edge
    rows = list_recent_paper_bets(days=7)
    assert len(rows) == 1 and rows[0].market_price == 0.97


async def test_wc_no_edge_still_vetoes_without_flag(fresh_db, settings):
    # default international_bet_every_match=False -> normal veto on no edge
    router = FakeRouter(_home_quote(0.97))
    n = await _score_and_log_one(MATCH, "WC", FakeForm(), StrategyEngine(settings),
                                 router, settings, False, False)
    assert n == 0


async def test_exposure_cap_blocks_logging(fresh_db, settings):
    # First bet logs; then force the cap so the next fixture is refused.
    await _score_and_log_one(MATCH, "PL", FakeForm(), StrategyEngine(settings),
                             FakeRouter(None), settings)
    settings.daily_exposure_cap_usd = 0.0  # any further stake exceeds the cap
    match2 = {**MATCH, "id": 9002}
    n = await _score_and_log_one(match2, "PL", FakeForm(), StrategyEngine(settings),
                                 FakeRouter(None), settings)
    assert n == 0
    assert len(list_recent_paper_bets(days=7)) == 1  # only the first


# --------------------------------------------------------------------------
# Multi-tenant placement: the decided bet is placed on EVERY funded user's own
# wallet (non-custodial), sized to that user's balance. These drive
# `_place_live_for_users` directly (the existing live tests cover the no-users
# fallback to the single global agent wallet).
# --------------------------------------------------------------------------
async def test_multi_user_places_on_each_funded_wallet(fresh_db, settings, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import betbot.wallet as wallet_mod
    from betbot.exchanges.base import MarketRef
    from betbot.main import _place_live_for_users
    from betbot.storage.repos import get_or_create_user

    secrets = tmp_path / "secrets"
    alice = get_or_create_user(111, "Alice", secrets_dir=str(secrets))
    get_or_create_user(222, "Bob", secrets_dir=str(secrets))  # under-funded -> skipped

    def fake_balance(address, chain, rpc):
        usd = 100.0 if address.lower() == alice.wallet_address.lower() else 0.0
        return wallet_mod.ChainBalance(chain, chain, usd, True)

    monkeypatch.setattr(wallet_mod, "usdc_balance", fake_balance)

    built = []

    def fake_make(*, signing_key, funder=None, limitless_creds=None):
        adapters = {ExchangeName.POLYMARKET: FakeAdapter(),
                    ExchangeName.LIMITLESS: FakeAdapter()}
        built.append((funder, adapters))
        return adapters

    route = RoutedQuote(
        adapter=FakeAdapter(),  # global fallback — must NOT be used when users exist
        market=MarketRef(ExchangeName.POLYMARKET, "m", "A vs B", {}),
        quote=_home_quote(0.50),
    )
    prediction = SimpleNamespace(competition_code="PL", fixture_id=9001)
    decision = SimpleNamespace(outcome=Outcome.HOME, stake_usd=10.0)

    await _place_live_for_users(route, prediction, decision, settings, True, fake_make, {})

    assert len(built) == 1                       # only Alice (funded) got an adapter
    funder, adapters = built[0]
    assert funder.lower() == alice.wallet_address.lower()
    placed = adapters[ExchangeName.POLYMARKET].placed
    assert len(placed) == 1
    assert placed[0][1] == 10.0                  # stake = min(10 stake, 100 balance)
    assert route.adapter.placed == []            # global fallback unused


async def test_multi_user_sizes_down_to_balance(fresh_db, settings, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import betbot.wallet as wallet_mod
    from betbot.exchanges.base import MarketRef
    from betbot.main import _place_live_for_users
    from betbot.storage.repos import get_or_create_user

    get_or_create_user(333, "Carol", secrets_dir=str(tmp_path / "secrets"))
    monkeypatch.setattr(
        wallet_mod, "usdc_balance",
        lambda a, c, r: wallet_mod.ChainBalance(c, c, 3.5, True),  # below the $10 stake
    )

    built = []

    def fake_make(*, signing_key, funder=None, limitless_creds=None):
        d = {ExchangeName.POLYMARKET: FakeAdapter(), ExchangeName.LIMITLESS: FakeAdapter()}
        built.append(d)
        return d

    route = RoutedQuote(adapter=FakeAdapter(),
                        market=MarketRef(ExchangeName.POLYMARKET, "m", "A vs B", {}),
                        quote=_home_quote(0.50))
    prediction = SimpleNamespace(competition_code="PL", fixture_id=9001)
    decision = SimpleNamespace(outcome=Outcome.HOME, stake_usd=10.0)

    await _place_live_for_users(route, prediction, decision, settings, True, fake_make, {})

    placed = built[0][ExchangeName.POLYMARKET].placed
    assert len(placed) == 1
    assert placed[0][1] == 3.5                   # capped to the user's balance


async def test_multi_user_wc_guard_blocks_all(fresh_db, settings, tmp_path, monkeypatch):
    """The World Cup hard-guard holds even with funded users (flag off)."""
    from types import SimpleNamespace

    import betbot.wallet as wallet_mod
    from betbot.exchanges.base import MarketRef
    from betbot.main import _place_live_for_users
    from betbot.storage.repos import get_or_create_user

    get_or_create_user(444, "Dave", secrets_dir=str(tmp_path / "secrets"))
    monkeypatch.setattr(wallet_mod, "usdc_balance",
                        lambda a, c, r: wallet_mod.ChainBalance(c, c, 100.0, True))

    built = []

    def fake_make(*, signing_key, funder=None, limitless_creds=None):
        d = {ExchangeName.POLYMARKET: FakeAdapter(), ExchangeName.LIMITLESS: FakeAdapter()}
        built.append(d)
        return d

    route = RoutedQuote(adapter=FakeAdapter(),
                        market=MarketRef(ExchangeName.POLYMARKET, "m", "A vs B", {}),
                        quote=_home_quote(0.50))
    prediction = SimpleNamespace(competition_code="WC", fixture_id=9001)
    decision = SimpleNamespace(outcome=Outcome.HOME, stake_usd=10.0)

    await _place_live_for_users(route, prediction, decision, settings, True, fake_make, {})
    assert built == []                           # no per-user adapters built; guard held
