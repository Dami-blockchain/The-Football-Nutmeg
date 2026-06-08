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
from betbot.exchanges.base import ExchangeName, OrderbookQuote, Outcome
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


class FakeRouter:
    def __init__(self, quote):
        self._quote = quote

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
