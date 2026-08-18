"""FootballDataCoUkProvider + OddsService.

Covers the parsing of the real feed's shape, de-vigging, the shared cache /
rate limiter (a 20-fixture Saturday must be ONE request), and every graceful
degradation path — a dead feed must never break an alert.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from betbot.data.odds import (
    FootballDataCoUkProvider,
    MatchOdds,
    OddsService,
    reset_shared_odds_service,
)
from betbot.data.odds_names import OddsNameResolver

CANON = ["ath madrid", "vallecano", "alaves", "barcelona", "espanol", "man city", "arsenal"]
ALIASES = {"Ath Madrid": ["Atl. Madrid"], "Vallecano": ["Rayo Vallecano"]}

# A trimmed but structurally faithful slice of the real fixtures.csv, BOM and
# all — the E2 row is a division we do not cover, Malaga is a club we have no
# ratings for, and 'Atl. Madrid' is football-data.co.uk disagreeing with its
# own historical files.
FIXTURES_CSV = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,Referee,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
    "E2,20/08/2026,20:00,Sheffield Wed,Bradford City,T Parsons,2.4,3.3,2.7,2.44,3.36,2.68\n"
    "SP1,19/08/2026,20:00,Atl. Madrid,Malaga,,1.33,5.25,9.5,1.3,5.27,10.19\n"
    "SP1,20/08/2026,20:00,Rayo Vallecano,Alaves,,2.25,3.0,3.6,2.23,2.99,3.52\n"
    "E0,22/08/2026,15:00,Arsenal,Man City,,3.1,3.4,2.3,3.05,3.45,2.35\n"
)


class _Settings:
    odds_anchor_enabled = True
    odds_anchor_market_weight = 1.0
    odds_cache_ttl_seconds = 3600.0
    odds_min_request_interval_seconds = 60.0
    odds_max_date_slack_days = 3
    leagues = ("PL", "PD", "BL1", "SA", "FL1", "CL")


@pytest.fixture(autouse=True)
def _reset_shared():
    reset_shared_odds_service()
    yield
    reset_shared_odds_service()


def _resolver() -> OddsNameResolver:
    return OddsNameResolver(CANON, aliases=ALIASES)


def _provider(payload: str | None = FIXTURES_CSV, counter: list | None = None):
    def fake_get(url, timeout):
        if counter is not None:
            counter.append(url)
        return payload

    return FootballDataCoUkProvider(_resolver(), fetcher=fake_get)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parses_covered_leagues_only_and_skips_unresolvable():
    p = _provider()
    rows = p.fetch(["PL", "PD", "BL1", "SA", "FL1"])
    keys = {(r.league, r.home, r.away) for r in rows}
    # Rayo/Alaves and Arsenal/Man City resolve; the E2 row is out of scope and
    # the Malaga row is skipped because we have no ratings for Malaga.
    assert keys == {("PD", "vallecano", "alaves"), ("PL", "arsenal", "man city")}
    assert "Malaga" in p.unresolved
    # Out-of-scope divisions must NOT pollute the unresolved report.
    assert "Sheffield Wed" not in p.unresolved


def test_atl_madrid_row_would_resolve_when_the_opponent_is_known():
    """Isolates the alias: the Malaga row above is skipped for the OPPONENT,
    not for Atletico."""
    p = _provider(
        "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        "SP1,19/08/2026,20:00,Atl. Madrid,Alaves,1.33,5.25,9.5\n"
    )
    rows = p.fetch(["PD"])
    assert [(r.home, r.away) for r in rows] == [("ath madrid", "alaves")]


def test_prefers_pinnacle_then_b365_then_average():
    p = _provider(
        "Div,Date,Time,HomeTeam,AwayTeam,PSH,PSD,PSA,B365H,B365D,B365A\n"
        "SP1,20/08/2026,20:00,Rayo Vallecano,Alaves,2.3,3.1,3.7,2.25,3.0,3.6\n"
    )
    rows = p.fetch(["PD"])
    assert rows[0].book == "PSH"
    assert rows[0].price_home == 2.3


def test_row_with_no_usable_prices_is_dropped():
    p = _provider(
        "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        "SP1,20/08/2026,20:00,Rayo Vallecano,Alaves,,,\n"
    )
    assert p.fetch(["PD"]) == []


def test_nonsense_prices_are_rejected():
    """Decimal odds must exceed 1.0; a 0.5 is a corrupt cell, not a 200% shot."""
    p = _provider(
        "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        "SP1,20/08/2026,20:00,Rayo Vallecano,Alaves,0.5,3.0,3.6\n"
    )
    assert p.fetch(["PD"]) == []


def test_dead_feed_returns_no_rows_rather_than_raising():
    assert _provider(payload=None).fetch(["PL"]) == []


def test_garbage_payload_returns_no_rows():
    assert _provider(payload="not,a,fixtures,file\n1,2,3,4\n").fetch(["PL"]) == []


# ---------------------------------------------------------------------------
# De-vigging
# ---------------------------------------------------------------------------
def test_devig_strips_the_overround():
    o = MatchOdds(
        league="PD", match_date=date(2026, 8, 20), home="vallecano", away="alaves",
        price_home=2.25, price_draw=3.0, price_away=3.6, source="t", book="B365H",
    )
    assert o.overround > 1.0  # the book's margin is really there
    probs = o.probabilities()
    assert sum(probs) == pytest.approx(1.0)
    # Ordering is preserved and the shortest price is the biggest probability.
    assert probs[0] > probs[1] > probs[2]
    assert probs[0] == pytest.approx((1 / 2.25) / (1 / 2.25 + 1 / 3.0 + 1 / 3.6))


# ---------------------------------------------------------------------------
# Shared cache + rate limiting
# ---------------------------------------------------------------------------
def test_twenty_fixture_saturday_is_one_http_get():
    calls: list[str] = []
    svc = OddsService(_Settings(), providers=[_provider(counter=calls)])

    async def go():
        for _ in range(20):
            await svc.prime(["PL", "PD"])

    asyncio.run(go())
    assert len(calls) == 1, f"expected 1 request for the batch, got {len(calls)}"


def test_stale_cache_refetches_but_the_rate_limiter_holds_it_back():
    calls: list[str] = []
    now = [1000.0]

    class S(_Settings):
        odds_cache_ttl_seconds = 10.0
        odds_min_request_interval_seconds = 100.0

    svc = OddsService(S(), providers=[_provider(counter=calls)], clock=lambda: now[0])
    asyncio.run(svc.prime(["PD"]))
    assert len(calls) == 1
    now[0] += 20.0  # cache stale...
    asyncio.run(svc.prime(["PD"]))
    assert len(calls) == 1, "rate limiter must suppress the early refetch"
    now[0] += 100.0  # ...and now the interval has elapsed
    asyncio.run(svc.prime(["PD"]))
    assert len(calls) == 2


def test_provider_that_raises_does_not_propagate():
    class Boom:
        name = "boom"

        def fetch(self, leagues):
            raise RuntimeError("feed on fire")

    svc = OddsService(_Settings(), providers=[Boom()])
    assert asyncio.run(svc.prime(["PL"])) == 0
    assert svc.quote("PL", date(2026, 8, 22), "Arsenal", "Man City") is None


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def _primed_service() -> OddsService:
    svc = OddsService(_Settings(), providers=[_provider()])
    asyncio.run(svc.prime(["PL", "PD"]))
    return svc


def test_quote_resolves_live_football_data_org_names():
    svc = _primed_service()
    q = svc.quote("PD", date(2026, 8, 20), "Rayo Vallecano", "Alaves")
    assert q is not None
    assert q.odds.price_home == 2.25
    assert sum(q.probabilities) == pytest.approx(1.0)


def test_quote_tolerates_a_utc_vs_local_date_offset():
    svc = _primed_service()
    assert svc.quote("PD", date(2026, 8, 21), "Rayo Vallecano", "Alaves") is not None


def test_quote_refuses_a_far_away_date():
    """Guards against matching the REVERSE fixture later in the season."""
    svc = _primed_service()
    assert svc.quote("PD", date(2027, 1, 20), "Rayo Vallecano", "Alaves") is None


def test_quote_is_side_sensitive():
    """Home/away must not be interchangeable — the price is not symmetric."""
    svc = _primed_service()
    assert svc.quote("PD", date(2026, 8, 20), "Alaves", "Rayo Vallecano") is None


def test_quote_for_unresolvable_team_is_none():
    svc = _primed_service()
    assert svc.quote("PD", date(2026, 8, 19), "Atl. Madrid", "Malaga") is None


def test_quote_for_league_we_do_not_cover_is_none():
    svc = _primed_service()
    assert svc.quote("CL", date(2026, 8, 22), "Arsenal", "Man City") is None
