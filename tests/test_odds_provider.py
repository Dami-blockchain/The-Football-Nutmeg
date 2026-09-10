"""FootballDataCoUkProvider + OddsService.

Covers the parsing of the real feed's shape, de-vigging, the shared cache /
rate limiter (a 20-fixture Saturday must be ONE request), and every graceful
degradation path — a dead feed must never break an alert.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

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


def test_dead_feed_returns_none_not_empty_list():
    """A None from ``_http_get`` (503/timeout/DNS) must surface from ``fetch``
    as None, distinct from a reached-but-empty ``[]``. OddsService relies on
    that distinction to keep its last-good cache instead of wiping it on an
    outage (Defect A)."""
    assert _provider(payload=None).fetch(["PL"]) is None


def test_garbage_payload_returns_no_rows():
    # A reached-but-malformed 200 (no fixtures.csv header) is a FAILURE, not a
    # reached-and-empty refresh: it must surface as None so the cache is kept.
    assert _provider(payload="not,a,fixtures,file\n1,2,3,4\n").fetch(["PL"]) is None


def test_reached_but_empty_fixtures_file_is_empty_list_not_none():
    """A 200 carrying only a header row = the feed was reached and lists no
    fixtures. That is ``[]`` (a legitimate refresh), NOT ``None`` (a failure)."""
    header_only = "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
    assert _provider(payload=header_only).fetch(["PL"]) == []


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


# ---------------------------------------------------------------------------
# Defect A — a failed refresh must NOT wipe the last-good cache
# ---------------------------------------------------------------------------
_EMPTY_FIXTURES = "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\n"


def _switchable_provider(state: dict, counter: list | None = None):
    """Provider whose HTTP payload can change between calls. ``state['payload']``
    is a CSV string (reached), ``''``/header-only (reached, empty), or None
    (unreachable — what ``_http_get`` returns on any exception)."""

    def fake_get(url, timeout):
        if counter is not None:
            counter.append(url)
        return state["payload"]

    return FootballDataCoUkProvider(_resolver(), fetcher=fake_get)


class _FastRetry(_Settings):
    odds_cache_ttl_seconds = 10.0
    odds_min_request_interval_seconds = 5.0


def _vallecano(svc: OddsService):
    return svc.quote("PD", date(2026, 8, 20), "Rayo Vallecano", "Alaves")


def test_503_mid_session_keeps_prior_index_and_does_not_advance_ttl():
    calls: list[str] = []
    now = [1000.0]
    state = {"payload": FIXTURES_CSV}
    svc = OddsService(_FastRetry(), providers=[_switchable_provider(state, calls)],
                      clock=lambda: now[0])
    primed = asyncio.run(svc.prime(["PD"]))
    assert primed >= 1 and _vallecano(svc) is not None

    # TTL expires AND the min-interval elapses, then the feed 503s.
    now[0] += 20.0
    state["payload"] = None
    retained = asyncio.run(svc.prime(["PD"]))

    assert retained == primed, "a 503 must NOT wipe the last-good index"
    assert _vallecano(svc) is not None, "the primed quote must still be served"
    assert len(calls) == 2, "it DID attempt the refresh"
    # TTL not advanced: the service still considers itself stale, so a recovered
    # feed is re-primed at the next opportunity rather than 6h from the outage.
    assert svc._is_stale() is True


def test_reached_but_empty_keeps_prior_index_and_does_not_advance_ttl():
    """INVERTED (was ...empties_cache_and_advances_ttl). A reached-but-empty
    file is fixtures.csv's NORMAL between-rounds / international-break state, not
    a genuine "no more fixtures ever" signal: replacing the cache on it lost a
    whole matchday's anchors whenever the feed regressed to an older/emptier
    snapshot (the 10 Sep incident). It must now be treated like a 503 — retain
    the last-good index and do NOT advance the TTL. ``quote``'s date guard makes
    the retained index inert once nothing in scope is in range, so a stale index
    goes useless, never wrong."""
    now = [1000.0]
    state = {"payload": FIXTURES_CSV}
    svc = OddsService(_FastRetry(), providers=[_switchable_provider(state)],
                      clock=lambda: now[0])
    primed = asyncio.run(svc.prime(["PD"]))
    assert primed >= 1 and _vallecano(svc) is not None

    now[0] += 20.0
    state["payload"] = _EMPTY_FIXTURES
    retained = asyncio.run(svc.prime(["PD"]))
    assert retained == primed, "a reached-but-empty refresh must NOT wipe the index"
    assert _vallecano(svc) is not None, "the primed quote must still be served"
    assert svc._is_stale() is True, "an empty refresh must NOT advance the TTL"


def test_empty_refresh_logs_this_refresh_coverage_not_lifetime_totals():
    """The provider's coverage counters are lifetime running totals, so the
    empty-cache log must report THIS refresh's delta: after a first prime that
    saw an in-scope card (attempted lifetime >= 1), an empty refresh must log
    ``attempted_this_refresh == 0`` — the operator's tell that the file carried
    no card for us, distinct from a card that failed name resolution. Logging
    the cumulative counter here would read >= 1 and mislead the go/no-go call."""
    import structlog

    now = [1000.0]
    state = {"payload": FIXTURES_CSV}
    provider = _switchable_provider(state)
    svc = OddsService(_FastRetry(), providers=[provider], clock=lambda: now[0])
    assert asyncio.run(svc.prime(["PD"])) >= 1
    assert provider.attempted_fixtures >= 1, "the first card must bump the lifetime total"

    now[0] += 20.0
    state["payload"] = _EMPTY_FIXTURES
    with structlog.testing.capture_logs() as logs:
        asyncio.run(svc.prime(["PD"]))
    evt = next(e for e in logs if e.get("event") == "odds_refresh_empty_cache_retained")
    assert evt["attempted_this_refresh"] == 0, "no in-scope card was seen this refresh"
    assert evt["skipped_this_refresh"] == 0
    assert evt["retained_rows"] >= 1, "the last-good index is what we retained"


def test_retry_backoff_respects_min_interval_after_a_failure():
    calls: list[str] = []
    now = [1000.0]
    state = {"payload": None}  # feed down from the first call

    class S(_Settings):
        odds_cache_ttl_seconds = 10.0
        odds_min_request_interval_seconds = 100.0

    svc = OddsService(S(), providers=[_switchable_provider(state, calls)],
                      clock=lambda: now[0])
    assert asyncio.run(svc.prime(["PD"])) == 0
    assert len(calls) == 1

    # Still stale (nothing ever loaded), but the min-interval must suppress the
    # retry so a persistent outage does not hammer the host.
    now[0] += 20.0
    assert asyncio.run(svc.prime(["PD"])) == 0
    assert len(calls) == 1, "min-interval must hold back the early retry"

    now[0] += 100.0  # interval elapsed -> retry allowed
    asyncio.run(svc.prime(["PD"]))
    assert len(calls) == 2


def test_recovered_feed_reindexes_normally():
    now = [1000.0]
    state = {"payload": None}
    svc = OddsService(_FastRetry(), providers=[_switchable_provider(state)],
                      clock=lambda: now[0])
    assert asyncio.run(svc.prime(["PD"])) == 0
    assert _vallecano(svc) is None

    now[0] += 20.0
    state["payload"] = FIXTURES_CSV  # feed recovers
    assert asyncio.run(svc.prime(["PD"])) >= 1
    assert _vallecano(svc) is not None
    assert svc._is_stale() is False


def test_malformed_200_keeps_prior_index_like_a_503():
    """A reached-but-malformed 200 (e.g. an HTML maintenance page — realistic
    while the host is flapping) must be treated as a failure, NOT a
    reached-and-empty refresh: the last-good index survives and the TTL is not
    advanced, exactly as for a 503 (Defect A, second door)."""
    now = [1000.0]
    state = {"payload": FIXTURES_CSV}
    svc = OddsService(_FastRetry(), providers=[_switchable_provider(state)],
                      clock=lambda: now[0])
    primed = asyncio.run(svc.prime(["PD"]))
    assert primed >= 1 and _vallecano(svc) is not None

    now[0] += 20.0
    state["payload"] = "<html><body>We are down for maintenance</body></html>"
    retained = asyncio.run(svc.prime(["PD"]))

    assert retained == primed, "a malformed body must NOT wipe the index"
    assert _vallecano(svc) is not None
    assert svc._is_stale() is True, "TTL must not advance on a malformed body"


def test_empty_then_nonempty_replaces_the_index():
    """The retain-on-empty rule must NOT strand a stale card: the very next
    NON-empty in-scope refresh replaces the whole index and advances the TTL."""
    now = [1000.0]
    state = {"payload": _EMPTY_FIXTURES}  # first refresh is reached-but-empty
    svc = OddsService(_FastRetry(), providers=[_switchable_provider(state)],
                      clock=lambda: now[0])
    assert asyncio.run(svc.prime(["PD"])) == 0
    assert _vallecano(svc) is None, "nothing to serve from an empty cold start"
    assert svc._is_stale() is True, "a cold empty start must NOT advance the TTL"

    now[0] += 20.0
    state["payload"] = FIXTURES_CSV  # a real card arrives
    assert asyncio.run(svc.prime(["PD"])) >= 1
    assert _vallecano(svc) is not None, "the real card must replace the index"
    assert svc._is_stale() is False, "a non-empty refresh advances the TTL"


# The ACTUAL 10 Sep 2026 incident file: football-data.co.uk recovered after
# four days down and served a structurally valid fixtures.csv (real header, 19
# rows) that is VACUOUS for us — every division is out of scope (E1, E2, G1,
# N1, P1, SC0), zero top-5 fixtures. Under the old empties-cache rule this wiped
# a live matchday's anchors. This is the incident as data, not a stand-in.
_LIVE_INCIDENT_FIXTURE = Path(__file__).parent / "fixtures" / "live_fixtures_20260910.csv"


def test_real_10sep_out_of_scope_file_retains_the_cache():
    live = _LIVE_INCIDENT_FIXTURE.read_text()
    # Pin that this file exercises the reached-but-EMPTY branch, not the
    # unreachable/malformed one: both retain the cache, so without this the test
    # would pass even if the header guard rejected the file as None. It must
    # pass the guard and parse to [] with zero in-scope fixtures attempted.
    probe = _provider(payload=live).fetch(list(_Settings.leagues))
    assert probe == [], "the incident file must parse to an empty in-scope list"

    now = [1000.0]
    state = {"payload": FIXTURES_CSV}
    svc = OddsService(_FastRetry(), providers=[_switchable_provider(state)],
                      clock=lambda: now[0])
    primed = asyncio.run(svc.prime(["PD"]))
    assert primed >= 1 and _vallecano(svc) is not None

    # The recovered-but-vacuous file lands: reached, structurally valid, zero
    # in-scope fixtures. The primed anchor MUST survive and the TTL must hold.
    now[0] += 20.0
    state["payload"] = live
    retained = asyncio.run(svc.prime(["PD"]))
    assert retained == primed, "the real incident file must NOT wipe the index"
    assert _vallecano(svc) is not None, "the live matchday's anchor must survive"
    assert svc._is_stale() is True, "a vacuous recovery must NOT advance the TTL"
