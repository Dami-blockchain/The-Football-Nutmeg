"""Regression tests for the top-4-league H2H routing defect.

For ~5 weeks ``polymarket_no_h2h_match`` fired on EPL / La Liga / Bundesliga /
Ligue 1 fixtures whose 1X2 markets provably exist on Polymarket. Root cause:
``GammaClient.list_soccer_events`` fetched only the *generic* soccer tag
(``100350``), ordered by start date and capped at ~200 events. That tag holds
2000+ open events dominated by long-running outright / season-winner / awards
markets, so the actual per-match H2H events overflowed the window and were never
discovered. The matcher then only ever saw outright candidates (which name both
teams via "Will <team> win the league?") and logged the no-H2H note.

The fix fetches each league's own (small, H2H-dense) ``primaryTagId`` directly
and merges those events in first, so per-match 1X2 events are always present.

These tests exercise the WHOLE defect path — ``/sports`` discovery →
``list_soccer_events`` → ``find_market`` — against RECORDED real Gamma responses
(``tests/fixtures/gamma/*.json``, captured 2026-08-24). No live network.

Serie A is the deliberate control: Polymarket genuinely lists NO per-match Serie
A 1X2 markets, only a season-winner outright. Those fixtures must keep missing
cleanly (the outright names both teams, so it classifies but scores 0 → the
existing ``polymarket_no_h2h_match``). A false H2H match is worse than a miss.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket import PolymarketAdapter
from betbot.exchanges.polymarket_gamma import GammaClient

FIXTURES = Path(__file__).parent / "fixtures" / "gamma"
KICKOFF = datetime(2026, 8, 24, 15, 0, tzinfo=timezone.utc)

# primaryTagId per league (matches tests/fixtures/gamma/sports.json).
_EPL_TAG = 306
_SERIE_A_TAG = 100618
_OTHER_LEAGUE_TAGS = {780, 1494, 102070, 1234}  # laliga, bundesliga, ligue1, ucl
_GENERIC_SOCCER_TAG = 100350


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


def _gamma_handler(request: httpx.Request) -> httpx.Response:
    """Route recorded fixtures the way the live Gamma API does.

    The key realism: the GENERIC soccer tag (page 0) returns only outright /
    awards noise — NONE of it is a top-4-league per-match H2H event, exactly as
    live. The per-match H2H events are reachable ONLY via the league tags.
    """
    path = request.url.path
    if path == "/sports":
        return httpx.Response(200, json=_load("sports.json"))
    if path == "/events":
        params = request.url.params
        if params.get("tag_slug"):  # WC slug tag — none in this scenario
            return httpx.Response(200, json=[])
        tag_id = int(params.get("tag_id", "0"))
        offset = int(params.get("offset", "0"))
        if tag_id == _EPL_TAG:
            return httpx.Response(200, json=_load("epl_h2h_event.json"))
        if tag_id == _SERIE_A_TAG:
            return httpx.Response(200, json=_load("seriea_outright_event.json"))
        if tag_id in _OTHER_LEAGUE_TAGS:
            return httpx.Response(200, json=[])
        if tag_id == _GENERIC_SOCCER_TAG:
            # Only page 0 has (noise) content; everything after is empty.
            return httpx.Response(200, json=_load("generic_tag_page0.json") if offset == 0 else [])
        return httpx.Response(200, json=[])
    return httpx.Response(404)


def _gamma() -> GammaClient:
    return GammaClient(client=httpx.AsyncClient(transport=httpx.MockTransport(_gamma_handler)))


def _adapter(gamma: GammaClient) -> PolymarketAdapter:
    # Empty (fuzzy-only) resolver: exercises the real normalisation path.
    return PolymarketAdapter(gamma, TeamAliasResolver())


# ----------------------------------------------------------------------
# discover_league_tags
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_discover_league_tags_from_sports():
    async with _gamma() as g:
        tags = await g.discover_league_tags()
    # All six league primaryTagIds, from the recorded /sports payload.
    assert set(tags) == {306, 780, 1494, 102070, 100618, 1234}


@pytest.mark.asyncio
async def test_discover_league_tags_falls_back_on_sports_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with GammaClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as g:
        tags = await g.discover_league_tags()
    # Verified-good fallback map still yields every league tag.
    assert set(tags) == {306, 780, 1494, 102070, 100618, 1234}


# ----------------------------------------------------------------------
# list_soccer_events — per-league H2H events must be merged in
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_soccer_events_includes_league_h2h_events():
    async with _gamma() as g:
        events = await g.list_soccer_events()
    slugs = {e.get("slug") for e in events}
    # The EPL per-match H2H event is present even though the generic tag's
    # first page (the only slice the old code fetched) never contains it.
    assert "epl-ful-che-2026-08-24" in slugs
    # The generic-tag noise is still merged (other-competition coverage).
    assert "ballon-dor-winner-2026" in slugs


@pytest.mark.asyncio
async def test_generic_tag_page_alone_lacks_top_league_h2h():
    """Guard on the fixtures themselves: the generic page is pure outright/awards
    noise, so a fix that fetched ONLY the generic tag would still miss. This
    proves the league-tag fetch is what makes routing succeed."""
    page0 = _load("generic_tag_page0.json")
    assert not any((e.get("slug") or "").startswith("epl-") for e in page0)


# ----------------------------------------------------------------------
# find_market — the end-to-end defect and its control
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_epl_fixture_now_routes_to_h2h_event():
    async with _gamma() as g:
        ref = await _adapter(g).find_market("Fulham FC", "Chelsea FC", KICKOFF)
    assert ref is not None, "EPL H2H market exists on Polymarket — must route"
    assert ref.market_id == "epl-ful-che-2026-08-24"
    assert ref.metadata["layout"] == "B"
    # HOME and AWAY YES tokens were classified from the real market questions.
    tokens = ref.metadata["outcome_tokens"]
    assert "HOME" in tokens and "AWAY" in tokens and "DRAW" in tokens


@pytest.mark.asyncio
async def test_serie_a_fixture_still_misses_cleanly():
    """Serie A genuinely has no per-match 1X2 market. The season-winner outright
    names both teams (so it classifies) but must NOT be routed as the fixture —
    a false H2H match is worse than a miss."""
    async with _gamma() as g:
        ref = await _adapter(g).find_market("Inter Milan", "AC Milan", KICKOFF)
    assert ref is None


@pytest.mark.asyncio
async def test_serie_a_outright_is_never_misrouted_as_home_away():
    """Even reversed / a different Serie A pairing pulled from the same outright
    must return no market, never the outright event's tokens."""
    async with _gamma() as g:
        ref = await _adapter(g).find_market("Napoli", "AS Roma", KICKOFF)
    assert ref is None


# ----------------------------------------------------------------------
# Paged league fetch — a full page 0 must not hide a page-1 H2H event
# ----------------------------------------------------------------------
def _prop_filler(i: int) -> dict:
    """A benign non-1X2 event (won't classify): props/side markets like the
    ones that inflate a real league tag and push main events past page 0."""
    return {
        "slug": f"epl-filler-{i}-exact-score",
        "title": f"Filler {i} - Exact Score",
        "markets": [],
    }


def _paged_epl_handler(request: httpx.Request) -> httpx.Response:
    """Same as the default handler, but tag 306 (EPL) returns a FULL page 0 of
    100 filler events with the real H2H event only on page 1 (offset 100)."""
    path = request.url.path
    if path == "/sports":
        return httpx.Response(200, json=_load("sports.json"))
    if path == "/events":
        params = request.url.params
        if params.get("tag_slug"):
            return httpx.Response(200, json=[])
        tag_id = int(params.get("tag_id", "0"))
        offset = int(params.get("offset", "0"))
        if tag_id == _EPL_TAG:
            if offset == 0:
                return httpx.Response(200, json=[_prop_filler(i) for i in range(100)])
            if offset == 100:
                return httpx.Response(200, json=_load("epl_h2h_event.json"))
            return httpx.Response(200, json=[])
        if tag_id == _SERIE_A_TAG:
            return httpx.Response(200, json=_load("seriea_outright_event.json"))
        if tag_id in _OTHER_LEAGUE_TAGS:
            return httpx.Response(200, json=[])
        if tag_id == _GENERIC_SOCCER_TAG:
            return httpx.Response(
                200, json=_load("generic_tag_page0.json") if offset == 0 else []
            )
        return httpx.Response(200, json=[])
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_league_tag_is_paged_past_a_full_first_page():
    """Regression for the Fable finding on 48b3241: a single-page league fetch
    would miss a main 1X2 event sitting behind a full page 0 of prop events.
    The EPL H2H event is on page 1 here and must still be discovered + routed."""
    gamma = GammaClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(_paged_epl_handler))
    )
    async with gamma as g:
        events = await g.list_soccer_events()
        slugs = {e.get("slug") for e in events}
        assert "epl-ful-che-2026-08-24" in slugs, "page-1 H2H event must be fetched"
        ref = await _adapter(g).find_market("Fulham FC", "Chelsea FC", KICKOFF)
    assert ref is not None
    assert ref.market_id == "epl-ful-che-2026-08-24"
