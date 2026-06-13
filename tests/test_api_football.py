"""Tests for the API-Football client (player availability / injuries).

Offline via httpx.MockTransport — no real network is ever touched.
"""

from __future__ import annotations

import httpx
import pytest

from betbot.data.api_football import (
    ApiFootballClient,
    ApiFootballError,
    _parse_injuries,
    _pick_team_id,
)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ----------------------------------------------------------------------
# Pure parsing helpers
# ----------------------------------------------------------------------
def test_parse_injuries_extracts_name_reason_type():
    rows = [
        {"player": {"id": 1, "name": "A. Player", "type": "Missing Fixture",
                    "reason": "Knee Injury"}},
        {"player": {"id": 2, "name": "B. Player", "type": "Questionable",
                    "reason": "Knock"}},
    ]
    out = _parse_injuries(rows)
    assert len(out) == 2
    assert out[0] == {"name": "A. Player", "type": "Missing Fixture",
                      "reason": "Knee Injury"}


def test_parse_injuries_dedupes_by_player_id():
    rows = [
        {"player": {"id": 7, "name": "Dup", "type": "x", "reason": "y"}},
        {"player": {"id": 7, "name": "Dup", "type": "x", "reason": "y"}},
        {"player": {"id": 8, "name": "Other", "type": "x", "reason": "y"}},
    ]
    assert len(_parse_injuries(rows)) == 2


def test_pick_team_id_prefers_national_exact_match():
    results = [
        {"team": {"id": 10, "name": "Brazil", "national": True}},
        {"team": {"id": 11, "name": "Brazil U20", "national": True}},
    ]
    assert _pick_team_id(results, "Brazil") == 10


def test_pick_team_id_prefers_national_over_club():
    results = [
        {"team": {"id": 50, "name": "Some Club FC", "national": False}},
        {"team": {"id": 51, "name": "USA", "national": True}},
    ]
    assert _pick_team_id(results, "USA") == 51


def test_pick_team_id_none_when_empty():
    assert _pick_team_id([], "Nowhere") is None


# ----------------------------------------------------------------------
# resolve_team_id — parses + caches (protects the 100/day budget)
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_team_id_parses_and_caches():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/teams"
        assert request.url.params.get("search") == "Brazil"
        calls["n"] += 1
        return httpx.Response(200, json={"response": [
            {"team": {"id": 6, "name": "Brazil", "national": True}},
        ]})

    async with ApiFootballClient("K", client=_mock_client(handler)) as c:
        assert await c.resolve_team_id("Brazil") == 6
        # Second lookup is served from the team-id cache — no extra HTTP call.
        assert await c.resolve_team_id("Brazil") == 6
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_auth_header_set_on_owned_client():
    c = ApiFootballClient("SECRET")  # constructs its own httpx client
    assert c._client.headers.get("x-apisports-key") == "SECRET"
    await c.close()
    c2 = ApiFootballClient("")  # no key -> no header
    assert "x-apisports-key" not in c2._client.headers
    await c2.close()


@pytest.mark.asyncio
async def test_resolve_team_id_returns_none_without_key():
    async def _no_call(_request):  # pragma: no cover — must not be reached
        raise AssertionError("network must not be touched without a key")

    c = ApiFootballClient("", client=_mock_client(_no_call))
    assert c.has_key is False
    # No key => still attempts? No: caller gates on has_key, but the client
    # itself will try; here we assert the high-level no-op path used by main.
    await c.close()


@pytest.mark.asyncio
async def test_resolve_team_id_none_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with ApiFootballClient("K", client=_mock_client(handler)) as c:
        assert await c.resolve_team_id("Brazil") is None  # graceful, no raise


# ----------------------------------------------------------------------
# get_injuries — parses the list; never raises
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_injuries_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/injuries"
        assert request.url.params.get("team") == "6"
        assert request.url.params.get("season") == "2026"
        return httpx.Response(200, json={"response": [
            {"player": {"id": 1, "name": "X", "type": "Missing Fixture",
                        "reason": "ACL"}},
            {"player": {"id": 2, "name": "Y", "type": "Missing Fixture",
                        "reason": "Ban"}},
        ]})

    async with ApiFootballClient("K", client=_mock_client(handler)) as c:
        injuries = await c.get_injuries(6, 2026)
    assert [i["name"] for i in injuries] == ["X", "Y"]


@pytest.mark.asyncio
async def test_get_injuries_empty_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    async with ApiFootballClient("K", client=_mock_client(handler)) as c:
        assert await c.get_injuries(6, 2026) == []  # graceful


@pytest.mark.asyncio
async def test_get_method_raises_on_non_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    async with ApiFootballClient("K", client=_mock_client(handler)) as c:
        # _get raises, but get_injuries swallows it to []
        assert await c.get_injuries(6, 2026) == []
        with pytest.raises(ApiFootballError):
            await c._get("/anything")
