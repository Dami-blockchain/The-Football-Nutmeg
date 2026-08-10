"""Offline tests for the api-football client + lineup_service helpers.

No live network: a stub httpx.AsyncClient returns canned api-football
envelopes so we pin the parsing / graceful-degradation contract.
"""

from __future__ import annotations

import pytest

from betbot.data.api_football import ApiFootballClient
from betbot.data.lineup_service import apply_injuries


class _StubResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _StubClient:
    """Minimal async httpx stand-in mapping path -> canned response."""

    def __init__(self, routes):
        self._routes = routes
        self.is_closed = False

    async def get(self, url, params=None):
        for path, resp in self._routes.items():
            if path in url:
                return resp
        return _StubResponse(200, {"response": [], "errors": []})

    async def aclose(self):
        self.is_closed = True


def _client(routes) -> ApiFootballClient:
    return ApiFootballClient("k", client=_StubClient(routes))


@pytest.mark.asyncio
async def test_get_lineups_parses_home_away_order():
    payload = {
        "errors": [],
        "response": [
            {"formation": "4-3-3", "startXI": [
                {"player": {"name": "A"}}, {"player": {"name": "B"}}]},
            {"formation": "5-4-1", "startXI": [
                {"player": {"name": "C"}}]},
        ],
    }
    c = _client({"/fixtures/lineups": _StubResponse(200, payload)})
    got = await c.get_lineups(123)
    assert got["home"]["formation"] == "4-3-3"
    assert got["home"]["xi"] == ["A", "B"]
    assert got["away"]["formation"] == "5-4-1"
    assert got["away"]["xi"] == ["C"]


@pytest.mark.asyncio
async def test_get_lineups_empty_response_is_none():
    c = _client({"/fixtures/lineups": _StubResponse(200, {"response": [], "errors": []})})
    assert await c.get_lineups(123) is None


@pytest.mark.asyncio
async def test_api_errors_return_none_not_raise():
    c = _client({"/fixtures/lineups": _StubResponse(
        200, {"response": [], "errors": {"token": "invalid"}})})
    assert await c.get_lineups(123) is None


@pytest.mark.asyncio
async def test_http_error_returns_none():
    c = _client({"/fixtures/lineups": _StubResponse(500, {})})
    assert await c.get_lineups(123) is None


@pytest.mark.asyncio
async def test_list_fixtures_shape():
    payload = {
        "errors": [],
        "response": [
            {"fixture": {"id": 42, "date": "2023-08-11T19:00:00+00:00"},
             "teams": {"home": {"name": "Burnley"}, "away": {"name": "Man City"}}},
        ],
    }
    c = _client({"/fixtures": _StubResponse(200, payload)})
    got = await c.list_fixtures(39, 2023, "2023-08-11", "2023-08-11")
    assert got == [{
        "af_fixture_id": 42, "home_name": "Burnley",
        "away_name": "Man City", "kickoff_iso": "2023-08-11T19:00:00+00:00",
    }]


@pytest.mark.asyncio
async def test_get_injuries_dedupes():
    payload = {"errors": [], "response": [
        {"player": {"name": "X"}}, {"player": {"name": "X"}},
        {"player": {"name": "Y"}}]}
    c = _client({"/injuries": _StubResponse(200, payload)})
    assert await c.get_injuries(50, 2025) == ["X", "Y"]


def test_apply_injuries_removes_injured_from_xi():
    xi = {"Kevin De Bruyne", "Rodri", "Haaland"}
    out = apply_injuries(xi, ["Rodri"])
    assert out == {"Kevin De Bruyne", "Haaland"}


def test_apply_injuries_noop_when_empty():
    xi = {"a", "b"}
    assert apply_injuries(xi, []) == xi
