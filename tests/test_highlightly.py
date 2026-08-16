"""Offline tests for the Highlightly Soccer client + lineup_service resolution.

No live network: a stub httpx.AsyncClient returns canned Highlightly envelopes so
we pin the parsing / graceful-degradation contract + the Cloudflare-UA header.
"""

from __future__ import annotations

import pytest

from betbot.data.highlightly import HighlightlyClient, highlightly_league_name


class _StubResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = str(payload)

    def json(self):
        return self._payload


class _StubClient:
    """Minimal async httpx stand-in: records requests, maps path -> response."""

    def __init__(self, routes, *, headers=None):
        self._routes = routes
        self.is_closed = False
        self.requests: list[tuple[str, dict | None]] = []
        # httpx.AsyncClient exposes default headers on .headers; the real client
        # merges these into every request, so recording them proves the UA + key
        # ride along.
        self.headers = headers or {}

    async def get(self, url, params=None):
        self.requests.append((url, params))
        for path, resp in self._routes.items():
            if path in url:
                return resp
        return _StubResponse(200, {"data": []})

    async def aclose(self):
        self.is_closed = True


def _client(routes, *, key="k") -> tuple[HighlightlyClient, _StubClient]:
    headers = {"User-Agent": "Mozilla/5.0 test-UA"}
    if key:
        headers["x-rapidapi-key"] = key
    stub = _StubClient(routes, headers=headers)
    return HighlightlyClient(key, client=stub), stub


# ----------------------------------------------------------------------
# Header contract: key + browser UA (Cloudflare 1010 without a UA)
# ----------------------------------------------------------------------
def test_default_headers_carry_key_and_browser_ua():
    # The real client (no injected stub) builds its own header set.
    c = HighlightlyClient("secret-key")
    hdrs = c._client.headers
    assert hdrs.get("x-rapidapi-key") == "secret-key"
    ua = hdrs.get("User-Agent") or hdrs.get("user-agent") or ""
    assert "Mozilla" in ua  # a browser UA, mandatory for Cloudflare


# ----------------------------------------------------------------------
# list_matches — parses {data: [...]} into {match_id, home_name, away_name, state}
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_matches_parses_data_wrapper():
    payload = {"data": [
        {"id": 111, "homeTeam": {"name": "Espanyol"},
         "awayTeam": {"name": "Levante"}, "state": {"description": "Not started"}},
        {"id": 222, "homeTeam": {"name": "Racing Santander"},
         "awayTeam": {"name": "Villarreal"}, "state": {"description": "Second half"}},
    ]}
    c, stub = _client({"/matches": _StubResponse(200, payload)})
    got = await c.list_matches("La Liga", "2026-08-16")
    assert got == [
        {"match_id": 111, "home_name": "Espanyol", "away_name": "Levante",
         "state": "Not started"},
        {"match_id": 222, "home_name": "Racing Santander",
         "away_name": "Villarreal", "state": "Second half"},
    ]
    # Request carried the league + date params.
    url, params = stub.requests[0]
    assert "/matches" in url
    assert params["leagueName"] == "La Liga" and params["date"] == "2026-08-16"


@pytest.mark.asyncio
async def test_list_matches_accepts_bare_list():
    payload = [{"id": 5, "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
                "state": {"description": "x"}}]
    c, _ = _client({"/matches": _StubResponse(200, payload)})
    got = await c.list_matches("La Liga", "2026-08-16")
    assert got[0]["match_id"] == 5


@pytest.mark.asyncio
async def test_list_matches_http_error_is_empty_not_raise():
    c, _ = _client({"/matches": _StubResponse(403, {"error": "1010"})})
    assert await c.list_matches("La Liga", "2026-08-16") == []


# ----------------------------------------------------------------------
# get_lineup — flatten initialLineup rows into 11 names; None when not posted
# ----------------------------------------------------------------------
def _lineup_payload():
    # 5 formation rows summing to 11 players per side (real Highlightly shape).
    home_rows = [
        [{"name": "Julen Agirrezabala", "number": 1, "position": "GK", "id": 1}],
        [{"name": "Jorge Salinas"}, {"name": "Pedro Felipe"},
         {"name": "Pablo Ramón"}, {"name": "Alvaro Mantilla"}],
        [{"name": "Sergio Martinez"}, {"name": "Gustavo Puerta"}],
        [{"name": "Iñigo Vicente"}, {"name": "Sergio Canales"},
         {"name": "Andrés Martín"}],
        [{"name": "Asier Villalibre"}],
    ]
    away_rows = [[{"name": f"Away{i}"} for i in range(11)]]
    return {
        "homeTeam": {"formation": "4-2-3-1", "initialLineup": home_rows},
        "awayTeam": {"formation": "4-4-2", "initialLineup": away_rows},
    }


@pytest.mark.asyncio
async def test_get_lineup_flattens_rows_into_eleven():
    c, _ = _client({"/lineups/": _StubResponse(200, _lineup_payload())})
    got = await c.get_lineup(1336359273)
    assert got["home"]["formation"] == "4-2-3-1"
    assert len(got["home"]["xi"]) == 11
    assert got["home"]["xi"][0] == "Julen Agirrezabala"
    assert "Asier Villalibre" in got["home"]["xi"]
    assert len(got["away"]["xi"]) == 11


@pytest.mark.asyncio
async def test_get_lineup_empty_rows_is_none():
    payload = {"homeTeam": {"formation": "Unknown", "initialLineup": []},
               "awayTeam": {"formation": "Unknown", "initialLineup": []}}
    c, _ = _client({"/lineups/": _StubResponse(200, payload)})
    assert await c.get_lineup(999) is None


@pytest.mark.asyncio
async def test_get_lineup_unknown_formation_side_is_dropped():
    payload = {
        "homeTeam": {"formation": "4-3-3", "initialLineup":
                     [[{"name": f"H{i}"} for i in range(11)]]},
        "awayTeam": {"formation": "Unknown", "initialLineup": []},
    }
    c, _ = _client({"/lineups/": _StubResponse(200, payload)})
    got = await c.get_lineup(1)
    assert len(got["home"]["xi"]) == 11
    assert got["away"]["xi"] == []  # away not posted -> empty, still non-None


@pytest.mark.asyncio
async def test_get_lineup_http_error_is_none():
    c, _ = _client({"/lineups/": _StubResponse(500, {})})
    assert await c.get_lineup(1) is None


def test_league_name_map():
    assert highlightly_league_name("PD") == "La Liga"
    assert highlightly_league_name("pl") == "Premier League"
    assert highlightly_league_name("CL") == "Champions League"
    assert highlightly_league_name("XYZ") is None
