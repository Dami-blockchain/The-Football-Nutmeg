"""Tests for the Gamma discovery client (Phase 2). Offline via MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from betbot.exchanges.polymarket_gamma import (
    GammaClient,
    _decode_market,
)


# ----------------------------------------------------------------------
# _decode_market — the #1 Gamma gotcha: JSON-string fields.
# ----------------------------------------------------------------------
def test_decode_market_parses_json_string_fields():
    raw = {
        "question": "Will X win?",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["111", "222"]',
        "outcomePrices": '["0.6", "0.4"]',
    }
    m = _decode_market(raw)
    assert m["outcomes"] == ["Yes", "No"]
    assert m["clobTokenIds"] == ["111", "222"]
    assert m["outcomePrices"] == ["0.6", "0.4"]
    # original dict is not mutated
    assert isinstance(raw["outcomes"], str)


def test_decode_market_tolerates_garbage():
    m = _decode_market({"outcomes": "not json", "clobTokenIds": '["1"]'})
    assert m["outcomes"] == []          # garbage -> empty, no raise
    assert m["clobTokenIds"] == ["1"]


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_list_events_decodes_nested_markets():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/events"
        body = [
            {
                "title": "A vs B",
                "markets": [
                    {"question": "Will A win?", "outcomes": '["Yes", "No"]',
                     "clobTokenIds": '["1", "2"]', "outcomePrices": '["0.5", "0.5"]'},
                ],
            }
        ]
        return httpx.Response(200, json=body)

    async with GammaClient(client=_mock_client(handler)) as g:
        events = await g.list_events(tag_id=100350)
    assert events[0]["markets"][0]["outcomes"] == ["Yes", "No"]
    assert events[0]["markets"][0]["clobTokenIds"] == ["1", "2"]


@pytest.mark.asyncio
async def test_discover_soccer_tag_intersects_league_tags():
    def handler(request: httpx.Request) -> httpx.Response:
        # epl and lal share tags 1 and 100350; 100350 is the soccer tag.
        sports = [
            {"sport": "epl", "tags": "1,82,100350,100639"},
            {"sport": "lal", "tags": "1,780,100350,100639"},
            {"sport": "ncaab", "tags": "1,100149"},
        ]
        return httpx.Response(200, json=sports)

    async with GammaClient(client=_mock_client(handler)) as g:
        tag = await g.discover_soccer_tag()
    # 100639 is also shared, but 100350 is our known soccer tag and is preferred.
    assert tag == 100350


@pytest.mark.asyncio
async def test_discover_soccer_tag_falls_back_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with GammaClient(client=_mock_client(handler)) as g:
        tag = await g.discover_soccer_tag()
    assert tag == 100350
