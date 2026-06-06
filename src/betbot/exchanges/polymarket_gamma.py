"""Polymarket Gamma API client — market discovery only (public, no auth).

Gamma (https://gamma-api.polymarket.com) is Polymarket's public catalogue API.
We use it to find football match events and their per-outcome CLOB token ids.
Order placement happens against the CLOB, not here.

Football match events use **Layout B**: one event per match with THREE binary
YES/NO markets — "Will <home> win?", "Will it end in a draw?", "Will <away>
win?". A few events use **Layout A**: a single market carrying three outcomes
and three token ids. This client stays thin — it fetches and JSON-decodes the
string fields (``clobTokenIds``, ``outcomes``, ``outcomePrices``); the adapter
classifies HOME/DRAW/AWAY.

Empirically (2026-06) the football/soccer tag is ``100350`` — the value
``100381`` named in older notes no longer returns football events, so we
auto-discover via ``/sports`` and fall back to ``100350``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from betbot.logging import get_logger

log = get_logger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Fallback soccer tag id, verified against the live API. Auto-discovery via
# /sports refines this when possible.
SOCCER_TAG_ID = 100350

# football-data competition_code -> Gamma `sport` slug, used to intersect tag
# sets during auto-discovery of the soccer tag.
_FOOTBALL_SPORT_SLUGS = frozenset({"epl", "lal", "sa", "bundesliga", "ligue1", "ucl", "wc"})

_JSON_STRING_FIELDS = ("clobTokenIds", "outcomes", "outcomePrices")


class GammaError(RuntimeError):
    """A Gamma API request failed."""


def _decode_market(market: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a market dict with JSON-string fields decoded.

    Gamma serialises ``clobTokenIds``/``outcomes``/``outcomePrices`` as JSON
    *strings* (e.g. ``'["Yes", "No"]'``). Decode them to real lists; on any
    parse error fall back to an empty list rather than raising, so one
    malformed market can't sink a whole discovery run.
    """
    decoded = dict(market)
    for field in _JSON_STRING_FIELDS:
        value = decoded.get(field)
        if isinstance(value, str):
            try:
                decoded[field] = json.loads(value)
            except (ValueError, TypeError):
                log.warning("gamma_json_decode_failed", field=field, raw=value[:80])
                decoded[field] = []
    return decoded


class GammaClient:
    """Thin async wrapper over the public Gamma endpoints we need."""

    def __init__(
        self,
        base_url: str = GAMMA_BASE,
        *,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def __aenter__(self) -> "GammaClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        try:
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise GammaError(f"GET {path} failed: {e}") from e

    # ------------------------------------------------------------------
    async def get_sports(self) -> list[dict[str, Any]]:
        data = await self._get("/sports")
        return data if isinstance(data, list) else []

    async def discover_soccer_tag(self) -> int:
        """Best-effort soccer tag id from /sports; fall back to the constant.

        Strategy: take the tag sets of the known football leagues and return
        the tag they share that isn't the generic catch-all. If anything is
        ambiguous or the call fails, return :data:`SOCCER_TAG_ID`.
        """
        try:
            sports = await self.get_sports()
        except GammaError:
            return SOCCER_TAG_ID

        tag_sets: list[set[int]] = []
        for s in sports:
            slug = (s.get("sport") or "").lower()
            if slug not in _FOOTBALL_SPORT_SLUGS:
                continue
            raw = s.get("tags") or ""
            try:
                tags = {int(t) for t in str(raw).split(",") if t.strip()}
            except ValueError:
                continue
            if tags:
                tag_sets.append(tags)

        if len(tag_sets) < 2:
            return SOCCER_TAG_ID
        common = set.intersection(*tag_sets)
        # Drop the generic "all sports" tag (id 1) and any obviously-wrong ids.
        common.discard(1)
        if SOCCER_TAG_ID in common:
            return SOCCER_TAG_ID
        return min(common) if common else SOCCER_TAG_ID

    # ------------------------------------------------------------------
    async def list_events(
        self,
        *,
        tag_id: int,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return events for a tag, each with its markets JSON-decoded."""
        events = await self._get(
            "/events",
            params={
                "tag_id": tag_id,
                "closed": "true" if closed else "false",
                "limit": limit,
                "offset": offset,
                "order": "startDate",
                "ascending": "true",
            },
        )
        if not isinstance(events, list):
            return []
        for e in events:
            e["markets"] = [_decode_market(m) for m in (e.get("markets") or [])]
        return events

    async def list_soccer_events(
        self, *, tag_id: int | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Open football events. Pages through results up to ``limit``."""
        resolved_tag = tag_id if tag_id is not None else await self.discover_soccer_tag()
        collected: list[dict[str, Any]] = []
        page = 100
        offset = 0
        while len(collected) < limit:
            batch = await self.list_events(
                tag_id=resolved_tag, closed=False, limit=page, offset=offset
            )
            if not batch:
                break
            collected.extend(batch)
            if len(batch) < page:
                break
            offset += page
        return collected[:limit]
