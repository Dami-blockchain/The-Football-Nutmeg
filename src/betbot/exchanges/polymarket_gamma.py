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

# Per-match WC events (slug pattern ``fifwc-<home>-<away>-<date>``, Layout B
# with 3 binary HOME/DRAW/AWAY markets) are NOT returned by the auto-discovered
# numeric soccer tag — they live under this tag SLUG. Verified live 2026-06:
# tag_slug=fifa-world-cup returns the fifwc-* match events. We merge these into
# the soccer-events feed so per-match WC markets are actually discoverable
# (otherwise the matcher falls back to the tournament-winner outright and the
# price-sanity guard rejects every WC fixture).
WC_TAG_SLUG = "fifa-world-cup"

# The football leagues we route fixtures for, keyed by their Gamma ``/sports``
# ``sport`` slug (verified live 2026-08-24). Used both to intersect tag sets
# when auto-discovering the generic soccer tag AND to look up each league's own
# per-match H2H tag in :meth:`GammaClient.discover_league_tags`.
#
# NOTE: these slugs changed at some point in Polymarket's Gamma restructuring —
# the old values ("sa", "bundesliga", "ligue1") no longer exist and matched
# nothing, which quietly broke tag discovery. Current slugs:
#   epl=Premier League, lal=LaLiga, bun=Bundesliga, fl1=Ligue 1,
#   sea=Serie A, ucl=UEFA Champions League.
_FOOTBALL_SPORT_SLUGS = frozenset({"epl", "lal", "bun", "fl1", "sea", "ucl"})

# Known-good ``primaryTagId`` per football-league slug (verified live
# 2026-08-24). Used as a fallback if ``/sports`` is unavailable or its shape
# changes, so per-match H2H discovery keeps working. These are the ONLY tags
# that reliably surface per-match 1X2 events — the generic soccer tag buries
# them behind thousands of long-running outright/awards markets.
_LEAGUE_TAG_FALLBACK: dict[str, int] = {
    "epl": 306,
    "lal": 780,
    "bun": 1494,
    "fl1": 102070,
    "sea": 100618,
    "ucl": 1234,
}

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

    async def discover_league_tags(self) -> list[int]:
        """Per-league ``primaryTagId`` for the leagues we route fixtures for.

        The generic soccer tag holds 2000+ open events dominated by
        long-running outright / season-winner / awards markets. Ordered by
        start date, the actual per-match H2H events (EPL, La Liga, Bundesliga,
        Ligue 1, + UCL) overflow any reasonable fetch window and were never
        discovered — the matcher then only ever saw outright candidates and
        logged ``polymarket_no_h2h_match``. Each league's own tag is small and
        H2H-dense, so we fetch those directly.

        Discovered from ``/sports`` by ``sport`` slug; falls back to the
        known-good ids in :data:`_LEAGUE_TAG_FALLBACK` for any league the
        endpoint doesn't return (or if ``/sports`` fails entirely). Serie A's
        tag is included even though it currently lists no H2H markets — fetching
        it yields only outright events, which the matcher correctly rejects, so
        Serie A keeps missing cleanly rather than being silently unqueried.
        """
        tags: dict[str, int] = {}
        try:
            sports = await self.get_sports()
        except GammaError:
            sports = []
        for s in sports:
            slug = (s.get("sport") or "").lower()
            if slug not in _FOOTBALL_SPORT_SLUGS:
                continue
            raw = s.get("primaryTagId")
            try:
                tags[slug] = int(raw)
            except (TypeError, ValueError):
                continue
        # Fill any league the endpoint didn't hand us from the verified map.
        for slug, tid in _LEAGUE_TAG_FALLBACK.items():
            tags.setdefault(slug, tid)
        return sorted(set(tags.values()))

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

    async def list_events_by_slug(
        self, tag_slug: str, *, closed: bool = False, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Open events for a tag SLUG (e.g. ``fifa-world-cup``), markets decoded.

        The numeric-tag listing misses per-match WC events; querying by the
        tag slug returns them. Pages up to ``limit``."""
        collected: list[dict[str, Any]] = []
        page = 100
        offset = 0
        while len(collected) < limit:
            events = await self._get(
                "/events",
                params={
                    "tag_slug": tag_slug,
                    "closed": "true" if closed else "false",
                    "limit": page,
                    "offset": offset,
                    "order": "startDate",
                    "ascending": "true",
                },
            )
            if not isinstance(events, list) or not events:
                break
            for e in events:
                e["markets"] = [_decode_market(m) for m in (e.get("markets") or [])]
            collected.extend(events)
            if len(events) < page:
                break
            offset += page
        return collected[:limit]

    async def list_soccer_events(
        self, *, tag_id: int | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Open football events for discovery, deduped by event slug/id.

        Three sources, merged in priority order:

        1. **Per-league H2H tags** (EPL/La Liga/Bundesliga/Ligue 1/Serie A/UCL).
           These are the routing targets. The generic soccer tag buries per-match
           1X2 events behind thousands of long-running outright/awards markets, so
           we fetch each league's own (small, H2H-dense) tag directly. Fetched
           FIRST so they can never be trimmed by a downstream cap.
        2. **The generic numeric soccer tag**, paged — coverage for any other
           competition the app may predict, and Layout-A markets.
        3. **Per-match WC events** (a slug tag the numeric tag misses).
        """
        resolved_tag = tag_id if tag_id is not None else await self.discover_soccer_tag()
        collected: list[dict[str, Any]] = []
        seen: set[Any] = set()

        def _merge(events: list[dict[str, Any]]) -> None:
            for e in events:
                key = e.get("slug") or e.get("id")
                if key in seen:
                    continue
                seen.add(key)
                collected.append(e)

        # 1) Per-league H2H tags — the fix for `polymarket_no_h2h_match`.
        try:
            league_tags = await self.discover_league_tags()
        except GammaError:
            league_tags = []
        for lt in league_tags:
            try:
                _merge(
                    await self.list_events(tag_id=lt, closed=False, limit=100)
                )
            except GammaError:
                continue  # one league failing must not sink discovery

        # 2) Generic numeric soccer tag, paged (bounded by its own quota).
        page = 100
        offset = 0
        fetched_generic = 0
        while fetched_generic < limit:
            batch = await self.list_events(
                tag_id=resolved_tag, closed=False, limit=page, offset=offset
            )
            if not batch:
                break
            _merge(batch)
            fetched_generic += len(batch)
            if len(batch) < page:
                break
            offset += page

        # 3) Per-match WC events (slug tag), which the numeric tag omits.
        try:
            wc = await self.list_events_by_slug(WC_TAG_SLUG, limit=limit)
        except GammaError:
            wc = []  # never let the WC supplement break club discovery
        _merge(wc)

        return collected
