"""Async client for API-Football (API-SPORTS) — player availability (injuries).

Mirrors :class:`betbot.data.football_data.FootballDataClient`: async httpx
client + a sliding-window rate limiter + a TTL cache. The only thing this
client does is supply a COARSE "how many players are unavailable" count per
national team, which the international engine turns into a small, conservative
negative Glicko adjustment (see :mod:`betbot.strategy.availability`).

Honesty / scope:
- Free tier is 100 requests/day. Team-id lookups are cached (TTLCache) so a
  repeated team name within a run/day doesn't burn the budget; injuries are
  fetched per fixture.
- This is an OPTIONAL integration. With no API key it is a graceful NO-OP at
  the caller (like the other optional integrations) — every method is written
  so a missing key / network blip can be caught and turned into "0 unavailable"
  WITHOUT ever breaking a prediction.
- Confirmed-XI / late team-sheet (who actually starts) is explicitly OUT of
  scope for v1; this is absence-count only. A v2 would weight by player
  importance (market value / minutes) and consume the confirmed lineup endpoint.

Auth: header ``x-apisports-key: <KEY>``. Base ``https://v3.football.api-sports.io``.
Responses are JSON of the form ``{"response": [...]}``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import httpx

from betbot.logging import get_logger
from betbot.utils.cache import TTLCache

log = get_logger(__name__)


class ApiFootballError(RuntimeError):
    """Raised on transport / API errors talking to API-Football."""


class _SlidingWindow:
    """N requests per 60s, blocking when full (same shape as football_data)."""

    def __init__(self, limit_per_min: int) -> None:
        self._limit = max(1, limit_per_min)
        self._timestamps: deque[float] = deque(maxlen=self._limit)

    async def acquire(self) -> None:
        loop = asyncio.get_event_loop()
        now = loop.time()
        while len(self._timestamps) >= self._limit:
            oldest = self._timestamps[0]
            wait = 60.0 - (now - oldest)
            if wait <= 0:
                self._timestamps.popleft()
                break
            log.debug("api_football_rate_limit_wait", seconds=round(wait, 2))
            await asyncio.sleep(wait + 0.05)
            now = loop.time()
        self._timestamps.append(now)


class ApiFootballClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://v3.football.api-sports.io",
        *,
        rate_limit_per_min: int = 10,
        # Team-id lookups are stable, so cache them aggressively (1h) to protect
        # the 100/day budget. Injuries change match-to-match; cache them briefly.
        team_id_ttl_seconds: float = 3600.0,
        cache_ttl_seconds: float = 300.0,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            log.warning(
                "api_football_no_api_key",
                note="injury adjustment will NO-OP (no key configured).",
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._limiter = _SlidingWindow(rate_limit_per_min)
        self._cache: TTLCache[Any] = TTLCache(ttl_seconds=cache_ttl_seconds)
        self._team_id_cache: TTLCache[int | None] = TTLCache(
            ttl_seconds=team_id_ttl_seconds
        )
        self._owns_client = client is None
        headers = {"x-apisports-key": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers=headers
        )

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    async def __aenter__(self) -> "ApiFootballClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        cache_key = (path, tuple(sorted((params or {}).items())))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._limiter.acquire()
        url = f"{self._base_url}{path}"
        log.debug("api_football_request", path=path, params=params)
        try:
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as e:
            raise ApiFootballError(f"network error calling {path}: {e}") from e

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "10"))
            log.warning("api_football_rate_limited", retry_after=retry_after)
            await asyncio.sleep(min(retry_after, 60.0))
            return await self._get(path, params)

        if resp.status_code >= 400:
            raise ApiFootballError(f"{path} -> {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise ApiFootballError(f"non-JSON response from {path}") from e

        self._cache.set(cache_key, data)
        return data

    # ------------------------------------------------------------------
    async def resolve_team_id(self, name: str) -> int | None:
        """Map a (national) team name to API-Football's numeric team id.

        Cached so repeated lookups within the TTL don't burn the 100/day
        budget. Tolerant of national-team naming: API-Football lists national
        sides under the country name (e.g. "Brazil", "USA"), so we pass the
        name straight to ``/teams?search=`` and prefer an exact (case-folded)
        match, falling back to the first national-team result.

        Returns ``None`` (never raises) when nothing matches, the lookup
        errors, or no key is configured — the caller treats None as "no
        injury data for this team".
        """
        key = (name or "").strip()
        if not key:
            return None
        cached = self._team_id_cache.get(key.casefold())
        if cached is not None:
            return cached
        # Distinguish a cached "miss" from "not cached": TTLCache returns None
        # for both, so a miss re-queries within the TTL. That's acceptable —
        # misses are rare (real WC teams resolve) and still cheap vs. the budget.
        try:
            data = await self._get("/teams", params={"search": key})
        except ApiFootballError as e:
            log.warning("api_football_team_lookup_failed", team=name, error=str(e))
            return None

        results = data.get("response") or []
        team_id = _pick_team_id(results, key)
        if team_id is not None:
            self._team_id_cache.set(key.casefold(), team_id)
        return team_id

    async def get_injuries(
        self, team_id: int, season: int
    ) -> list[dict[str, Any]]:
        """Return the list of unavailable players for a team this season.

        Each item is a small dict ``{"name": str, "reason": str, "type": str}``.
        Never raises: on any API error returns ``[]`` so the caller falls back
        to "no adjustment". The caller is responsible for the higher-level
        try/except too, but this method is defensive on its own.
        """
        try:
            data = await self._get(
                "/injuries", params={"team": int(team_id), "season": int(season)}
            )
        except ApiFootballError as e:
            log.warning("api_football_injuries_failed", team_id=team_id, error=str(e))
            return []
        return _parse_injuries(data.get("response") or [])


# ----------------------------------------------------------------------
# Pure parsing helpers (no IO) — easy to unit-test.
# ----------------------------------------------------------------------
def _pick_team_id(results: list[dict[str, Any]], name: str) -> int | None:
    """Pick the best team id from a ``/teams?search=`` response.

    Prefers an exact (case-folded) name match; otherwise prefers a national
    team; otherwise the first result. Returns None if nothing usable.
    """
    target = name.strip().casefold()
    national: int | None = None
    first: int | None = None
    for item in results:
        team = item.get("team") or {}
        tid = team.get("id")
        if not isinstance(tid, int):
            continue
        if first is None:
            first = tid
        tname = str(team.get("name", "")).strip().casefold()
        is_national = bool(team.get("national"))
        if tname == target:
            # Exact match wins outright, but prefer the national variant if the
            # exact match is a club sharing the name and a national side exists.
            if is_national or national is None:
                return tid
        if is_national and national is None:
            national = tid
    return national if national is not None else first


def _parse_injuries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten API-Football /injuries rows into ``{name, reason, type}`` dicts.

    De-duplicates by player id (the endpoint can list a player once per
    fixture), so the count reflects DISTINCT unavailable players, not rows.
    """
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in rows:
        player = row.get("player") or {}
        pid = player.get("id")
        marker = pid if pid is not None else str(player.get("name", "")).casefold()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(
            {
                "name": player.get("name") or "",
                # /injuries: player.type is the status (e.g. "Missing Fixture"),
                # player.reason is the cause (e.g. "Knee Injury").
                "type": player.get("type") or "",
                "reason": player.get("reason") or "",
            }
        )
    return out
