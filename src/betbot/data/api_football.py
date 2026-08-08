"""Async client for api-football (api-sports.io) v3 — FREE tier only.

FREE-tier resource for the R4a lineup layer:
  * confirmed starting XI + formations (``GET /fixtures/lineups``),
  * injuries (``GET /injuries``),
  * season player-minutes for "expected regular" weighting
    (``GET /players``, paginated).

Budget-aware: FREE tier is ~100 requests/day, ~10/min. We reuse the same
sliding-window limiter shape as :mod:`betbot.data.football_data` and cache
within a run. Every public method is **graceful**: on any transport / API /
parse error it logs and returns ``None`` / an empty container — it NEVER
raises up to a caller, so a missing lineup silently falls back to the
baseline (lineup-free) prediction.

Auth: header ``x-apisports-key: <API_FOOTBALL_KEY>``. Response envelope is
``{"response": [...], "errors": ...}``; ``response`` is empty (not an error)
when a lineup has not yet been posted (~>60min before kickoff).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import httpx

from betbot.logging import get_logger
from betbot.utils.cache import TTLCache

log = get_logger(__name__)

# api-football hard-caps player pagination; we cap at 3 pages to protect the
# 100/day budget (each page is one request).
_MAX_PLAYER_PAGES = 3


class _SlidingWindow:
    """N requests per 60s, blocking when full (shared shape with football_data)."""

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
            log.debug("af_rate_limit_wait", seconds=round(wait, 2))
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
        cache_ttl_seconds: float = 300.0,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            log.warning("api_football_no_api_key", note="Calls will fail; graceful.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._limiter = _SlidingWindow(rate_limit_per_min)
        self._cache: TTLCache[Any] = TTLCache(ttl_seconds=cache_ttl_seconds)
        self._owns_client = client is None
        headers = {"x-apisports-key": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers=headers
        )
        # How many live HTTP requests this client has actually issued — lets
        # scripts report their budget spend.
        self.request_count = 0

    async def __aenter__(self) -> "ApiFootballClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    @classmethod
    def from_settings(cls, settings) -> "ApiFootballClient":
        return cls(
            settings.api_football_key,
            settings.api_football_base_url,
            rate_limit_per_min=settings.api_football_rate_limit_per_min,
        )

    # ------------------------------------------------------------------
    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> list[Any] | None:
        """GET a path -> the ``response`` list, or ``None`` on any error.

        Never raises. An empty ``response`` (e.g. lineup not posted yet) is a
        valid non-error result and returns ``[]``.
        """
        cache_key = (path, tuple(sorted((params or {}).items())))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._limiter.acquire()
        url = f"{self._base_url}{path}"
        log.debug("af_request", path=path, params=params)
        try:
            self.request_count += 1
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as e:
            log.warning("af_network_error", path=path, error=str(e))
            return None

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "10"))
            log.warning("af_rate_limited", retry_after=retry_after)
            await asyncio.sleep(min(retry_after, 60.0))
            return await self._get(path, params)

        if resp.status_code >= 400:
            log.warning(
                "af_http_error", path=path, status=resp.status_code,
                body=resp.text[:200],
            )
            return None

        try:
            data = resp.json()
        except ValueError:
            log.warning("af_non_json", path=path)
            return None

        # api-football signals validation problems in ``errors`` while still
        # returning HTTP 200. A dict of errors (or a non-empty list) means the
        # request was rejected (bad key, exhausted quota, bad param).
        errors = data.get("errors")
        if errors:
            log.warning("af_api_errors", path=path, errors=errors)
            return None

        response = data.get("response")
        if not isinstance(response, list):
            log.warning("af_unexpected_shape", path=path)
            return None

        self._cache.set(cache_key, response)
        return response

    # ------------------------------------------------------------------
    async def list_fixtures(
        self, af_league_id: int, season: int, date_from: str, date_to: str
    ) -> list[dict[str, Any]]:
        """Fixtures in a league within a date window.

        -> ``[{af_fixture_id, home_name, away_name, kickoff_iso}]``. Empty on
        error or no fixtures.
        """
        resp = await self._get(
            "/fixtures",
            params={
                "league": af_league_id,
                "season": season,
                "from": date_from,
                "to": date_to,
            },
        )
        if not resp:
            return []
        out: list[dict[str, Any]] = []
        for item in resp:
            try:
                fixture = item.get("fixture") or {}
                teams = item.get("teams") or {}
                out.append(
                    {
                        "af_fixture_id": fixture.get("id"),
                        "home_name": (teams.get("home") or {}).get("name") or "",
                        "away_name": (teams.get("away") or {}).get("name") or "",
                        "kickoff_iso": fixture.get("date") or "",
                    }
                )
            except AttributeError:
                continue
        return [f for f in out if f["af_fixture_id"] is not None]

    async def get_lineups(self, af_fixture_id: int) -> dict[str, Any] | None:
        """Confirmed starting XI for a fixture.

        -> ``{"home": {"formation": str, "xi": [names]}, "away": {...}}`` or
        ``None`` when the lineup has not been posted yet (empty response) /
        on error. api-football orders ``response[0]``=home, ``[1]``=away.
        """
        resp = await self._get(
            "/fixtures/lineups", params={"fixture": af_fixture_id}
        )
        if not resp or len(resp) < 2:
            # Empty (not posted yet) or partial -> no usable lineup.
            return None

        def _side(entry: dict[str, Any]) -> dict[str, Any]:
            xi: list[str] = []
            for slot in entry.get("startXI") or []:
                name = ((slot or {}).get("player") or {}).get("name")
                if name:
                    xi.append(name)
            return {"formation": entry.get("formation") or "", "xi": xi}

        try:
            return {"home": _side(resp[0]), "away": _side(resp[1])}
        except (AttributeError, IndexError):
            return None

    async def get_injuries(self, af_team_id: int, season: int) -> list[str]:
        """Player names currently listed as injured/out for a team. Empty on error."""
        resp = await self._get(
            "/injuries", params={"team": af_team_id, "season": season}
        )
        if not resp:
            return []
        names: list[str] = []
        for item in resp:
            name = ((item or {}).get("player") or {}).get("name")
            if name:
                names.append(name)
        # De-duplicate while preserving order (a player can appear per-fixture).
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    async def get_team_player_minutes(
        self, af_team_id: int, season: int
    ) -> dict[str, int]:
        """Season minutes played per player for a team.

        -> ``{player_name: minutes}``. Walks pages until an empty page or the
        3-page cap (each page is one request; the cap protects the 100/day
        budget). Empty on error.
        """
        minutes: dict[str, int] = {}
        for page in range(1, _MAX_PLAYER_PAGES + 1):
            resp = await self._get(
                "/players",
                params={"team": af_team_id, "season": season, "page": page},
            )
            if not resp:
                break
            for item in resp:
                player = (item or {}).get("player") or {}
                name = player.get("name")
                if not name:
                    continue
                stats = item.get("statistics") or []
                mins = 0
                if stats:
                    games = (stats[0] or {}).get("games") or {}
                    raw = games.get("minutes")
                    try:
                        mins = int(raw) if raw is not None else 0
                    except (TypeError, ValueError):
                        mins = 0
                # Sum across appearances if a name recurs; keep the max page total.
                minutes[name] = minutes.get(name, 0) + mins
            if len(resp) < 20:  # api-football returns 20/page; short page = last.
                break
        return minutes
