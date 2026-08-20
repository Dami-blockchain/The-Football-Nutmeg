"""Async client for football-data.org v4 API.

- Sliding-window rate limiter (10 req/min on free tier by default).
- TTL cache to amortise repeated calls within a scoring run.
- 429 backoff that respects Retry-After when present.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import httpx

from betbot.logging import get_logger
from betbot.utils.cache import TTLCache

log = get_logger(__name__)


class FootballDataError(RuntimeError):
    """Raised on transport / API errors talking to football-data.org."""


class _SlidingWindow:
    """N requests per 60s, blocking when full."""

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
            log.debug("rate_limit_wait", seconds=round(wait, 2))
            await asyncio.sleep(wait + 0.05)
            now = loop.time()
        self._timestamps.append(now)


class FootballDataClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.football-data.org/v4",
        *,
        rate_limit_per_min: int = 10,
        cache_ttl_seconds: float = 60.0,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            log.warning("football_data_no_api_key", note="Public endpoints only.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._limiter = _SlidingWindow(rate_limit_per_min)
        self._cache: TTLCache[Any] = TTLCache(ttl_seconds=cache_ttl_seconds)
        self._owns_client = client is None
        headers = {"X-Auth-Token": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers=headers
        )

    async def __aenter__(self) -> "FootballDataClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Any:
        cache_key = (path, tuple(sorted((params or {}).items())))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._limiter.acquire()
        url = f"{self._base_url}{path}"
        log.debug("football_data_request", path=path, params=params)
        try:
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as e:
            raise FootballDataError(f"network error calling {path}: {e}") from e

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "10"))
            log.warning("football_data_rate_limited", retry_after=retry_after)
            await asyncio.sleep(min(retry_after, 60.0))
            return await self._get(path, params)

        if resp.status_code >= 400:
            raise FootballDataError(
                f"{path} -> {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise FootballDataError(f"non-JSON response from {path}") from e

        self._cache.set(cache_key, data)
        return data

    # ------------------------------------------------------------------
    async def list_matches(
        self,
        competition_code: str,
        date_from: str,
        date_to: str,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Every match a competition has in the date range, in one call.

        ``status=None`` means no filter, which is what the kickoff re-sync
        needs: a fixture that has since been POSTPONED, or promoted from
        SCHEDULED to TIMED, must still come back. A ``status=SCHEDULED`` filter
        silently drops those, and the re-sync would read the absence as "not in
        this window any more" rather than as a match that just moved.
        """
        params: dict[str, Any] = {"dateFrom": date_from, "dateTo": date_to}
        if status is not None:
            params["status"] = status
        data = await self._get(
            f"/competitions/{competition_code}/matches", params=params
        )
        matches = data.get("matches") or []
        return [m for m in matches if isinstance(m, dict)]

    async def list_scheduled_matches(
        self, competition_code: str, date_from: str, date_to: str
    ) -> list[dict[str, Any]]:
        return await self.list_matches(
            competition_code, date_from, date_to, status="SCHEDULED"
        )

    async def list_team_recent_matches(
        self, team_id: int, limit: int = 5, before: str | None = None
    ) -> list[dict[str, Any]]:
        # football-data rejects dateTo without dateFrom; pass either both
        # (a generous 365-day lookback) or neither. We always sort + slice
        # on our side so the API's natural ordering doesn't matter.
        from datetime import date as _date, timedelta as _td

        params: dict[str, Any] = {"status": "FINISHED", "limit": max(limit, 10)}
        if before:
            params["dateTo"] = before
            try:
                params["dateFrom"] = (
                    _date.fromisoformat(before) - _td(days=365)
                ).isoformat()
            except ValueError:
                # If the caller passed a non-ISO string, just drop the bound.
                params.pop("dateTo", None)

        data = await self._get(f"/teams/{team_id}/matches", params=params)
        matches = data.get("matches") or []
        sorted_m = sorted(
            (m for m in matches if isinstance(m, dict)),
            key=lambda m: m.get("utcDate", ""),
            reverse=True,
        )
        return sorted_m[:limit]

    async def get_standings(
        self, competition_code: str
    ) -> list[dict[str, Any]]:
        data = await self._get(f"/competitions/{competition_code}/standings")
        standings = data.get("standings") or []
        for s in standings:
            if s.get("type") == "TOTAL":
                table = s.get("table") or []
                return [row for row in table if isinstance(row, dict)]
        return []

    async def get_team(self, team_id: int) -> dict[str, Any]:
        return await self._get(f"/teams/{team_id}")

    async def list_competition_teams(
        self, competition_code: str
    ) -> list[dict[str, Any]]:
        data = await self._get(f"/competitions/{competition_code}/teams")
        return data.get("teams") or []

    async def get_match(self, match_id: int) -> dict[str, Any] | None:
        try:
            return await self._get(f"/matches/{match_id}")
        except FootballDataError as e:
            if "404" in str(e):
                return None
            raise
