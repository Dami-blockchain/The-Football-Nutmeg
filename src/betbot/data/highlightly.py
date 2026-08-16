"""Async client for Highlightly Soccer (soccer.highlightly.net) — FREE tier.

Confirmed-lineup source for the pre-match alert. api-football's FREE tier only
serves seasons 2022-2024 (it blocks the CURRENT season with "Free plans do not
have access to this season"), so the live lineup path there fails. Highlightly's
FREE tier serves the CURRENT season, so it drives confirmed XIs; api-football is
kept ONLY for prior-season (2024) player-minutes (importance weighting), which
still works.

Two endpoints:
  * ``GET /matches?date&leagueName&limit`` -> the day's matches for a league,
  * ``GET /lineups/{matchId}`` -> the confirmed XI per side (empty / formation
    "Unknown" until the club posts it, ~55-70 min before kickoff).

CRITICAL — Cloudflare: the host sits behind Cloudflare, which returns HTTP 403
"error code: 1010" for a request without a browser ``User-Agent``. We ALWAYS
send both the API key (``x-rapidapi-key``) and a browser UA header.

Budget: the FREE tier is ~100 requests/day, so callers cache aggressively (one
``/matches`` per league per alert batch, one ``/lineups`` per fixture). Every
public method is **graceful**: on any transport / HTTP / parse error it logs and
returns ``None`` / ``[]`` — it NEVER raises up to a caller, so a missing lineup
silently falls back to the baseline (lineup-free) prediction.
"""

from __future__ import annotations

from typing import Any

import httpx

from betbot.logging import get_logger
from betbot.utils.cache import TTLCache

log = get_logger(__name__)

# A real browser User-Agent is MANDATORY: Cloudflare 403s ("error code: 1010")
# a request without one, even with a valid API key.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# Our internal competition codes -> Highlightly ``leagueName`` values (verified
# live: today's La Liga matches returned).
HIGHLIGHTLY_LEAGUE_NAMES: dict[str, str] = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "Champions League",
}


def highlightly_league_name(competition_code: str) -> str | None:
    """Map an internal competition code to its Highlightly ``leagueName``."""
    return HIGHLIGHTLY_LEAGUE_NAMES.get((competition_code or "").upper())


class HighlightlyClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://soccer.highlightly.net",
        *,
        cache_ttl_seconds: float = 300.0,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            log.warning("highlightly_no_api_key", note="Calls will fail; graceful.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._cache: TTLCache[Any] = TTLCache(ttl_seconds=cache_ttl_seconds)
        self._owns_client = client is None
        headers = {"User-Agent": _BROWSER_UA}
        if api_key:
            headers["x-rapidapi-key"] = api_key
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers=headers
        )
        # Live HTTP requests actually issued — lets callers report budget spend.
        self.request_count = 0

    async def __aenter__(self) -> "HighlightlyClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    @classmethod
    def from_settings(cls, settings) -> "HighlightlyClient":
        return cls(
            settings.highlightly_api_key,
            settings.highlightly_base_url,
        )

    # ------------------------------------------------------------------
    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Any | None:
        """GET a path -> parsed JSON, or ``None`` on any error. Never raises.

        A browser UA + the API key are already on the shared client's default
        headers, so this satisfies Cloudflare on every request.
        """
        cache_key = (path, tuple(sorted((params or {}).items())))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self._base_url}{path}"
        log.debug("highlightly_request", path=path, params=params)
        try:
            self.request_count += 1
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as e:
            log.warning("highlightly_network_error", path=path, error=str(e))
            return None

        if resp.status_code >= 400:
            log.warning(
                "highlightly_http_error", path=path, status=resp.status_code,
                body=resp.text[:200],
            )
            return None

        try:
            data = resp.json()
        except ValueError:
            log.warning("highlightly_non_json", path=path)
            return None

        self._cache.set(cache_key, data)
        return data

    # ------------------------------------------------------------------
    async def list_matches(self, league_name: str, date: str) -> list[dict[str, Any]]:
        """Matches for a league on a date.

        -> ``[{match_id, home_name, away_name, state}]``. Empty on error or when
        no matches. Highlightly wraps the list in ``{"data": [...]}`` (verified
        live), but we also accept a bare list defensively.
        """
        data = await self._get(
            "/matches",
            params={"leagueName": league_name, "date": date, "limit": 100},
        )
        if data is None:
            return []
        if isinstance(data, dict):
            items = data.get("data") or data.get("matches") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        out: list[dict[str, Any]] = []
        for m in items:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if mid is None:
                continue
            out.append(
                {
                    "match_id": mid,
                    "home_name": (m.get("homeTeam") or {}).get("name") or "",
                    "away_name": (m.get("awayTeam") or {}).get("name") or "",
                    "state": (m.get("state") or {}).get("description") or "",
                }
            )
        return out

    async def get_lineup(self, match_id) -> dict[str, Any] | None:
        """Confirmed starting XI for a match.

        -> ``{"home": {"formation": str, "xi": [names]}, "away": {...}}`` or
        ``None`` when the club has not posted the XI yet (empty
        ``initialLineup`` / formation "Unknown") or on error.

        Highlightly's ``initialLineup`` is a LIST OF FORMATION ROWS, each row a
        list of ``{name, number, position, id}``; we flatten the rows into the
        11 starter names.
        """
        data = await self._get(f"/lineups/{match_id}")
        if not isinstance(data, dict):
            return None

        def _side(entry: Any) -> dict[str, Any] | None:
            if not isinstance(entry, dict):
                return None
            formation = str(entry.get("formation") or "").strip()
            rows = entry.get("initialLineup") or []
            xi: list[str] = []
            for row in rows:
                players = row if isinstance(row, list) else [row]
                for p in players:
                    if isinstance(p, dict):
                        name = p.get("name")
                        if name:
                            xi.append(str(name))
            # Not posted yet: empty rows or the "Unknown" placeholder formation.
            if not xi or formation.lower() == "unknown":
                return None
            return {"formation": formation, "xi": xi}

        home = _side(data.get("homeTeam"))
        away = _side(data.get("awayTeam"))
        if home is None and away is None:
            return None
        return {
            "home": home or {"formation": "", "xi": []},
            "away": away or {"formation": "", "xi": []},
        }
