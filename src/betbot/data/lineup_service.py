"""Bridge our football-data.org fixtures to Highlightly lineups + api-football minutes.

Two jobs:

1. **Match id resolution** — given one of our fixtures (competition code,
   home/away names, kickoff date) find the matching HIGHLIGHTLY match id via
   ``list_matches`` (the day's league fixtures) + :class:`TeamAliasResolver` name
   matching. Resolutions are cached in-process (they never change for a fixture).
2. **Lineup adjustments** — :func:`adjustments_for_fixture` fetches the confirmed
   XI from Highlightly (+ injuries, folded in as extra absences) and the on-disk
   player-minutes cache, then delegates the pure math to
   :func:`betbot.strategy.lineup.lineup_rating_adjustment`. Returns ``(0.0, 0.0)``
   gracefully whenever the lineup isn't out yet or any data is missing.

Why Highlightly for the XI: api-football's FREE tier only serves seasons
2022-2024 (it blocks the CURRENT season), so its live lineup path fails
("Free plans do not have access to this season"). Highlightly's FREE tier serves
the current season, so it is the confirmed-XI source. api-football is kept ONLY
for the prior-season (2024) player-minutes cache (importance weighting), which
still works on the free tier.

The player-minutes cache lives at ``data/af_player_minutes/<CODE>_<season>.json``
and is written by ``scripts/fetch_player_minutes.py``. At season start the new
season has ~0 minutes, so a team whose current-season total is near zero falls
back to the prior season's cache for "expected regular" importance.

All network here is best-effort: the underlying :class:`HighlightlyClient` and
:class:`ApiFootballClient` never raise, and this module treats any gap as
"no adjustment".
"""

from __future__ import annotations

import json
from pathlib import Path

from betbot.data.api_football import ApiFootballClient
from betbot.data.highlightly import HighlightlyClient, highlightly_league_name
from betbot.exchanges.matcher import TeamAliasResolver, normalize
from betbot.logging import get_logger
from betbot.strategy.lineup import lineup_rating_adjustment

log = get_logger(__name__)

# Our internal competition codes -> api-football league ids (verified facts).
# Retained ONLY for the player-minutes fetch script; the XI source is Highlightly.
AF_LEAGUE_IDS: dict[str, int] = {
    "PL": 39,
    "PD": 140,
    "BL1": 78,
    "SA": 135,
    "FL1": 61,
    "CL": 2,
}

# A team whose current-season minutes total falls below this is treated as
# "not enough data yet" (season just started) -> fall back to the prior season.
# One team match produces ~990 player-minutes (11 starters x 90), so a threshold
# in the hundreds would flip to current-season data after a SINGLE match — far
# too thin to rank expected regulars. 4500 ≈ 5 team matches (mirroring the
# last-5 form window); below that the prior completed season's minutes win.
_MIN_SEASON_MINUTES = 4500

# How many prior seasons to walk back looking for a populated minutes cache. The
# free api-football tier only serves ~2022-2024, so with a 2026 current season
# the newest usable prior cache is 2024 (2 seasons back).
_PRIOR_SEASON_LOOKBACK = 3

PLAYER_MINUTES_DIR = Path("data/af_player_minutes")


def af_league_id(competition_code: str) -> int | None:
    """Map an internal competition code to its api-football league id."""
    return AF_LEAGUE_IDS.get(competition_code.upper())


def player_minutes_path(code: str, season: int) -> Path:
    return PLAYER_MINUTES_DIR / f"{code.upper()}_{season}.json"


def _load_minutes_cache(code: str, season: int) -> dict[str, dict[str, int]]:
    """Load ``{team_key: {player_norm: minutes}}`` for a league+season ('' if none)."""
    path = player_minutes_path(code, season)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, dict[str, int]] = {}
    for team_key, players in (raw or {}).items():
        if not isinstance(players, dict):
            continue
        out[str(team_key)] = {
            str(p): int(m) for p, m in players.items()
            if isinstance(m, (int, float))
        }
    return out


def _team_minutes(
    cache: dict[str, dict[str, int]], team_name: str
) -> dict[str, int]:
    """Find a team's ``{player: minutes}`` in a cache by normalized name match."""
    target = normalize(team_name)
    # Exact normalized key first, then substring both ways (handles "Man City"
    # vs "Manchester City" style cache keys).
    for key, players in cache.items():
        if normalize(key) == target:
            return players
    for key, players in cache.items():
        nk = normalize(key)
        if nk and (nk in target or target in nk):
            return players
    return {}


class LineupService:
    def __init__(
        self,
        settings,
        *,
        client: ApiFootballClient | None = None,
        highlightly: HighlightlyClient | None = None,
        resolver: TeamAliasResolver | None = None,
    ) -> None:
        self._settings = settings
        # api-football kept ONLY for the (prior-season) minutes cache script.
        self._client = client or ApiFootballClient.from_settings(settings)
        self._highlightly = highlightly or HighlightlyClient.from_settings(settings)
        self._resolver = (
            resolver if resolver is not None
            else TeamAliasResolver.from_yaml("config/team_aliases.yaml")
        )
        self._season = settings.api_football_season
        # (code, home_norm, away_norm, date) -> highlightly_match_id | None
        self._fixture_cache: dict[tuple[str, str, str, str], object | None] = {}
        # (league_name, date) -> the day's matches (cached to stay budget-cheap:
        # one /matches per league per alert batch).
        self._matches_cache: dict[tuple[str, str], list[dict]] = {}

    async def close(self) -> None:
        await self._client.close()
        await self._highlightly.close()

    # ------------------------------------------------------------------
    async def _day_matches(self, league_name: str, date: str) -> list[dict]:
        """The league's matches for a date, cached across an alert batch."""
        key = (league_name, date)
        cached = self._matches_cache.get(key)
        if cached is not None:
            return cached
        matches = await self._highlightly.list_matches(league_name, date)
        # Never cache an EMPTY list: list_matches returns [] on transient API
        # errors too, and this service is now a process-wide singleton — a
        # poisoned (league, date) entry would kill lineups for every later
        # alert that day (including the confirmed-XI update). Retry next call.
        if matches:
            self._matches_cache[key] = matches
        return matches

    async def resolve_match_id(
        self,
        competition_code: str,
        home_name: str,
        away_name: str,
        kickoff_date: str,
    ) -> object | None:
        """Find the Highlightly match id for one of our fixtures (cached).

        Resolves via the day's ``/matches`` list for the mapped ``leagueName``
        plus :class:`TeamAliasResolver` on BOTH team names. Returns the match id
        (opaque int) or ``None`` when unmapped / unmatched / no data.
        """
        league_name = highlightly_league_name(competition_code)
        if league_name is None:
            return None
        key = (
            competition_code.upper(),
            normalize(home_name),
            normalize(away_name),
            kickoff_date,
        )
        if key in self._fixture_cache:
            return self._fixture_cache[key]

        candidates = await self._day_matches(league_name, kickoff_date)
        found: object | None = None
        for m in candidates:
            hl_home = m.get("home_name") or ""
            hl_away = m.get("away_name") or ""
            if self._resolver.same_team(home_name, hl_home) and self._resolver.same_team(
                away_name, hl_away
            ):
                found = m.get("match_id")
                break
        if found is None and candidates:
            log.info(
                "highlightly_match_unresolved",
                code=competition_code,
                home=home_name,
                away=away_name,
                date=kickoff_date,
                candidates=len(candidates),
            )
        # Cache a NEGATIVE resolution only when the day's list actually had
        # candidates (a real alias miss is stable); an empty/errored list must
        # stay retryable — the singleton would otherwise pin None for good.
        if found is not None or candidates:
            self._fixture_cache[key] = found
        return found

    # Back-compat alias: callers (daily_jobs closure) used ``resolve_fixture_id``.
    async def resolve_fixture_id(
        self,
        competition_code: str,
        home_name: str,
        away_name: str,
        kickoff_date: str,
    ) -> object | None:
        return await self.resolve_match_id(
            competition_code, home_name, away_name, kickoff_date
        )

    # ------------------------------------------------------------------
    def _minutes_for(
        self, competition_code: str, team_name: str
    ) -> dict[str, int]:
        """Player-minutes for a team, with prior-season fallback at season start.

        At season start the current-season cache is empty (~0 minutes), so a team
        below ``_MIN_SEASON_MINUTES`` (~10 full player-games) falls back to the
        most recent PRIOR season that actually has minutes for it. We walk back a
        few seasons because the api-football FREE tier only serves ~2022-2024, so
        the immediate prior season (e.g. 2025) may be unavailable and the newest
        backfilled cache is 2024.
        """
        cur = _load_minutes_cache(competition_code, self._season)
        team_cur = _team_minutes(cur, team_name)
        if sum(team_cur.values()) > _MIN_SEASON_MINUTES:
            return team_cur
        for back in range(1, _PRIOR_SEASON_LOOKBACK + 1):
            prev = _load_minutes_cache(competition_code, self._season - back)
            team_prev = _team_minutes(prev, team_name)
            if sum(team_prev.values()) > 0:
                return team_prev
        return team_cur

    async def get_confirmed_xi(
        self,
        competition_code: str,
        home_name: str,
        away_name: str,
        kickoff_date: str,
        match_id: object | None = None,
    ) -> dict | None:
        """Confirmed XI + formation per side, or ``None`` when not posted yet.

        -> ``{"home": {"formation", "xi": [names]}, "away": {...}}`` (the shape
        :func:`betbot.tips.format_prediction_with_lineup` expects). Resolves the
        Highlightly match id if not supplied. ``None`` on any gap (unmapped
        league / unresolved fixture / XI not posted / error).
        """
        if match_id is None:
            match_id = await self.resolve_match_id(
                competition_code, home_name, away_name, kickoff_date
            )
        if match_id is None:
            return None
        return await self._highlightly.get_lineup(match_id)

    async def adjustments_for_fixture(
        self,
        competition_code: str,
        home_name: str,
        away_name: str,
        kickoff_date: str,
        match_id: object | None = None,
        lineups: dict | None = None,
        home_injured: list[str] | None = None,
        away_injured: list[str] | None = None,
    ) -> tuple[float, float]:
        """(home_adj, away_adj) lineup rating shifts; (0.0, 0.0) when unavailable.

        Uses the Highlightly confirmed XI (resolved via ``match_id`` if not
        supplied, or the pre-fetched ``lineups`` to avoid a duplicate call) and
        the on-disk player-minutes cache. Injured players are folded in as extra
        absences: an injured player is removed from the effective confirmed XI
        (via :func:`apply_injuries`) so a late scratch still counts against the
        team even if the posted XI listed them.
        """
        s = self._settings
        if lineups is None:
            lineups = await self.get_confirmed_xi(
                competition_code, home_name, away_name, kickoff_date, match_id=match_id
            )
        if not lineups:
            return (0.0, 0.0)  # not posted yet / unresolved -> baseline

        home_mins = self._minutes_for(competition_code, home_name)
        away_mins = self._minutes_for(competition_code, away_name)

        home_xi = apply_injuries(
            set((lineups.get("home") or {}).get("xi") or []), home_injured or []
        )
        away_xi = apply_injuries(
            set((lineups.get("away") or {}).get("xi") or []), away_injured or []
        )

        home_adj = lineup_rating_adjustment(
            home_xi, home_mins, max_penalty=s.lineup_max_penalty
        )
        away_adj = lineup_rating_adjustment(
            away_xi, away_mins, max_penalty=s.lineup_max_penalty
        )
        return (home_adj, away_adj)


# ----------------------------------------------------------------------
# Injury folding helper (kept module-level + pure-ish so it is reusable by the
# minutes-fetch script and any future caller). Given a confirmed XI and a set of
# injured player names, an injured expected-regular is simply absent from the XI
# already; injuries mostly matter when a player is neither in the XI nor injured
# vs benched. We expose the removal so callers can subtract injured names from a
# candidate XI when reasoning about *why* a regular is out.
def apply_injuries(confirmed_xi: set[str], injured: list[str]) -> set[str]:
    """Return ``confirmed_xi`` with any injured player names removed.

    An injured player should never count as present even in the unlikely event
    the posted XI still lists them (late scratch). Matching is normalized.
    """
    if not injured:
        return set(confirmed_xi)
    injured_norm = {normalize(n) for n in injured if n}
    return {p for p in confirmed_xi if normalize(p) not in injured_norm}
