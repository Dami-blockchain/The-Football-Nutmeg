"""Bridge our football-data.org fixtures to api-football lineups + minutes.

Two jobs:

1. **Fixture id resolution** — given one of our fixtures (competition code,
   home/away names, kickoff date) find the matching api-football fixture id via
   ``list_fixtures`` + :class:`TeamAliasResolver` name matching. Resolutions are
   cached in-process (they never change for a past/settled fixture).
2. **Lineup adjustments** — :func:`adjustments_for_fixture` fetches the confirmed
   XI (+ injuries, folded in as extra absences) and the on-disk player-minutes
   cache, then delegates the pure math to
   :func:`betbot.strategy.lineup.lineup_rating_adjustment`. Returns ``(0.0, 0.0)``
   gracefully whenever the lineup isn't out yet or any data is missing.

The player-minutes cache lives at ``data/af_player_minutes/<CODE>_<season>.json``
and is written by ``scripts/fetch_player_minutes.py``. At season start the new
season has ~0 minutes, so a team whose current-season total is near zero falls
back to the prior season's cache for "expected regular" importance.

All network here is best-effort: the underlying :class:`ApiFootballClient` never
raises, and this module treats any gap as "no adjustment".
"""

from __future__ import annotations

import json
from pathlib import Path

from betbot.data.api_football import ApiFootballClient
from betbot.exchanges.matcher import TeamAliasResolver, normalize
from betbot.logging import get_logger
from betbot.strategy.lineup import lineup_rating_adjustment

log = get_logger(__name__)

# Our internal competition codes -> api-football league ids (verified facts).
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
        resolver: TeamAliasResolver | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or ApiFootballClient.from_settings(settings)
        self._resolver = (
            resolver if resolver is not None
            else TeamAliasResolver.from_yaml("config/team_aliases.yaml")
        )
        self._season = settings.api_football_season
        # (code, home_norm, away_norm, date) -> af_fixture_id | None
        self._fixture_cache: dict[tuple[str, str, str, str], int | None] = {}

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    async def resolve_fixture_id(
        self,
        competition_code: str,
        home_name: str,
        away_name: str,
        kickoff_date: str,
    ) -> int | None:
        """Find the api-football fixture id for one of our fixtures (cached)."""
        league_id = af_league_id(competition_code)
        if league_id is None:
            return None
        key = (
            competition_code.upper(),
            normalize(home_name),
            normalize(away_name),
            kickoff_date,
        )
        if key in self._fixture_cache:
            return self._fixture_cache[key]

        candidates = await self._client.list_fixtures(
            league_id, self._season, kickoff_date, kickoff_date
        )
        found: int | None = None
        for fx in candidates:
            af_home = fx.get("home_name") or ""
            af_away = fx.get("away_name") or ""
            if self._resolver.same_team(home_name, af_home) and self._resolver.same_team(
                away_name, af_away
            ):
                found = fx.get("af_fixture_id")
                break
        if found is None and candidates:
            log.info(
                "af_fixture_unresolved",
                code=competition_code,
                home=home_name,
                away=away_name,
                date=kickoff_date,
                candidates=len(candidates),
            )
        self._fixture_cache[key] = found
        return found

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

    async def adjustments_for_fixture(
        self,
        competition_code: str,
        home_name: str,
        away_name: str,
        kickoff_date: str,
        af_fixture_id: int | None = None,
        home_injured: list[str] | None = None,
        away_injured: list[str] | None = None,
    ) -> tuple[float, float]:
        """(home_adj, away_adj) lineup rating shifts; (0.0, 0.0) when unavailable.

        Injured players are folded in as extra absences: an injured player is
        removed from the effective confirmed XI (via :func:`apply_injuries`) so a
        late scratch still counts against the team even if the posted XI listed
        them. Pass ``home_injured`` / ``away_injured`` (names from
        :meth:`ApiFootballClient.get_injuries`); ``None`` skips that fetch to
        protect the request budget.
        """
        s = self._settings
        if af_fixture_id is None:
            af_fixture_id = await self.resolve_fixture_id(
                competition_code, home_name, away_name, kickoff_date
            )
        if af_fixture_id is None:
            return (0.0, 0.0)

        lineups = await self._client.get_lineups(af_fixture_id)
        if not lineups:
            return (0.0, 0.0)  # not posted yet -> baseline prediction

        home_mins = self._minutes_for(competition_code, home_name)
        away_mins = self._minutes_for(competition_code, away_name)

        home_xi = apply_injuries(
            set(lineups.get("home", {}).get("xi") or []), home_injured or []
        )
        away_xi = apply_injuries(
            set(lineups.get("away", {}).get("xi") or []), away_injured or []
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
