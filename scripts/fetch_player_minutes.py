"""Fetch / refresh the api-football season player-minutes cache.

Writes ``data/af_player_minutes/<CODE>_<season>.json`` for the configured
leagues + season:

    {af_team_id: {player_norm: minutes}}

We key by api-football team id (stable) but also record the team's display name
so the on-disk cache is human-inspectable and the LineupService can match by
name. Respects the FREE-tier request budget (100/day) — each team costs up to
3 requests (paginated players) plus one fixtures call per league to discover
teams; the script logs how many live calls it spent and stops if a soft budget
cap is hit.

Usage:
    python scripts/fetch_player_minutes.py                # all leagues, configured season
    python scripts/fetch_player_minutes.py --codes PL PD  # subset
    python scripts/fetch_player_minutes.py --season 2025  # override season
    python scripts/fetch_player_minutes.py --budget 60    # stop after ~60 calls

The weekly refresh job (daily_jobs) will call ``run`` — R4b wires scheduling.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from betbot.config import get_settings
from betbot.data.api_football import ApiFootballClient
from betbot.data.lineup_service import PLAYER_MINUTES_DIR, af_league_id
from betbot.exchanges.matcher import normalize
from betbot.logging import get_logger

log = get_logger(__name__)

# Default soft cap so a full-season refresh can't blow the whole 100/day budget
# in one run. Each league ~ 1 (fixtures/teams discovery) + 3*num_teams.
DEFAULT_BUDGET = 90


async def _discover_team_ids(
    client: ApiFootballClient, league_id: int, season: int
) -> dict[int, str]:
    """Team id -> name for a league+season via the fixtures endpoint.

    ``/teams?league=&season=`` is the natural call but ``/fixtures`` is already
    verified free and returns every team across the season, so we harvest team
    ids from a wide fixtures window to avoid depending on an unverified endpoint.
    """
    # A full-season window; api-football returns the whole schedule.
    fixtures = await client.list_fixtures(
        league_id, season, f"{season}-07-01", f"{season + 1}-06-30"
    )
    # list_fixtures strips team ids, so pull the raw response too.
    raw = await client._get(  # noqa: SLF001 - intentional internal reuse
        "/fixtures",
        params={
            "league": league_id,
            "season": season,
            "from": f"{season}-07-01",
            "to": f"{season + 1}-06-30",
        },
    )
    ids: dict[int, str] = {}
    for item in raw or []:
        teams = item.get("teams") or {}
        for side in ("home", "away"):
            t = teams.get(side) or {}
            tid, name = t.get("id"), t.get("name")
            if tid is not None and name:
                ids[int(tid)] = str(name)
    log.info("af_teams_discovered", league=league_id, teams=len(ids),
             fixtures=len(fixtures))
    return ids


async def run(
    codes: list[str], season: int, budget: int = DEFAULT_BUDGET
) -> dict[str, int]:
    """Refresh minutes caches for ``codes``. Returns {code: teams_written}."""
    settings = get_settings()
    PLAYER_MINUTES_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    async with ApiFootballClient.from_settings(settings) as client:
        for code in codes:
            league_id = af_league_id(code)
            if league_id is None:
                log.warning("af_unknown_code", code=code)
                continue
            if client.request_count >= budget:
                log.warning("af_budget_reached", spent=client.request_count)
                break

            team_ids = await _discover_team_ids(client, league_id, season)
            out: dict[str, dict[str, int]] = {}
            for tid, name in team_ids.items():
                if client.request_count >= budget:
                    log.warning(
                        "af_budget_reached_midleague",
                        code=code, spent=client.request_count,
                    )
                    break
                mins = await client.get_team_player_minutes(tid, season)
                # Store under both the team id and a name tag for legibility.
                out[str(tid)] = {normalize(p): m for p, m in mins.items()}
                out[f"name:{name}"] = {normalize(p): m for p, m in mins.items()}

            path = PLAYER_MINUTES_DIR / f"{code}_{season}.json"
            path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            written[code] = len([k for k in out if not k.startswith("name:")])
            log.info("af_minutes_written", code=code, path=str(path),
                     teams=written[code])

        log.info("af_minutes_done", calls_spent=client.request_count)
        print(f"api-football calls spent: {client.request_count}")
    return written


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Refresh api-football player-minutes cache.")
    p.add_argument(
        "--codes", nargs="*", default=list(settings.leagues),
        help="Internal competition codes (default: all configured leagues).",
    )
    p.add_argument(
        "--season", type=int, default=settings.api_football_season,
        help="Season START year (2025/26 -> 2025).",
    )
    p.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET,
        help="Soft cap on live api-football requests this run.",
    )
    args = p.parse_args()
    # CL squads are the domestic clubs already cached under their leagues, and
    # api-football's league=2 player minutes are CL-only (tiny); skip by default
    # unless explicitly requested.
    codes = [c.upper() for c in args.codes]
    result = asyncio.run(run(codes, args.season, budget=args.budget))
    print(f"minutes written per league: {result}")


if __name__ == "__main__":
    main()
