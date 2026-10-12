#!/usr/bin/env python3
"""Daily self-built cross-league Elo builder (OPERATOR-RUN, FREE data only).

Writes the fallback snapshot ``data/clubelo_selfbuilt_latest.csv`` (ClubElo
schema) + a rolling Elo state file. NEVER touches the DB or any live service.
chdirs to the repo root so data/ + config/ resolve regardless of cwd. Run from
a systemd USER timer AFTER the daily scrape (see deploy/selfbuilt-elo.*).

Sources (free, keyless):
* football-data.co.uk mmz4281 season CSVs for the domestic divisions;
* football-data.org (free tier, via FootballDataClient, rate-limited) for
  Champions League FINISHED matches — the cross-league linkage.

Europa League / Conference League FINISHED results are NOT on a free feed, so
the cross-league linkage is CL-only. openfootball/europe does NOT publish the
UEFA files (all 404), so it is not used.

Every fetch is 200-not-payload-guarded; a parse failure degrades to "no new
rows" (keep the prior snapshot), never a crash/empty file. The whole run
GATHERS all sources first and applies Elo ONCE in global date order (so no
source drops another's rows), advances ``From`` only on applied rows, and
EXITS NON-ZERO WITHOUT writing a snapshot if zero CL rows are ingested.
Run scripts/seed_selfbuilt_elo.py ONCE first to burn in history.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from betbot.config import get_settings  # noqa: E402
from betbot.data.football_data import FootballDataClient, FootballDataError  # noqa: E402
from betbot.data.selfbuilt_feed import (  # noqa: E402
    DIVISION_COUNTRY,
    EloState,
    FeedError,
    align_to_state,
    bridge_to_clubelo,
    load_clubelo_rows,
    parse_fdorg_matches,
    parse_football_data_co_uk,
    update_elo,
    write_clubelo_snapshot,
)
from betbot.logging import get_logger  # noqa: E402

log = get_logger("build_selfbuilt_elo")

STATE_PATH = Path("data/selfbuilt_elo_state.json")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
FD_DIVISIONS = list(DIVISION_COUNTRY.keys())

#: football-data.co.uk TOP-flight divisions. The seed (schochastics) is
#: top-flight only, so ONLY these (and CL) are date-filtered against the seed
#: cutoff; second divisions are not in the seed, so their rows are applied in
#: full (seen-set deduped) rather than filtered out — avoiding the one-time loss
#: of their pre-cutoff rows (§g).
TOP_FLIGHT = {"E0", "SP1", "D1", "I1", "F1", "N1", "P1", "B1", "T1", "SC0", "G1"}

#: football-data.org area.name -> the schochastics SEED country a club's idents
#: live under, when a UEFA club plays its league in another country. Monaco
#: plays Ligue 1, so its seed ident is "(France)". (Welsh clubs in the English
#: pyramid are not CL participants, and real Welsh-league CL entrants keep
#: Wales, so Wales is deliberately NOT remapped.)
AREA_TO_SEED_COUNTRY = {"Monaco": "France"}


def _season_codes(from_year: int, to_year: int) -> list[str]:
    """mmz4281 codes for each season start-year in [from_year, to_year]."""
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(from_year, to_year + 1)]


def _season_start_year(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def _fetch(url: str) -> tuple[str, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https host)
        return resp.read().decode("utf-8", errors="replace"), resp.headers.get("Content-Type")


def _load_state() -> EloState:
    if not STATE_PATH.exists():
        log.error("selfbuilt_state_missing", path=str(STATE_PATH),
                  hint="run scripts/seed_selfbuilt_elo.py first")
        return EloState()
    d = json.loads(STATE_PATH.read_text())
    return EloState(
        ratings={k: float(v) for k, v in d.get("ratings", {}).items()},
        games={k: int(v) for k, v in d.get("games", {}).items()},
        last_date=date.fromisoformat(d["last_date"]) if d.get("last_date") else None,
        seen=set(d.get("seen", [])),
    )


def _save_state(state: EloState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "ratings": {k: round(v, 4) for k, v in state.ratings.items()},
        "games": state.games,
        "last_date": None if state.last_date is None else state.last_date.isoformat(),
        "seen": sorted(state.seen),
    }))


def _selfbuilt_aliases() -> dict[str, list[str]]:
    # ONLY the self-built table (ClubElo-key canonicals). team_aliases.yaml is
    # the engine's football-data.org->market table — a different canonical
    # space — and merging it collides with these ClubElo keys (e.g. "Paris SG"
    # is a canonical here but an alt of "Paris Saint-Germain FC" there), which
    # AliasFolder now rejects. The engine serve path still uses team_aliases via
    # its own resolver; only this build-time bridge uses the self-built table.
    fp = Path("config/selfbuilt_elo_aliases.yaml")
    data = yaml.safe_load(fp.read_text()) if fp.exists() else {}
    return (data or {}).get("aliases") or {}


async def _fetch_cl(seasons: list[int], today: date) -> tuple[list, dict[str, str]]:
    """CL FINISHED matches + a {team: area.name} country map, per season window
    (football-data.org free tier). The country map (one extra call/season) is
    what lets CL rows route through the country-restricted alignment."""
    settings = get_settings()
    results: list = []
    team_country: dict[str, str] = {}
    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        for start_year in seasons:
            try:
                teams = (await client._get(
                    "/competitions/CL/teams", {"season": start_year})).get("teams") or []
                for t in teams:
                    nm = (t.get("name") or "").strip()
                    area = (t.get("area") or {}).get("name") or ""
                    if nm and area:
                        team_country[nm] = AREA_TO_SEED_COUNTRY.get(area, area)
            except FootballDataError as e:
                log.warning("cl_teams_fetch_skipped", season=start_year, error=str(e))
            d_from = f"{start_year}-07-01"
            d_to = min(date(start_year + 1, 6, 30), today).isoformat()
            try:
                matches = await client.list_matches("CL", d_from, d_to, status="FINISHED")
            except FootballDataError as e:
                log.warning("cl_fetch_skipped", season=start_year, error=str(e))
                continue
            results.extend(parse_fdorg_matches(matches, competition="UEFA CL"))
    return results, team_country


def main() -> int:
    today = date.today()
    state = _load_state()
    if not state.ratings:
        log.error("selfbuilt_build_no_state", hint="seed first")
        return 1

    aliases = _selfbuilt_aliases()
    state_idents = list(state.ratings.keys())
    before_idents = set(state.ratings)
    cutoff = state.last_date   # the seed authoritatively covers up to here
    from_year = _season_start_year(cutoff) if cutoff else _season_start_year(today)
    to_year = _season_start_year(today)

    # Seed-covered rows (top-flight domestic + CL) are date-filtered against the
    # cutoff; second-division rows (NOT in the top-flight-only seed) are applied
    # in full so their pre-cutoff rows are not lost (§g). Both are seen-set
    # deduped, so a second run still applies nothing.
    filtered: list = []       # top-flight + CL -> > cutoff
    unfiltered: list = []     # second divisions -> all (seen-set dedup)

    for code in _season_codes(from_year, to_year):
        for div in FD_DIVISIONS:
            url = f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
            try:
                body, ct = _fetch(url)
                res = parse_football_data_co_uk(body, content_type=ct)
            except (FeedError, OSError, ValueError) as e:
                log.warning("fd_fetch_skipped", season=code, div=div, error=str(e))
                continue
            res = align_to_state(res, state_idents, DIVISION_COUNTRY[div],
                                 alias_map=aliases, games=state.games)
            (filtered if div in TOP_FLIGHT else unfiltered).extend(res)

    # Cross-league: CL FINISHED matches, routed through the country-restricted
    # alignment using each team's football-data.org area.name. The window spans
    # the PREVIOUS season too, so the zero-CL guard only trips when BOTH seasons
    # return nothing (not every Aug->mid-Sep when the new season has no CL yet).
    cl_seasons = list(range(from_year - 1, to_year + 1))
    cl, team_country = asyncio.run(_fetch_cl(cl_seasons, today))
    uefa_rows = len(cl)
    filtered.extend(align_to_state(
        cl, state_idents, team_country=team_country, alias_map=aliases, games=state.games))

    # Fail CLOSED: a build with zero CL rows has NO cross-league linkage, so we
    # do NOT rewrite the snapshot (keep the prior one).
    if uefa_rows == 0:
        log.error("selfbuilt_build_no_uefa",
                  hint="zero CL rows -> no linkage; snapshot NOT rewritten (feed outage?)")
        return 1

    if cutoff is not None:
        filtered = [r for r in filtered if r.match_date > cutoff]

    applied = update_elo(state, filtered + unfiltered)
    if applied == 0:
        log.warning("selfbuilt_build_no_new_rows", from_date=str(state.last_date))
    # New idents = clubs that did not fold onto a seed ident. Log at WARNING so
    # fragmentation is visible in journalctl (ideally only genuinely new clubs).
    new_idents = {k: state.games.get(k, 0) for k in set(state.ratings) - before_idents}
    if new_idents:
        log.warning("selfbuilt_new_idents", count=len(new_idents),
                    idents=sorted(new_idents.items(), key=lambda kv: -kv[1])[:40])
    # Round in memory so the snapshot is byte-stable when nothing was applied.
    state.ratings = {k: round(v, 4) for k, v in state.ratings.items()}

    settings = get_settings()
    ce_rows = load_clubelo_rows(Path(settings.cl_snapshot_path))
    snap, unresolved = bridge_to_clubelo(state.ratings, state.games, ce_rows, aliases)
    out = Path(settings.cl_selfbuilt_snapshot_path)
    write_clubelo_snapshot(out, snap, from_date=state.last_date,
                           country_of={club: code for club, code in ce_rows})
    _save_state(state)
    log.info("selfbuilt_build_done", applied=applied, uefa_rows=uefa_rows,
             bridged=len(snap), unresolved=len(unresolved),
             from_date=str(state.last_date), out=str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
