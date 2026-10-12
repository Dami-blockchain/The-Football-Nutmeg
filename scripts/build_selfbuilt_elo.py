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
    align_any_country,
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


def _merged_aliases() -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for p in ("config/team_aliases.yaml", "config/selfbuilt_elo_aliases.yaml"):
        fp = Path(p)
        if fp.exists():
            data = yaml.safe_load(fp.read_text()) or {}
            for canon, alts in (data.get("aliases") or {}).items():
                aliases.setdefault(canon, [])
                aliases[canon].extend(alts or [])
    return aliases


async def _fetch_cl(seasons: list[int], today: date) -> list:
    """CL FINISHED matches per season window via football-data.org free tier."""
    settings = get_settings()
    results = []
    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        for start_year in seasons:
            d_from = f"{start_year}-07-01"
            d_to = min(date(start_year + 1, 6, 30), today).isoformat()
            try:
                matches = await client.list_matches("CL", d_from, d_to, status="FINISHED")
            except FootballDataError as e:
                log.warning("cl_fetch_skipped", season=start_year, error=str(e))
                continue
            results.extend(parse_fdorg_matches(matches, competition="UEFA CL"))
    return results


def main() -> int:
    today = date.today()
    state = _load_state()
    if not state.ratings:
        log.error("selfbuilt_build_no_state", hint="seed first")
        return 1

    aliases = _merged_aliases()
    state_idents = list(state.ratings.keys())
    from_year = _season_start_year(state.last_date) if state.last_date else _season_start_year(today)
    to_year = _season_start_year(today)
    seasons = list(range(from_year, to_year + 1))

    all_results: list = []

    # Domestic: every division across every in-range season.
    for code in _season_codes(from_year, to_year):
        for div in FD_DIVISIONS:
            url = f"https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
            try:
                body, ct = _fetch(url)
                res = parse_football_data_co_uk(body, content_type=ct)
            except (FeedError, OSError, ValueError) as e:
                log.warning("fd_fetch_skipped", season=code, div=div, error=str(e))
                continue
            res = align_to_state(res, state_idents, DIVISION_COUNTRY[div], aliases)
            all_results.extend(res)

    # Cross-league: CL FINISHED matches (football-data.org free tier).
    cl = asyncio.run(_fetch_cl(seasons, today))
    cl = align_any_country(cl, state_idents, aliases)
    uefa_rows = len(cl)
    all_results.extend(cl)

    if uefa_rows == 0:
        log.error("selfbuilt_build_no_uefa",
                  hint="no CL rows ingested -> no cross-league linkage; snapshot NOT rewritten")
        return 1

    applied = update_elo(state, all_results)
    if applied == 0:
        log.warning("selfbuilt_build_no_new_rows", from_date=str(state.last_date))

    settings = get_settings()
    ce_rows = load_clubelo_rows(Path(settings.cl_snapshot_path))
    snap, unresolved = bridge_to_clubelo(state.ratings, state.games, ce_rows, aliases)
    out = Path(settings.cl_selfbuilt_snapshot_path)
    write_clubelo_snapshot(out, snap, from_date=state.last_date)
    _save_state(state)
    log.info("selfbuilt_build_done", applied=applied, uefa_rows=uefa_rows,
             bridged=len(snap), unresolved=len(unresolved),
             from_date=str(state.last_date), out=str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
