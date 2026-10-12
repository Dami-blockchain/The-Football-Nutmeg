#!/usr/bin/env python3
"""Daily self-built cross-league Elo builder (OPERATOR-RUN, FREE data only).

Writes the fallback snapshot ``data/clubelo_selfbuilt_latest.csv`` (ClubElo
schema) + a rolling Elo state file. NEVER touches the DB or any live service.
Intended to run from a systemd USER timer on the droplet AFTER the daily
scrape, e.g.:

    ~/tfsm/.venv/bin/python scripts/build_selfbuilt_elo.py

Sources (free, keyless): football-data.co.uk mmz4281 season CSVs for the
domestic divisions, plus openfootball/europe for the cross-league UEFA results.
Every fetch is 200-not-payload-guarded (betbot.data.selfbuilt_feed); a parse
failure degrades to "no new rows" (keep yesterday's snapshot), never a crash or
an empty file. Run scripts/seed_selfbuilt_elo.py ONCE first to burn in history.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from betbot.config import get_settings  # noqa: E402
from betbot.data.selfbuilt_feed import (  # noqa: E402
    EloState,
    FeedError,
    align_to_state,
    bridge_to_clubelo,
    parse_football_data_co_uk,
    parse_openfootball_txt,
    update_elo,
    write_clubelo_snapshot,
)
from betbot.exchanges.matcher import TeamAliasResolver  # noqa: E402
from betbot.logging import get_logger  # noqa: E402
from betbot.strategy.cl_engine import _load_snapshot  # noqa: E402

log = get_logger("build_selfbuilt_elo")

STATE_PATH = Path("data/selfbuilt_elo_state.json")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# football-data.co.uk division codes to pull (the ~22 free leagues; minnow
# leagues it lacks fall through to naive, per §3 — acceptable, low value).
FD_DIVISIONS = [
    "E0", "E1", "SP1", "SP2", "D1", "D2", "I1", "I2", "F1", "F2",
    "N1", "P1", "B1", "T1", "SC0", "G1",
]
# openfootball/europe UEFA competition files for the current season.
OF_BASE = "https://raw.githubusercontent.com/openfootball/europe/master"


def _season_code(today: date) -> str:
    """mmz4281 season code, e.g. 2026-08 -> '2627'."""
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _of_season(today: date) -> str:
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _fetch(url: str) -> tuple[str, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https hosts)
        ct = resp.headers.get("Content-Type")
        return resp.read().decode("utf-8", errors="replace"), ct


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
    )


def _save_state(state: EloState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "ratings": {k: round(v, 4) for k, v in state.ratings.items()},
        "games": state.games,
        "last_date": None if state.last_date is None else state.last_date.isoformat(),
    }))


def _merged_resolver() -> TeamAliasResolver:
    aliases: dict[str, list[str]] = {}
    for p in ("config/team_aliases.yaml", "config/selfbuilt_elo_aliases.yaml"):
        fp = Path(p)
        if fp.exists():
            data = yaml.safe_load(fp.read_text()) or {}
            for canon, alts in (data.get("aliases") or {}).items():
                aliases.setdefault(canon, [])
                aliases[canon].extend(alts or [])
    return TeamAliasResolver(aliases=aliases)


def main() -> int:
    today = date.today()
    state = _load_state()
    resolver = _merged_resolver()
    existing_keys = list(state.ratings.keys())
    fetched_rows = 0

    season = _season_code(today)
    for div in FD_DIVISIONS:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
        try:
            body, ct = _fetch(url)
            results = parse_football_data_co_uk(body, content_type=ct)
        except (FeedError, OSError, ValueError) as e:
            log.warning("fd_fetch_skipped", div=div, error=str(e))
            continue
        results = align_to_state(results, existing_keys, resolver)
        fetched_rows += update_elo(state, results)

    of_season = _of_season(today)
    for comp, fname in (("UEFA CL", "cl"), ("UEFA EL", "el"), ("UEFA ECL", "ecl")):
        url = f"{OF_BASE}/{of_season}/{fname}.txt"
        try:
            body, ct = _fetch(url)
            if ct and "html" in ct.lower():
                raise FeedError(f"openfootball html body: {ct}")
            results = parse_openfootball_txt(body, competition=comp,
                                             default_year=int(of_season[:4]))
        except (FeedError, OSError, ValueError) as e:
            log.warning("openfootball_fetch_skipped", comp=comp, error=str(e))
            continue
        results = align_to_state(results, existing_keys, resolver)
        fetched_rows += update_elo(state, results)

    if state.last_date is None or not state.ratings:
        log.error("selfbuilt_build_empty", hint="no state; seed first")
        return 1

    # Bridge to ClubElo keys using the live site scrape's club list.
    settings = get_settings()
    ce_clubs, _ = _load_snapshot(Path(settings.cl_snapshot_path))
    snap, unresolved = bridge_to_clubelo(
        state.ratings, state.games, list(ce_clubs.keys()), resolver,
    )
    out = Path(settings.cl_selfbuilt_snapshot_path)
    write_clubelo_snapshot(out, snap, from_date=state.last_date)
    _save_state(state)
    log.info("selfbuilt_build_done", new_rows=fetched_rows, bridged=len(snap),
             unresolved=len(unresolved), from_date=str(state.last_date),
             out=str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
