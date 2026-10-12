#!/usr/bin/env python3
"""ONE-OFF self-built Elo burn-in seed (OPERATOR-RUN, FREE data, NO DB).

Burns the cross-league Elo in from 1990 so ratings are settled long before any
CL fixture is priced, then writes the initial state file + fallback snapshot
that scripts/build_selfbuilt_elo.py extends daily. Run ONCE:

    ~/tfsm/.venv/bin/python scripts/seed_selfbuilt_elo.py

Source: schochastics/football-data games.parquet (free GitHub dump, ~1.31M
matches) exported to data/eu_results.csv via duckdb (same as the research
prototype export.py). This script NEVER touches the betbot DB or any live
service — it only writes data/ files. If data/eu_results.csv already exists it
is reused (re-pullable, static).

duckdb is only needed for the parquet export step; if data/eu_results.csv is
already present, duckdb is not imported.
"""
from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from betbot.config import get_settings  # noqa: E402
from betbot.data.selfbuilt_feed import (  # noqa: E402
    EloState,
    Result,
    bridge_to_clubelo,
    write_clubelo_snapshot,
)
from betbot.exchanges.matcher import TeamAliasResolver  # noqa: E402
from betbot.logging import get_logger  # noqa: E402
from betbot.strategy.cl_engine import _load_snapshot  # noqa: E402

log = get_logger("seed_selfbuilt_elo")

EU_RESULTS = Path("data/eu_results.csv")
PARQUET = Path("data/games.parquet")
STATE_PATH = Path("data/selfbuilt_elo_state.json")


def _export_eu_results() -> None:
    """Export the European results from games.parquet via duckdb (free)."""
    import duckdb  # imported only when the CSV must be built

    EU_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          SELECT date, home_ident AS home, away_ident AS away, gh, ga,
                 competition, home_country, away_country
          FROM read_parquet('{PARQUET}')
          WHERE gh IS NOT NULL AND ga IS NOT NULL
            AND ((competition LIKE 'UEFA%') OR (level='national' AND continent='Europe'))
            AND date >= DATE '1990-01-01'
          ORDER BY date, home
        ) TO '{EU_RESULTS}' (HEADER, DELIMITER ',')
    """)


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
    if not EU_RESULTS.exists():
        if not PARQUET.exists():
            log.error("seed_source_missing", parquet=str(PARQUET),
                      hint="download schochastics/football-data games.parquet into data/")
            return 1
        log.info("seed_exporting_parquet", parquet=str(PARQUET))
        _export_eu_results()

    state = EloState()
    country_of: dict[str, str] = {}
    applied = 0
    # The CSV is already date-ordered, so stream it straight in (no look-ahead).
    with EU_RESULTS.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                d = date.fromisoformat(r["date"][:10])
                gh, ga = int(r["gh"]), int(r["ga"])
            except (KeyError, ValueError, TypeError):
                continue
            home, away = (r.get("home") or "").strip(), (r.get("away") or "").strip()
            if not home or not away:
                continue
            state.apply(Result(d, home, away, gh, ga, r.get("competition", "")))
            applied += 1
            if r.get("home_country"):
                country_of.setdefault(home, r["home_country"])
            if r.get("away_country"):
                country_of.setdefault(away, r["away_country"])

    if not state.ratings or state.last_date is None:
        log.error("seed_empty")
        return 1

    settings = get_settings()
    ce_clubs, _ = _load_snapshot(Path(settings.cl_snapshot_path))
    resolver = _merged_resolver()
    snap, unresolved = bridge_to_clubelo(
        state.ratings, state.games, list(ce_clubs.keys()), resolver,
    )
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    STATE_PATH.write_text(json.dumps({
        "ratings": {k: round(v, 4) for k, v in state.ratings.items()},
        "games": state.games,
        "last_date": state.last_date.isoformat(),
    }))
    out = Path(settings.cl_selfbuilt_snapshot_path)
    write_clubelo_snapshot(out, snap, from_date=state.last_date)
    log.info("seed_done", applied=applied, teams=len(state.ratings),
             bridged=len(snap), unresolved=len(unresolved),
             from_date=str(state.last_date), out=str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
