"""Download historical Champions League results -> data/cl_results.csv.

Source: football-data.org (the same API the live loop polls; free tier serves
finished CL matches for recent seasons). Used by scripts/backtest_cl.py to
gate the cross-league Elo engine. Columns match club_results.csv minus odds
(the free tier carries no usable odds for CL):

    date,home_team,away_team,home_score,away_score,league

Run (repo root, venv active):
    python scripts/fetch_cl_results.py
    python scripts/fetch_cl_results.py --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient


async def _fetch(seasons: list[int]) -> list[dict]:
    s = get_settings()
    rows: list[dict] = []
    async with FootballDataClient(
        api_key=s.football_data_api_key,
        base_url=s.football_data_base_url,
        rate_limit_per_min=s.football_data_rate_limit_per_min,
    ) as client:
        for season in seasons:
            data = await client._get(
                "/competitions/CL/matches",
                params={"season": season, "status": "FINISHED"},
            )
            matches = data.get("matches") or []
            n = 0
            for m in matches:
                ft = (m.get("score") or {}).get("fullTime") or {}
                hs, as_ = ft.get("home"), ft.get("away")
                home = (m.get("homeTeam") or {}).get("name")
                away = (m.get("awayTeam") or {}).get("name")
                d = (m.get("utcDate") or "")[:10]
                if None in (hs, as_) or not (home and away and d):
                    continue
                rows.append({
                    "date": d, "home_team": home, "away_team": away,
                    "home_score": int(hs), "away_score": int(as_), "league": "CL",
                })
                n += 1
            print(f"  season {season}: {n} finished matches")
    rows.sort(key=lambda r: r["date"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--out", type=Path, default=Path("data/cl_results.csv"))
    args = ap.parse_args()

    rows = asyncio.run(_fetch(args.seasons))
    if not rows:
        raise SystemExit("no CL matches fetched")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    teams = {r["home_team"] for r in rows} | {r["away_team"] for r in rows}
    print(f"wrote {args.out}: {len(rows)} matches, {len(teams)} clubs")


if __name__ == "__main__":
    main()
