"""Download historical club 1X2 odds from football-data.co.uk -> data/club_odds.csv.

Free, no key, no signup — the same archives ``fetch_club_results.py`` already
uses, but keeping BOTH price vintages per match:

* **prematch** (``PSH``/``B365H``/``AvgH``) — the early-week price. This is the
  same column family the live ``fixtures.csv`` feed publishes, so it is an
  honest stand-in for what we would actually have at T-24h.
* **closing** (``PSCH``/``B365CH``/``AvgCH``) — the kickoff price. NOT available
  pre-match. A backtest anchored on closing odds is OPTIMISTIC; it is written
  out only so the backtest can QUANTIFY that optimism, never to ship on.

Team names are pushed through the same explicit :class:`OddsNameResolver` the
live path uses, so a row that would not resolve live does not silently resolve
here either — the backtest measures the coverage we would really get.

Run (repo root, venv active):
    python scripts/fetch_club_odds.py
    python scripts/fetch_club_odds.py --seasons 2425 2526 --out data/club_odds.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from betbot.config import get_settings
from betbot.data.odds import DIV_TO_LEAGUE, FootballDataCoUkProvider
from betbot.data.odds_names import OddsNameResolver

DEFAULT_SEASONS = ("2021", "2122", "2223", "2324", "2425", "2526", "2627")

FIELDS = [
    "date", "league", "home", "away",
    "pre_home", "pre_draw", "pre_away", "pre_book",
    "close_home", "close_draw", "close_away", "close_book",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    ap.add_argument("--out", type=Path, default=Path("data/club_odds.csv"))
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="polite pause between requests (seconds)")
    args = ap.parse_args()

    s = get_settings()
    resolver = OddsNameResolver.from_files(
        s.odds_team_alias_path, s.club_name_map_path
    )
    provider = FootballDataCoUkProvider(resolver, timeout=60.0)

    merged: dict[tuple[str, str, str, str], dict] = {}
    for season in args.seasons:
        for league in DIV_TO_LEAGUE.values():
            for kind in ("prematch", "closing"):
                rows = provider.fetch_season(season, league, kind=kind)
                time.sleep(args.sleep)
                for r in rows:
                    key = (r.match_date.isoformat(), r.league, r.home, r.away)
                    rec = merged.setdefault(key, {
                        "date": key[0], "league": key[1], "home": key[2], "away": key[3],
                    })
                    prefix = "pre" if kind == "prematch" else "close"
                    rec[f"{prefix}_home"] = r.price_home
                    rec[f"{prefix}_draw"] = r.price_draw
                    rec[f"{prefix}_away"] = r.price_away
                    rec[f"{prefix}_book"] = r.book
            print(f"  {season} {league}: cumulative {len(merged)} fixtures")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(merged):
            w.writerow(merged[key])

    n_pre = sum(1 for v in merged.values() if v.get("pre_home"))
    n_close = sum(1 for v in merged.values() if v.get("close_home"))
    print(f"\nwrote {len(merged)} fixtures -> {args.out}")
    print(f"  with prematch prices: {n_pre}")
    print(f"  with closing prices : {n_close}")
    if provider.unresolved:
        print("\nUNRESOLVED team spellings (skipped, never guessed):")
        for name, count in sorted(provider.unresolved.items(), key=lambda kv: -kv[1]):
            print(f"  {name!r}: {count} rows")


if __name__ == "__main__":
    main()
