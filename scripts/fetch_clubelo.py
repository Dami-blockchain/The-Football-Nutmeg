"""Fetch ClubElo ratings snapshots -> data/clubelo/ (+ data/clubelo_latest.csv).

ClubElo (clubelo.com) publishes free point-in-time Elo ratings for ~600
European clubs, trained on domestic AND European results — which makes club
strengths comparable ACROSS leagues. That is exactly what Champions League
pricing needs and what our per-league Glicko can't do.

Two modes:

* ``--latest``: fetch today's snapshot to ``data/clubelo_latest.csv`` (the
  live CL engine reads this; refresh daily/weekly via cron or the daemon).
* ``--for-dates <csv>``: for the backtest — fetch one snapshot per unique
  month found in a results CSV (Elo moves slowly; a <=1-month-old snapshot
  mirrors how live would use a periodically refreshed file). Snapshots are
  cached in ``data/clubelo/YYYY-MM-DD.csv`` and never re-fetched.

Be polite to the free service: sequential fetches, small pause, cache on disk.

Run (repo root, venv active):
    python scripts/fetch_clubelo.py --latest
    python scripts/fetch_clubelo.py --for-dates data/cl_results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
import urllib.request
from datetime import date
from pathlib import Path

BASE = "http://api.clubelo.com/{d}"
CACHE_DIR = Path("data/clubelo")
LATEST = Path("data/clubelo_latest.csv")


def _fetch_snapshot(d: str, dest: Path, timeout: int = 30) -> bool:
    try:
        raw = urllib.request.urlopen(BASE.format(d=d), timeout=timeout).read()  # noqa: S310
    except Exception as e:  # noqa: BLE001
        print(f"  fetch {d} failed: {e}")
        return False
    text = raw.decode("utf-8", errors="replace")
    if not text.startswith("Rank,"):
        print(f"  fetch {d}: unexpected payload")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    n = max(text.count("\n") - 1, 0)
    print(f"  {d}: {n} clubs -> {dest}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--for-dates", type=Path,
                    help="results CSV; fetch a snapshot per unique month in it")
    ap.add_argument("--pause", type=float, default=1.5)
    args = ap.parse_args()

    if args.latest:
        today = date.today().isoformat()
        ok = _fetch_snapshot(today, LATEST)
        raise SystemExit(0 if ok else 1)

    if not args.for_dates:
        raise SystemExit("pass --latest or --for-dates <results.csv>")

    months: set[str] = set()
    for r in csv.DictReader(args.for_dates.open()):
        d = (r.get("date") or "")[:7]
        if d:
            months.add(d)
    wanted = sorted(f"{m}-01" for m in months)
    print(f"{len(wanted)} monthly snapshots to ensure cached")
    for d in wanted:
        dest = CACHE_DIR / f"{d}.csv"
        if dest.exists():
            continue
        _fetch_snapshot(d, dest)
        time.sleep(args.pause)
    print("done")


if __name__ == "__main__":
    main()
