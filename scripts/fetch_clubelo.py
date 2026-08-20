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
from datetime import date
from pathlib import Path

from betbot.data.clubelo import refresh_latest, snapshot_status

CACHE_DIR = Path("data/clubelo")
LATEST = Path("data/clubelo_latest.csv")


def _fetch_snapshot(d: str, dest: Path) -> bool:
    """Delegate to betbot.data.clubelo so there is ONE hardened fetch path.

    This script used to carry its own copy: single attempt, no retry, no
    payload validation, non-atomic write. That duplicate is exactly how the
    two paths drifted, so it is gone — retry/backoff, validation and the
    atomic write now apply here too.
    """
    ok = refresh_latest(dest, snapshot_date=date.fromisoformat(d))
    if ok:
        st = snapshot_status(dest)
        print(f"  {d}: {st.clubs} clubs -> {dest}")
    else:
        print(f"  fetch {d} failed (see log)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--for-dates", type=Path,
                    help="results CSV; fetch a snapshot per unique month in it")
    ap.add_argument("--pause", type=float, default=1.5)
    args = ap.parse_args()

    if args.latest:
        ok = refresh_latest(LATEST)
        st = snapshot_status(LATEST)
        if ok:
            print(f"  {st.snapshot_date}: {st.clubs} clubs -> {LATEST}")
        else:
            # Loud on the way out: the operator running this by hand should see
            # what the CL engine is actually left holding.
            age = "no file" if st.age_days is None else f"{st.age_days:.1f} days old"
            print(f"  refresh FAILED; existing snapshot is {age} ({st.reason})")
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
