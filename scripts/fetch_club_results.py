"""Download historical club results from football-data.co.uk -> data/club_results.csv.

Source: the free, well-known football-data.co.uk season archives (one CSV per
league per season, full-time scores + closing odds). Used to (a) seed per-club
Glicko-2 ratings and (b) fit the club Dixon-Coles goal model, and to give the
club backtest a real market baseline (Pinnacle/Bet365 closing lines).

We fetch the top-5 European leagues for the last few completed seasons and
normalise to a single tidy CSV:

    date,home_team,away_team,home_score,away_score,league,ps_home,ps_draw,ps_away

``league`` is our football-data.org competition code (PL/PD/BL1/SA/FL1) so the
downstream seeding lines up with the live scoring loop. ``ps_*`` are the closing
decimal odds (Pinnacle, falling back to Bet365 then the market average) — used
ONLY by the backtest as a market reference, never for training.

Run (repo root, venv active):
    python scripts/fetch_club_results.py
    python scripts/fetch_club_results.py --seasons 2223 2324 2425 --out data/club_results.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

# football-data.co.uk division code -> our football-data.org competition code.
DIV_TO_LEAGUE = {
    "E0": "PL",   # England Premier League
    "SP1": "PD",  # Spain La Liga (Primera Division)
    "D1": "BL1",  # Germany Bundesliga
    "I1": "SA",   # Italy Serie A
    "F1": "FL1",  # France Ligue 1
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

# Completed seasons PLUS the current one (football-data.co.uk 4-digit form:
# "2324" = 2023-24; "2627" = 2026-27, published live from matchday 1). Adding
# the current season is what lets the weekly re-seed advance in-season instead
# of freezing ratings at the end of 2025-26 every Monday.
DEFAULT_SEASONS = ("2021", "2122", "2223", "2324", "2425", "2526", "2627")


def _iso_date(raw: str) -> str | None:
    """football-data.co.uk uses dd/mm/yyyy (older files dd/mm/yy)."""
    raw = raw.strip()
    for fmt_len, century in ((10, None), (8, 2000)):
        parts = raw.split("/")
        if len(parts) != 3:
            return None
        d, m, y = parts
        if not (d.isdigit() and m.isdigit() and y.isdigit()):
            return None
        year = int(y)
        if len(y) == 2:
            year += 2000
        return f"{year:04d}-{int(m):02d}-{int(d):02d}"
    return None


def _first_float(row: dict, keys: tuple[str, ...]) -> str:
    """Return the first present, parseable odds column (as a string), else ''."""
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            try:
                float(v)
                return v
            except ValueError:
                continue
    return ""


def _fetch(url: str, timeout: int) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 — a missing season is not fatal
        print(f"  skip {url}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    ap.add_argument("--out", type=Path, default=Path("data/club_results.csv"))
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    out_rows: list[dict] = []
    for season in args.seasons:
        for div, league in DIV_TO_LEAGUE.items():
            url = BASE_URL.format(season=season, div=div)
            raw = _fetch(url, args.timeout)
            if not raw:
                continue
            n_before = len(out_rows)
            for r in csv.DictReader(io.StringIO(raw)):
                d = _iso_date(r.get("Date", ""))
                home, away = (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
                fthg, ftag = (r.get("FTHG") or "").strip(), (r.get("FTAG") or "").strip()
                if not (d and home and away and fthg and ftag):
                    continue
                try:
                    hs, as_ = int(float(fthg)), int(float(ftag))
                except ValueError:
                    continue
                out_rows.append({
                    "date": d,
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": as_,
                    "league": league,
                    "ps_home": _first_float(r, ("PSCH", "PSH", "B365H", "AvgH", "BbAvH")),
                    "ps_draw": _first_float(r, ("PSCD", "PSD", "B365D", "AvgD", "BbAvD")),
                    "ps_away": _first_float(r, ("PSCA", "PSA", "B365A", "AvgA", "BbAvA")),
                })
            print(f"  {season} {div}->{league}: +{len(out_rows) - n_before} matches")

    if not out_rows:
        print("no rows fetched — aborting (network? season codes?)")
        sys.exit(1)

    out_rows.sort(key=lambda r: (r["date"], r["league"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    leagues = sorted({r["league"] for r in out_rows})
    seasons = sorted({r["date"][:4] for r in out_rows})
    teams = {r["home_team"] for r in out_rows} | {r["away_team"] for r in out_rows}
    print(f"\nwrote {args.out}: {len(out_rows)} matches, "
          f"{len(teams)} clubs, leagues={leagues}, years={seasons}")


if __name__ == "__main__":
    main()
