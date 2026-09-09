"""Reconstruct site-scale ClubElo history snapshots from per-club pages.

WHY THIS EXISTS
---------------
api.clubelo.com (the machine CSV feed the CL engine was originally tuned on)
was deactivated upstream, and its dated ranking pages (``/2024-10-01/``) 404 —
so there is no site-scale HISTORY available in bulk. But each per-club page
(``https://clubelo.com/<Slug>``) embeds that club's COMPLETE site-scale Elo
history as a Vega spec (``var vegaJson = {...}``): one ``values`` series of
``{Date, Elo, Golo, segment_id}`` per club, ~2022-09 to present, one row per
match, single segment. This script fetches each CL club's page ONCE (politely,
cached), verifies the series broadly, and rebuilds point-in-time snapshots in
the exact CSV format ``data/clubelo/*.csv`` uses, so the existing backtest and
engine tooling read them unchanged.

VERIFIED (2026-09, 61 clubs, zero mismatches): every linked CL club has a
single-segment series covering the full snapshot window; the per-page <h1> is
the canonical club name (guards against country homonyms like LiverpoolUY /
BarcelonaEC); current Elos all sit in the elite band. Kairat (FK Kairat, KAZ)
is the sole gap — an UNLINKED minnow with no page, so it is absent from the
reconstructed HISTORY. The LIVE table scrape (data/clubelo_scrape_latest.csv)
DOES capture Kairat, so only the historical backtest lacks it.

Politeness: browser UA, >=2.5s between fetches, cached to disk and never
re-fetched. ~61 pages, ~35 MB. Do not hammer a small free site.

Run (repo root):
    python scripts/reconstruct_clubelo_site.py \
        --api-dir data/clubelo --out-dir data/clubelo_site \
        --cache-dir .cache/clubelo_pages

The 27 snapshot dates are taken from --api-dir so the site reconstruction lines
up 1:1 with the api-scale snapshots for a head-to-head (scripts/compare_cl_scales.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: ClubElo CSV Club name -> clubelo.com per-club page slug. Built from the
#: homepage ranking anchors (NOT difflib), token-verified, and country-guarded
#: against each club's api-snapshot country + its page <h1>. Kairat is omitted:
#: it is an unlinked <span> with no page (see module docstring).
SLUG_MAP: dict[str, str] = {
    "Ajax": "Ajax", "Antwerp": "Antwerp", "Arsenal": "Arsenal",
    "Aston Villa": "AstonVilla", "Atalanta": "atalanta", "Atletico": "atletico",
    "Barcelona": "Barcelona", "Bayern": "Bayern", "Benfica": "benfica",
    "Bilbao": "athletic-club", "Bodoe Glimt": "BodoeGlimt", "Bologna": "Bologna",
    "Brest": "Brest", "Brugge": "Brugge", "Celtic": "Celtic", "Chelsea": "Chelsea",
    "Crvena Zvezda": "CrvenaZvezda", "Dinamo Zagreb": "DinamoZagreb",
    "Dortmund": "Dortmund", "FC Kobenhavn": "FCKobenhavn", "Feyenoord": "Feyenoord",
    "Frankfurt": "Frankfurt", "Galatasaray": "Galatasaray", "Girona": "Girona",
    "Inter": "Inter", "Juventus": "Juventus", "Karabakh Agdam": "KarabakhAgdam",
    "Lazio": "Lazio", "Lens": "Lens", "Leverkusen": "Leverkusen", "Lille": "Lille",
    "Liverpool": "Liverpool", "Man City": "ManCity", "Man United": "ManUnited",
    "Marseille": "Marseille", "Milan": "Milan", "Monaco": "Monaco", "Napoli": "Napoli",
    "Newcastle": "Newcastle", "Olympiakos": "Olympiakos", "PSV": "PSV",
    "Paphos": "Paphos", "Paris SG": "ParisSG", "Porto": "porto",
    "RB Leipzig": "RBLeipzig", "Real Madrid": "realmadrid", "Salzburg": "rbsalzburg",
    "Sevilla": "Sevilla", "Shakhtar": "Shakhtar", "Slavia Praha": "SlaviaPraha",
    "Slovan Bratislava": "SlovanBratislava", "Sociedad": "Sociedad",
    "Sparta Praha": "SpartaPraha", "Sporting": "Sporting", "St Gillis": "StGillis",
    "Sturm Graz": "SturmGraz", "Stuttgart": "Stuttgart", "Tottenham": "Tottenham",
    "Union Berlin": "UnionBerlin", "Villarreal": "Villarreal", "Young Boys": "YoungBoys",
}

_VEGA_RE = re.compile(r"var vegaJson\s*=\s*(\{.*?\})\s*;", re.S)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)


def fetch_page(slug: str, cache_dir: Path, *, spacing: float = 2.5) -> str:
    dest = cache_dir / f"{slug}.html"
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(  # noqa: S310 — https clubelo only
        f"https://clubelo.com/{slug}", headers={"User-Agent": BROWSER_UA}
    )
    with urllib.request.urlopen(req, timeout=40) as resp:  # noqa: S310
        data = resp.read()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    time.sleep(spacing)
    return data.decode("utf-8", errors="replace")


def parse_series(html: str) -> list[tuple[date, float]] | None:
    """Return the club's [(date, elo), ...] site-scale series, date-sorted."""
    m = _VEGA_RE.search(html)
    if not m:
        return None
    spec = json.loads(m.group(1))
    for values in spec.get("datasets", {}).values():
        if (isinstance(values, list) and values
                and isinstance(values[0], dict)
                and "Elo" in values[0] and "Date" in values[0]):
            rows = [
                (datetime.fromisoformat(r["Date"]).date(), float(r["Elo"]))
                for r in values
            ]
            rows.sort()
            return rows
    return None


def page_h1(html: str) -> str:
    m = _H1_RE.search(html)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""


def _n_segments(html: str) -> int:
    """Distinct segment_id count in the club's Vega series (expect 1)."""
    m = _VEGA_RE.search(html)
    if not m:
        return 0
    spec = json.loads(m.group(1))
    for values in spec.get("datasets", {}).values():
        if (isinstance(values, list) and values
                and isinstance(values[0], dict) and "segment_id" in values[0]):
            return len({r.get("segment_id") for r in values})
    return 0


def _elo_at(rows: list[tuple[date, float]], d: date) -> float | None:
    """Latest Elo with row-date <= d (point-in-time; no lookahead)."""
    val = None
    for dt, elo in rows:
        if dt <= d:
            val = elo
        else:
            break
    return val


def _api_meta(api_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Country + Level per club, from the most recent api snapshot that lists it."""
    country: dict[str, str] = {}
    level: dict[str, int] = {}
    for snap in sorted(api_dir.glob("*.csv")):
        for r in csv.DictReader(snap.open()):
            c = (r.get("Club") or "").strip()
            if c in SLUG_MAP:
                country.setdefault(c, (r.get("Country") or "").strip())
                if c not in level:
                    try:
                        level[c] = int(float(r.get("Level") or 1))
                    except (TypeError, ValueError):
                        level[c] = 1
    return country, level


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-dir", type=Path, default=Path("data/clubelo"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/clubelo_site"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/clubelo_pages"))
    ap.add_argument("--no-fetch", action="store_true",
                    help="use only cached pages; never hit the network")
    args = ap.parse_args()

    dates = sorted(p.stem for p in args.api_dir.glob("*.csv"))
    need_min, need_max = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
    country, level = _api_meta(args.api_dir)

    # --- fetch + verify each club -----------------------------------------
    series: dict[str, list[tuple[date, float]]] = {}
    problems: list[str] = []
    for csvname, slug in sorted(SLUG_MAP.items()):
        cached = args.cache_dir / f"{slug}.html"
        if args.no_fetch and not cached.exists():
            problems.append(f"{csvname}({slug}): no cached page and --no-fetch")
            continue
        html = fetch_page(slug, args.cache_dir)
        s = parse_series(html)
        if not s:
            problems.append(f"{csvname}({slug}): no vega series")
            continue
        n_seg = _n_segments(html)
        if n_seg != 1:
            problems.append(f"{csvname}({slug}): {n_seg} segments (expected 1)")
        mn, mx = s[0][0], s[-1][0]
        if not (mn <= need_min and mx >= need_max):
            problems.append(f"{csvname}({slug}): coverage {mn}..{mx} "
                            f"(need {need_min}..{need_max})")
        h1 = page_h1(html)
        if not h1:
            problems.append(f"{csvname}({slug}): no <h1> identity")
        series[csvname] = s
        print(f"  {csvname:18s}{slug:18s} n={len(s):>4d} {mn}..{mx} "
              f"h1={h1[:24]!r} cur={s[-1][1]:.0f}")

    if problems:
        print("\nVERIFICATION PROBLEMS:")
        for p in problems:
            print("  ", p)
        print("Refusing to reconstruct with unverified clubs. Fix and re-run.")
        raise SystemExit(1)

    # --- reconstruct point-in-time snapshots ------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ds in dates:
        d = date.fromisoformat(ds)
        recs = []
        for csvname, rows in series.items():
            elo = _elo_at(rows, d)
            if elo is None:
                continue
            frm = max(dt for dt, _ in rows if dt <= d)
            recs.append((csvname, country.get(csvname, "???"),
                         level.get(csvname, 1), elo, frm))
        recs.sort(key=lambda r: -r[3])
        with (args.out_dir / f"{ds}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Rank", "Club", "Country", "Level", "Elo", "From", "To"])
            for rank, (c, co, lv, elo, frm) in enumerate(recs, 1):
                w.writerow([rank, c, co, lv, f"{elo:.2f}", frm.isoformat(), ds])
    print(f"\nwrote {len(dates)} site-scale snapshots to {args.out_dir} "
          f"({len(series)} clubs each; Kairat excluded — pageless)")


if __name__ == "__main__":
    main()
