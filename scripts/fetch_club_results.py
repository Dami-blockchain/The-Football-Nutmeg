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

CURRENT-SEASON FALLBACK
-----------------------
football-data.co.uk goes down for days at a time (an IONOS shield answering
HTTP 503 for every UA/path). When that happens the whole file used to be
rewritten empty and the script exited 1, which silently froze the weekly Glicko
re-seed and Dixon-Coles refit. So:

* football-data.co.uk stays AUTHORITATIVE for historical seasons and for the
  ``ps_*`` closing-odds columns (the odds anchor's backtest depends on those).
* Merge is KEY-LEVEL (per fixture), not partition-level: a fresh .co.uk row
  wins for any fixture it carries, and an existing row it does not carry is
  kept. So a failed, truncated, lagging, or cross-source-mismatched body can
  ADD or OVERRIDE rows but can never DELETE one. A body that parses to ZERO
  rows (an IONOS shield HTML page served with HTTP 200) is refused outright.
  Documented residual: a genuinely wrong existing row can no longer be removed
  by a re-fetch — accepted, since losing history is worse than a stale row.
* For the CURRENT season only, when .co.uk cannot supply it we fall back to
  football-data.org (already keyed, already rate-limited, free tier covers
  current-season results for all five domestic leagues) for FINISHED results.
  Those rows carry no closing odds (``ps_*`` empty) — odds stay a .co.uk-only
  concern. Names are mapped football-data.org -> dataset names with the SAME
  alias resolver the market matcher uses; any club that fails to map is kept
  (under its football-data.org name, so it never silently vanishes) and
  SURFACED loudly plus written to ``data/club_fallback_report.json`` for the
  daemon to page on.

Run (repo root, venv active):
    python scripts/fetch_club_results.py
    python scripts/fetch_club_results.py --seasons 2223 2324 2425 --out data/club_results.csv
    python scripts/fetch_club_results.py --no-fallback   # .co.uk only (offline test)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
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

# The season whose results the football-data.org fallback is allowed to supply.
# Keep in lockstep with the newest entry in DEFAULT_SEASONS.
CURRENT_SEASON = "2627"

# Where the fallback records its coverage so the daemon can page on unmapped
# clubs / a degraded (.co.uk-down) run without parsing captured stdout.
REPORT_PATH = Path("data/club_fallback_report.json")


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


def _season_code(date_iso: str) -> str | None:
    """ISO date -> football-data.co.uk 4-digit season code.

    Seasons run Aug->May; we cut at July. ``2026-08-15`` -> ``2627``,
    ``2026-03-01`` -> ``2526``. Used to partition existing rows so a
    successful .co.uk fetch replaces exactly the season it covers and a
    failed one leaves every other season's rows untouched.
    """
    try:
        y, m, _ = date_iso.split("-")
        year, month = int(y), int(m)
    except (ValueError, AttributeError):
        return None
    start = year if month >= 7 else year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


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


def _fixture_key(row: dict) -> tuple:
    """Dedupe/identity key for a fixture, spelling-insensitive on team names."""
    from betbot.exchanges.matcher import normalize
    return (
        row["league"],
        row["date"],
        normalize(str(row["home_team"])),
        normalize(str(row["away_team"])),
    )


def _load_existing(path: Path) -> list[dict]:
    """Last-known-good rows, so a failed fetch never drops history."""
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                try:
                    r["home_score"] = int(float(r["home_score"]))
                    r["away_score"] = int(float(r["away_score"]))
                except (ValueError, KeyError, TypeError):
                    continue
                rows.append(r)
    except OSError as e:
        print(f"  could not read existing {path}: {e}")
    return rows


def _fetch_couk(seasons, timeout, existing_counts):
    """Fetch .co.uk. Returns (rows, fetched_partitions, rejected).

    ``fetched_partitions`` is the set of (league, season_code) whose fresh file
    is TRUSTED and therefore replaces existing rows. ``rejected`` is a list of
    partitions whose fresh file was refused (and whose existing rows are kept),
    with the reason — surfaced in the report so the daemon can page.

    Why a partition can be refused even on HTTP 200: football-data.co.uk sits
    behind an IONOS shield that answers requests with a 200 HTML page that
    parses to ZERO result rows (the same "a 200 that is not your data"
    footgun that has already bitten the odds provider and the ClubElo scrape).
    Marking such a partition "served" would DROP every existing row for it. So
    we refuse a partition whose fresh row count is 0, or is SMALLER than what
    we already hold (a completed season never shrinks). ``existing_counts`` is
    the last-known-good per-partition count.
    """
    rows: list[dict] = []
    fetched: set[tuple[str, str]] = set()
    rejected: list[dict] = []
    for season in seasons:
        for div, league in DIV_TO_LEAGUE.items():
            url = BASE_URL.format(season=season, div=div)
            raw = _fetch(url, timeout)
            if not raw:
                continue
            parsed: list[dict] = []
            for r in csv.DictReader(io.StringIO(raw)):
                d = _iso_date(r.get("Date", ""))
                home = (r.get("HomeTeam") or "").strip()
                away = (r.get("AwayTeam") or "").strip()
                fthg, ftag = (r.get("FTHG") or "").strip(), (r.get("FTAG") or "").strip()
                if not (d and home and away and fthg and ftag):
                    continue
                try:
                    hs, as_ = int(float(fthg)), int(float(ftag))
                except ValueError:
                    continue
                parsed.append({
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
            have = existing_counts.get((league, season), 0)
            if not parsed:
                # 200 (or otherwise) but zero parseable rows — a shield page or
                # an empty file. Never let it delete the partition.
                rejected.append({
                    "league": league, "season": season,
                    "reason": "empty", "fresh": 0, "existing": have})
                print(f"  {season} {div}->{league}: 0 rows — REJECTED "
                      f"(kept {have} existing)")
                continue
            # A short (but non-empty) body is TRUSTED at key level — main()
            # merges per fixture, so it can only ADD or OVERRIDE rows, never
            # delete one. So we do NOT reject a shrink. But a COMPLETED season
            # that shrinks is genuinely anomalous (a provider correction, or a
            # partial body) and worth a page, so we record it as a diagnostic.
            # The CURRENT season legitimately shrinks across sources — .co.uk's
            # D1 omits the relegation play-off FD.org counts (308 vs 306), and
            # .co.uk lags FD.org mid-week — so it must NEVER be flagged.
            if len(parsed) < have and season != CURRENT_SEASON:
                rejected.append({
                    "league": league, "season": season,
                    "reason": "shrink", "fresh": len(parsed), "existing": have})
                print(f"  {season} {div}->{league}: {len(parsed)} < {have} rows "
                      f"— completed-season SHRINK (diagnostic; rows still merged)")
            rows.extend(parsed)
            fetched.add((league, season))
            print(f"  {season} {div}->{league}: +{len(parsed)} matches")
    return rows, fetched, rejected


def _fetch_fallback_current(dataset_names, present_keys, leagues=None):
    """football-data.org FINISHED results for the CURRENT season.

    Maps football-data.org names -> dataset (.co.uk) names via the market
    matcher's alias resolver so a club's history is not split across two keys.
    Only adds fixtures not already present (``present_keys``). ``leagues`` is
    the subset to fetch (the ones .co.uk did NOT serve this run); ``None`` means
    all domestic leagues. Returns ``(rows, report)`` where ``report`` records
    coverage and any unmapped club.
    """
    import asyncio

    from betbot.config import LEAGUE_CODES, get_settings
    from betbot.exchanges.matcher import TeamAliasResolver, normalize
    from betbot.data.football_data import FootballDataClient

    settings = get_settings()
    resolver = TeamAliasResolver.from_yaml("config/team_aliases.yaml")
    # normalised dataset name -> canonical dataset spelling (for exact hits).
    norm_to_dataset = {}
    for n in dataset_names:
        norm_to_dataset.setdefault(normalize(n), n)
    dataset_list = list(dataset_names)

    domestic = tuple(c for c in LEAGUE_CODES if c not in ("WC", "CL"))
    leagues = tuple(leagues) if leagues is not None else domestic
    # First of July of the current season's start year bounds the window.
    start_year = 2000 + int(CURRENT_SEASON[:2])
    date_from = f"{start_year}-07-01"
    date_to = datetime.now(timezone.utc).date().isoformat()

    def _resolve(name):
        nf = normalize(name)
        return norm_to_dataset.get(nf) or resolver.match(name, dataset_list)

    async def _run():
        rows = []
        unmapped = []
        mapped = 0
        async with FootballDataClient(
            api_key=settings.football_data_api_key,
            base_url=settings.football_data_base_url,
            rate_limit_per_min=settings.football_data_rate_limit_per_min,
        ) as client:
            for league in leagues:
                try:
                    matches = await client.list_matches(
                        league, date_from, date_to, status="FINISHED")
                except Exception as e:  # noqa: BLE001 — one league failing isn't fatal
                    print(f"  fallback {league}: FD.org error {type(e).__name__}: {e}")
                    continue
                added = 0
                for m in matches:
                    ft = (m.get("score") or {}).get("fullTime") or {}
                    hs, as_ = ft.get("home"), ft.get("away")
                    if hs is None or as_ is None:
                        continue
                    utc = m.get("utcDate") or ""
                    d = utc[:10]
                    if not d:
                        continue
                    raw_home = ((m.get("homeTeam") or {}).get("name") or "").strip()
                    raw_away = ((m.get("awayTeam") or {}).get("name") or "").strip()
                    if not (raw_home and raw_away):
                        continue
                    mh, ma = _resolve(raw_home), _resolve(raw_away)
                    # Never let an unmapped club vanish: keep the FD.org name and
                    # surface it so a human can add an alias.
                    for raw, mapped_name in ((raw_home, mh), (raw_away, ma)):
                        if mapped_name is None:
                            unmapped.append(f"{raw} ({league})")
                        else:
                            mapped += 1
                    row = {
                        "date": d,
                        "home_team": mh or raw_home,
                        "away_team": ma or raw_away,
                        "home_score": int(hs),
                        "away_score": int(as_),
                        "league": league,
                        "ps_home": "",
                        "ps_draw": "",
                        "ps_away": "",
                    }
                    if _fixture_key(row) in present_keys:
                        continue
                    present_keys.add(_fixture_key(row))
                    rows.append(row)
                    added += 1
                print(f"  fallback {league}: +{added} FINISHED matches (FD.org)")
        total_names = mapped + len(unmapped)
        coverage = (mapped / total_names) if total_names else 1.0
        report = {
            "mapped": mapped,
            "unmapped": sorted(set(unmapped)),
            "coverage": round(coverage, 4),
            "rows": len(rows),
        }
        return rows, report

    return asyncio.run(_run())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    ap.add_argument("--out", type=Path, default=Path("data/club_results.csv"))
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--no-fallback", action="store_true",
                    help="skip the football-data.org current-season fallback.")
    args = ap.parse_args()

    existing = _load_existing(args.out)

    # Per-partition counts of the last-known-good file, so _fetch_couk can
    # refuse a fresh file that is empty or has shrunk (a shield 200).
    existing_counts: dict[tuple[str, str], int] = {}
    for r in existing:
        sc = _season_code(r["date"])
        if sc is not None:
            existing_counts[(r["league"], sc)] = \
                existing_counts.get((r["league"], sc), 0) + 1

    couk_rows, fetched, rejected = _fetch_couk(
        args.seasons, args.timeout, existing_counts)

    # KEY-LEVEL merge (not partition-level): a fresh .co.uk row WINS for any
    # fixture it carries — so its closing odds land, even over a prior FD.org
    # fallback row for the same fixture — while an existing row the fresh file
    # does NOT carry is kept. A truncated, lagging, or cross-source-mismatched
    # body can therefore add or override rows but can never DELETE one.
    # Residual tradeoff: a genuinely wrong existing row can no longer be
    # removed by a re-fetch. Accepted — losing history is worse than keeping a
    # single stale row. ``fetched`` is now used only to compute missing_current.
    fresh_keys = {_fixture_key(r) for r in couk_rows}
    merged = [r for r in existing if _fixture_key(r) not in fresh_keys]
    merged.extend(couk_rows)

    # Which current-season leagues did .co.uk NOT serve this run? (Files are
    # published at different times, so a partial current season is normal.)
    missing_current = [
        lg for lg in DIV_TO_LEAGUE.values()
        if (lg, CURRENT_SEASON) not in fetched
    ]
    couk_has_current = not missing_current
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "couk_ok": bool(couk_rows),
        "couk_has_current": couk_has_current,
        "couk_missing_leagues": missing_current,
        "rejected_partitions": rejected,
        "fallback_used": False,
        "mapped": 0,
        "unmapped": [],
        "coverage": None,
        "fallback_rows": 0,
    }

    # Current-season fallback: for exactly the leagues .co.uk did not serve.
    if not args.no_fallback and missing_current:
        present = {_fixture_key(r) for r in merged}
        dataset_names = (
            {r["home_team"] for r in merged} | {r["away_team"] for r in merged}
        )
        print(f"\nfootball-data.co.uk missing season {CURRENT_SEASON} for "
              f"{missing_current} — falling back to football-data.org")
        try:
            fb_rows, fb_report = _fetch_fallback_current(
                dataset_names, present, leagues=missing_current)
            merged.extend(fb_rows)
            report.update(
                fallback_used=True,
                mapped=fb_report["mapped"],
                unmapped=fb_report["unmapped"],
                coverage=fb_report["coverage"],
                fallback_rows=len(fb_rows),
            )
            print(f"  fallback added {len(fb_rows)} rows; "
                  f"name-map coverage {fb_report['coverage']:.1%} "
                  f"({fb_report['mapped']} mapped, "
                  f"{len(fb_report['unmapped'])} unmapped)")
            if fb_report["unmapped"]:
                print("  UNMAPPED CLUBS (kept under FD.org name, add aliases):")
                for u in fb_report["unmapped"]:
                    print(f"    - {u}")
        except Exception as e:  # noqa: BLE001 — fallback failure must not wipe history
            print(f"  fallback FAILED ({type(e).__name__}: {e}); "
                  f"keeping last-known-good rows")

    # Write the coverage report for the daemon to page on (best-effort).
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"  could not write {REPORT_PATH}: {e}")

    if not merged:
        print("no rows (fetch failed AND no existing file) — aborting")
        sys.exit(1)

    # Dedupe (existing + fresh could overlap on a partial partition) and sort.
    seen = set()
    deduped = []
    for r in merged:
        k = _fixture_key(r)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    deduped.sort(key=lambda r: (r["date"], r["league"]))

    fieldnames = ["date", "home_team", "away_team", "home_score",
                  "away_score", "league", "ps_home", "ps_draw", "ps_away"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(deduped)

    leagues = sorted({r["league"] for r in deduped})
    seasons = sorted({r["date"][:4] for r in deduped})
    teams = {r["home_team"] for r in deduped} | {r["away_team"] for r in deduped}
    print(f"\nwrote {args.out}: {len(deduped)} matches, "
          f"{len(teams)} clubs, leagues={leagues}, years={seasons}")


if __name__ == "__main__":
    main()
