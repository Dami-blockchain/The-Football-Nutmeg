"""Bootstrap Glicko-2 ratings for CLUB teams from data/club_results.csv.

Mirror of ``seed_glicko.py`` (internationals) but for the top-5 domestic
leagues. Two things differ:

* the history comes from ``data/club_results.csv`` (football-data.co.uk via
  ``scripts/fetch_club_results.py``), whose club names ("Man City", "Bayern
  Munich") differ from the football-data.org names the live loop looks up
  ("Manchester City FC", "FC Bayern München"); so after the replay we copy each
  live team's rating under its football-data.org name via the same alias
  resolver the market matcher uses. Without this bridge, live club lookups
  silently fall back to the default rating.
* Glicko itself is venue-neutral (home advantage is applied at PREDICT time in
  the engine, not baked into ratings), so the replay is identical maths.

Ratings are stored keyed BOTH by the dataset name and the football-data.org
name, so lookups from either side resolve. Idempotent; safe to re-run.

Run (repo root, venv active):
    python scripts/seed_glicko_club.py
    python scripts/seed_glicko_club.py --years-back 6 --no-alias   # offline
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import defaultdict
from pathlib import Path

from betbot.config import LEAGUE_CODES, get_settings
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import upsert_rating_if_fresher
from betbot.strategy.glicko import Glicko2Rating, update_rating

log = get_logger("seed_glicko_club")

# football-data.org-name -> dataset-name map (normalised keys) written here so
# the club engine can resolve Dixon-Coles team keys (which are dataset-named).
NAME_MAP_PATH = Path("data/club_name_map.json")

# Domestic leagues we hold club history for (excludes WC and CL).
CLUB_LEAGUES = tuple(c for c in LEAGUE_CODES if c not in ("WC", "CL"))


def _outcome(hs: int, as_: int) -> str:
    return "HOME" if hs > as_ else ("AWAY" if as_ > hs else "DRAW")


def _replay(rows: list[tuple[str, str, str, int, int]], settings) -> dict[str, Glicko2Rating]:
    """Forward Glicko-2 replay, one rating period per match date."""
    default = Glicko2Rating(
        settings.glicko_default_rating, settings.glicko_default_rd, settings.glicko_default_vol
    )
    ratings: dict[str, Glicko2Rating] = {}
    by_date: dict[str, list] = defaultdict(list)
    for d, h, a, hs, as_ in rows:
        by_date[d].append((h, a, _outcome(hs, as_)))

    for d in sorted(by_date):
        matches = by_date[d]
        teams = {t for h, a, _ in matches for t in (h, a)}
        cur = {t: ratings.get(t, default) for t in teams}
        per: dict[str, list] = {t: [] for t in teams}
        for h, a, o in matches:
            sh = 1.0 if o == "HOME" else (0.5 if o == "DRAW" else 0.0)
            per[h].append((cur[a].rating, cur[a].rd, sh))
            per[a].append((cur[h].rating, cur[h].rd, 1.0 - sh if o != "DRAW" else 0.5))
        for t in teams:
            ratings[t] = update_rating(cur[t], per[t], tau=settings.glicko_tau, period=d)
    return ratings


def _load_rows(path: Path, years_back: int) -> list[tuple[str, str, str, int, int]]:
    rows: list[tuple[str, str, str, int, int]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                d = r["date"]
                hs, as_ = int(float(r["home_score"])), int(float(r["away_score"]))
            except (KeyError, ValueError):
                continue
            rows.append((d, r["home_team"].strip(), r["away_team"].strip(), hs, as_))
    rows.sort(key=lambda x: x[0])
    if rows:
        cutoff = str(int(rows[-1][0][:4]) - years_back)
        rows = [r for r in rows if r[0][:4] >= cutoff]
    return rows


async def _alias_to_footballdata(ratings: dict[str, Glicko2Rating], settings) -> int:
    """Copy each live football-data.org club rating from its dataset match."""
    from betbot.data.football_data import FootballDataClient
    from betbot.exchanges.matcher import TeamAliasResolver, normalize

    resolver = TeamAliasResolver.from_yaml("config/team_aliases.yaml")
    dataset_names = list(ratings)
    # Map every dataset name's normalised form back to its raw spelling, so we
    # can copy the rating to the live football-data.org name even when the two
    # normalise identically (e.g. dataset "Arsenal" vs live "Arsenal FC").
    norm_to_dataset: dict[str, str] = {}
    for n_ in dataset_names:
        norm_to_dataset.setdefault(normalize(n_), n_)
    # Identity entries so exact-spelling teams resolve without a bridge.
    name_map: dict[str, str] = {normalize(n): normalize(n) for n in dataset_names}
    n = 0
    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        for league in CLUB_LEAGUES:
            try:
                teams = await client.list_competition_teams(league)
            except Exception as e:  # noqa: BLE001 — one league failing isn't fatal
                log.warning("club_teams_fetch_failed", league=league, error=str(e))
                continue
            for t in teams:
                name = t.get("name")
                if not name:
                    continue
                nf = normalize(name)
                # Prefer an exact normalised hit (covers "Arsenal FC"->"Arsenal");
                # fall back to fuzzy/alias resolution for the rest.
                match = norm_to_dataset.get(nf) or resolver.match(name, dataset_names)
                if match is not None:
                    # Always store the rating under the LIVE football-data.org
                    # name — that's the key the scoring loop looks up.
                    upsert_rating_if_fresher(
                        name, ratings[match], team_id=t.get("id"))
                    name_map[nf] = normalize(match)
                    log.info("club_glicko_alias_seeded", league=league,
                             fd_team=name, dataset_team=match,
                             rating=round(ratings[match].rating))
                    n += 1
                else:
                    log.warning("club_glicko_alias_unresolved", league=league, fd_team=name)
    NAME_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    NAME_MAP_PATH.write_text(json.dumps(name_map, indent=0, sort_keys=True), encoding="utf-8")
    log.info("club_name_map_written", path=str(NAME_MAP_PATH), entries=len(name_map))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/club_results.csv"))
    ap.add_argument("--years-back", type=int, default=6)
    ap.add_argument("--no-alias", action="store_true",
                    help="skip the football-data.org name bridge (offline).")
    args = ap.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} missing — run scripts/fetch_club_results.py first")

    rows = _load_rows(args.csv, args.years_back)
    ratings = _replay(rows, settings)
    # MERGE, don't SET: never regress a rating the in-season settlement
    # nudge has advanced past what this CSV knows (see
    # repos.upsert_rating_if_fresher).
    for name, r in ratings.items():
        upsert_rating_if_fresher(name, r)
    print(f"seeded {len(ratings)} club ratings from {len(rows)} matches "
          f"(last {args.years_back} yrs)")

    aliased = 0
    if not args.no_alias:
        aliased = asyncio.run(_alias_to_footballdata(ratings, settings))
        print(f"bridged {aliased} football-data.org club names")

    top = sorted(ratings.items(), key=lambda kv: -kv[1].rating)[:12]
    print("\nstrongest clubs (dataset names):")
    for name, r in top:
        print(f"  {name:22s} {r.rating:7.1f}  (RD {r.rd:5.1f})")


if __name__ == "__main__":
    main()
