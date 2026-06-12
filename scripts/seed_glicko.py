"""Bootstrap Glicko-2 ratings for international teams (Phase 5.5).

Two paths (the operator supplies the data — we don't scrape a source):

* **Path 1 (preferred):** point ``BETBOT_GLICKO_RESULTS_CSV`` at an international
  results dataset (columns: ``date,home_team,away_team,home_score,away_score``,
  e.g. the well-known open "results.csv"). We replay Glicko-2 forward over the
  last few years (one rating period per match date) and store the final ratings.
* **Path 2 (fallback):** no CSV → seed every current World-Cup team (from
  football-data) at the default rating with a high RD, so they at least exist;
  ratings then sharpen as tournament matches settle.

Run: ``python scripts/seed_glicko.py``  (idempotent; overwrites the table).
"""

from __future__ import annotations

import asyncio
import csv
from collections import defaultdict
from pathlib import Path

from betbot.config import get_settings
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import upsert_rating
from betbot.strategy.glicko import Glicko2Rating, update_rating

log = get_logger("seed_glicko")


def _outcome(hs: int, as_: int) -> str:
    return "HOME" if hs > as_ else ("AWAY" if as_ > hs else "DRAW")


def _seed_from_csv(path: Path, settings) -> int:
    """In-memory Glicko bootstrap over a results CSV (fast — one DB write/team)."""
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                d = r["date"]
                hs, as_ = int(float(r["home_score"])), int(float(r["away_score"]))
            except (KeyError, ValueError):
                continue
            rows.append((d, r["home_team"].strip(), r["away_team"].strip(), hs, as_))
    rows.sort(key=lambda x: x[0])
    # keep the most recent ~6 years of history
    if rows:
        cutoff = str(int(rows[-1][0][:4]) - 6)
        rows = [r for r in rows if r[0][:4] >= cutoff]

    ratings: dict[str, Glicko2Rating] = {}
    default = Glicko2Rating(settings.glicko_default_rating, settings.glicko_default_rd,
                            settings.glicko_default_vol)

    # group consecutive rows by date into rating periods
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

    for name, r in ratings.items():
        upsert_rating(name, r)

    # The results dataset and football-data.org spell some nations differently
    # (e.g. dataset "Czech Republic" vs WC "Czechia", "DR Congo" vs "Congo DR").
    # The engine looks up ratings by the WC fixture's football-data name, so
    # also store each WC team's rating under THAT canonical name — resolved via
    # the same alias table the market matcher uses (exact/alias first, fuzzy
    # fallback). Without this, mis-spelled teams silently fall back to default.
    aliased = _alias_wc_teams(ratings)
    return len(ratings) + aliased


def _alias_wc_teams(ratings: dict[str, "Glicko2Rating"]) -> int:
    """Copy each WC team's history rating to its football-data name."""
    from betbot.exchanges.matcher import TeamAliasResolver, normalize

    fund_csv = Path("data/fundamentals_2026.csv")
    if not fund_csv.exists():
        return 0
    wc_names = [row["team"] for row in csv.DictReader(fund_csv.open())]
    resolver = TeamAliasResolver.from_yaml("config/team_aliases.yaml")
    dataset_names = list(ratings)
    norm_existing = {normalize(n) for n in dataset_names}

    n = 0
    for wc in wc_names:
        if normalize(wc) in norm_existing:
            continue  # the dataset already uses this exact spelling
        match = resolver.match(wc, dataset_names)
        if match is not None:
            upsert_rating(wc, ratings[match])
            log.info("glicko_alias_seeded", wc_team=wc, dataset_team=match,
                     rating=round(ratings[match].rating))
            n += 1
        else:
            log.warning("glicko_alias_unresolved", wc_team=wc)
    return n


async def _seed_wc_teams(settings) -> int:
    from betbot.data.football_data import FootballDataClient

    default = Glicko2Rating(settings.glicko_default_rating, settings.glicko_default_rd,
                            settings.glicko_default_vol)
    n = 0
    async with FootballDataClient(
        api_key=settings.football_data_api_key, base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        try:
            teams = await client.list_competition_teams("WC")
        except Exception as e:  # noqa: BLE001
            log.warning("seed_wc_failed", error=str(e))
            return 0
        for t in teams:
            name = t.get("name")
            if name:
                upsert_rating(name, default, team_id=t.get("id"))
                n += 1
    return n


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)

    csv_path = settings.glicko_results_csv
    if csv_path and Path(csv_path).exists():
        n = _seed_from_csv(Path(csv_path), settings)
        print(f"Path 1: seeded {n} teams from {csv_path}")
    else:
        n = asyncio.run(_seed_wc_teams(settings))
        print(f"Path 2: seeded {n} World-Cup teams at default rating "
              f"({settings.glicko_default_rating}/RD{settings.glicko_default_rd}). "
              "Set BETBOT_GLICKO_RESULTS_CSV for a real history bootstrap.")


if __name__ == "__main__":
    main()
