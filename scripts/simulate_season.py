"""Season-title Monte Carlo CLI — "who will win <league>?".

Pulls a competition's full season match list from football-data.org (ONE API
call), prices every remaining fixture off the LIVE club ensemble, rolls the
season ``--sims`` times, prints a ranked title table, and caches the result to
``data/season_title_<CODE>.json`` so ``/title`` is instant.

Run (repo root, venv active):
    python scripts/simulate_season.py --league PD
    python scripts/simulate_season.py --league PL --season 2025 --sims 20000
"""

from __future__ import annotations

import argparse
import asyncio

from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient
from betbot.logging import configure_logging, get_logger
from betbot.season_service import (
    LEAGUE_NAMES,
    fetch_season_inputs,
    run_season_sim,
    save_cache,
)
from betbot.storage.db import init_engine
from betbot.strategy.club_engine import ClubStrategyEngine

log = get_logger("simulate_season")


async def _run(code: str, season: int | None, sims: int, seed: int) -> None:
    settings = get_settings()
    init_engine(settings.db_path)  # get_rating reads the ratings DB
    engine = ClubStrategyEngine(settings)

    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        inputs = await fetch_season_inputs(client, code, season)

    if not inputs.remaining and not inputs.played:
        log.warning("no_matches", league=code, season=season)
        print(f"No matches found for {code} (season={season}).")
        return

    result = run_season_sim(inputs, engine, settings, n_sims=sims, seed=seed)
    path = save_cache(settings, code, result)

    name = LEAGUE_NAMES.get(code, code)
    md = result["matchday"]
    print(f"\n{name} ({code}) title race — {result['n_teams']} teams, "
          f"{result['played']} played / {result['remaining']} remaining"
          + (f", after matchday {md}" if md else "")
          + f"  [{sims} sims]")
    if result["unrated"]:
        print(f"  unrated (neutral fallback): {', '.join(result['unrated'])}")
    print(f"\n  {'Team':<28} {'P(title)':>9} {'P(top4)':>9} {'exp_pts':>8}")
    print("  " + "-" * 56)
    for row in result["table"][:12]:
        print(f"  {row['team']:<28} {row['p_title']*100:>8.1f}% "
              f"{row['p_top4']*100:>8.1f}% {row['exp_points']:>8.1f}")
    print(f"\n  cached -> {path}\n")


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description="Season-title Monte Carlo.")
    ap.add_argument("--league", default="PD",
                    help="football-data.org competition code (PD/PL/BL1/SA/FL1)")
    ap.add_argument("--season", type=int, default=None,
                    help="season start year (default: current season)")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()
    asyncio.run(_run(args.league.upper(), args.season, args.sims, args.seed))


if __name__ == "__main__":
    main()
