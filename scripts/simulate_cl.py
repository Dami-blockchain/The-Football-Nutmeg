"""Champions League winner Monte Carlo CLI — "who will win the CL?".

Probes football-data.org for a live CL draw; if none exists yet (pre-season,
draw not published) it falls back to a ClubElo-seeded 36-team bracket. Either
way it prices every knockout tie off the same ClubElo Elo model the live CL
engine uses, Monte-Carlos the bracket ``--sims`` times, prints a ranked
``Team  P(win CL)`` table, and caches to ``data/cl_winner.json`` so ``/title CL``
is instant.

Run (repo root, venv active):
    python scripts/simulate_cl.py --sims 10000
    python scripts/simulate_cl.py --season 2026 --sims 20000
"""

from __future__ import annotations

import argparse
import asyncio

from betbot.cl_service import fetch_cl_inputs, run_cl_sim, save_cache
from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient
from betbot.logging import configure_logging, get_logger

log = get_logger("simulate_cl")


async def _run(season: int | None, sims: int, seed: int) -> None:
    settings = get_settings()

    fetched: dict = {"entrants": [], "has_upcoming": False, "source": "none"}
    try:
        async with FootballDataClient(
            api_key=settings.football_data_api_key,
            base_url=settings.football_data_base_url,
            rate_limit_per_min=settings.football_data_rate_limit_per_min,
        ) as client:
            fetched = await fetch_cl_inputs(client, season)
    except Exception as e:  # noqa: BLE001 — network/tier issues must not abort
        log.warning("cl_fetch_failed", error=str(e))

    result = run_cl_sim(settings, fetched, n_sims=sims, seed=seed)
    path = save_cache(settings, result)

    mode = "PRE-DRAW estimate (ClubElo-seeded)" if result["pre_draw"] else "live bracket"
    print(f"\nChampions League winner projection — {mode}  [{sims} sims]")
    print(f"  entrants resolved: {result['n_entrants']}  |  unrated/skipped: "
          f"{result['n_unrated']}  |  source: {result['source']}  |  "
          f"ClubElo snapshot: {result['snapshot_date']}")
    if result.get("unrated"):
        print(f"  unrated (skipped): {', '.join(result['unrated'])}")
    print(f"\n  {'Team':<28} {'P(win CL)':>10}")
    print("  " + "-" * 40)
    for row in result["table"][:15]:
        print(f"  {row['team']:<28} {row['p_win']*100:>9.1f}%")
    print(f"\n  cached -> {path}\n")


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description="Champions League winner Monte Carlo.")
    ap.add_argument("--season", type=int, default=None,
                    help="season start year (default: current per football-data)")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()
    asyncio.run(_run(args.season, args.sims, args.seed))


if __name__ == "__main__":
    main()
