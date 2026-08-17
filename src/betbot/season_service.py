"""Wire the real club engine + football-data.org fixtures into the season sim.

This is the I/O layer the pure :mod:`betbot.strategy.season_sim` is deliberately
free of. It:

* builds a ``match_prob_fn`` from the LIVE :class:`ClubStrategyEngine` — reusing
  its :meth:`probability_triple` (the exact Glicko + Dixon-Coles log-pool the
  live loop prices with), so the season projection and the per-match alerts
  agree. Ties where either side is unrated fall back to a neutral triple
  (home-advantage-only) and are collected in ``unrated`` so the caller can
  caveat them;
* pulls a competition's full season match list in ONE football-data.org call
  (``/competitions/<CODE>/matches?season=<YEAR>``), splitting FINISHED results
  (seed points) from SCHEDULED/TIMED fixtures (still to play);
* runs the Monte Carlo and shapes the ranked title table;
* caches the result to ``data/season_title_<CODE>.json`` so ``/title`` is
  instant and a weekly daemon job can refresh it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from betbot.data.football_data import FootballDataClient
from betbot.logging import get_logger
from betbot.strategy.club_engine import ClubStrategyEngine
from betbot.strategy.glicko import Glicko2Rating, match_probabilities
from betbot.strategy.season_sim import Fixture, PlayedResult, Triple, simulate_season

log = get_logger(__name__)

# football-data.org competition code -> display name, for the /title header.
LEAGUE_NAMES: dict[str, str] = {
    "PD": "La Liga",
    "PL": "Premier League",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
}

_FINISHED = {"FINISHED", "AWARDED"}
_UPCOMING = {
    "SCHEDULED", "TIMED", "POSTPONED", "SUSPENDED", "IN_PLAY", "PAUSED", "LIVE",
}


@dataclass
class SeasonInputs:
    teams: set[str] = field(default_factory=set)
    played: list[PlayedResult] = field(default_factory=list)
    remaining: list[Fixture] = field(default_factory=list)
    played_count: int = 0
    matchday: int | None = None


def _cache_path(settings, code: str) -> Path:
    base = Path(getattr(settings, "dc_params_club_path", "data/x")).parent
    return base / f"season_title_{code}.json"


def parse_season_matches(matches: list[dict[str, Any]]) -> SeasonInputs:
    """Split a competition match list into played results + remaining fixtures.

    Pure (no engine, no network) so it can be unit-tested off a fixture blob.
    Teams are the union of every side that appears, so a promoted club with no
    fixtures-yet still shows up once it plays.
    """
    out = SeasonInputs()
    max_finished_md = 0
    for m in matches:
        home = ((m.get("homeTeam") or {}).get("name")) or ""
        away = ((m.get("awayTeam") or {}).get("name")) or ""
        if not home or not away:
            continue  # placeholder fixture (e.g. TBD knockout leg)
        out.teams.add(home)
        out.teams.add(away)
        status = (m.get("status") or "").upper()
        if status in _FINISHED:
            ft = (m.get("score") or {}).get("fullTime") or {}
            hg, ag = ft.get("home"), ft.get("away")
            if hg is None or ag is None:
                continue
            out.played.append((home, away, int(hg), int(ag)))
            out.played_count += 1
            md = m.get("matchday")
            if isinstance(md, int):
                max_finished_md = max(max_finished_md, md)
        elif status in _UPCOMING:
            out.remaining.append((home, away))
    out.matchday = max_finished_md or None
    return out


def build_match_prob_fn(
    engine: ClubStrategyEngine, settings
) -> tuple[Callable[[str, str], Triple], set[str]]:
    """A per-fixture pricer over the live club engine + the set of unrated teams.

    Unrated ties (either side has no real rating) fall back to a neutral triple
    priced from default Glicko ratings with only the club home-field boost — the
    same conservative default the engine would degrade to — and the unrated side
    names are collected so the caller can caveat the projection.
    """
    unrated: set[str] = set()
    default = Glicko2Rating(
        settings.glicko_default_rating,
        settings.glicko_default_rd,
        settings.glicko_default_vol,
    )
    neutral_triple = match_probabilities(
        default, default,
        home_field_mu=settings.glicko_club_home_mu,
        draw_rho=settings.glicko_club_draw_rho,
    )

    def match_prob_fn(home: str, away: str) -> Triple:
        if engine.is_rated(home, away):
            triple, _hx, _ax = engine.probability_triple(home, away)
            return triple
        if not engine.is_rated(home, home):
            unrated.add(home)
        if not engine.is_rated(away, away):
            unrated.add(away)
        return neutral_triple

    return match_prob_fn, unrated


async def fetch_season_inputs(
    client: FootballDataClient, code: str, season: int | None
) -> SeasonInputs:
    """ONE football-data.org call for a competition's full season match list."""
    params: dict[str, Any] = {}
    if season is not None:
        params["season"] = season
    data = await client._get(f"/competitions/{code}/matches", params=params)
    matches = [m for m in (data.get("matches") or []) if isinstance(m, dict)]
    return parse_season_matches(matches)


def run_season_sim(
    inputs: SeasonInputs,
    engine: ClubStrategyEngine,
    settings,
    *,
    n_sims: int = 10000,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Run the Monte Carlo and shape a ranked, cache-ready result dict."""
    match_prob_fn, unrated = build_match_prob_fn(engine, settings)
    table = simulate_season(
        teams=inputs.teams,
        played=inputs.played,
        remaining=inputs.remaining,
        match_prob_fn=match_prob_fn,
        n_sims=n_sims,
        seed=seed,
    )
    ranked = sorted(
        (
            {
                "team": t,
                "p_title": round(v["p_title"], 4),
                "p_top4": round(v["p_top4"], 4),
                "p_relegation": round(v["p_relegation"], 4),
                "exp_points": round(v["exp_points"], 1),
            }
            for t, v in table.items()
        ),
        key=lambda r: (-r["p_title"], -r["exp_points"], r["team"]),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matchday": inputs.matchday,
        "played": inputs.played_count,
        "remaining": len(inputs.remaining),
        "n_teams": len(inputs.teams),
        "n_sims": n_sims,
        "unrated": sorted(unrated),
        "table": ranked,
    }


def save_cache(settings, code: str, result: dict[str, Any]) -> Path:
    path = _cache_path(settings, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


def load_cache(settings, code: str) -> dict[str, Any] | None:
    path = _cache_path(settings, code)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
