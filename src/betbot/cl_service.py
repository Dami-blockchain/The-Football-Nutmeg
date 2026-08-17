"""Wire ClubElo + football-data.org into the Champions League winner sim.

This is the I/O layer the pure :mod:`betbot.strategy.tournament_sim` is free of.
Its job is to turn "who wins the Champions League?" into a bracket the Monte
Carlo can roll, and to be honest about how much real information it has.

**Why the CL is special.** The 2024/25+ format is not a points title: 36 teams
play a single 8-game league phase, the top 8 seed straight into the Round of 16,
teams 9-24 contest a two-legged knockout play-off for the other 8 R16 slots,
then a standard R16->QF->SF->Final knockout follows. So the winner needs a
*tournament* sim, not the round-robin ``season_sim``.

**Data reality (v1).** At build time (mid-August) the upcoming season's CL draw
is not published — football-data.org returns 404 for ``season=<next>`` and its
un-parametered CL match list is the *previous, completed* season. So there are
no fixtures to simulate. We therefore ship a **pre-draw seeded-knockout
approximation** as the primary path and clearly flag it (``pre_draw=True``):

* take the CL participants — ``/competitions/CL/teams`` for the season if the
  free tier exposes it, else the top ``field_size`` clubs by ClubElo from
  ``clubelo_latest.csv`` (top-1/2 domestic sides broadly mirror who qualifies);
* seed them by ClubElo (best first) and Monte-Carlo a single-elimination
  bracket to a champion.

If a real draw/fixtures *do* appear (``fetch_cl_inputs`` finds SCHEDULED CL
matches), the caller can pass ``pre_draw=False`` — the same ClubElo pricer still
drives the bracket; only the entrant set / seeding changes. A full league-phase
points sim feeding the knockout is left for v2 (documented deviation).

**Modelling a knockout tie from Elo.** ``_elo_probs`` (reused verbatim from
:mod:`betbot.strategy.cl_engine`) prices a single match as (H, D, A). A CL
knockout tie cannot end level — it goes to extra time / penalties — so we
**renormalise the draw out**: ``P(A advances) = p_home / (p_home + p_away)`` with
A at home. This treats the draw mass as splitting by the same H/A ratio (a
near-50/50 side stays near-50/50, which is a defensible proxy for the coin-flip
of a shoot-out between even teams), and avoids inventing an untested ET model.
For a two-legged tie we take the *neutral*-ish average of A-home and B-home to
remove the single-venue home-field skew (the play-off round is two-legged; R16+
in this projection we also treat as neutral single ties for simplicity — noted).

Unresolved/unrated entrants (no ClubElo rating after alias resolution) are
skipped and logged, so the bracket only ever contains priced teams.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from betbot.data.football_data import FootballDataClient
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.logging import get_logger
from betbot.strategy.cl_engine import _elo_probs, _load_snapshot, _newest_clubelo_dir
from betbot.strategy.tournament_sim import simulate_knockout

log = get_logger(__name__)

# UEFA Champions League league-phase field (2024/25 format).
CL_FIELD_SIZE = 36
CL_CODE = "CL"

_FINISHED = {"FINISHED", "AWARDED"}
_UPCOMING = {"SCHEDULED", "TIMED", "POSTPONED", "SUSPENDED", "IN_PLAY", "PAUSED", "LIVE"}


def _cache_path(settings) -> Path:
    base = Path(getattr(settings, "dc_params_club_path", "data/x")).parent
    return base / "cl_winner.json"


def load_snapshot(settings) -> tuple[dict[str, float], Any]:
    """Load the ClubElo snapshot exactly as EuropeanStrategyEngine does."""
    path = Path(settings.clubelo_latest_path)
    if not path.exists():
        alt = _newest_clubelo_dir(Path("data/clubelo"))
        if alt is not None:
            path = alt
    return _load_snapshot(path)


def build_advance_prob_fn(
    snapshot: dict[str, float],
    resolver: TeamAliasResolver,
    settings,
) -> tuple[Callable[[str, str], float], Callable[[str], str | None]]:
    """P(a advances past b) from ClubElo, reusing cl_engine._elo_probs.

    The draw is renormalised out (knockouts can't end level); for a two-legged /
    venue-neutral tie we average the a-home and b-home renormalised win probs so
    no side gets a spurious single-venue home boost.

    Returns ``(advance_prob_fn, resolve)`` where ``resolve(name)`` is the
    memoised ClubElo-name resolver (used by the caller to filter entrants).
    """
    clubs = list(snapshot.keys())
    res_cache: dict[str, str | None] = {}

    def resolve(name: str) -> str | None:
        if name in res_cache:
            return res_cache[name]
        hit = resolver.match(name, clubs) if clubs else None
        res_cache[name] = hit
        return hit

    def _p_home_only(elo_a: float, elo_b: float) -> float:
        # renormalise the draw out of a single a-home match
        p_h, _p_d, p_a = _elo_probs(
            elo_a, elo_b, settings.cl_elo_home_adv, settings.cl_elo_draw_rho
        )
        denom = p_h + p_a
        return p_h / denom if denom > 0 else 0.5

    def advance_prob_fn(a: str, b: str) -> float:
        ha, hb = resolve(a), resolve(b)
        if ha is None or hb is None:
            return 0.5  # should not happen: caller pre-filters to resolved teams
        ea, eb = snapshot[ha], snapshot[hb]
        # venue-neutral two-way average (a at home vs b at home)
        p_a_home = _p_home_only(ea, eb)
        p_b_home = _p_home_only(eb, ea)
        return 0.5 * (p_a_home + (1.0 - p_b_home))

    return advance_prob_fn, resolve


async def fetch_cl_inputs(
    client: FootballDataClient, season: int | None
) -> dict[str, Any]:
    """Probe football-data.org for a live CL draw/fixtures.

    Returns ``{"entrants": [...], "has_upcoming": bool, "source": str}``.
    ``entrants`` are club names (football-data spellings) drawn from the season's
    team list if available, else from any upcoming fixtures. Empty / no-upcoming
    means the draw isn't out yet -> caller falls back to ClubElo seeding.
    """
    out: dict[str, Any] = {"entrants": [], "has_upcoming": False, "source": "none"}
    params: dict[str, Any] = {}
    if season is not None:
        params["season"] = season

    # 1) team list for the season (best entrant source when the draw is set)
    try:
        data = await client._get(f"/competitions/{CL_CODE}/teams", params=params)
        teams = [t.get("name") for t in (data.get("teams") or []) if t.get("name")]
        if teams:
            out["entrants"] = teams
            out["source"] = "football_data_teams"
    except Exception as e:  # noqa: BLE001 — pre-draw 404 is expected
        log.info("cl_teams_unavailable", season=season, error=str(e))

    # 2) matches — do any UPCOMING (this-season) fixtures exist?
    try:
        data = await client._get(f"/competitions/{CL_CODE}/matches", params=params)
        matches = [m for m in (data.get("matches") or []) if isinstance(m, dict)]
        upcoming = [m for m in matches if (m.get("status") or "").upper() in _UPCOMING]
        if upcoming:
            out["has_upcoming"] = True
            names: set[str] = set()
            for m in upcoming:
                for side in ("homeTeam", "awayTeam"):
                    nm = (m.get(side) or {}).get("name")
                    if nm:
                        names.add(nm)
            if not out["entrants"]:
                out["entrants"] = sorted(names)
                out["source"] = "football_data_fixtures"
    except Exception as e:  # noqa: BLE001
        log.info("cl_matches_unavailable", season=season, error=str(e))

    return out


def _seed_from_clubelo(
    snapshot: dict[str, float], resolve, entrants: list[str], field_size: int
) -> tuple[list[str], list[str]]:
    """Resolve + sort entrants best-first by ClubElo; split into (seeded, unrated).

    With no entrant list (pure pre-draw), fall back to the top ``field_size``
    ClubElo clubs directly.
    """
    if entrants:
        rated: list[tuple[str, float]] = []
        unrated: list[str] = []
        for name in entrants:
            hit = resolve(name)
            if hit is None:
                unrated.append(name)
            else:
                rated.append((name, snapshot[hit]))
        rated.sort(key=lambda kv: -kv[1])
        return [n for n, _ in rated], unrated

    # Pure pre-draw: top clubs by ClubElo become the projected field.
    top = sorted(snapshot.items(), key=lambda kv: -kv[1])[:field_size]
    return [club for club, _ in top], []


def run_cl_sim(
    settings,
    fetched: dict[str, Any],
    *,
    n_sims: int = 10000,
    seed: int = 20260817,
    field_size: int = CL_FIELD_SIZE,
    resolver: TeamAliasResolver | None = None,
) -> dict[str, Any]:
    """Build the bracket + Monte-Carlo it to a champion. Cache-ready dict."""
    snapshot, snap_date = load_snapshot(settings)
    resolver = resolver or TeamAliasResolver.from_yaml("config/team_aliases.yaml")
    advance_prob_fn, resolve = build_advance_prob_fn(snapshot, resolver, settings)

    entrants_in = list(fetched.get("entrants") or [])
    pre_draw = not fetched.get("has_upcoming", False)

    seeded, unrated = _seed_from_clubelo(snapshot, resolve, entrants_in, field_size)
    if unrated:
        log.info("cl_unrated_entrants", count=len(unrated), teams=unrated[:20])

    if not seeded:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pre_draw": pre_draw,
            "source": fetched.get("source", "none"),
            "n_entrants": 0,
            "n_unrated": len(unrated),
            "n_sims": n_sims,
            "snapshot_date": str(snap_date) if snap_date else None,
            "table": [],
        }

    p_win = simulate_knockout(
        entrants=seeded, advance_prob_fn=advance_prob_fn, n_sims=n_sims, seed=seed
    )
    ranked = sorted(
        ({"team": t, "p_win": round(p, 4)} for t, p in p_win.items()),
        key=lambda r: (-r["p_win"], r["team"]),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pre_draw": pre_draw,
        "source": fetched.get("source", "clubelo_topN" if pre_draw else "none"),
        "n_entrants": len(seeded),
        "n_unrated": len(unrated),
        "unrated": unrated[:40],
        "n_sims": n_sims,
        "snapshot_date": str(snap_date) if snap_date else None,
        "table": ranked,
    }


def save_cache(settings, result: dict[str, Any]) -> Path:
    path = _cache_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


def load_cache(settings) -> dict[str, Any] | None:
    path = _cache_path(settings)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
