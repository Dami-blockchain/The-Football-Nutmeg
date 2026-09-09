"""Walk-forward Champions League backtest — the R2 Elo gate.

Prices each CL match from a point-in-time ClubElo snapshot (the snapshot dated
the FIRST OF THE MATCH'S MONTH — no lookahead), tunes home-advantage and the
draw-model rho on seasons 2023+2024, optionally blends in the club Dixon-Coles
goal model, then scores ONCE on the held-out 2025 season against the naive
form engine baseline. The cross-league Elo engine ships only if the bootstrap
95% CI on the per-match RPS improvement (naive - elo) has a lower bound > 0.

ClubElo is a single cross-league rating scale (unlike the per-league-seeded
Glicko club ratings), which is exactly why it can price CL matches that mix
teams from different domestic leagues — the whole point of R2.

Team-name bridging: fd.org names ("Manchester City FC") -> ClubElo names
("Man City") via normalize() + TeamAliasResolver against each snapshot's club
list. Unresolved teams are counted and skipped from Elo scoring.

Run (repo root, venv active):
    python scripts/backtest_cl.py
    python scripts/backtest_cl.py --cl-csv data/cl_results.csv --test-from 2025-07-01

--------------------------------------------------------------------------------
GATE FRAMING — READ BEFORE JUDGING A SITE-vs-API COMPARISON AS "no improvement"
--------------------------------------------------------------------------------
The naive-vs-Elo gate above (CI on naive - elo > 0) is a SUPERIORITY test and
is fine for that big effect (~0.03 RPS/match). It is the WRONG test for the
site-scale-vs-api-scale question, which is a NON-INFERIORITY question.

Why: the per-match RPS diff between two ClubElo scalings has SD ~= 0.035, so on
~190 held-out test matches/season the standard error is ~0.0026/match and the
minimum effect detectable at 80% power is ~0.0073/match. The staleness effect
actually at stake is ~0.002-0.004/match. A superiority gate on this corpus
therefore CANNOT reach significance on a realistic effect — "CI spans zero" is
no evidence of no difference, only an underpowered test. (You would need
n ~= 600-2400 test matches; one season yields ~190.)

So the site switch is gated as NON-INFERIORITY against the FROZEN api pin (the
real production alternative — api.clubelo stopped refreshing, so the pin ages):
the switch ships if the site feed is not worse than the frozen pin by more than
the measured staleness cost (~0.004 RPS/match). scripts/compare_cl_scales.py
runs that comparison (paired diff + the p>=0.65 subset). The engine's SHIPPED
site constants are the MEASURED rescale of the incumbent (scale 400*0.78=312,
home_adv 65*0.78=51, rho unchanged) — NOT the SCALE_GRID's train-tuned value,
which is unstable on n=314 and only used here for the sensitivity check.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_club import (  # noqa: E402
    Form,
    OUT_IDX,
    _bootstrap_ci,
    _fixture_form,
    _load,
    _outcome,
)

from betbot.config import get_settings  # noqa: E402
from betbot.exchanges.matcher import TeamAliasResolver, normalize  # noqa: E402
from betbot.strategy import dixon_coles as dc  # noqa: E402
from betbot.strategy.engine import StrategyEngine  # noqa: E402
from betbot.strategy.ensemble import log_pool, ranked_probability_score  # noqa: E402
from betbot.strategy.glicko import DRAW_CAP, DRAW_FLOOR  # noqa: E402

HA_GRID = [0.0, 25.0, 50.0, 65.0, 80.0, 100.0]
RHO_GRID = [0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]
# Logistic Elo divisor. 400 = classic api.clubelo scale. Smaller divisors
# sharpen the same gap — needed for compressed site-scale snapshots. Tuned on
# TRAIN only, alongside home-advantage and draw-rho. Grid spans 200-450 (step
# 25) so the site-scale interior optimum (~275) is not a boundary selection.
# CAUTION: train-tuning this on n=314 is unstable and inflates the served call
# count; the SHIPPED site config does NOT use the tuned value — it uses the
# measured rescale of the incumbent (400*0.78=312). See the NON-INFERIORITY note
# in the module docstring.
SCALE_GRID = [float(x) for x in range(200, 451, 25)]
DC_WEIGHT_GRID = [0.3, 0.6, 1.0]
TRAIN_CUTOFF = date(2025, 7, 1)


def _elo_probs(elo_home: float, elo_away: float, ha: float, rho: float,
               scale: float = 400.0):
    """Elo 1X2 probabilities; draw split mirrors glicko.match_probabilities.

    ``scale`` is the logistic divisor (classic Elo = 400). Compressed snapshot
    scales (e.g. ClubElo site-scale) need a smaller divisor to recover sharpness.
    Mirrors betbot.strategy.cl_engine._elo_probs / cl_elo_scale.
    """
    d = elo_home + ha - elo_away
    p_home_raw = 1.0 / (1.0 + 10.0 ** (-d / scale))
    p_draw = rho * (1.0 - abs(p_home_raw - 0.5) * 2.0)
    p_draw = min(DRAW_CAP, max(DRAW_FLOOR, p_draw))
    p_home = (1.0 - p_draw) * p_home_raw
    p_away = (1.0 - p_draw) * (1.0 - p_home_raw)
    return p_home, p_draw, p_away


def _load_snapshot(month_first: date, cache: dict[date, dict[str, float]],
                   clubelo_dir: Path) -> dict[str, float] | None:
    """ClubElo snapshot for the first-of-month, {club_name: elo}. Cached."""
    if month_first in cache:
        return cache[month_first]
    path = clubelo_dir / f"{month_first.isoformat()}.csv"
    if not path.exists():
        cache[month_first] = {}
        return {}
    snap: dict[str, float] = {}
    for row in csv.DictReader(path.open()):
        club = (row.get("Club") or "").strip()
        try:
            elo = float(row["Elo"])
        except (KeyError, ValueError, TypeError):
            continue
        if club:
            snap[club] = elo
    cache[month_first] = snap
    return snap


def _resolve(name: str, snap: dict[str, float], resolver: TeamAliasResolver,
             cache: dict[tuple[str, date], str | None], key_month: date,
             clubs: list[str]) -> str | None:
    ck = (name, key_month)
    if ck in cache:
        return cache[ck]
    hit = resolver.match(name, clubs)
    cache[ck] = hit
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cl-csv", type=Path, default=Path("data/cl_results.csv"))
    ap.add_argument("--clubelo-dir", type=Path, default=Path("data/clubelo"))
    ap.add_argument("--dc-params", type=Path, default=Path("data/dc_params_club.json"))
    ap.add_argument("--name-map", type=Path, default=Path("data/club_name_map.json"))
    ap.add_argument("--aliases", default="config/team_aliases.yaml")
    ap.add_argument("--test-from", default="2025-07-01")
    args = ap.parse_args()

    s = get_settings()
    cutoff = date.fromisoformat(args.test_from)
    rows = _load(args.cl_csv)
    resolver = TeamAliasResolver.from_yaml(args.aliases)

    # DC params + fd->dataset name map (same bridging club_engine uses).
    dc_params = dc.DCParams.from_json(args.dc_params.read_text(encoding="utf-8"))
    name_map: dict[str, str] = {}
    try:
        import json
        name_map = {str(k): str(v)
                    for k, v in json.loads(args.name_map.read_text()).items()}
    except (OSError, ValueError):
        pass

    def dc_key(name: str) -> str:
        n = normalize(name)
        return name_map.get(n, n)

    snap_cache: dict[date, dict[str, float]] = {}
    res_cache: dict[tuple[str, date], str | None] = {}
    clubs_cache: dict[date, list[str]] = {}

    def month_first(d: date) -> date:
        return date(d.year, d.month, 1)

    def elo_lookup(name: str, d: date) -> float | None:
        mf = month_first(d)
        snap = _load_snapshot(mf, snap_cache, args.clubelo_dir)
        if not snap:
            return None
        clubs = clubs_cache.get(mf)
        if clubs is None:
            clubs = list(snap.keys())
            clubs_cache[mf] = clubs
        hit = _resolve(name, snap, resolver, res_cache, mf, clubs)
        if hit is None:
            return None
        return snap[hit]

    unresolved: set[str] = set()
    unresolved_matches = 0

    def probe_resolution(rows_subset):
        nonlocal unresolved_matches
        n = 0
        for r in rows_subset:
            eh = elo_lookup(r["home"], r["date"])
            ea = elo_lookup(r["away"], r["date"])
            if eh is None:
                unresolved.add(r["home"])
            if ea is None:
                unresolved.add(r["away"])
            if eh is None or ea is None:
                unresolved_matches += 1
            else:
                n += 1
        return n

    train = [r for r in rows if r["date"] < cutoff]
    test = [r for r in rows if r["date"] >= cutoff]
    print(f"CL matches: {len(rows)} total, train {len(train)} (<{cutoff}), "
          f"test {len(test)} (>= {cutoff})")

    # ---- Tune HA + rho on train (Elo-only) -----------------------------
    train_scored = [
        r for r in train
        if elo_lookup(r["home"], r["date"]) is not None
        and elo_lookup(r["away"], r["date"]) is not None
    ]
    print(f"train matches with both Elo ratings: {len(train_scored)}/{len(train)}")

    best = None  # (mean_rps, ha, rho, scale)
    for ha in HA_GRID:
        for rho in RHO_GRID:
            for scale in SCALE_GRID:
                tot = 0.0
                for r in train_scored:
                    eh = elo_lookup(r["home"], r["date"])
                    ea = elo_lookup(r["away"], r["date"])
                    oi = OUT_IDX[_outcome(r["hs"], r["as"])]
                    tot += ranked_probability_score(
                        _elo_probs(eh, ea, ha, rho, scale), oi)
                mrps = tot / max(len(train_scored), 1)
                if best is None or mrps < best[0]:
                    best = (mrps, ha, rho, scale)
    best_rps, ha_star, rho_star, scale_star = best
    print(f"tuned Elo: HA={ha_star:.0f} rho={rho_star:.2f} scale={scale_star:.0f} "
          f"train mean RPS={best_rps:.5f}")

    # ---- Tune Elo+DC blend weight on train -----------------------------
    def dc_probs_or_none(r):
        kh, ka = dc_key(r["home"]), dc_key(r["away"])
        if kh in dc_params.teams and ka in dc_params.teams:
            return dc.match_probabilities(dc_params, kh, ka, home_field=True)
        return None

    best_blend = None  # (mean_rps, dc_w)
    for dc_w in DC_WEIGHT_GRID:
        tot = 0.0
        for r in train_scored:
            eh = elo_lookup(r["home"], r["date"])
            ea = elo_lookup(r["away"], r["date"])
            oi = OUT_IDX[_outcome(r["hs"], r["as"])]
            elo_p = _elo_probs(eh, ea, ha_star, rho_star, scale_star)
            dcp = dc_probs_or_none(r)
            if dcp is not None:
                probs = log_pool([(1.0, elo_p), (dc_w, dcp)])
            else:
                probs = elo_p
            tot += ranked_probability_score(probs, oi)
        mrps = tot / max(len(train_scored), 1)
        if best_blend is None or mrps < best_blend[0]:
            best_blend = (mrps, dc_w)
    blend_rps, dc_w_star = best_blend
    blend_beats_elo = blend_rps < best_rps
    print(f"tuned Elo+DC blend: dc_weight={dc_w_star:.1f} "
          f"train mean RPS={blend_rps:.5f} "
          f"(beats pure Elo on train: {blend_beats_elo})")

    # ---- Warm naive form engine on ALL earlier CL matches --------------
    naive = StrategyEngine(s)
    form = Form()
    train_by_date: dict[date, list] = defaultdict(list)
    for r in train:
        train_by_date[r["date"]].append(r)
    for d in sorted(train_by_date):
        for r in train_by_date[d]:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))

    # ---- FINAL GATE — score once on the held-out test season ----------
    variants = ["naive", "elo"]
    if blend_beats_elo:
        variants.append("elo_dc")
    stats = {m: {"n": 0, "hit": 0, "rps": 0.0, "ll": 0.0} for m in variants}
    # per-match RPS diffs vs naive, only where the Elo variant is scorable.
    diffs = {m: [] for m in variants if m != "naive"}

    test_by_date: dict[date, list] = defaultdict(list)
    for r in test:
        test_by_date[r["date"]].append(r)

    for d in sorted(test_by_date):
        day = test_by_date[d]
        for r in day:
            home, away = r["home"], r["away"]
            oi = OUT_IDX[_outcome(r["hs"], r["as"])]

            # naive baseline is always scorable (form-warmed, walking forward).
            ff = _fixture_form(home, away, form)
            fp = naive.predict(ff)
            naive_p = (fp.p_home, fp.p_draw, fp.p_away)

            eh = elo_lookup(home, d)
            ea = elo_lookup(away, d)
            if eh is None:
                unresolved.add(home)
            if ea is None:
                unresolved.add(away)
            scorable = eh is not None and ea is not None
            if not scorable:
                continue  # skip from Elo scoring; counted below via probe

            elo_p = _elo_probs(eh, ea, ha_star, rho_star, scale_star)
            probs_by = {"naive": naive_p, "elo": elo_p}
            if "elo_dc" in variants:
                dcp = dc_probs_or_none(r)
                probs_by["elo_dc"] = (
                    log_pool([(1.0, elo_p), (dc_w_star, dcp)]) if dcp is not None
                    else elo_p
                )

            n_rps = ranked_probability_score(naive_p, oi)
            for m in variants:
                p = probs_by[m]
                st = stats[m]
                st["n"] += 1
                st["hit"] += int(max(range(3), key=lambda i: p[i]) == oi)
                st["rps"] += ranked_probability_score(p, oi)
                st["ll"] += -math.log(max(p[oi], 1e-9))
                if m != "naive":
                    diffs[m].append(n_rps - ranked_probability_score(p, oi))

        for r in day:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))

    # ---- Resolution report (train + test) ------------------------------
    probe_resolution(rows)
    print(f"\nname resolution: {len(unresolved)} distinct teams unresolved, "
          f"{unresolved_matches} matches unscorable by Elo")
    if unresolved:
        print("  unresolved teams: " + ", ".join(sorted(unresolved)))

    # ---- Report --------------------------------------------------------
    print("\n=== held-out test season results (>= 2025-07-01) ===")
    print(f"{'model':8s} {'n':>5s} {'acc%':>7s} {'meanRPS':>9s} {'logloss':>9s}")
    for m in variants:
        st = stats[m]
        n = max(st["n"], 1)
        print(f"{m:8s} {st['n']:>5d} {100 * st['hit'] / n:>7.2f} "
              f"{st['rps'] / n:>9.4f} {st['ll'] / n:>9.4f}")

    print("\nbootstrap 95% CI on per-match RPS improvement (naive - variant):")
    gate_pass = False
    for m in diffs:
        dl = diffs[m]
        if not dl:
            continue
        impr = sum(dl) / len(dl)
        lo, hi = _bootstrap_ci(dl)
        sig = lo > 0
        if m == "elo":
            gate_pass = sig
        print(f"  {m:8s} improvement {impr:+.5f}/match  "
              f"CI [{lo:+.5f}, {hi:+.5f}]  "
              f"({'CI>0' if sig else 'includes 0'})")

    print(f"\ntuned params: HA={ha_star:.0f}, rho={rho_star:.2f}, "
          f"scale={scale_star:.0f}, dc_weight={dc_w_star:.1f} "
          f"(blend shipped: {blend_beats_elo})")
    print(f"\nFINAL VERDICT: {'GATE PASSED' if gate_pass else 'GATE FAILED'}")


if __name__ == "__main__":
    main()
