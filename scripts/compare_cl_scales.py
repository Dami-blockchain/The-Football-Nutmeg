"""Head-to-head: SITE-scale CL engine vs the api.clubelo incumbent (NON-INFERIORITY).

This is the gate for the 2026-09 switch of the CL engine from the ageing
api.clubelo pin to the fresh site-scale scrape. It is framed as NON-INFERIORITY,
NOT superiority — see the WHY below and the note in scripts/backtest_cl.py.

WHY NON-INFERIORITY (not "does site beat api, CI>0")
----------------------------------------------------
The per-match RPS diff between two ClubElo scalings has SD ~= 0.035, so on the
~190 held-out test matches a season yields, the SE is ~0.0026/match and the
minimum detectable effect at 80% power is ~0.0073/match. The staleness effect at
stake is ~0.002-0.004/match. A superiority gate on this corpus therefore cannot
reach significance on a realistic effect: "CI spans zero" is an underpowered
test, not evidence of no difference. So we instead ask: is the fresh site feed
NOT WORSE than the frozen api pin by more than the staleness cost (margin ~0.004
RPS/match)? Production's real alternative is the FROZEN pin (api.clubelo stopped
refreshing ~2026-08-31 and now ages), which is why the frozen-pin comparison is
the one that matters.

The SITE config here is the MEASURED rescale of the incumbent (scale 312, home
51, rho 0.26 = the api constants times the 0.78 spread ratio), NOT a train-tuned
value — tuning on n=314 is unstable.

Run (repo root, venv active):
    python scripts/compare_cl_scales.py \
        --api-dir data/clubelo --site-dir data/clubelo_site \
        --cl-csv data/cl_results.csv --pin-date 2025-05-01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from backtest_club import OUT_IDX, _load, _outcome  # noqa: E402
from backtest_cl import _elo_probs  # noqa: E402
from betbot.exchanges.matcher import TeamAliasResolver, normalize  # noqa: E402
from betbot.strategy import dixon_coles as dc  # noqa: E402
from betbot.strategy.ensemble import log_pool, ranked_probability_score  # noqa: E402

# Shipped configs: (home_adv, draw_rho, scale). Site = measured rescale (x0.78).
API_HA, API_RHO, API_SCALE = 65.0, 0.26, 400.0
SITE_HA, SITE_RHO, SITE_SCALE = 51.0, 0.26, 312.0
NI_MARGIN = 0.004  # non-inferiority margin (RPS/match) = the staleness cost


def _month_first(d: date) -> date:
    return date(d.year, d.month, 1)


def _snapshot_loader(directory: Path, resolver: TeamAliasResolver):
    cache: dict[date, dict[str, float]] = {}

    def load(m: date) -> dict[str, float]:
        if m in cache:
            return cache[m]
        p = directory / f"{m.isoformat()}.csv"
        d: dict[str, float] = {}
        if p.exists():
            for r in csv.DictReader(p.open()):
                try:
                    d[(r.get("Club") or "").strip()] = float(r["Elo"])
                except (KeyError, ValueError, TypeError):
                    pass
        cache[m] = d
        return d

    rc: dict[tuple[str, date], str | None] = {}

    def look(name: str, snap_month: date) -> float | None:
        snap = load(snap_month)
        if not snap:
            return None
        ck = (name, snap_month)
        if ck not in rc:
            rc[ck] = resolver.match(name, list(snap.keys()))
        hit = rc[ck]
        return snap.get(hit) if hit else None

    return look


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _boot_ci(diffs: list[float], b: int = 5000, seed: int = 12345):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b)
    )
    return means[int(0.025 * b)], means[int(0.975 * b)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-dir", type=Path, default=Path("data/clubelo"))
    ap.add_argument("--site-dir", type=Path, default=Path("data/clubelo_site"))
    ap.add_argument("--cl-csv", type=Path, default=Path("data/cl_results.csv"))
    ap.add_argument("--dc-params", type=Path, default=Path("data/dc_params_club.json"))
    ap.add_argument("--name-map", type=Path, default=Path("data/club_name_map.json"))
    ap.add_argument("--aliases", default="config/team_aliases.yaml")
    ap.add_argument("--test-from", default="2025-07-01")
    ap.add_argument("--pin-date", default="2025-05-01",
                    help="freeze the api snapshot at this date (the ageing pin)")
    args = ap.parse_args()

    cut = date.fromisoformat(args.test_from)
    pin_month = _month_first(date.fromisoformat(args.pin_date))
    rows = _load(args.cl_csv)
    test = [r for r in rows if r["date"] >= cut]
    resolver = TeamAliasResolver.from_yaml(args.aliases)

    dc_params = dc.DCParams.from_json(args.dc_params.read_text(encoding="utf-8"))
    name_map = {str(k): str(v) for k, v in
                json.loads(args.name_map.read_text()).items()}

    def dc_key(n: str) -> str:
        x = normalize(n)
        return name_map.get(x, x)

    def dc_probs(r):
        kh, ka = dc_key(r["home"]), dc_key(r["away"])
        if kh in dc_params.teams and ka in dc_params.teams:
            return dc.match_probabilities(dc_params, kh, ka, home_field=True)
        return None

    look_api = _snapshot_loader(args.api_dir, resolver)
    look_site = _snapshot_loader(args.site_dir, resolver)

    def triple(look, snap_month, r, ha, rho, scale):
        eh = look(r["home"], snap_month)
        ea = look(r["away"], snap_month)
        if eh is None or ea is None:
            return None
        ep = _elo_probs(eh, ea, ha, rho, scale)
        dp = dc_probs(r)
        return log_pool([(1.0, ep), (1.0, dp)]) if dp else ep

    # configs: label -> (loader, snap_month_fn, ha, rho, scale)
    configs = {
        "api_fresh":  (look_api,  lambda r: _month_first(r["date"]), API_HA, API_RHO, API_SCALE),
        "api_frozen": (look_api,  lambda r: pin_month,               API_HA, API_RHO, API_SCALE),
        "site_fresh": (look_site, lambda r: _month_first(r["date"]), SITE_HA, SITE_RHO, SITE_SCALE),
    }

    def probs(label, r):
        look, mf, ha, rho, scale = configs[label]
        return triple(look, mf(r), r, ha, rho, scale)

    # common scorable set: every config scores it (excludes pageless Kairat etc.)
    common = [r for r in test if all(probs(k, r) is not None for k in configs)]
    print(f"test matches {len(test)}; common scorable across all configs {len(common)}\n")

    stats = {k: {"n": 0, "hit": 0, "rps": 0.0, "ll": 0.0, "hn": 0, "hh": 0}
             for k in configs}
    per = {k: [] for k in configs}
    for r in common:
        oi = OUT_IDX[_outcome(r["hs"], r["as"])]
        for k in configs:
            p = probs(k, r)
            st = stats[k]
            st["n"] += 1
            pick = max(range(3), key=lambda i: p[i])
            st["hit"] += int(pick == oi)
            st["rps"] += ranked_probability_score(p, oi)
            st["ll"] += -math.log(max(p[oi], 1e-9))
            if p[pick] >= 0.65:
                st["hn"] += 1
                st["hh"] += int(pick == oi)
            per[k].append(ranked_probability_score(p, oi))

    print(f"{'config':11s}{'n':>5s}{'acc%':>8s}{'RPS':>9s}{'logloss':>9s}"
          f"   p>=0.65: n hit% Wilson95")
    for k in configs:
        st = stats[k]
        n = max(st["n"], 1)
        lo, hi = _wilson(st["hh"], st["hn"])
        print(f"{k:11s}{st['n']:>5d}{100*st['hit']/n:>8.2f}{st['rps']/n:>9.4f}"
              f"{st['ll']/n:>9.4f}   {st['hn']:>3d} "
              f"{100*st['hh']/max(st['hn'],1):>5.1f} "
              f"[{100*lo:.1f},{100*hi:.1f}]")

    print(f"\nNON-INFERIORITY of site_fresh vs each api baseline "
          f"(margin {NI_MARGIN:+.4f} RPS/match):")
    for base in ("api_fresh", "api_frozen"):
        diffs = [per[base][i] - per["site_fresh"][i] for i in range(len(common))]
        mean = sum(diffs) / len(diffs)
        lo, hi = _boot_ci(diffs)
        # site is non-inferior if it is not worse than base by > margin, i.e.
        # (base_rps - site_rps) lower CI bound > -margin.
        ni = lo > -NI_MARGIN
        better = lo > 0
        verdict = ("SUPERIOR (CI>0)" if better
                   else "NON-INFERIOR" if ni
                   else "FAILS non-inferiority")
        print(f"  {base:11s}: (base-site) mean {mean:+.5f}/match  "
              f"CI[{lo:+.5f},{hi:+.5f}]  -> {verdict}")


if __name__ == "__main__":
    main()
