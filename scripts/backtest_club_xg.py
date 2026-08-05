"""R3 gate: does TRUE xG beat goals in the club goal model?

Self-contained on data/club_xg.csv (Understat: real match xG + goals + result
for the SAME fixtures, so goals-vs-xG is apples-to-apples with no name-bridge).

Walk-forward, held-out 2024/25 season (train < 2024-07-01, test 2024-07-01 ..
2025-07-01 — the partial 2025/26 backfill is excluded). Everything shares one
goals-Glicko rating core + a recent-form component; only the Dixon-Coles input
changes, isolating xG's effect:

    baseline : Glicko + goals-DC + form   (== the current club ensemble)
    xG       : Glicko + xG-DC    + form
    blend    : Glicko + goals-DC + xG-DC + form
    forecast : Understat's own model (reference yardstick, not ours)

Metrics: accuracy, mean RPS, log-loss + bootstrap 95% CI on the per-match RPS
improvement (baseline - variant). Gate passes iff CI lower bound > 0. Also
prints predicted expected-goals (DC lambda) for a few marquee ties.

Run: python scripts/backtest_club_xg.py
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
    OUT_IDX, Form, Glicko, _bootstrap_ci, _fixture_form, _outcome,
)

from betbot.config import get_settings  # noqa: E402
from betbot.exchanges.matcher import normalize  # noqa: E402
from betbot.strategy import dixon_coles as dc  # noqa: E402
from betbot.strategy.engine import StrategyEngine  # noqa: E402
from betbot.strategy.ensemble import log_pool, ranked_probability_score  # noqa: E402
from betbot.strategy.glicko import match_probabilities  # noqa: E402

TEST_FROM = date(2024, 7, 1)
TEST_TO = date(2025, 7, 1)


def _load(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(path.open()):
        try:
            d = date.fromisoformat(r["date"])
            hg, ag = int(float(r["home_goals"])), int(float(r["away_goals"]))
            hx, ax = float(r["home_xg"]), float(r["away_xg"])
        except (KeyError, ValueError, TypeError):
            continue
        fc = None
        try:
            fc = (float(r["f_home"]), float(r["f_draw"]), float(r["f_away"]))
        except (KeyError, ValueError, TypeError):
            fc = None
        rows.append({"date": d, "home": r["home_team"].strip(), "away": r["away_team"].strip(),
                     "hg": hg, "ag": ag, "hx": hx, "ax": ax, "league": r["league"], "fc": fc})
    rows.sort(key=lambda x: x["date"])
    return rows


def _dc_matches(rows, *, use_xg: bool):
    out = []
    for r in rows:
        hv, av = (r["hx"], r["ax"]) if use_xg else (r["hg"], r["ag"])
        out.append(dc.DCMatch(date=r["date"], home=normalize(r["home"]),
                              away=normalize(r["away"]), home_goals=hv, away_goals=av,
                              neutral=False, friendly=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/club_xg.csv"))
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args()
    s = get_settings()

    rows = _load(args.csv)
    train = [r for r in rows if r["date"] < TEST_FROM]
    test = [r for r in rows if TEST_FROM <= r["date"] < TEST_TO]
    print(f"train {len(train)} | test(2024/25) {len(test)}")

    goals_dc = dc.fit(_dc_matches(train, use_xg=False), priors={}, iterations=args.iterations)
    xg_dc = dc.fit(_dc_matches(train, use_xg=True), priors={}, iterations=args.iterations)
    print(f"goals-DC home_adv={goals_dc.home_adv:.3f} rho={goals_dc.rho:.2f} | "
          f"xG-DC home_adv={xg_dc.home_adv:.3f} rho={xg_dc.rho:.2f}")

    # Warm goals-Glicko + form on train.
    glk, form = Glicko(s), Form()
    bydate = defaultdict(list)
    for r in train:
        bydate[r["date"]].append(r)
    for d in sorted(bydate):
        glk.update_day([(r["home"], r["away"], _outcome(r["hg"], r["ag"])) for r in bydate[d]],
                       d.isoformat())
        for r in bydate[d]:
            form.push(r["home"], r["away"], _outcome(r["hg"], r["ag"]))

    naive = StrategyEngine(s)
    wg, wd, wf = s.club_weight_glicko, s.club_weight_dc, s.club_weight_form

    def dc_probs(params, r):
        return dc.match_probabilities(params, normalize(r["home"]), normalize(r["away"]),
                                      home_field=True)

    names = ("baseline", "xG", "blend", "forecast")
    stat = {n: {"n": 0, "hit": 0, "rps": 0.0, "ll": 0.0} for n in names}
    rps_series = {n: [] for n in names}

    tbd = defaultdict(list)
    for r in test:
        tbd[r["date"]].append(r)
    for d in sorted(tbd):
        for r in tbd[d]:
            oi = OUT_IDX[_outcome(r["hg"], r["ag"])]
            rh, ra = glk.get(r["home"]), glk.get(r["away"])
            glk_p = match_probabilities(rh, ra, home_field_mu=s.glicko_club_home_mu,
                                        draw_rho=s.glicko_club_draw_rho)
            fp = naive.predict(_fixture_form(r["home"], r["away"], form))
            form_p = (fp.p_home, fp.p_draw, fp.p_away)
            gdc, xdc = dc_probs(goals_dc, r), dc_probs(xg_dc, r)
            preds = {
                "baseline": log_pool([(wg, glk_p), (wd, gdc), (wf, form_p)]),
                "xG": log_pool([(wg, glk_p), (wd, xdc), (wf, form_p)]),
                "blend": log_pool([(wg, glk_p), (wd, gdc), (wd, xdc), (wf, form_p)]),
                "forecast": r["fc"],
            }
            for n in names:
                p = preds[n]
                if p is None:
                    continue
                st = stat[n]
                st["n"] += 1
                st["hit"] += int(max(range(3), key=lambda i: p[i]) == oi)
                rps = ranked_probability_score(p, oi)
                st["rps"] += rps
                st["ll"] += -math.log(max(p[oi], 1e-9))
                rps_series[n].append((r["date"], rps))
        glk.update_day([(r["home"], r["away"], _outcome(r["hg"], r["ag"])) for r in tbd[d]],
                       d.isoformat())
        for r in tbd[d]:
            form.push(r["home"], r["away"], _outcome(r["hg"], r["ag"]))

    print(f"\n=== held-out 2024/25 (n={stat['baseline']['n']}) ===")
    print(f"{'model':10s} {'acc%':>7s} {'meanRPS':>9s} {'logloss':>9s}")
    for n in names:
        st = stat[n]
        c = max(st["n"], 1)
        print(f"{n:10s} {100*st['hit']/c:>7.2f} {st['rps']/c:>9.4f} {st['ll']/c:>9.4f}")

    # Bootstrap CI on baseline - variant, aligned per match.
    base_map = dict(zip(range(len(rps_series["baseline"])),
                        [x[1] for x in rps_series["baseline"]]))
    for variant in ("xG", "blend"):
        diffs = [b - v for (b, v) in zip([x[1] for x in rps_series["baseline"]],
                                         [x[1] for x in rps_series[variant]])]
        impr = sum(diffs) / len(diffs)
        lo, hi = _bootstrap_ci(diffs)
        verdict = "GATE PASSED" if lo > 0 else ("regression" if hi < 0 else "not distinguishable")
        print(f"\n{variant} vs baseline: RPS improvement {impr:+.5f}/match  "
              f"CI95 [{lo:+.5f}, {hi:+.5f}] -> {verdict}")

    # Predicted expected-goals sample from the xG-DC model.
    print("\npredicted xG (xG-DC lambda, home advantage on):")
    for h, a in [("Manchester City", "Liverpool"), ("Real Madrid", "Barcelona"),
                 ("Bayern Munich", "Dortmund")]:
        lam_h, lam_a = dc.expected_goals(xg_dc, normalize(h), normalize(a), home_field=True)
        print(f"  {h} {lam_h:.2f} - {lam_a:.2f} {a}")


if __name__ == "__main__":
    main()
