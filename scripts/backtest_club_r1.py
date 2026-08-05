"""R1 sweep for the club ensemble: MOV-Glicko, DC half-life, weights, calibration.

Three-way chronological split to avoid tuning on the final gate:

* train  : matches before 2024-07-01   (warm ratings, fit DC)
* val    : 2024-07-01 .. 2025-06-30    (grid-sweep configs, fit calibration)
* test   : 2025-07-01 onwards          (final gate — evaluated ONCE, at the end)

Sweep axes: glicko variant (standard vs margin-of-victory), club home_mu,
draw_rho, DC time-decay half-life, and the glicko/dc/form log-pool weights.
The best-on-val config is then re-evaluated walk-forward on the untouched
test season against the CURRENT live config (the shipped defaults), with a
bootstrap CI on the per-match RPS difference. Optionally writes the isotonic
calibration artifact fit on val predictions (--write-calibration).

Run (repo root, venv active; ~2-4 min):
    python scripts/backtest_club_r1.py
    python scripts/backtest_club_r1.py --write-calibration
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_club import (  # noqa: E402
    OUT_IDX,
    Form,
    Glicko,
    _bootstrap_ci,
    _fixture_form,
    _load,
    _market_probs,
    _outcome,
)

from betbot.config import get_settings  # noqa: E402
from betbot.exchanges.matcher import normalize  # noqa: E402
from betbot.strategy import dixon_coles as dc  # noqa: E402
from betbot.strategy.engine import StrategyEngine  # noqa: E402
from betbot.strategy.ensemble import (  # noqa: E402
    IsotonicCalibrator,
    calibrate,
    log_pool,
    ranked_probability_score,
)
from betbot.strategy.glicko import Glicko2Rating, match_probabilities  # noqa: E402
from betbot.strategy.glicko_mov import update_rating_mov  # noqa: E402

VAL_FROM = date(2024, 7, 1)
TEST_FROM = date(2025, 7, 1)

HOME_MUS = (0.20, 0.30, 0.40)
DRAW_RHOS = (0.24, 0.28, 0.32)
HALF_LIVES = (270.0, 390.0, 540.0, 720.0)
W_GLICKO = (0.6, 1.0, 1.4)
W_DC = (0.6, 1.0, 1.4)
W_FORM = (0.0, 0.25, 0.5)

# The shipped live defaults (config.py) — the incumbent the sweep must beat.
BASELINE = {"variant": "std", "mu": 0.30, "rho": 0.28, "hl": 540.0,
            "wg": 1.0, "wd": 1.0, "wf": 0.5}


class MovGlicko(Glicko):
    """Walk-forward Glicko-2 with margin-of-victory scaled updates."""

    def update_day_scores(self, matches: list[tuple[str, str, int, int]], d: str) -> None:
        teams = {t for h, a, _, _ in matches for t in (h, a)}
        cur = {t: self.get(t) for t in teams}
        per: dict[str, list] = {t: [] for t in teams}
        for h, a, hs, as_ in matches:
            o = _outcome(hs, as_)
            gd = abs(hs - as_)
            sh = 1.0 if o == "HOME" else (0.5 if o == "DRAW" else 0.0)
            per[h].append((cur[a].rating, cur[a].rd, sh, gd))
            per[a].append((cur[h].rating, cur[h].rd, 1.0 - sh if o != "DRAW" else 0.5, gd))
        for t in teams:
            self.r[t] = update_rating_mov(cur[t], per[t], tau=self.s.glicko_tau, period=d)


def _dc_matches(rows: list[dict]) -> list[dc.DCMatch]:
    return [
        dc.DCMatch(date=r["date"], home=normalize(r["home"]), away=normalize(r["away"]),
                   home_goals=r["hs"], away_goals=r["as"], neutral=False, friendly=False)
        for r in rows
    ]


def _walk(rows, glk_std, glk_mov, form, naive, dc_by_hl):
    """Walk a split in date order; per match, record every component's probs
    BEFORE folding the day's results into ratings/form."""
    records = []
    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)
    for d in sorted(by_date):
        day = by_date[d]
        for r in day:
            home, away = r["home"], r["away"]
            g_probs = {}
            for variant, glk in (("std", glk_std), ("mov", glk_mov)):
                rh, ra = glk.get(home), glk.get(away)
                for mu in HOME_MUS:
                    for rho in DRAW_RHOS:
                        g_probs[(variant, mu, rho)] = match_probabilities(
                            rh, ra, home_field_mu=mu, draw_rho=rho)
            d_probs = {
                hl: dc.match_probabilities(params, normalize(home), normalize(away),
                                           home_field=True)
                for hl, params in dc_by_hl.items()
            }
            fp = naive.predict(_fixture_form(home, away, form))
            records.append({
                "oi": OUT_IDX[_outcome(r["hs"], r["as"])],
                "g": g_probs, "d": d_probs,
                "f": (fp.p_home, fp.p_draw, fp.p_away),
                "mkt": _market_probs(r["ps"]),
                "known": (glk_std.get(home).rd < glk_std.default.rd
                          and glk_std.get(away).rd < glk_std.default.rd),
            })
        glk_std.update_day([(r["home"], r["away"], _outcome(r["hs"], r["as"]))
                            for r in day], d.isoformat())
        glk_mov.update_day_scores([(r["home"], r["away"], r["hs"], r["as"])
                                   for r in day], d.isoformat())
        for r in day:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))
    return records


def _config_probs(rec, cfg):
    comps = [(cfg["wg"], rec["g"][(cfg["variant"], cfg["mu"], cfg["rho"])]),
             (cfg["wd"], rec["d"][cfg["hl"]])]
    if cfg["wf"] > 0:
        comps.append((cfg["wf"], rec["f"]))
    return log_pool(comps)


def _mean_rps(records, cfg, calibrators=None):
    tot = 0.0
    for rec in records:
        p = _config_probs(rec, cfg)
        if calibrators is not None:
            p = calibrate(p, calibrators)
        tot += ranked_probability_score(p, rec["oi"])
    return tot / max(len(records), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/club_results.csv"))
    ap.add_argument("--write-calibration", action="store_true")
    ap.add_argument("--calib-out", type=Path,
                    default=Path("data/ensemble_calibration_club.json"))
    args = ap.parse_args()

    s = get_settings()
    rows = _load(args.csv)
    train = [r for r in rows if r["date"] < VAL_FROM]
    val = [r for r in rows if VAL_FROM <= r["date"] < TEST_FROM]
    test = [r for r in rows if r["date"] >= TEST_FROM]
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")

    # ---- Phase 1: warm ratings on train; fit DC per half-life on train ----
    glk_std, glk_mov, form = Glicko(s), MovGlicko(s), Form()
    by_date = defaultdict(list)
    for r in train:
        by_date[r["date"]].append(r)
    for d in sorted(by_date):
        day = by_date[d]
        glk_std.update_day([(r["home"], r["away"], _outcome(r["hs"], r["as"]))
                            for r in day], d.isoformat())
        glk_mov.update_day_scores([(r["home"], r["away"], r["hs"], r["as"])
                                   for r in day], d.isoformat())
        for r in day:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))
    train_dc = _dc_matches(train)
    dc_by_hl = {}
    for hl in HALF_LIVES:
        dc_by_hl[hl] = dc.fit(train_dc, priors={}, half_life_days=hl, iterations=200)
        print(f"  DC(train) hl={hl:.0f}: home_adv={dc_by_hl[hl].home_adv:.3f}")

    naive = StrategyEngine(s)

    # ---- Phase 2: walk val, recording all component probs ----
    val_recs = _walk(val, glk_std, glk_mov, form, naive, dc_by_hl)

    # ---- Phase 3: grid sweep on val ----
    results = []
    for variant in ("std", "mov"):
        for mu in HOME_MUS:
            for rho in DRAW_RHOS:
                for hl in HALF_LIVES:
                    for wg in W_GLICKO:
                        for wd in W_DC:
                            for wf in W_FORM:
                                cfg = {"variant": variant, "mu": mu, "rho": rho,
                                       "hl": hl, "wg": wg, "wd": wd, "wf": wf}
                                results.append((_mean_rps(val_recs, cfg), cfg))
    results.sort(key=lambda x: x[0])
    base_val = _mean_rps(val_recs, BASELINE)
    print(f"\nval sweep ({len(results)} configs): baseline(live cfg) RPS {base_val:.4f}")
    print("top 10 on val:")
    for rps, cfg in results[:10]:
        print(f"  {rps:.4f}  {cfg['variant']:3s} mu={cfg['mu']:.2f} rho={cfg['rho']:.2f} "
              f"hl={cfg['hl']:.0f} w=({cfg['wg']},{cfg['wd']},{cfg['wf']})")
    best = results[0][1]

    # ---- Phase 4: fit isotonic calibration on val preds of the best config ----
    preds = {k: [] for k in range(3)}
    obs = {k: [] for k in range(3)}
    for rec in val_recs:
        p = _config_probs(rec, best)
        for k in range(3):
            preds[k].append(p[k])
            obs[k].append(1.0 if rec["oi"] == k else 0.0)
    calibs = tuple(IsotonicCalibrator().fit(preds[k], obs[k]) for k in range(3))
    cal_val = _mean_rps(val_recs, best, calibs)
    print(f"\nbest on val: {results[0][0]:.4f} -> with calibration {cal_val:.4f} "
          f"({'keep' if cal_val < results[0][0] else 'calibration HURTS on val'})")
    use_calib = cal_val < results[0][0]

    # ---- Phase 5: final gate on untouched test season ----
    # Ratings/form walked through val already; refit DC on train+val at the
    # baseline hl and the best hl.
    tv_dc = _dc_matches(train + val)
    dc_final = {}
    for hl in {BASELINE["hl"], best["hl"]}:
        dc_final[hl] = dc.fit(tv_dc, priors={}, half_life_days=hl, iterations=200)
    test_recs = _walk(test, glk_std, glk_mov, form, naive, dc_final)

    def stats(cfg, cal):
        n = hit = 0
        rps_list = []
        ll = 0.0
        for rec in test_recs:
            p = _config_probs(rec, cfg)
            if cal is not None:
                p = calibrate(p, cal)
            n += 1
            hit += int(max(range(3), key=lambda i: p[i]) == rec["oi"])
            rps_list.append(ranked_probability_score(p, rec["oi"]))
            ll += -math.log(max(p[rec["oi"]], 1e-9))
        return n, hit, rps_list, ll

    n_b, hit_b, rps_b, ll_b = stats(BASELINE, None)
    n_t, hit_t, rps_t, ll_t = stats(best, calibs if use_calib else None)
    mkt_n = mkt_hit = 0
    mkt_rps = mkt_ll = 0.0
    for rec in test_recs:
        if rec["mkt"] is None:
            continue
        mkt_n += 1
        mkt_hit += int(max(range(3), key=lambda i: rec["mkt"][i]) == rec["oi"])
        mkt_rps += ranked_probability_score(rec["mkt"], rec["oi"])
        mkt_ll += -math.log(max(rec["mkt"][rec["oi"]], 1e-9))

    print(f"\n=== FINAL GATE (test season, n={n_b}) ===")
    print(f"{'model':22s} {'acc%':>7s} {'meanRPS':>9s} {'logloss':>9s}")
    print(f"{'live club cfg (base)':22s} {100*hit_b/n_b:>7.2f} "
          f"{sum(rps_b)/n_b:>9.4f} {ll_b/n_b:>9.4f}")
    print(f"{'R1 tuned':22s} {100*hit_t/n_t:>7.2f} "
          f"{sum(rps_t)/n_t:>9.4f} {ll_t/n_t:>9.4f}")
    print(f"{'market':22s} {100*mkt_hit/max(mkt_n,1):>7.2f} "
          f"{mkt_rps/max(mkt_n,1):>9.4f} {mkt_ll/max(mkt_n,1):>9.4f}")

    diffs = [b - t for b, t in zip(rps_b, rps_t)]
    impr = sum(diffs) / len(diffs)
    lo, hi = _bootstrap_ci(diffs)
    print(f"\nRPS improvement (base - tuned): {impr:+.5f}  CI95 [{lo:+.5f}, {hi:+.5f}]"
          f"  -> {'GATE PASSED' if lo > 0 else 'GATE FAILED (keep incumbent)'}")
    print(f"best cfg: {best}  calibration={'ON' if use_calib else 'OFF'}")

    if args.write_calibration and use_calib:
        payload = {k: json.loads(c.to_json()) for k, c in
                   zip(("home", "draw", "away"), calibs)}
        args.calib_out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"wrote {args.calib_out}")


if __name__ == "__main__":
    main()
