"""Walk-forward backtest of the ensemble on the 2022 World Cup (Qatar).

Replays international results chronologically: Glicko-2 ratings update after
every match date and the Dixon-Coles model is refit from scratch before each
WC match day (data strictly BEFORE the day — no leakage). For every WC match
it scores pure Glicko vs the Glicko+DC ensemble on RPS, Brier and log-loss,
and fits per-outcome isotonic calibration on a pre-tournament window
(2021-01 → WC start), persisting it to data/ensemble_calibration.json.

HONESTY NOTES (per CLAUDE.md):
- No fundamentals priors here: the Klement CSV is the 2026 cohort (2026 FIFA
  points, USA/MEX/CAN hosts) and cannot be retro-applied to Qatar 2022. The
  fundamentals layer's cold-start value is therefore NOT measured by this
  backtest.
- Beating pure Glicko is a MODEL improvement, not market edge. The market
  benchmark needs historical closing odds we don't have a free source for;
  supply them via --odds-csv (date,home_team,away_team,p_home,p_draw,p_away,
  vig included is fine — we de-vig) and the report adds the market RPS.

Run (repo root, venv active; a few minutes — DC refits per WC match day):
    python scripts/backtest_ensemble.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betbot.exchanges.matcher import normalize  # noqa: E402
from betbot.strategy import dixon_coles as dc  # noqa: E402
from betbot.strategy.ensemble import (  # noqa: E402
    EnsembleWeights,
    IsotonicCalibrator,
    blend,
    calibrate,
    de_vig,
    ranked_probability_score,
)
from betbot.strategy.glicko import (  # noqa: E402
    Glicko2Rating,
    match_probabilities,
    update_rating,
)

CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
HISTORY_FROM = "2016-01-01"
CALIB_FROM = "2021-01-01"
WC_START, WC_END = "2022-11-20", "2022-12-18"
HOSTS_2022 = {"qatar"}
HOST_MU = 0.2
DEFAULT = Glicko2Rating(1500, 200, 0.06)
DC_REFIT_DAYS_CALIB = 90  # quarterly refits in the calibration window


def _outcome_index(hs: int, as_: int) -> int:
    return 0 if hs > as_ else (2 if as_ > hs else 1)


_MAJOR_KEYS = ("euro", "copa am", "african cup", "afc asian cup",
               "gold cup", "world cup")


def _is_major(r: dict) -> bool:
    """Continental/world FINALS tournament — the closest pre-WC proxy for WC
    conditions (competitive, neutral/hosted venues). Qualifiers excluded:
    they are ordinary home-and-away fixtures."""
    t = r.get("tournament", "").lower()
    return any(k in t for k in _MAJOR_KEYS) and "qualif" not in t


def _load_rows(raw: str) -> list[dict]:
    rows = [r for r in csv.DictReader(io.StringIO(raw))
            if HISTORY_FROM <= r["date"] <= WC_END]
    rows.sort(key=lambda r: r["date"])
    return rows


def _to_dc_match(r: dict) -> dc.DCMatch:
    return dc.DCMatch(
        date=date.fromisoformat(r["date"]),
        home=normalize(r["home_team"]),
        away=normalize(r["away_team"]),
        home_goals=int(float(r["home_score"])),
        away_goals=int(float(r["away_score"])),
        neutral=(r.get("neutral", "").strip().upper() == "TRUE"),
        friendly=(r.get("tournament", "").strip().lower() == "friendly"),
    )


class Replay:
    """Chronological Glicko replay with one rating period per match date."""

    def __init__(self) -> None:
        self.ratings: dict[str, Glicko2Rating] = {}

    def get(self, team: str) -> Glicko2Rating:
        return self.ratings.get(team, DEFAULT)

    def apply_day(self, day_rows: list[dict]) -> None:
        results = []
        for r in day_rows:
            try:
                hs, as_ = int(float(r["home_score"])), int(float(r["away_score"]))
            except ValueError:
                continue
            results.append((normalize(r["home_team"]), normalize(r["away_team"]),
                            _outcome_index(hs, as_)))
        teams = {t for h, a, _ in results for t in (h, a)}
        cur = {t: self.get(t) for t in teams}
        per: dict[str, list] = {t: [] for t in teams}
        for h, a, oi in results:
            sh = 1.0 if oi == 0 else (0.5 if oi == 1 else 0.0)
            per[h].append((cur[a].rating, cur[a].rd, sh))
            per[a].append((cur[h].rating, cur[h].rd, 1.0 - sh))
        for t in teams:
            self.ratings[t] = update_rating(cur[t], per[t])

    def probs(self, r: dict) -> tuple[float, float, float]:
        # A non-neutral match means the home side is genuinely at home —
        # at the WC that's only the host, elsewhere it's most matches.
        h, a = normalize(r["home_team"]), normalize(r["away_team"])
        neutral = r.get("neutral", "").strip().upper() == "TRUE"
        return match_probabilities(
            self.get(h), self.get(a), home_field_mu=0.0 if neutral else HOST_MU
        )


def _dc_probs(params: dc.DCParams, r: dict) -> tuple[float, float, float]:
    h, a = normalize(r["home_team"]), normalize(r["away_team"])
    neutral = r.get("neutral", "").strip().upper() == "TRUE"
    return dc.match_probabilities(params, h, a, home_field=not neutral)


def _scores(preds: list[tuple[tuple[float, float, float], int]]) -> dict[str, float]:
    n = len(preds)
    rps = sum(ranked_probability_score(p, oi) for p, oi in preds) / n
    brier = sum(
        sum((p[i] - (1.0 if i == oi else 0.0)) ** 2 for i in range(3))
        for p, oi in preds
    ) / n
    logloss = sum(-math.log(max(p[oi], 1e-12)) for p, oi in preds) / n
    hit = sum(1.0 for p, oi in preds if max(range(3), key=lambda i: p[i]) == oi) / n
    return {"rps": rps, "brier": brier, "logloss": logloss, "hit": hit}


def _load_market(path: str | None) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            probs = de_vig([float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])])
            out[(r["date"], normalize(r["home_team"]), normalize(r["away_team"]))] = tuple(probs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="Local results.csv (skips the download).")
    ap.add_argument("--odds-csv", help="Optional WC2022 closing-odds CSV (see docstring).")
    ap.add_argument("--w-dc", type=float, default=None,
                    help="DC log-pool weight (glicko=1); default: sweep the "
                         "calibration window and use the best.")
    # Written as a .candidate so it never auto-deploys: the engine only loads
    # data/ensemble_calibration.json, and promoting the candidate is an
    # explicit operator `mv` after reading this report.
    ap.add_argument("--calibration-out",
                    default="data/ensemble_calibration.candidate.json", type=Path)
    args = ap.parse_args()

    if args.csv:
        raw = Path(args.csv).read_text(encoding="utf-8")
    else:
        print("downloading international results…")
        raw = urllib.request.urlopen(CSV_URL, timeout=60).read().decode("utf-8")
    rows = _load_rows(raw)
    market = _load_market(args.odds_csv)

    by_date: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)
    days = sorted(by_date)

    replay = Replay()
    dc_params: dc.DCParams | None = None
    dc_fit_day: date | None = None
    history: list[dc.DCMatch] = []

    # (glicko_probs, dc_probs, outcome_index) per scored match.
    Triple = tuple[float, float, float]
    calib_comp: list[tuple[Triple, Triple, int]] = []
    calib_major: list[tuple[Triple, Triple, int]] = []
    wc_comp: list[tuple[Triple, Triple, int]] = []
    wc_market: list[tuple[Triple, int]] = []

    for day in days:
        day_rows = [r for r in by_date[day]
                    if r.get("home_score") not in (None, "", "NA")]
        has_wc = WC_START <= day <= WC_END and any(
            r.get("tournament") == "FIFA World Cup" for r in day_rows
        )
        in_calib = CALIB_FROM <= day < WC_START

        # Refit DC before predicting: quarterly in the calibration window,
        # before every match day inside the WC. Data strictly before `day`.
        if in_calib or has_wc:
            d = date.fromisoformat(day)
            stale = dc_fit_day is None or (
                (d - dc_fit_day).days >= (0 if has_wc else DC_REFIT_DAYS_CALIB)
            )
            if has_wc or stale:
                dc_params = dc.fit(history)
                dc_fit_day = d

        for r in day_rows:
            try:
                oi = _outcome_index(int(float(r["home_score"])),
                                    int(float(r["away_score"])))
            except ValueError:
                continue
            is_wc = has_wc and r.get("tournament") == "FIFA World Cup"
            # Calibrate on competitive matches only — friendlies have their
            # own draw/effort profile and would skew the WC mapping.
            is_calib = (in_calib and dc_params is not None
                        and r.get("tournament", "").strip().lower() != "friendly")
            if (is_calib or is_wc) and dc_params is not None:
                comp = (replay.probs(r), _dc_probs(dc_params, r), oi)
                if is_calib:
                    calib_comp.append(comp)
                    if _is_major(r):
                        calib_major.append(comp)
                if is_wc:
                    wc_comp.append(comp)
                    key = (r["date"], normalize(r["home_team"]),
                           normalize(r["away_team"]))
                    if key in market:
                        wc_market.append((market[key], oi))

        replay.apply_day(day_rows)
        history.extend(_to_dc_match(r) for r in day_rows
                       if r.get("home_score") not in (None, "", "NA"))

    # ---- weight sweep on the calibration window (walk-forward: the window
    # precedes the WC, so picking w_dc here is leakage-free) ---------------
    def _blend_all(comps, w_dc):
        w = EnsembleWeights(glicko=1.0, dixon_coles=w_dc, market=0.0)
        return [(blend(g, d, weights=w), oi) for g, d, oi in comps]

    # Sweep on the major-tournament subset when it's big enough — the
    # closest leakage-free proxy for WC conditions; competitive otherwise.
    sweep_set = calib_major if len(calib_major) >= 150 else calib_comp
    sweep = {}
    for w_dc in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        sweep[w_dc] = _scores(_blend_all(sweep_set, w_dc))["rps"]
    best_w_dc = min(sweep, key=sweep.get) if args.w_dc is None else args.w_dc
    calib_preds = _blend_all(sweep_set, best_w_dc)
    wc_glicko = [(g, oi) for g, _, oi in wc_comp]
    wc_ens = _blend_all(wc_comp, best_w_dc)

    # ---- calibration fit on the pre-tournament window -------------------
    cals = []
    for i, name in enumerate(("home", "draw", "away")):
        cal = IsotonicCalibrator().fit(
            [p[i] for p, _ in calib_preds],
            [1.0 if oi == i else 0.0 for _, oi in calib_preds],
        )
        cals.append((name, cal))
    args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_out.write_text(json.dumps(
        {name: json.loads(cal.to_json()) for name, cal in cals}, indent=1
    ), encoding="utf-8")
    calibrators = tuple(cal for _, cal in cals)
    wc_ens_cal = [(calibrate(p, calibrators), oi) for p, oi in wc_ens]

    # ---- report ----------------------------------------------------------
    subset = "major-tournament" if sweep_set is calib_major else "competitive"
    print(f"\nw_dc sweep on {subset} subset (RPS): "
          + "  ".join(f"{k}:{v:.4f}" for k, v in sweep.items())
          + f"  -> using w_dc={best_w_dc}")
    print(f"calibration window: {len(calib_preds)} {subset} matches "
          f"({CALIB_FROM} -> {WC_START}); WC matches scored: {len(wc_ens)}")
    print(f"{'model':<22}{'RPS':>8}{'Brier':>8}{'LogLoss':>9}{'Hit%':>7}")
    for label, preds in (
        ("pure Glicko", wc_glicko),
        ("ensemble (G+DC)", wc_ens),
        ("ensemble calibrated", wc_ens_cal),
    ):
        s = _scores(preds)
        print(f"{label:<22}{s['rps']:>8.4f}{s['brier']:>8.4f}"
              f"{s['logloss']:>9.4f}{s['hit']*100:>6.1f}%")
    if wc_market:
        s = _scores(wc_market)
        print(f"{'market (de-vigged)':<22}{s['rps']:>8.4f}{s['brier']:>8.4f}"
              f"{s['logloss']:>9.4f}{s['hit']*100:>6.1f}%")
    else:
        print("market benchmark: SKIPPED (no --odds-csv; model-vs-model only — "
              "this does NOT demonstrate edge over market prices)")

    # Paired per-match comparison — 64 matches is a SMALL sample; report
    # the uncertainty rather than letting a point estimate over-claim.
    diffs = [
        ranked_probability_score(pe, oi) - ranked_probability_score(pg, oi)
        for (pe, oi), (pg, _) in zip(wc_ens_cal, wc_glicko)
    ]
    n = len(diffs)
    mean_d = sum(diffs) / n
    sd = math.sqrt(sum((x - mean_d) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    g, e = _scores(wc_glicko)["rps"], _scores(wc_ens_cal)["rps"]
    verdict = "ENSEMBLE BEATS PURE GLICKO" if e < g else "NO IMPROVEMENT — keep Glicko"
    print(f"\nverdict (RPS, lower=better): {verdict}  ({e:.4f} vs {g:.4f})")
    print(f"paired RPS diff: {mean_d:+.4f} ± {1.96 * se:.4f} (95% CI, n={n}) — "
          + ("statistically meaningful" if abs(mean_d) > 1.96 * se
             else "NOT statistically distinguishable on this sample"))
    print(f"calibration written -> {args.calibration_out}")


if __name__ == "__main__":
    main()
