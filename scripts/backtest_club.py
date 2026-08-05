"""Walk-forward backtest: club ensemble vs naive form engine vs the market.

Gate for shipping the club ensemble. Splits data/club_results.csv into a train
prefix and a held-out test season, then walks the test season IN DATE ORDER,
predicting each match BEFORE its result is known and only then folding the
result into the Glicko ratings and form windows (no lookahead). Three models
are scored on identical inputs:

* **naive**    — the current form-only StrategyEngine;
* **ensemble** — Glicko(club) + Dixon-Coles(club) + form, log-pooled (exactly
  the ClubStrategyEngine formula: same weights, home_mu, draw_rho);
* **market**   — de-vigged closing odds (Pinnacle/Bet365), the honest yardstick.

Metrics: accuracy (argmax), mean RPS (lower better), mean log-loss. A bootstrap
95% CI on the per-match RPS improvement (naive - ensemble) says whether the
ensemble's edge over naive is real or noise.

Run (repo root, venv active):
    python scripts/backtest_club.py
    python scripts/backtest_club.py --test-from 2025-07-01 --dc-iterations 200
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from betbot.config import get_settings
from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.strategy import dixon_coles as dc
from betbot.strategy.engine import StrategyEngine
from betbot.exchanges.matcher import normalize
from betbot.strategy.ensemble import log_pool, ranked_probability_score
from betbot.strategy.glicko import Glicko2Rating, match_probabilities, update_rating

OUT_IDX = {"HOME": 0, "DRAW": 1, "AWAY": 2}


def _outcome(hs: int, as_: int) -> str:
    return "HOME" if hs > as_ else ("AWAY" if as_ > hs else "DRAW")


def _load(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(path.open()):
        try:
            d = date.fromisoformat(r["date"])
            hs, as_ = int(float(r["home_score"])), int(float(r["away_score"]))
        except (KeyError, ValueError):
            continue
        rows.append({
            "date": d, "home": r["home_team"].strip(), "away": r["away_team"].strip(),
            "hs": hs, "as": as_, "league": r.get("league", ""),
            "ps": (r.get("ps_home", ""), r.get("ps_draw", ""), r.get("ps_away", "")),
        })
    rows.sort(key=lambda x: x["date"])
    return rows


def _market_probs(ps: tuple[str, str, str]) -> tuple[float, float, float] | None:
    try:
        o = [float(x) for x in ps]
    except (ValueError, TypeError):
        return None
    if any(v <= 1.0 for v in o):
        return None
    inv = [1.0 / v for v in o]
    s = sum(inv)
    return (inv[0] / s, inv[1] / s, inv[2] / s)


class Glicko:
    """In-memory walk-forward Glicko-2, one rating period per match date."""

    def __init__(self, settings):
        self.s = settings
        self.default = Glicko2Rating(
            settings.glicko_default_rating, settings.glicko_default_rd,
            settings.glicko_default_vol,
        )
        self.r: dict[str, Glicko2Rating] = {}

    def get(self, name: str) -> Glicko2Rating:
        return self.r.get(name, self.default)

    def update_day(self, matches: list[tuple[str, str, str]], d: str) -> None:
        teams = {t for h, a, _ in matches for t in (h, a)}
        cur = {t: self.get(t) for t in teams}
        per: dict[str, list] = {t: [] for t in teams}
        for h, a, o in matches:
            sh = 1.0 if o == "HOME" else (0.5 if o == "DRAW" else 0.0)
            per[h].append((cur[a].rating, cur[a].rd, sh))
            per[a].append((cur[h].rating, cur[h].rd, 1.0 - sh if o != "DRAW" else 0.5))
        for t in teams:
            self.r[t] = update_rating(cur[t], per[t], tau=self.s.glicko_tau, period=d)


class Form:
    """Rolling last-K points per team -> a weighted_points proxy for the naive
    engine (both models see the identical FixtureForm, so this is fair)."""

    def __init__(self, k: int = 5):
        self.k = k
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=k))

    def points(self, team: str) -> tuple[float, int]:
        h = self.hist[team]
        if not h:
            return 1.0, 0  # neutral prior
        return sum(h) / len(h), len(h)

    def push(self, home: str, away: str, o: str) -> None:
        self.hist[home].append(3 if o == "HOME" else (1 if o == "DRAW" else 0))
        self.hist[away].append(3 if o == "AWAY" else (1 if o == "DRAW" else 0))


def _fixture_form(home: str, away: str, form: Form) -> FixtureForm:
    hp, hn = form.points(home)
    ap, an = form.points(away)
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=0, home_team=ht, away_team=at, kickoff=None, competition_code="PL")
    return FixtureForm(
        fixture=fx,
        home_form=FormSnapshot(team=ht, weighted_points=hp, raw_points=0, matches_considered=hn),
        away_form=FormSnapshot(team=at, weighted_points=ap, raw_points=0, matches_considered=an),
    )


def _bootstrap_ci(diffs: list[float], iters: int = 5000, seed: int = 12345) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/club_results.csv"))
    ap.add_argument("--test-from", default="2025-07-01",
                    help="matches on/after this date are the held-out test set")
    ap.add_argument("--dc-iterations", type=int, default=200)
    args = ap.parse_args()

    s = get_settings()
    rows = _load(args.csv)
    cutoff = date.fromisoformat(args.test_from)
    train = [r for r in rows if r["date"] < cutoff]
    test = [r for r in rows if r["date"] >= cutoff]
    print(f"train {len(train)} matches (<{cutoff}), test {len(test)} matches (>= {cutoff})")

    # Fit Dixon-Coles once on the train prefix (dataset-normalised keys).
    dc_matches = [
        dc.DCMatch(date=r["date"], home=normalize(r["home"]), away=normalize(r["away"]),
                   home_goals=r["hs"], away_goals=r["as"], neutral=False, friendly=False)
        for r in train
    ]
    dc_params = dc.fit(dc_matches, priors={}, iterations=args.dc_iterations)
    print(f"DC fit on train: teams={len(dc_params.teams)}, home_adv={dc_params.home_adv:.3f}")

    # Replay train to warm Glicko + form windows.
    glk = Glicko(s)
    form = Form()
    by_date: dict[date, list] = defaultdict(list)
    for r in train:
        by_date[r["date"]].append(r)
    for d in sorted(by_date):
        day = [(r["home"], r["away"], _outcome(r["hs"], r["as"])) for r in by_date[d]]
        glk.update_day(day, d.isoformat())
        for r in by_date[d]:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))

    naive = StrategyEngine(s)
    weights = [
        (s.club_weight_glicko, "glicko"),
        (s.club_weight_dc, "dc"),
        (s.club_weight_form, "form"),
    ]

    # Accumulators per model.
    stats: dict[str, dict] = {
        m: {"n": 0, "hit": 0, "rps": 0.0, "ll": 0.0} for m in ("naive", "ensemble", "market")
    }
    per_league: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "naive_rps": 0.0, "ens_rps": 0.0}
    )
    rps_diffs: list[float] = []  # naive_rps - ens_rps, per test match (ensemble better => positive)

    test_by_date: dict[date, list] = defaultdict(list)
    for r in test:
        test_by_date[r["date"]].append(r)

    for d in sorted(test_by_date):
        day = test_by_date[d]
        # Predict every match on this date BEFORE updating ratings with the day.
        for r in day:
            home, away = r["home"], r["away"]
            oi = OUT_IDX[_outcome(r["hs"], r["as"])]
            ff = _fixture_form(home, away, form)

            fp = naive.predict(ff)
            naive_probs = (fp.p_home, fp.p_draw, fp.p_away)

            rh, ra = glk.get(home), glk.get(away)
            glicko_probs = match_probabilities(
                rh, ra, home_field_mu=s.glicko_club_home_mu, draw_rho=s.glicko_club_draw_rho
            )
            dc_probs = dc.match_probabilities(
                dc_params, normalize(home), normalize(away), home_field=True
            )
            comp = {"glicko": glicko_probs, "dc": dc_probs, "form": naive_probs}
            ens_probs = log_pool([(w, comp[name]) for w, name in weights if w > 0])

            mkt = _market_probs(r["ps"])

            for name, probs in (("naive", naive_probs), ("ensemble", ens_probs)):
                st = stats[name]
                st["n"] += 1
                st["hit"] += int(max(range(3), key=lambda i: probs[i]) == oi)
                st["rps"] += ranked_probability_score(probs, oi)
                st["ll"] += -math.log(max(probs[oi], 1e-9))
            if mkt is not None:
                st = stats["market"]
                st["n"] += 1
                st["hit"] += int(max(range(3), key=lambda i: mkt[i]) == oi)
                st["rps"] += ranked_probability_score(mkt, oi)
                st["ll"] += -math.log(max(mkt[oi], 1e-9))

            n_rps = ranked_probability_score(naive_probs, oi)
            e_rps = ranked_probability_score(ens_probs, oi)
            rps_diffs.append(n_rps - e_rps)
            pl = per_league[r["league"]]
            pl["n"] += 1
            pl["naive_rps"] += n_rps
            pl["ens_rps"] += e_rps

        # Now fold the day's results into ratings + form (walk-forward).
        glk.update_day([(r["home"], r["away"], _outcome(r["hs"], r["as"])) for r in day],
                       d.isoformat())
        for r in day:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))

    # ---- Report --------------------------------------------------------
    print("\n=== held-out test results ===")
    print(f"{'model':10s} {'n':>5s} {'acc%':>7s} {'meanRPS':>9s} {'logloss':>9s}")
    for m in ("naive", "ensemble", "market"):
        st = stats[m]
        n = max(st["n"], 1)
        print(f"{m:10s} {st['n']:>5d} {100*st['hit']/n:>7.2f} "
              f"{st['rps']/n:>9.4f} {st['ll']/n:>9.4f}")

    n = max(stats["ensemble"]["n"], 1)
    naive_mrps = stats["naive"]["rps"] / n
    ens_mrps = stats["ensemble"]["rps"] / n
    impr = naive_mrps - ens_mrps
    lo, hi = _bootstrap_ci(rps_diffs)
    print(f"\nRPS improvement (naive - ensemble): {impr:+.5f} per match")
    print(f"  bootstrap 95% CI: [{lo:+.5f}, {hi:+.5f}]  "
          f"({'SIGNIFICANT — ensemble wins' if lo > 0 else 'includes 0 — not distinguishable'})")
    rel = 100 * impr / naive_mrps if naive_mrps else 0.0
    print(f"  relative RPS reduction vs naive: {rel:+.2f}%")

    print("\nper-league mean RPS (naive -> ensemble):")
    for lg in sorted(per_league):
        pl = per_league[lg]
        c = max(pl["n"], 1)
        nr, er = pl["naive_rps"] / c, pl["ens_rps"] / c
        print(f"  {lg:4s} n={pl['n']:>4d}  {nr:.4f} -> {er:.4f}  "
              f"({100*(nr-er)/nr:+.1f}%)")


if __name__ == "__main__":
    main()

