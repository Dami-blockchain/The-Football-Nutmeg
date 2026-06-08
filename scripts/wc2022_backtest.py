"""Backtest the Glicko engine on the 2022 World Cup (Qatar): bot vs actual.

Bootstraps Glicko-2 from international results BEFORE the tournament, then
walk-forward predicts each match (updating ratings after every matchday) and
compares the bot's predicted outcome to what actually happened. Reports the
favourite hit-rate, multiclass Brier score, and a per-match table.

Data: martj42/international_results results.csv (public). Host bump (Qatar)
applied to non-neutral matches via the CSV's `neutral` flag.

Run: python scripts/wc2022_backtest.py
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections import defaultdict

from betbot.strategy.glicko import Glicko2Rating, match_probabilities, update_rating

CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
WC_START, WC_END = "2022-11-20", "2022-12-18"
HOST_MU = 0.2
TAU = 0.5
DEFAULT = Glicko2Rating(1500, 200, 0.06)


def _outcome(hs: int, as_: int) -> str:
    return "HOME" if hs > as_ else ("AWAY" if as_ > hs else "DRAW")


def main() -> None:
    print("downloading international results…")
    raw = urllib.request.urlopen(CSV_URL, timeout=60).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    rows.sort(key=lambda r: r["date"])

    ratings: dict[str, Glicko2Rating] = {}

    def get(t):
        return ratings.get(t, DEFAULT)

    def apply_day(matches):
        teams = {t for h, a, *_ in matches for t in (h, a)}
        cur = {t: get(t) for t in teams}
        per = {t: [] for t in teams}
        for h, a, o in matches:
            sh = 1.0 if o == "HOME" else (0.5 if o == "DRAW" else 0.0)
            per[h].append((cur[a].rating, cur[a].rd, sh))
            per[a].append((cur[h].rating, cur[h].rd, 1.0 - sh if o != "DRAW" else 0.5))
        for t in teams:
            ratings[t] = update_rating(cur[t], per[t], tau=TAU)

    # --- bootstrap from history before the tournament ---
    boot_by_date = defaultdict(list)
    wc_rows = []
    for r in rows:
        try:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
        except (ValueError, KeyError):
            continue
        d = r["date"]
        if d < WC_START:
            boot_by_date[d].append((r["home_team"], r["away_team"], _outcome(hs, as_)))
        elif WC_START <= d <= WC_END and r.get("tournament") == "FIFA World Cup":
            wc_rows.append(r)

    for d in sorted(boot_by_date):
        apply_day(boot_by_date[d])
    print(f"bootstrapped {len(ratings)} teams from {len(boot_by_date)} pre-WC match days")
    print(f"2022 World Cup matches found: {len(wc_rows)}\n")

    # --- walk-forward over the tournament ---
    wc_by_date = defaultdict(list)
    for r in wc_rows:
        wc_by_date[r["date"]].append(r)

    correct = brier_sum = n = 0
    draws_actual = 0
    table = []
    for d in sorted(wc_by_date):
        day_results = []
        for r in wc_by_date[d]:
            h, a = r["home_team"], r["away_team"]
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            actual = _outcome(hs, as_)
            neutral = (r.get("neutral", "").upper() == "TRUE")
            hf = 0.0 if neutral else HOST_MU
            p_home, p_draw, p_away = match_probabilities(get(h), get(a), home_field_mu=hf,
                                                         draw_rho=0.28)
            probs = {"HOME": p_home, "DRAW": p_draw, "AWAY": p_away}
            pred = max(probs, key=probs.get)
            n += 1
            correct += int(pred == actual)
            draws_actual += int(actual == "DRAW")
            brier_sum += sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2 for k in probs)
            table.append((d, f"{h} {hs}-{as_} {a}", pred, f"{probs[pred]:.0%}", actual,
                          "OK" if pred == actual else ""))
            day_results.append((h, a, actual))
        apply_day(day_results)

    print(f"{'date':<11}{'match':<34}{'pred':<6}{'p':<5}{'actual':<7}hit")
    for d, m, pred, p, actual, hit in table:
        print(f"{d:<11}{m[:33]:<34}{pred:<6}{p:<5}{actual:<7}{hit}")

    print("\n================ SUMMARY ================")
    print(f"matches:            {n}")
    print(f"favourite correct:  {correct}/{n} = {correct/n:.1%}")
    print(f"actual draws:       {draws_actual}/{n} = {draws_actual/n:.1%}")
    print(f"multiclass Brier:   {brier_sum/n:.3f}  (lower better; 0.667 = uniform 1/3)")
    print("note: ~50-55% favourite accuracy is the known ceiling for football models.")


if __name__ == "__main__":
    main()
