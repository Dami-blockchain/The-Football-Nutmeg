"""Fit the Dixon-Coles model on international results -> data/dc_params.json.

Data: martj42/international_results results.csv (public; same source as
wc2022_backtest.py and the Glicko seed). Team parameters are shrunk toward
priors from the Klement fundamentals layer (data/fundamentals_2026.csv) so
sparse-data qualifiers start from a structurally sensible strength.

Run (repo root, venv active; takes a couple of minutes — pure-Python MLE):
    python scripts/fit_dixon_coles.py
    python scripts/fit_dixon_coles.py --csv path/to/results.csv   # offline
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betbot.exchanges.matcher import normalize  # noqa: E402
from betbot.strategy.dixon_coles import (  # noqa: E402
    DCMatch,
    fit,
    match_probabilities,
    priors_from_scores,
)
from betbot.strategy.fundamentals import composite_scores, load_fundamentals  # noqa: E402

CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
YEARS_BACK = 8


def _load_matches(raw: str) -> list[DCMatch]:
    rows = list(csv.DictReader(io.StringIO(raw)))
    rows.sort(key=lambda r: r["date"])
    latest = int(rows[-1]["date"][:4])
    out: list[DCMatch] = []
    for r in rows:
        try:
            d = date.fromisoformat(r["date"])
            hs, as_ = int(float(r["home_score"])), int(float(r["away_score"]))
        except (KeyError, ValueError):
            continue
        if d.year < latest - YEARS_BACK:
            continue
        out.append(DCMatch(
            date=d,
            home=normalize(r["home_team"]),
            away=normalize(r["away_team"]),
            home_goals=hs,
            away_goals=as_,
            neutral=(r.get("neutral", "").strip().upper() == "TRUE"),
            friendly=(r.get("tournament", "").strip().lower() == "friendly"),
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="Local results.csv (skips the download).")
    ap.add_argument("--fundamentals", default="data/fundamentals_2026.csv")
    ap.add_argument("--out", default="data/dc_params.json", type=Path)
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args()

    if args.csv:
        raw = Path(args.csv).read_text(encoding="utf-8")
    else:
        print("downloading international results…")
        raw = urllib.request.urlopen(CSV_URL, timeout=60).read().decode("utf-8")
    matches = _load_matches(raw)
    print(f"{len(matches)} matches in the last {YEARS_BACK} years")

    priors = {}
    fund_path = Path(args.fundamentals)
    if fund_path.exists():
        scores = composite_scores(load_fundamentals(fund_path))
        priors = priors_from_scores(scores)
        print(f"fundamentals priors for {len(priors)} teams")
    else:
        print(f"WARNING: {fund_path} missing — fitting without fundamentals priors")

    params = fit(matches, priors=priors, iterations=args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(params.to_json(), encoding="utf-8")
    print(f"wrote {args.out}  (teams={len(params.teams)}, "
          f"base_mu={params.base_mu:.3f}, home_adv={params.home_adv:.3f}, "
          f"rho={params.rho:.2f})")

    strength = {n: t.attack + t.defence for n, t in params.teams.items()}
    top = sorted(strength.items(), key=lambda kv: -kv[1])[:10]
    print("\nstrongest (attack+defence):")
    for n, s in top:
        t = params.teams[n]
        print(f"  {n:20s} atk={t.attack:+.2f} dfn={t.defence:+.2f} ({s:+.2f})")
    ph, pd, pa = match_probabilities(params, normalize("Brazil"), normalize("Bolivia"))
    print(f"\nsanity Brazil v Bolivia (neutral): {ph:.2f}/{pd:.2f}/{pa:.2f}")


if __name__ == "__main__":
    main()
