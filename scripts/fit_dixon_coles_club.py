"""Fit the Dixon-Coles goal model on CLUB results -> data/dc_params_club.json.

Club sibling of ``scripts/fit_dixon_coles.py``. Differences from the
internationals fit:

* data is ``data/club_results.csv`` (football-data.co.uk), not the nations set;
* every league match is a genuine home fixture, so ``neutral=False`` — the
  model LEARNS a real club home advantage (nations are mostly neutral-venue);
* nothing is a "friendly", and there are no fundamentals priors (that CSV is
  World-Cup specific), so team strengths come purely from the goal data — which
  is dense for clubs (36-38 games/team/season), unlike sparse nation data.

Team params are only meaningful *within* a league (clubs don't play across
leagues in this dataset), which is exactly how the club engine uses them — it
prices domestic league fixtures only. Run (repo root, venv active):

    python scripts/fit_dixon_coles_club.py
    python scripts/fit_dixon_coles_club.py --years-back 5 --iterations 300
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betbot.exchanges.matcher import normalize  # noqa: E402
from betbot.strategy.dixon_coles import DCMatch, fit, match_probabilities  # noqa: E402


def _load_matches(path: Path, years_back: int) -> list[DCMatch]:
    rows = list(csv.DictReader(path.open()))
    rows.sort(key=lambda r: r["date"])
    if not rows:
        return []
    latest = int(rows[-1]["date"][:4])
    out: list[DCMatch] = []
    for r in rows:
        try:
            d = date.fromisoformat(r["date"])
            hs, as_ = int(float(r["home_score"])), int(float(r["away_score"]))
        except (KeyError, ValueError):
            continue
        if d.year < latest - years_back:
            continue
        out.append(DCMatch(
            date=d,
            home=normalize(r["home_team"]),
            away=normalize(r["away_team"]),
            home_goals=hs,
            away_goals=as_,
            neutral=False,   # every league match has a real home side
            friendly=False,
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/club_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/dc_params_club.json"))
    ap.add_argument("--years-back", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=250)
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} missing — run scripts/fetch_club_results.py first")

    matches = _load_matches(args.csv, args.years_back)
    print(f"{len(matches)} club matches in the last {args.years_back} years")

    params = fit(matches, priors={}, iterations=args.iterations)
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
        print(f"  {n:22s} atk={t.attack:+.2f} dfn={t.defence:+.2f} ({s:+.2f})")

    ph, pd_, pa = match_probabilities(
        params, normalize("Man City"), normalize("Southampton"), home_field=True
    )
    print(f"\nsanity Man City (H) v Southampton: {ph:.2f}/{pd_:.2f}/{pa:.2f}")


if __name__ == "__main__":
    main()
