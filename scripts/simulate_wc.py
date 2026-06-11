"""Monte Carlo the 2026 World Cup -> ranked outright/futures table.

Loads the fitted Dixon-Coles artifact (scripts/fit_dixon_coles.py writes it)
and the public group draw (data/wc2026_groups.csv, group/team columns), runs
the pure simulator in betbot.strategy.simulate, and prints win/final/semi
probabilities per team. Hosts (USA/Mexico/Canada) get the DC home-field
boost in the group stage only — see the simulate module docstring.

Run (repo root, venv active; 20k sims take a few seconds — distributions are
cached per pairing, see _MatchSampler):
    python scripts/simulate_wc.py
    python scripts/simulate_wc.py --sims 5000 --seed 7 --penalty-randomness 1.0
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betbot.exchanges.matcher import normalize  # noqa: E402
from betbot.strategy.dixon_coles import DCParams  # noqa: E402
from betbot.strategy.simulate import WC2026_BRACKET, simulate_tournament  # noqa: E402

HOST_NATIONS = ("United States", "Mexico", "Canada")


def _load_groups(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """CSV (group,team) -> ({group: [normalized names]}, {normalized: display}).

    Names are matcher-normalised so they line up with the DC artifact, which
    scripts/fit_dixon_coles.py keys the same way.
    """
    groups: dict[str, list[str]] = {}
    display: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["team"].strip()
            norm = normalize(name)
            groups.setdefault(row["group"].strip(), []).append(norm)
            display[norm] = name
    return groups, display


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dc-params", default="data/dc_params.json", type=Path)
    ap.add_argument("--groups", default="data/wc2026_groups.csv", type=Path)
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--penalty-randomness", type=float, default=0.5,
        help="1.0 = shootouts are coin flips, 0.0 = decided by DC strength",
    )
    args = ap.parse_args()

    params = DCParams.from_json(args.dc_params.read_text(encoding="utf-8"))
    groups, display = _load_groups(args.groups)

    unknown = sorted(
        t for ts in groups.values() for t in ts if t not in params.teams
    )
    if unknown:
        print(f"WARNING: no DC params for {len(unknown)} teams "
              f"(neutral prior used): {', '.join(unknown)}")

    probs = simulate_tournament(
        groups,
        params,
        random.Random(args.seed),
        bracket=WC2026_BRACKET,
        n_sims=args.sims,
        hosts=frozenset(normalize(n) for n in HOST_NATIONS),
        penalty_randomness=args.penalty_randomness,
    )

    ranked = sorted(probs.values(), key=lambda p: (-p.win, -p.final, p.team))
    print(f"\n2026 World Cup — {args.sims} sims, seed {args.seed}, "
          f"penalty randomness {args.penalty_randomness}\n")
    print(f"{'#':>2}  {'team':24s} {'win%':>6} {'final%':>7} {'semi%':>6}")
    for i, p in enumerate(ranked, 1):
        print(f"{i:>2}  {display.get(p.team, p.team):24s} "
              f"{100 * p.win:6.2f} {100 * p.final:7.2f} {100 * p.semi:6.2f}")


if __name__ == "__main__":
    main()
