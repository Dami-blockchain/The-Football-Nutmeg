"""Seed margin-of-victory Glicko-2 ratings -> data/glicko_mov.json (challenger).

Mirrors scripts/seed_glicko.py exactly — same CSV, same 6-year window, same
rating-period grouping, same WC-name aliasing — but replays through
``update_rating_mov`` so the goal margin scales each rating move. The output is
written to a JSON file (NOT the live ratings table) so the engine can dual-log
the MOV challenger without touching the live Glicko ratings.

Run: ``python scripts/seed_glicko_mov.py``  (idempotent; overwrites the JSON).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from betbot.config import get_settings
from betbot.exchanges.matcher import TeamAliasResolver, normalize
from betbot.strategy.glicko import Glicko2Rating
from betbot.strategy.glicko_mov import update_rating_mov


def _outcome(hs: int, as_: int) -> str:
    return "HOME" if hs > as_ else ("AWAY" if as_ > hs else "DRAW")


def main() -> None:
    s = get_settings()
    csv_path = Path(s.glicko_results_csv)
    if not (csv_path and csv_path.exists()):
        print(f"no results CSV at {csv_path} — cannot build MOV ratings")
        return

    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                d = r["date"]
                hs, as_ = int(float(r["home_score"])), int(float(r["away_score"]))
            except (KeyError, ValueError):
                continue
            rows.append((d, r["home_team"].strip(), r["away_team"].strip(), hs, as_))
    rows.sort(key=lambda x: x[0])
    if rows:
        cutoff = str(int(rows[-1][0][:4]) - 6)
        rows = [r for r in rows if r[0][:4] >= cutoff]

    by_date: dict[str, list] = defaultdict(list)
    for d, h, a, hs, as_ in rows:
        by_date[d].append((h, a, hs, as_))

    default = Glicko2Rating(s.glicko_default_rating, s.glicko_default_rd, s.glicko_default_vol)
    ratings: dict[str, Glicko2Rating] = {}
    for d in sorted(by_date):
        matches = by_date[d]
        teams = {t for h, a, _, _ in matches for t in (h, a)}
        cur = {t: ratings.get(t, default) for t in teams}
        per: dict[str, list] = {t: [] for t in teams}
        for h, a, hs, as_ in matches:
            o = _outcome(hs, as_)
            sh = 1.0 if o == "HOME" else (0.5 if o == "DRAW" else 0.0)
            gd = abs(hs - as_)
            per[h].append((cur[a].rating, cur[a].rd, sh, gd))
            per[a].append((cur[h].rating, cur[h].rd,
                           1.0 - sh if o != "DRAW" else 0.5, gd))
        for t in teams:
            ratings[t] = update_rating_mov(cur[t], per[t], tau=s.glicko_tau, period=d)

    out = {n: [r.rating, r.rd, r.volatility] for n, r in ratings.items()}

    # Alias WC fixture names (football-data spelling) to the dataset rating,
    # exactly like seed_glicko's _alias_wc_teams, so the engine resolves them.
    fund = Path("data/fundamentals_2026.csv")
    aliased = 0
    if fund.exists():
        wc_names = [row["team"] for row in csv.DictReader(fund.open())]
        resolver = TeamAliasResolver.from_yaml("config/team_aliases.yaml")
        dataset_names = list(ratings)
        norm_existing = {normalize(n) for n in dataset_names}
        for wc in wc_names:
            if normalize(wc) in norm_existing:
                continue
            m = resolver.match(wc, dataset_names)
            if m is not None:
                out[wc] = [ratings[m].rating, ratings[m].rd, ratings[m].volatility]
                aliased += 1

    dest = Path(s.glicko_mov_path)
    dest.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {len(out)} MOV ratings ({aliased} WC-aliased) -> {dest}")


if __name__ == "__main__":
    main()
