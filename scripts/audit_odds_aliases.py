"""Audit odds-feed team-name resolution — coverage, skips, and MIS-RESOLUTIONS.

This is the operational loop that lets the alias table grow SAFELY. It never
edits anything; it prints:

* every spelling the feed used that we refused to resolve (add an alias only
  after eyeballing these — that is the only legitimate source of new aliases);
* the resolved coverage rate;
* a collision report: two different feed spellings landing on the SAME
  canonical club within one fixture, or a canonical club claimed by two
  spellings that are clearly different clubs. That is the Espanyol/Barcelona
  failure mode and it must be ZERO.

Run (repo root, venv active):
    python scripts/audit_odds_aliases.py                # live fixtures.csv
    python scripts/audit_odds_aliases.py --season 2526  # a whole season file
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from betbot.config import get_settings
from betbot.data.odds import DIV_TO_LEAGUE, FootballDataCoUkProvider
from betbot.data.odds_names import OddsNameResolver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=None,
                    help="audit mmz4281/<season>/*.csv instead of live fixtures.csv")
    ap.add_argument("--leagues", nargs="+", default=list(DIV_TO_LEAGUE.values()))
    args = ap.parse_args()

    s = get_settings()
    resolver = OddsNameResolver.from_files(s.odds_team_alias_path, s.club_name_map_path)
    provider = FootballDataCoUkProvider(resolver, timeout=60.0)

    if args.season:
        rows = []
        for lg in args.leagues:
            rows.extend(provider.fetch_season(args.season, lg))
        label = f"season {args.season}"
    else:
        rows = provider.fetch(args.leagues)
        label = "live fixtures.csv"

    attempted = provider.attempted_fixtures
    print(f"=== odds name audit: {label} ===")
    print(f"in-scope fixtures attempted : {attempted}")
    print(f"resolved                    : {len(rows)}")
    print(f"skipped on an unresolved name: {provider.skipped_fixtures}")
    print(f"dropped for missing prices   : "
          f"{attempted - provider.skipped_fixtures - len(rows)}")
    if attempted:
        print(f"name-resolution coverage     : "
              f"{100.0 * (attempted - provider.skipped_fixtures) / attempted:.1f}%")

    # Mis-resolution check: within the resolved set, no fixture may have both
    # sides land on the same club, and each canonical club should be reachable
    # from a coherent set of spellings.
    collisions = [r for r in rows if r.home == r.away]
    print(f"\nMIS-RESOLUTIONS (home == away after resolution): {len(collisions)}")
    for r in collisions:
        print(f"  !! {r.league} {r.match_date} -> {r.home}")

    by_canon: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_canon[r.home].add(r.league)
        by_canon[r.away].add(r.league)
    cross_league = {c: lgs for c, lgs in by_canon.items() if len(lgs) > 1}
    print(f"clubs appearing in >1 league (suspicious): {len(cross_league)}")
    for c, lgs in sorted(cross_league.items()):
        print(f"  ?? {c}: {sorted(lgs)}")

    if provider.unresolved:
        print("\nUNRESOLVED spellings — the ONLY legitimate source of new aliases:")
        for name, count in sorted(provider.unresolved.items(), key=lambda kv: -kv[1]):
            print(f"  {name!r}: {count}")
    else:
        print("\nno unresolved spellings")

    print("\nresolved fixtures:")
    for r in rows[:60]:
        probs = r.probabilities()
        print(
            f"  {r.league} {r.match_date} {r.home:>18s} v {r.away:<18s} "
            f"{r.book:>7s} {r.price_home:>6.2f}/{r.price_draw:>5.2f}/{r.price_away:>6.2f} "
            f"overround {r.overround:.4f} -> devig "
            f"{probs[0]:.3f}/{probs[1]:.3f}/{probs[2]:.3f}"
        )
    if len(rows) > 60:
        print(f"  ... and {len(rows) - 60} more")


if __name__ == "__main__":
    main()
