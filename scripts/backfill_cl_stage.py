"""Backfill the CL ``stage`` column on existing prediction rows (operator-run).

Why this exists
---------------
``feat/cl-stage-logging`` adds a nullable ``stage`` column to ``predictions``
and ``prediction_outcomes`` and populates it going forward for Champions League
fixtures (LEAGUE_STAGE / PLAYOFFS / LAST_16 / QUARTER_FINALS / SEMI_FINALS /
FINAL). Rows written BEFORE the deploy carry NULL. This script fills them in
from football-data.org — the same free feed the live loop polls — so the
historical knockout vs league-phase split is complete.

It is deliberately a SEPARATE, operator-run script: the live startup path stays
write-free. It is DRY-RUN by default and only writes with ``--apply``.

IMPORTANT: deploy the ``stage`` column FIRST. The column is created at
service startup (the additive boot migration), so you must merge and RESTART
the services BEFORE running this script — even for a dry-run, which reads the
column. If the column is absent the script exits immediately with a message.

Source is cheap: one ``/competitions/CL/matches?season=<yr>`` call per season
returns every fixture that season with its ``id`` and ``stage`` in a single
request, so the whole backfill costs a handful of calls, not one per row.

Run (repo root, live venv active)::

    # see what WOULD change + the current KO/league split, write nothing:
    python scripts/backfill_cl_stage.py

    # actually write stage onto the live DB (operator only):
    python scripts/backfill_cl_stage.py --apply

    # limit / override the seasons probed (season = start year):
    python scripts/backfill_cl_stage.py --seasons 2023 2024 2025 2026

    # just print the knockout-vs-league accuracy split, no fetching:
    python scripts/backfill_cl_stage.py --report-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import inspect, select, text, update

from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient, extract_stage
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionOutcome, PredictionRow

_KNOCKOUT = {"PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"}


def _cl_years_in_db() -> list[int]:
    """Distinct kickoff years of CL rows, so we probe only seasons we hold.

    football-data's ``season`` is the start year; a match played in Jan-Jul
    belongs to the PREVIOUS start year, so we probe year and year-1 for safety.
    """
    years: set[int] = set()
    with session_scope() as s:
        for row in s.execute(
            select(PredictionRow.kickoff).where(PredictionRow.competition_code == "CL")
        ):
            ko = row[0]
            if ko is not None:
                years.add(ko.year)
                years.add(ko.year - 1)
        for row in s.execute(
            select(PredictionOutcome.kickoff).where(
                PredictionOutcome.competition_code == "CL"
            )
        ):
            ko = row[0]
            if ko is not None:
                years.add(ko.year)
                years.add(ko.year - 1)
    return sorted(y for y in years if y >= 2000)


async def _fetch_stage_map(seasons: list[int]) -> dict[int, str]:
    """``{fixture_id: stage}`` for every CL match across the given seasons."""
    s = get_settings()
    out: dict[int, str] = {}
    async with FootballDataClient(
        api_key=s.football_data_api_key,
        base_url=s.football_data_base_url,
        rate_limit_per_min=s.football_data_rate_limit_per_min,
    ) as client:
        for year in seasons:
            try:
                data = await client._get(
                    "/competitions/CL/matches", params={"season": year}
                )
            except Exception as e:  # noqa: BLE001 — a missing season must not abort
                print(f"  season {year}: fetch failed ({e})")
                continue
            matches = [m for m in (data.get("matches") or []) if isinstance(m, dict)]
            n = 0
            for m in matches:
                mid = m.get("id")
                stage = extract_stage(m, "CL")
                if mid is not None and stage is not None:
                    out[int(mid)] = stage
                    n += 1
            print(f"  season {year}: {n} CL fixtures with a stage")
    return out


def _report_split() -> None:
    """Print the knockout vs league-phase accuracy split for CL outcomes."""
    print("\n=== CL record by stage (prediction_outcomes) ===")
    with session_scope() as s:
        rows = s.execute(
            text(
                "SELECT COALESCE(stage, '(null)') AS stage, "
                "COUNT(*) AS n, SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS hits "
                "FROM prediction_outcomes WHERE competition_code = 'CL' "
                "GROUP BY COALESCE(stage, '(null)') ORDER BY n DESC"
            )
        ).all()
    if not rows:
        print("  no CL outcomes yet.")
        return
    ko_n = ko_h = lg_n = lg_h = 0
    for stage, n, hits in rows:
        hits = hits or 0
        rate = (hits / n * 100) if n else 0.0
        print(f"  {stage:<16} n={n:<5} hit={hits:<5} {rate:5.1f}%")
        if stage in _KNOCKOUT:
            ko_n += n
            ko_h += hits
        elif stage == "LEAGUE_STAGE":
            lg_n += n
            lg_h += hits
    print("  " + "-" * 40)
    if lg_n:
        print(f"  LEAGUE PHASE     n={lg_n:<5} hit={lg_h:<5} {lg_h / lg_n * 100:5.1f}%")
    if ko_n:
        print(f"  KNOCKOUT         n={ko_n:<5} hit={ko_h:<5} {ko_h / ko_n * 100:5.1f}%")


def _backfill(stage_map: dict[int, str], apply: bool) -> None:
    for model, label in ((PredictionRow, "predictions"), (PredictionOutcome, "prediction_outcomes")):
        with session_scope() as s:
            rows = s.execute(
                select(model.fixture_id).where(
                    model.competition_code == "CL", model.stage.is_(None)
                )
            ).all()
            fixture_ids = {r[0] for r in rows}
            matched = {fid: stage_map[fid] for fid in fixture_ids if fid in stage_map}
            print(
                f"{label}: {len(fixture_ids)} CL rows with NULL stage, "
                f"{len(matched)} resolvable from the provider"
            )
            if apply and matched:
                for fid, stage in matched.items():
                    s.execute(
                        update(model)
                        .where(model.fixture_id == fid, model.stage.is_(None))
                        .values(stage=stage)
                    )
                print(f"  -> wrote stage to {len(matched)} {label} fixtures")
            elif matched:
                print(f"  (dry-run; re-run with --apply to write {len(matched)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Restart the services BEFORE running this (even a dry-run): the "
            "stage column is created at service startup. Absent column -> exit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=None, help="DB path (default: settings.db_path)")
    ap.add_argument("--seasons", type=int, nargs="*", default=None,
                    help="season start years to probe (default: derived from DB)")
    ap.add_argument("--apply", action="store_true", help="write changes (default dry-run)")
    ap.add_argument("--report-only", action="store_true",
                    help="only print the KO/league split, no fetch or write")
    args = ap.parse_args()

    settings = get_settings()
    engine = init_engine(args.db or settings.db_path)
    cols = {c["name"] for c in inspect(engine).get_columns("predictions")}
    if "stage" not in cols:
        sys.exit("stage column absent: deploy + restart services first")

    if args.report_only:
        _report_split()
        return

    seasons = args.seasons or _cl_years_in_db()
    if not seasons:
        print("No CL rows in the DB (nothing to probe).")
        _report_split()
        return
    print(f"Probing CL seasons: {seasons}")
    stage_map = asyncio.run(_fetch_stage_map(seasons))
    print(f"Provider returned stage for {len(stage_map)} CL fixtures.\n")

    _backfill(stage_map, apply=args.apply)
    _report_split()


if __name__ == "__main__":
    main()
