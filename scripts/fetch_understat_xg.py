"""Fetch true match xG from Understat via the Apify managed scraper.

Actor: ``constructive_calm/understat-football-analytics`` (mode=league_matches)
-> one row per played match with home/away xG. Big-5 leagues, 2014/15+.
Pay-per-event ($0.004/match); every run is capped with ``maxTotalChargeUsd``
so a wrong input can never overspend. The token is read straight from
``~/tfsm/.env`` (APIFY_TOKEN) so it never transits code or argv.

Writes data/club_xg.csv:  date,home_team,away_team,home_xg,away_xg,
home_goals,away_goals,league   (league = our football-data.org code)

Run (repo root, venv active):
    python scripts/fetch_understat_xg.py --probe            # EPL 2024, cap $2
    python scripts/fetch_understat_xg.py --seasons 2022 2023 2024 2025 --cap 28
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

ACTOR = "constructive_calm~understat-football-analytics"
API = "https://api.apify.com/v2"

# Understat league code -> our football-data.org competition code.
UND_TO_LEAGUE = {
    "EPL": "PL", "La_liga": "PD", "Bundesliga": "BL1", "Serie_A": "SA", "Ligue_1": "FL1",
}


def _token() -> str:
    env = pathlib.Path(".env").read_text()
    m = re.search(r"^\s*APIFY_TOKEN\s*=\s*(\S+)", env, re.M)
    if not m:
        raise SystemExit("APIFY_TOKEN not in .env")
    return m.group(1).strip().strip("\"'")


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return json.load(r)


def _get(url: str) -> object:
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
        return json.load(r)


def run_actor(token: str, actor_input: dict, cap_usd: float, *, poll: int = 5,
              timeout_s: int = 900) -> tuple[list[dict], dict]:
    """Start the actor async, poll to completion, return (items, run-info)."""
    start = _post(
        f"{API}/acts/{ACTOR}/runs?token={token}&maxTotalChargeUsd={cap_usd}",
        actor_input,
    )["data"]
    run_id = start["id"]
    waited = 0
    while True:
        info = _get(f"{API}/actor-runs/{run_id}?token={token}")["data"]
        status = info["status"]
        if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
            break
        if waited >= timeout_s:
            print(f"  timeout after {waited}s (status {status})")
            break
        time.sleep(poll)
        waited += poll
    ds = info.get("defaultDatasetId")
    items = _get(f"{API}/datasets/{ds}/items?token={token}&clean=true") if ds else []
    return items, info


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_rows(items: list[dict], league: str) -> list[dict]:
    """Map the actor payload (homeTeam/awayTeam = {title,goals,xg}) to our schema.

    Only finished matches (isResult=True) are kept. Understat's own forecast
    (winHome/draw/winAway) is carried through as a bonus signal.
    """
    out = []
    for it in items:
        if not it.get("isResult"):
            continue
        h = it.get("homeTeam") or {}
        a = it.get("awayTeam") or {}
        hx, ax = _num(h.get("xg")), _num(a.get("xg"))
        home, away = (h.get("title") or "").strip(), (a.get("title") or "").strip()
        dt = (it.get("datetime") or "")[:10]
        if None in (hx, ax) or not (home and away and dt):
            continue
        fc = it.get("forecast") or {}
        out.append({
            "date": dt, "home_team": home, "away_team": away,
            "home_xg": hx, "away_xg": ax,
            "home_goals": _num(h.get("goals")), "away_goals": _num(a.get("goals")),
            "league": league,
            "f_home": _num(fc.get("winHome")), "f_draw": _num(fc.get("draw")),
            "f_away": _num(fc.get("winAway")),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true", help="EPL 2024 only, cap $2")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--leagues", nargs="+", default=list(UND_TO_LEAGUE))
    ap.add_argument("--cap", type=float, default=28.0)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/club_xg.csv"))
    args = ap.parse_args()
    token = _token()

    if args.probe:
        items, info = run_actor(token, {"mode": "league_matches",
                                        "leagues": ["EPL"], "seasons": [2024]}, cap_usd=2.0)
        print(f"probe status={info.get('status')} items={len(items)} "
              f"charge_usd≈{info.get('usageTotalUsd')}")
        if items:
            print("first item keys:", sorted(items[0].keys()))
            print("sample:", json.dumps(items[0], default=str)[:400])
            rows = _norm_rows(items, "PL")
            print(f"normalised rows: {len(rows)}")
            if rows:
                print("normalised sample:", rows[0])
        return

    all_rows: list[dict] = []
    per_call_cap = round(args.cap / max(len(args.leagues), 1), 2)
    for und, league in UND_TO_LEAGUE.items():
        if und not in args.leagues and league not in args.leagues:
            continue
        items, info = run_actor(token, {"mode": "league_matches",
                                        "leagues": [und], "seasons": args.seasons},
                                cap_usd=per_call_cap)
        rows = _norm_rows(items, league)
        all_rows.extend(rows)
        print(f"  {und}->{league}: {len(rows)} matches "
              f"(status {info.get('status')}, ~${info.get('usageTotalUsd')})")

    if not all_rows:
        raise SystemExit("no rows fetched — check probe output / input schema")
    all_rows.sort(key=lambda r: (r["date"], r["league"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    teams = {r["home_team"] for r in all_rows} | {r["away_team"] for r in all_rows}
    yrs = sorted({r["date"][:4] for r in all_rows})
    print(f"\nwrote {args.out}: {len(all_rows)} matches, {len(teams)} clubs, years={yrs}")


if __name__ == "__main__":
    main()
