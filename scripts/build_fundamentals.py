"""Build data/fundamentals_2026.csv for the Klement fundamentals layer.

Pulls the qualified WC team list from football-data.org (needs
FOOTBALL_DATA_API_KEY in .env) and GDP per capita + population from the
World Bank API (no key). Average temperature and FIFA ranking points come
from the static tables below — they move slowly; refresh the FIFA points
snapshot before a new tournament (last refresh noted on the table).

Usage (from the repo root, venv active):
    python scripts/build_fundamentals.py                 # full build
    python scripts/build_fundamentals.py --teams "Brazil,Japan,..."  # offline list
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from betbot.config import get_settings  # noqa: E402
from betbot.exchanges.matcher import normalize  # noqa: E402

WORLD_BANK = "https://api.worldbank.org/v2/country/{iso3}/indicator/{ind}"
GDP_PC, POP = "NY.GDP.PCAP.CD", "SP.POP.TOTL"

HOSTS = {"united states", "usa", "canada", "mexico"}

# normalised team name -> ISO3 (World Bank country code). The UK home nations
# map to GBR — country-level economics is the best available proxy, same call
# Klement's country-level model makes.
ISO3 = {
    "algeria": "DZA", "argentina": "ARG", "australia": "AUS", "austria": "AUT",
    "belgium": "BEL", "bolivia": "BOL", "bosnia-herzegovina": "BIH",
    "bosnia herzegovina": "BIH", "bosnia and herzegovina": "BIH",
    "brazil": "BRA", "cameroon": "CMR", "canada": "CAN", "cape verde": "CPV",
    "cape verde islands": "CPV", "chile": "CHL", "colombia": "COL",
    "congo dr": "COD", "dr congo": "COD", "costa rica": "CRI",
    "croatia": "HRV", "curacao": "CUW", "czechia": "CZE", "czech republic": "CZE",
    "denmark": "DNK", "ecuador": "ECU", "egypt": "EGY", "england": "GBR",
    "finland": "FIN", "france": "FRA", "germany": "DEU", "ghana": "GHA",
    "greece": "GRC", "haiti": "HTI", "honduras": "HND", "hungary": "HUN",
    "iran": "IRN", "iraq": "IRQ", "ireland": "IRL", "italy": "ITA",
    "ivory coast": "CIV", "cote divoire": "CIV", "jamaica": "JAM",
    "japan": "JPN", "jordan": "JOR", "mali": "MLI", "mexico": "MEX",
    "morocco": "MAR", "netherlands": "NLD", "new zealand": "NZL",
    "nigeria": "NGA", "north macedonia": "MKD", "norway": "NOR",
    "panama": "PAN", "paraguay": "PRY", "peru": "PER", "poland": "POL",
    "portugal": "PRT", "qatar": "QAT", "romania": "ROU", "saudi arabia": "SAU",
    "scotland": "GBR", "senegal": "SEN", "serbia": "SRB", "slovakia": "SVK",
    "slovenia": "SVN", "south africa": "ZAF", "south korea": "KOR",
    "korea republic": "KOR", "spain": "ESP", "sweden": "SWE",
    "switzerland": "CHE", "tunisia": "TUN", "turkey": "TUR", "turkiye": "TUR",
    "ukraine": "UKR", "united arab emirates": "ARE", "united states": "USA",
    "usa": "USA", "uruguay": "URY", "uzbekistan": "UZB", "venezuela": "VEN",
    "wales": "GBR",
}

# Annual mean temperature, C (World Bank Climate Knowledge Portal, rounded).
AVG_TEMP_C = {
    "algeria": 23.0, "argentina": 14.8, "australia": 21.7, "austria": 7.1,
    "belgium": 10.5, "bolivia": 21.0, "bosnia-herzegovina": 10.0,
    "bosnia herzegovina": 10.0, "bosnia and herzegovina": 10.0,
    "brazil": 25.0, "cameroon": 24.6, "canada": -5.3, "cape verde": 23.3,
    "cape verde islands": 23.3, "chile": 9.3, "colombia": 24.5,
    "congo dr": 24.0, "dr congo": 24.0, "costa rica": 24.8,
    "croatia": 11.0, "curacao": 27.9, "czechia": 8.0, "czech republic": 8.0,
    "denmark": 8.5, "ecuador": 21.8, "egypt": 22.1, "england": 9.8,
    "finland": 2.1, "france": 11.5, "germany": 9.5, "ghana": 27.2,
    "greece": 15.5, "haiti": 24.9, "honduras": 23.5, "hungary": 10.5,
    "iran": 17.2, "iraq": 21.4, "ireland": 9.6, "italy": 13.5,
    "ivory coast": 26.3, "cote divoire": 26.3, "jamaica": 27.0,
    "japan": 11.2, "jordan": 18.3, "mali": 28.3, "mexico": 21.0,
    "morocco": 17.5, "netherlands": 10.0, "new zealand": 10.5,
    "nigeria": 26.8, "north macedonia": 9.8, "norway": 1.5,
    "panama": 25.5, "paraguay": 23.6, "peru": 19.6, "poland": 8.5,
    "portugal": 15.9, "qatar": 27.1, "romania": 9.5, "saudi arabia": 25.0,
    "scotland": 8.0, "senegal": 27.8, "serbia": 11.0, "slovakia": 6.8,
    "slovenia": 9.0, "south africa": 17.8, "south korea": 11.5,
    "korea republic": 11.5, "spain": 13.3, "sweden": 2.1,
    "switzerland": 5.5, "tunisia": 19.2, "turkey": 11.1, "turkiye": 11.1,
    "ukraine": 8.3, "united arab emirates": 27.0, "united states": 8.6,
    "usa": 8.6, "uruguay": 17.5, "uzbekistan": 12.9, "venezuela": 25.4,
    "wales": 9.2,
}

# FIFA men's ranking points snapshot. APPROXIMATE values, last refreshed
# 2026-06 from the most recent published ranking; update before reuse.
FIFA_POINTS = {
    "algeria": 1521, "argentina": 1867, "australia": 1554, "austria": 1580,
    "belgium": 1740, "bolivia": 1308, "bosnia-herzegovina": 1395,
    "bosnia herzegovina": 1395, "bosnia and herzegovina": 1395,
    "brazil": 1776, "cameroon": 1480, "canada": 1558, "cape verde": 1420,
    "cape verde islands": 1420, "chile": 1461, "colombia": 1679,
    "congo dr": 1430, "dr congo": 1430, "costa rica": 1431,
    "croatia": 1698, "curacao": 1305, "czechia": 1491, "czech republic": 1491,
    "denmark": 1627, "ecuador": 1589, "egypt": 1518, "england": 1820,
    "finland": 1350, "france": 1862, "germany": 1724, "ghana": 1450,
    "greece": 1498, "haiti": 1289, "honduras": 1380, "hungary": 1503,
    "iran": 1637, "iraq": 1413, "ireland": 1412, "italy": 1702,
    "ivory coast": 1487, "cote divoire": 1487, "jamaica": 1370,
    "japan": 1652, "jordan": 1389, "mali": 1460, "mexico": 1660,
    "morocco": 1694, "netherlands": 1758, "new zealand": 1390,
    "nigeria": 1481, "north macedonia": 1378, "norway": 1519,
    "panama": 1456, "paraguay": 1475, "peru": 1470, "poland": 1517,
    "portugal": 1778, "qatar": 1410, "romania": 1480, "saudi arabia": 1418,
    "scotland": 1480, "senegal": 1645, "serbia": 1497, "slovakia": 1470,
    "slovenia": 1462, "south africa": 1445, "south korea": 1574,
    "korea republic": 1574, "spain": 1880, "sweden": 1536,
    "switzerland": 1635, "tunisia": 1482, "turkey": 1551, "turkiye": 1551,
    "ukraine": 1535, "united arab emirates": 1382, "united states": 1673,
    "usa": 1673, "uruguay": 1679, "uzbekistan": 1437, "venezuela": 1476,
    "wales": 1535,
}

# Total squad market value in MILLIONS of EUR. APPROXIMATE Transfermarkt-style
# total-squad valuations, snapshotted 2026-06; well-known public figures rounded
# to round numbers. Squad value is a slow-moving prior — approximate is fine.
# The elite (Spain/France/England/Brazil/Portugal/Germany) cluster ~900M-1.1B;
# the minnows (Curacao, Cape Verde, etc.) sit ~15-40M. Refresh before reuse.
SQUAD_VALUE_EUR_M = {
    "algeria": 230, "argentina": 720, "australia": 110, "austria": 480,
    "belgium": 640, "bolivia": 25, "bosnia-herzegovina": 250,
    "bosnia herzegovina": 250, "bosnia and herzegovina": 250,
    "brazil": 1000, "cameroon": 320, "canada": 180, "cape verde": 40,
    "cape verde islands": 40, "chile": 130, "colombia": 430,
    "congo dr": 280, "dr congo": 280, "costa rica": 45,
    "croatia": 420, "curacao": 20, "czechia": 320, "czech republic": 320,
    "denmark": 560, "ecuador": 380, "egypt": 220, "england": 1500,
    "finland": 130, "france": 1400, "germany": 1000, "ghana": 290,
    "haiti": 30, "honduras": 30, "hungary": 230,
    "iran": 90, "iraq": 50, "ireland": 220, "italy": 720,
    "ivory coast": 360, "cote divoire": 360, "jamaica": 110,
    "japan": 360, "jordan": 25, "mali": 280, "mexico": 220,
    "morocco": 520, "netherlands": 880, "new zealand": 40,
    "nigeria": 470, "north macedonia": 110, "norway": 620,
    "panama": 35, "paraguay": 110, "peru": 80, "poland": 350,
    "portugal": 1100, "qatar": 35, "romania": 200, "saudi arabia": 50,
    "scotland": 290, "senegal": 600, "serbia": 480, "slovakia": 200,
    "slovenia": 220, "south africa": 90, "south korea": 200,
    "korea republic": 200, "spain": 1400, "sweden": 360,
    "switzerland": 400, "tunisia": 130, "turkey": 540, "turkiye": 540,
    "ukraine": 350, "united arab emirates": 30, "united states": 320,
    "usa": 320, "uruguay": 380, "uzbekistan": 60, "venezuela": 130,
    "wales": 280,
}


def _squad_value_eur(key: str) -> float:
    """Total squad market value in EUR for a normalised team name (0.0 if
    unknown — treated as no-signal downstream)."""
    return float(SQUAD_VALUE_EUR_M.get(key, 0)) * 1_000_000


async def _wc_team_names() -> list[str]:
    from betbot.data.football_data import FootballDataClient

    s = get_settings()
    async with FootballDataClient(
        api_key=s.football_data_api_key,
        base_url=s.football_data_base_url,
        rate_limit_per_min=s.football_data_rate_limit_per_min,
    ) as client:
        teams = await client.list_competition_teams("WC")
    return [t["name"] for t in teams if t.get("name")]


async def _world_bank_latest(client: httpx.AsyncClient, iso3: str, ind: str) -> float:
    """Most recent non-null value for an indicator. Tries mrnev=1 first, then
    a date-range query; retries — the World Bank API throws sporadic 400s."""
    url = WORLD_BANK.format(iso3=iso3, ind=ind)
    queries = [
        {"format": "json", "mrnev": 1},
        {"format": "json", "date": "2015:2026", "per_page": 20},
    ]
    last_err: Exception | None = None
    for attempt in range(3):
        for params in queries:
            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                body = r.json()
                rows = body[1] if len(body) > 1 and body[1] else []
                for row in rows:  # newest first; skip null years
                    if row.get("value") is not None:
                        return float(row["value"])
            except (httpx.HTTPError, ValueError) as e:  # noqa: PERF203
                last_err = e
        await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"World Bank has no {ind} for {iso3}: {last_err}")


async def build(team_names: list[str], out_path: Path) -> None:
    missing = sorted(
        n for n in team_names
        if normalize(n) not in ISO3
        or normalize(n) not in AVG_TEMP_C
        or normalize(n) not in FIFA_POINTS
    )
    if missing:
        sys.exit(
            "Missing static data for: " + ", ".join(missing)
            + "\nAdd them to ISO3 / AVG_TEMP_C / FIFA_POINTS in this script."
        )

    rows = []
    async with httpx.AsyncClient(timeout=30) as client:
        for name in sorted(team_names):
            key = normalize(name)
            iso3 = ISO3[key]
            gdp = await _world_bank_latest(client, iso3, GDP_PC)
            pop = await _world_bank_latest(client, iso3, POP)
            rows.append({
                "team": name,
                "iso3": iso3,
                "gdp_per_capita_usd": round(gdp, 2),
                "population": int(pop),
                "avg_temp_c": AVG_TEMP_C[key],
                "fifa_points": FIFA_POINTS[key],
                "host": str(key in HOSTS).lower(),
                "squad_value_eur": _squad_value_eur(key),
            })
            print(f"  {name:24s} {iso3}  gdp_pc={gdp:>10.0f}  pop={int(pop):>11d}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} teams -> {out_path}")


def augment_squad_value(path: Path) -> None:
    """Add (or refresh) the squad_value_eur column on an EXISTING CSV without
    touching the network. Reads the current rows, fills squad_value_eur from
    SQUAD_VALUE_EUR_M keyed on the normalised team name, and rewrites the file.

    This is how the committed data/fundamentals_2026.csv gets the new column
    without re-running the World Bank / football-data pull.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    if "squad_value_eur" not in fields:
        fields.append("squad_value_eur")
    for row in rows:
        row["squad_value_eur"] = _squad_value_eur(normalize(row["team"]))

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Augmented squad_value_eur for {len(rows)} teams -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--teams",
        help="Comma-separated team list (skips the football-data.org pull).",
    )
    ap.add_argument(
        "--out", default="data/fundamentals_2026.csv", type=Path,
        help="Output CSV path (default: data/fundamentals_2026.csv)",
    )
    ap.add_argument(
        "--augment-only", action="store_true",
        help="No network: just add/refresh the squad_value_eur column on --out.",
    )
    args = ap.parse_args()

    if args.augment_only:
        augment_squad_value(args.out)
        return

    if args.teams:
        names = [t.strip() for t in args.teams.split(",") if t.strip()]
    else:
        names = asyncio.run(_wc_team_names())
        print(f"football-data.org returned {len(names)} WC teams")

    asyncio.run(build(names, args.out))


if __name__ == "__main__":
    main()
