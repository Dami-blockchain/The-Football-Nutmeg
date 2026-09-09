"""football-data.org fallback for CURRENT-SEASON club results.

Guards the outage that froze the club learning loop: football-data.co.uk went
HTTP 503 for days, and the fetch script both (a) exited 1 and (b) would have
rewritten the whole CSV empty. These tests pin the fix:

* the football-data.org fallback fires when .co.uk cannot serve the current
  season, and NOT when it can;
* a total .co.uk outage never drops historical rows or their closing odds;
* a club the resolver cannot name-map is KEPT (never silently vanishes) and is
  surfaced in the coverage report.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "fcr_under_test", _REPO / "scripts" / "fetch_club_results.py"
)
fcr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fcr)


# ----------------------------------------------------------------------
# season partitioning
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "date_iso,code",
    [
        ("2026-08-15", "2627"),
        ("2026-12-31", "2627"),
        ("2027-05-20", "2627"),
        ("2026-03-01", "2526"),
        ("2026-07-01", "2627"),
        ("2026-06-30", "2526"),
    ],
)
def test_season_code(date_iso, code):
    assert fcr._season_code(date_iso) == code


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
_COUK_CSV = (
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,PSCH,PSCD,PSCA\n"
    "31/08/2026,Arsenal,Chelsea,2,1,1.80,3.50,4.20\n"
)


class _FakeFDClient:
    """Async-context stand-in for FootballDataClient."""

    def __init__(self, matches_by_league):
        self._m = matches_by_league

    def __call__(self, *a, **k):  # constructed as FootballDataClient(...)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def list_matches(self, league, date_from, date_to, *, status=None):
        return self._m.get(league, [])


def _fd_match(home, away, hs, as_, date="2026-09-06"):
    return {
        "utcDate": f"{date}T14:00:00Z",
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": {"fullTime": {"home": hs, "away": as_}},
    }


# ----------------------------------------------------------------------
# 1. primary wins when available -> fallback NOT used
# ----------------------------------------------------------------------
def test_primary_wins_when_available(monkeypatch, tmp_path):
    out = tmp_path / "club.csv"

    # Serve ALL five current-season divisions so coverage is complete and the
    # per-league fallback (FIX B) has nothing missing to fetch.
    monkeypatch.setattr(
        fcr, "_fetch",
        lambda url, timeout: _COUK_CSV if "2627" in url else None,
    )

    def _boom(*a, **k):  # fallback must never be consulted here
        raise AssertionError("fallback fired even though .co.uk served 2627")

    monkeypatch.setattr(fcr, "_fetch_fallback_current", _boom)
    monkeypatch.setattr(fcr, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(
        "sys.argv",
        ["fetch", "--out", str(out), "--seasons", "2627", "--timeout", "5"],
    )

    fcr.main()

    body = out.read_text()
    assert "Arsenal,Chelsea,2,1" in body
    assert "1.8" in body  # closing odds preserved from .co.uk
    import json
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["couk_has_current"] is True
    assert rep["fallback_used"] is False


# ----------------------------------------------------------------------
# 2. fallback fires when primary unreachable
# ----------------------------------------------------------------------
def test_fallback_fires_when_primary_unreachable(monkeypatch, tmp_path):
    out = tmp_path / "club.csv"
    monkeypatch.setattr(fcr, "_fetch", lambda url, timeout: None)  # all 503

    called = {}

    def _fake_fb(dataset_names, present_keys, leagues=None):
        called["yes"] = True
        rows = [{
            "date": "2026-09-06", "home_team": "Arsenal", "away_team": "Chelsea",
            "home_score": 2, "away_score": 1, "league": "PL",
            "ps_home": "", "ps_draw": "", "ps_away": "",
        }]
        return rows, {"mapped": 2, "unmapped": [], "coverage": 1.0, "rows": 1}

    monkeypatch.setattr(fcr, "_fetch_fallback_current", _fake_fb)
    monkeypatch.setattr(fcr, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(
        "sys.argv",
        ["fetch", "--out", str(out), "--seasons", "2627", "--timeout", "5"],
    )

    fcr.main()

    assert called.get("yes"), "fallback did not fire despite total .co.uk outage"
    assert "Arsenal,Chelsea,2,1,PL" in out.read_text()
    import json
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["fallback_used"] is True
    assert rep["fallback_rows"] == 1


# ----------------------------------------------------------------------
# 3. total outage never drops history or its closing odds
# ----------------------------------------------------------------------
def test_history_preserved_on_total_outage(monkeypatch, tmp_path):
    out = tmp_path / "club.csv"
    # A historical row WITH odds already on disk (a completed season).
    out.write_text(
        "date,home_team,away_team,home_score,away_score,league,ps_home,ps_draw,ps_away\n"
        "2024-05-01,Man City,Liverpool,3,1,PL,1.50,4.10,6.00\n"
    )
    monkeypatch.setattr(fcr, "_fetch", lambda url, timeout: None)  # all 503
    # Fallback yields nothing (e.g. FD.org also hiccups) — history must survive.
    monkeypatch.setattr(
        fcr, "_fetch_fallback_current",
        lambda dn, pk, leagues=None: (
            [], {"mapped": 0, "unmapped": [], "coverage": 1.0, "rows": 0}),
    )
    monkeypatch.setattr(fcr, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(
        "sys.argv", ["fetch", "--out", str(out), "--timeout", "5"],
    )

    fcr.main()

    body = out.read_text()
    assert "Man City,Liverpool,3,1,PL,1.50,4.10,6.00" in body, (
        "historical row (and its closing odds) was dropped on a total outage"
    )


# ----------------------------------------------------------------------
# 4. unmapped club is surfaced AND kept (never vanishes)
# ----------------------------------------------------------------------
def test_unmapped_club_surfaces_and_is_kept(monkeypatch):
    fake = _FakeFDClient({
        "PL": [_fd_match("Arsenal", "Totally Unknown Zzz FC", 1, 0)],
    })
    import betbot.data.football_data as fd
    monkeypatch.setattr(fd, "FootballDataClient", fake)
    # Constrain the run to one league so we do not hit the network for others.
    import betbot.config as cfg
    monkeypatch.setattr(cfg, "LEAGUE_CODES", ("PL",))

    rows, report = fcr._fetch_fallback_current({"Arsenal"}, set())

    # The row is KEPT, with the unmapped side under its FD.org name.
    assert len(rows) == 1
    kept = rows[0]
    assert kept["home_team"] == "Arsenal"
    assert kept["away_team"] == "Totally Unknown Zzz FC"
    # ... and the club is surfaced, not silently swallowed.
    assert any("Totally Unknown Zzz FC" in u for u in report["unmapped"])
    assert report["coverage"] < 1.0


# ----------------------------------------------------------------------
# 5. FIX A — a 200-but-malformed body must NOT delete the partition
# ----------------------------------------------------------------------
def test_malformed_200_preserves_partition(monkeypatch, tmp_path):
    out = tmp_path / "club.csv"
    # A completed-season partition already on disk (PL 2025-26 = code 2526).
    out.write_text(
        "date,home_team,away_team,home_score,away_score,league,ps_home,ps_draw,ps_away\n"
        "2026-05-01,Arsenal,Chelsea,2,1,PL,1.80,3.50,4.20\n"
    )
    # .co.uk answers HTTP 200 with a shield/HTML body that parses to 0 rows.
    shield = "<html><body>Service temporarily unavailable</body></html>"
    monkeypatch.setattr(
        fcr, "_fetch",
        lambda url, timeout: shield if ("2526" in url and "E0" in url) else None,
    )
    monkeypatch.setattr(
        fcr, "_fetch_fallback_current",
        lambda dn, pk, leagues=None: (
            [], {"mapped": 0, "unmapped": [], "coverage": 1.0, "rows": 0}),
    )
    monkeypatch.setattr(fcr, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(
        "sys.argv", ["fetch", "--out", str(out), "--seasons", "2526", "--timeout", "5"],
    )

    fcr.main()

    body = out.read_text()
    assert "Arsenal,Chelsea,2,1,PL,1.80,3.50,4.20" in body, (
        "a 200-but-malformed body DELETED the historical partition"
    )
    import json
    rep = json.loads((tmp_path / "report.json").read_text())
    rej = rep["rejected_partitions"]
    assert any(r["league"] == "PL" and r["season"] == "2526" and r["reason"] == "empty"
               for r in rej), "the rejected partition was not surfaced in the report"


# ----------------------------------------------------------------------
# 6. FIX B — a PARTIAL current season runs the fallback for exactly the
#    leagues .co.uk did not serve
# ----------------------------------------------------------------------
def test_partial_current_season_falls_back_for_missing_leagues(monkeypatch, tmp_path):
    out = tmp_path / "club.csv"
    # Only 2627/E0 (PL) is served; the other four divisions 503.
    couk_pl = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,PSCH,PSCD,PSCA\n"
        "01/09/2026,Arsenal,Chelsea,2,1,1.80,3.50,4.20\n"
    )
    monkeypatch.setattr(
        fcr, "_fetch",
        lambda url, timeout: couk_pl if ("2627" in url and "E0" in url) else None,
    )

    seen = {}

    def _fake_fb(dataset_names, present_keys, leagues=None):
        seen["leagues"] = list(leagues) if leagues is not None else None
        return [], {"mapped": 0, "unmapped": [], "coverage": 1.0, "rows": 0}

    monkeypatch.setattr(fcr, "_fetch_fallback_current", _fake_fb)
    monkeypatch.setattr(fcr, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(
        "sys.argv", ["fetch", "--out", str(out), "--seasons", "2627", "--timeout", "5"],
    )

    fcr.main()

    assert seen.get("leagues") is not None, "fallback did not run for a partial season"
    assert set(seen["leagues"]) == {"PD", "BL1", "SA", "FL1"}, (
        f"fallback ran for the wrong leagues: {seen['leagues']}"
    )
    assert "PL" not in seen["leagues"], "fallback re-fetched a league .co.uk served"


# ----------------------------------------------------------------------
# 7. a later .co.uk row SUPERSEDES a prior FD.org fallback row (no dupe)
# ----------------------------------------------------------------------
def test_couk_supersedes_fallback_row_without_duplication(monkeypatch, tmp_path):
    out = tmp_path / "club.csv"
    # Last week's fallback wrote this current-season fixture with NO odds.
    out.write_text(
        "date,home_team,away_team,home_score,away_score,league,ps_home,ps_draw,ps_away\n"
        "2026-09-01,Arsenal,Chelsea,2,1,PL,,,\n"
    )
    # This week .co.uk publishes the SAME fixture, now WITH closing odds.
    couk_pl = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,PSCH,PSCD,PSCA\n"
        "01/09/2026,Arsenal,Chelsea,2,1,1.80,3.50,4.20\n"
    )
    monkeypatch.setattr(
        fcr, "_fetch",
        lambda url, timeout: couk_pl if ("2627" in url and "E0" in url) else None,
    )
    # PD/BL1/SA/FL1 are "missing" but the fallback adds nothing new.
    monkeypatch.setattr(
        fcr, "_fetch_fallback_current",
        lambda dn, pk, leagues=None: (
            [], {"mapped": 0, "unmapped": [], "coverage": 1.0, "rows": 0}),
    )
    monkeypatch.setattr(fcr, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(
        "sys.argv", ["fetch", "--out", str(out), "--seasons", "2627", "--timeout", "5"],
    )

    fcr.main()

    lines = [ln for ln in out.read_text().splitlines() if "Arsenal,Chelsea" in ln]
    assert len(lines) == 1, f"fixture duplicated across sources: {lines}"
    assert "1.80,3.50,4.20" in lines[0], "the .co.uk row (with odds) did not win"
