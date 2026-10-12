"""Tests for the free-data self-built Elo feed/builder helpers.

Pure, no network: fetch bodies are passed as strings. Covers the
200-not-payload guard, the football-data.co.uk + openfootball parsers, the Elo
update (determinism + no look-ahead / no double-count), the ClubElo name
bridge, state alignment (anti-fragmentation), and the snapshot round-trip.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from betbot.data.selfbuilt_feed import (
    EloState,
    FeedError,
    Result,
    align_to_state,
    bridge_to_clubelo,
    expected_home,
    gmult,
    parse_football_data_co_uk,
    parse_openfootball_txt,
    update_elo,
    validate_csv_body,
    write_clubelo_snapshot,
)
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.strategy.cl_engine import _load_snapshot

_FD_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "E0,15/08/2026,Arsenal,Chelsea,2,1,H\n"
    "E0,16/08/2026,Liverpool,Everton,3,0,H\n"
    "E0,16/08/2026,Spurs,,1,1,D\n"          # blank away -> skipped
    "E0,bad-date,Fulham,Brentford,0,0,D\n"  # bad date -> skipped
    "E0,17/08/2026,Leeds,Burnley,x,0,A\n"   # bad score -> skipped
)


# ---- 200-not-payload guard -------------------------------------------
def test_validate_rejects_html_content_type():
    with pytest.raises(FeedError):
        validate_csv_body("Date,x\n1,2\n", content_type="text/html",
                          required_cols=["Date"])


def test_validate_rejects_html_body():
    with pytest.raises(FeedError):
        validate_csv_body("<!DOCTYPE html><html>shield</html>",
                          content_type=None, required_cols=["Date"])


def test_validate_rejects_missing_column():
    with pytest.raises(FeedError):
        validate_csv_body("A,B\n1,2\n", content_type="text/csv",
                          required_cols=["Date"])


def test_validate_rejects_too_few_rows():
    with pytest.raises(FeedError):
        validate_csv_body("Date\n", content_type="text/csv",
                          required_cols=["Date"], min_rows=1)


def test_validate_parses_good_csv():
    rows = validate_csv_body("Date,X\n01/01/2026,5\n", content_type="text/csv",
                             required_cols=["Date", "X"])
    assert rows[0]["X"] == "5"


# ---- football-data.co.uk parser --------------------------------------
def test_parse_fd_skips_bad_rows():
    res = parse_football_data_co_uk(_FD_CSV, content_type="text/csv")
    assert len(res) == 2
    assert res[0] == Result(date(2026, 8, 15), "Arsenal", "Chelsea", 2, 1, "DOM")


def test_parse_fd_rejects_shield_html():
    with pytest.raises(FeedError):
        parse_football_data_co_uk("<html>Access denied</html>", content_type="text/html")


# ---- openfootball .txt parser ----------------------------------------
def test_parse_openfootball_iso_and_dow_dates():
    txt = (
        "» Matchday 1\n"
        "2026-09-16\n"
        "  Bayern            3-1  Chelsea\n"
        "Tue Sep/17\n"
        "  Real Madrid       2-2  Juventus\n"
        "garbage line without score\n"
    )
    res = parse_openfootball_txt(txt, competition="UEFA CL", default_year=2026)
    assert len(res) == 2
    assert res[0] == Result(date(2026, 9, 16), "Bayern", "Chelsea", 3, 1, "UEFA CL")
    assert res[1].match_date == date(2026, 9, 17)


# ---- Elo update ------------------------------------------------------
def test_gmult_and_expected_symmetry():
    assert gmult(0) == 1.0 and gmult(2) == 1.5 and gmult(3) == 1.75
    assert abs(expected_home(1500, 1500) - expected_home(1500, 1500)) < 1e-12
    # Equal ratings: home-advantage makes the home expectation > 0.5.
    assert expected_home(1500, 1500) > 0.5


def test_update_elo_no_lookahead_and_no_double_count():
    s = EloState()
    r1 = Result(date(2026, 8, 1), "A", "B", 3, 0, "DOM")
    r2 = Result(date(2026, 8, 8), "A", "B", 0, 1, "DOM")
    assert update_elo(s, [r2, r1]) == 2          # applied in date order
    assert s.last_date == date(2026, 8, 8)
    a_after = s.ratings["A"]
    # Re-running with the same (now <= last_date) rows applies nothing.
    assert update_elo(s, [r1, r2]) == 0
    assert s.ratings["A"] == a_after


def test_update_elo_deterministic():
    rows = [Result(date(2026, 8, d), "A", "B", 2, 1, "DOM") for d in range(1, 6)]
    s1, s2 = EloState(), EloState()
    update_elo(s1, rows)
    update_elo(s2, list(reversed(rows)))
    assert s1.ratings == s2.ratings


# ---- bridge ----------------------------------------------------------
def test_bridge_resolves_drops_unresolved_and_lowgames():
    resolver = TeamAliasResolver(aliases={"Man City": ["Manchester City"]})
    ratings = {"Manchester City": 1850.0, "Nowhere United": 1500.0, "Tiny FC": 1600.0}
    games = {"Manchester City": 40, "Nowhere United": 30, "Tiny FC": 5}
    snap, unresolved = bridge_to_clubelo(
        ratings, games, ["Man City", "Arsenal"], resolver, min_games=15,
    )
    assert snap == {"Man City": 1850.0}       # resolved + enough games
    assert "Nowhere United" in unresolved      # no ClubElo candidate
    assert "Tiny FC" not in snap               # below min_games (not even tried)


def test_bridge_collision_prefers_more_games():
    resolver = TeamAliasResolver(aliases={"Inter": ["Internazionale", "Inter Milan"]})
    ratings = {"Internazionale": 1800.0, "Inter Milan": 1700.0}
    games = {"Internazionale": 50, "Inter Milan": 20}
    snap, _ = bridge_to_clubelo(ratings, games, ["Inter"], resolver, min_games=15)
    assert snap == {"Inter": 1800.0}           # the 50-game entry wins


# ---- align_to_state (anti-fragmentation) -----------------------------
def test_align_rewrites_onto_existing_key():
    resolver = TeamAliasResolver(aliases={})
    existing = ["Paris Saint-Germain"]
    rows = [Result(date(2026, 9, 1), "Paris Saint-Germain FC", "Monaco", 2, 0, "DOM")]
    aligned = align_to_state(rows, existing, resolver, threshold=70)
    assert aligned[0].home == "Paris Saint-Germain"   # folded onto the seed key
    assert aligned[0].away == "Monaco"                # unresolved stays raw


def test_align_noop_when_no_existing_keys():
    rows = [Result(date(2026, 9, 1), "X", "Y", 1, 0, "DOM")]
    assert align_to_state(rows, [], TeamAliasResolver(aliases={})) == rows


# ---- snapshot round-trip ---------------------------------------------
def test_write_snapshot_reads_back_with_from_date(tmp_path):
    out = tmp_path / "selfbuilt.csv"
    snap = {"Man City": 1850.0, "Arsenal": 1800.0}
    write_clubelo_snapshot(out, snap, from_date=date(2026, 10, 1),
                           country_of={"Man City": "ENG"})
    loaded, newest = _load_snapshot(Path(out))
    assert loaded["Man City"] == 1850.0 and loaded["Arsenal"] == 1800.0
    assert newest == date(2026, 10, 1)
