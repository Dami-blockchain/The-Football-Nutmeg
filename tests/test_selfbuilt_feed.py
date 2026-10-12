"""Tests for the free-data self-built Elo feed/builder helpers.

Pure, no network. Covers the 200-not-payload guard, the football-data.co.uk +
football-data.org parsers, the Elo update (determinism, no look-ahead, no
double-count, MULTI-SOURCE gather-then-apply), the COUNTRY-RESTRICTED ClubElo
name bridge + collision safety using REAL ident spellings, state alignment, the
snapshot round-trip, and that every shipped alias target exists in ClubElo.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import json
import pytest
import yaml

from betbot.data.selfbuilt_feed import (
    EloState,
    FeedError,
    Result,
    align_to_state,
    bridge_to_clubelo,
    expected_home,
    gmult,
    load_clubelo_rows,
    parse_fdorg_matches,
    parse_football_data_co_uk,
    update_elo,
    validate_csv_body,
    write_clubelo_snapshot,
)

_REPO = Path(__file__).resolve().parents[1]

_FD_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "E0,15/08/2026,Arsenal,Chelsea,2,1,H\n"
    "E0,16/08/2026,Liverpool,Everton,3,0,H\n"
    "E0,16/08/2026,Spurs,,1,1,D\n"          # blank away -> skipped
    "E0,bad-date,Fulham,Brentford,0,0,D\n"  # bad date -> skipped
    "E0,17/08/2026,Leeds,Burnley,x,0,A\n"   # bad score -> skipped
)


def _aliases() -> dict[str, list[str]]:
    # ONLY the self-built ClubElo-key table (team_aliases.yaml is a different
    # canonical space and AliasFolder now rejects the cross-table collisions).
    data = yaml.safe_load((_REPO / "config/selfbuilt_elo_aliases.yaml").read_text())
    return (data or {}).get("aliases") or {}


# ---- 200-not-payload guard -------------------------------------------
def test_validate_rejects_html_content_type():
    with pytest.raises(FeedError):
        validate_csv_body("Date,x\n1,2\n", content_type="text/html", required_cols=["Date"])


def test_validate_rejects_html_body():
    with pytest.raises(FeedError):
        validate_csv_body("<!DOCTYPE html><html>shield</html>", content_type=None,
                          required_cols=["Date"])


def test_validate_rejects_missing_column():
    with pytest.raises(FeedError):
        validate_csv_body("A,B\n1,2\n", content_type="text/csv", required_cols=["Date"])


def test_validate_rejects_too_few_rows():
    with pytest.raises(FeedError):
        validate_csv_body("Date\n", content_type="text/csv",
                          required_cols=["Date"], min_rows=1)


def test_validate_parses_good_csv():
    rows = validate_csv_body("Date,X\n01/01/2026,5\n", content_type="text/csv",
                             required_cols=["Date", "X"])
    assert rows[0]["X"] == "5"


# ---- parsers ---------------------------------------------------------
def test_parse_fd_skips_bad_rows():
    res = parse_football_data_co_uk(_FD_CSV, content_type="text/csv")
    assert len(res) == 2
    assert res[0] == Result(date(2026, 8, 15), "Arsenal", "Chelsea", 2, 1, "DOM")


def test_parse_fd_rejects_shield_html():
    with pytest.raises(FeedError):
        parse_football_data_co_uk("<html>Access denied</html>", content_type="text/html")


def test_parse_fdorg_matches_finished_only():
    matches = [
        {"status": "FINISHED", "utcDate": "2026-09-16T19:00:00Z",
         "homeTeam": {"name": "Manchester City FC"}, "awayTeam": {"name": "Paris Saint-Germain FC"},
         "score": {"fullTime": {"home": 2, "away": 1}}},
        {"status": "SCHEDULED", "utcDate": "2026-09-17T19:00:00Z",
         "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
         "score": {"fullTime": {"home": None, "away": None}}},
    ]
    res = parse_fdorg_matches(matches)
    assert len(res) == 1
    assert res[0] == Result(date(2026, 9, 16), "Manchester City FC",
                            "Paris Saint-Germain FC", 2, 1, "UEFA CL")


# ---- Elo update ------------------------------------------------------
def test_gmult_and_expected_home_advantage():
    assert gmult(0) == 1.0 and gmult(2) == 1.5 and gmult(3) == 1.75
    assert expected_home(1500, 1500) > 0.5


def test_update_elo_no_lookahead_and_no_double_count():
    s = EloState()
    r1 = Result(date(2026, 8, 1), "A", "B", 3, 0, "DOM")
    r2 = Result(date(2026, 8, 8), "A", "B", 0, 1, "DOM")
    assert update_elo(s, [r2, r1]) == 2
    assert s.last_date == date(2026, 8, 8)
    a_after = s.ratings["A"]
    assert update_elo(s, [r1, r2]) == 0       # seen-set dedupe
    assert s.ratings["A"] == a_after


def test_update_elo_multi_source_overlapping_dates_applies_all_once():
    # F1 regression: a domestic source and a CL source sharing dates must BOTH
    # apply in one pass (the old per-source cutoff dropped the second source),
    # and a duplicate match across sources applies exactly once.
    dom = [
        Result(date(2026, 9, 1), "A", "B", 1, 0, "DOM"),
        Result(date(2026, 9, 2), "C", "D", 2, 2, "DOM"),
    ]
    cl = [
        Result(date(2026, 9, 1), "A", "C", 0, 1, "UEFA CL"),   # same date, new match
        Result(date(2026, 9, 2), "B", "D", 3, 1, "UEFA CL"),
        Result(date(2026, 9, 1), "A", "B", 1, 0, "UEFA CL"),   # duplicate of dom[0]
    ]
    s = EloState()
    applied = update_elo(s, dom + cl)
    assert applied == 4                        # 5 rows, 1 is a dup of dom[0]
    assert s.last_date == date(2026, 9, 2)
    # Re-running either source alone applies nothing new.
    assert update_elo(s, dom) == 0
    assert update_elo(s, cl) == 0


def test_update_elo_does_not_advance_from_when_nothing_applied():
    s = EloState()
    update_elo(s, [Result(date(2026, 8, 1), "A", "B", 1, 0, "DOM")])
    before = s.last_date
    assert update_elo(s, [Result(date(2026, 8, 1), "A", "B", 1, 0, "DOM")]) == 0
    assert s.last_date == before


# ---- country-restricted bridge (real ident spellings) ----------------
def test_bridge_real_idents_resolve_within_country():
    aliases = _aliases()
    ratings = {
        "Paris Saint-Germain (France)": 1980.0,
        "Manchester City (England)": 2030.0,
        "Qarabağ (Azerbaijan)": 1620.0,
        "Sporting Braga (Portugal)": 1700.0,
        "Crvena Zvezda (Serbia)": 1660.0,
        "FC København (Denmark)": 1680.0,
        "Real Sociedad (Spain)": 1770.0,
    }
    games = {k: 40 for k in ratings}
    clubelo = [
        ("Paris SG", "FRA"), ("Paris FC", "FRA"), ("Man City", "ENG"),
        ("Karabakh Agdam", "AZE"), ("Braga", "POR"), ("Crvena Zvezda", "SRB"),
        ("FC Kobenhavn", "DEN"), ("Real Sociedad", "ESP"),
    ]
    snap, unresolved = bridge_to_clubelo(ratings, games, clubelo, aliases)
    assert snap["Paris SG"] == 1980.0 and "Paris FC" in unresolved
    assert snap["Man City"] == 2030.0
    assert snap["Karabakh Agdam"] == 1620.0
    assert snap["Braga"] == 1700.0
    assert snap["Crvena Zvezda"] == 1660.0
    assert snap["FC Kobenhavn"] == 1680.0
    assert snap["Real Sociedad"] == 1770.0


def test_bridge_rejects_wrong_country_and_single_token_collision():
    # Sheffield United vs Sheffield Wednesday share ONE token -> no cross-map;
    # Chievo (Italy) must not steal Verona's rating.
    aliases = _aliases()
    ratings = {
        "Sheffield United (England)": 1500.0,
        "Chievo (Italy)": 1450.0,
        "Manchester City (England)": 2030.0,   # different country decoy for Verona
    }
    games = {k: 40 for k in ratings}
    clubelo = [("Sheffield Wednesday", "ENG"), ("Verona", "ITA")]
    snap, unresolved = bridge_to_clubelo(ratings, games, clubelo, aliases)
    assert "Sheffield Wednesday" in unresolved   # not fed from Sheffield United
    assert "Verona" in unresolved                # not fed from Chievo
    assert snap == {}


def test_bridge_collision_leaves_key_unresolved_without_alias():
    # Two idents both plausibly "United (England)"; one ClubElo key. Greedy 1:1
    # assigns the exact match; a second contesting ident cannot also take it.
    ratings = {"Leeds United (England)": 1600.0, "Leeds (England)": 1590.0}
    games = {k: 40 for k in ratings}
    snap, _ = bridge_to_clubelo(ratings, games, [("Leeds", "ENG")], {})
    # "Leeds" exact-matches "Leeds (England)"; "Leeds United" shares 1 token -> no.
    assert snap == {"Leeds": 1590.0}


def test_bridge_ignores_low_game_idents():
    snap, unresolved = bridge_to_clubelo(
        {"Manchester City (England)": 2030.0}, {"Manchester City (England)": 5},
        [("Man City", "ENG")], _aliases(), min_games=15,
    )
    assert snap == {} and "Man City" in unresolved


# ---- alignment (country-restricted token-subset, NO alias) -----------
def test_align_domestic_token_subset_and_translit():
    existing = [
        "Leicester City (England)", "Arsenal FC (England)",
        "Deportivo Alavés (Spain)", "FC Koeln (Germany)",
    ]
    rows = [
        Result(date(2026, 9, 1), "Leicester", "Arsenal", 1, 1, "DOM"),
    ]
    aligned = align_to_state(rows, existing, "England")
    assert aligned[0].home == "Leicester City (England)"   # {leicester} subset, unique
    assert aligned[0].away == "Arsenal FC (England)"


def test_align_alaves_and_koeln_via_translit():
    existing = ["Deportivo Alavés (Spain)", "FC Koeln (Germany)"]
    a = align_to_state([Result(date(2026, 9, 1), "Alaves", "Vallecano", 1, 0, "DOM")],
                       existing, "Spain")
    assert a[0].home == "Deportivo Alavés (Spain)"
    assert a[0].away == "Vallecano (Spain)"                # not in state -> suffixed
    k = align_to_state([Result(date(2026, 9, 1), "FC Koln", "Mainz", 2, 2, "DOM")],
                       existing, "Germany")
    assert k[0].home == "FC Koeln (Germany)"               # Koln -> Koeln via translit


def test_align_verona_resolves_to_hellas_via_alias():
    # Chievo is defunct and only Hellas carries the "Verona" alias, so the exact
    # alias-fold resolves Verona onto Hellas; Chievo (no alias) is NOT grabbed.
    existing = ["Hellas Verona (Italy)", "Chievo Verona (Italy)"]
    rows = [Result(date(2026, 9, 1), "Verona", "Napoli", 1, 0, "DOM")]
    aligned = align_to_state(rows, existing, "Italy", alias_map=_aliases())
    assert aligned[0].home == "Hellas Verona (Italy)"   # resolved via alias
    assert aligned[0].away == "Napoli (Italy)"


def test_align_refuses_ambiguous_without_alias():
    # Same shape but NO alias: token-subset ties two clubs -> refused -> suffixed.
    existing = ["Hellas Verona (Italy)", "Chievo Verona (Italy)"]
    aligned = align_to_state([Result(date(2026, 9, 1), "Verona", "Napoli", 1, 0, "DOM")],
                             existing, "Italy")
    assert aligned[0].home == "Verona (Italy)"


def test_align_exact_collision_prefers_more_games():
    # Two seed spellings of the SAME club fold to one canonical -> pick the
    # better-sampled (Sunderland / Atromitos class).
    existing = ["Sunderland (England)", "Sunderland AFC (England)"]
    games = {"Sunderland (England)": 1200, "Sunderland AFC (England)": 30}
    aligned = align_to_state([Result(date(2026, 9, 1), "Sunderland", "X", 1, 0, "DOM")],
                             existing, "England", games=games)
    assert aligned[0].home == "Sunderland (England)"


def test_align_alias_merges_inter_and_man_city():
    # Token rules can't merge inter<->internazionale (Milano/Milan clash) or
    # man<->manchester; the exact alias-fold pass does, country-restricted.
    aliases = _aliases()
    existing = ["Inter (Italy)", "Milan (Italy)", "Manchester City (England)"]
    it = align_to_state([Result(date(2026, 9, 1), "FC Internazionale Milano", "Milan", 1, 0, "UEFA CL")],
                        existing, team_country={"FC Internazionale Milano": "Italy", "Milan": "Italy"},
                        alias_map=aliases)
    assert it[0].home == "Inter (Italy)"      # merged, NOT a fresh fragment
    assert it[0].away == "Milan (Italy)"      # Milano did NOT also grab Milan
    en = align_to_state([Result(date(2026, 9, 1), "Man City", "Spurs", 2, 0, "DOM")],
                        existing, "England", alias_map=aliases)
    assert en[0].home == "Manchester City (England)"   # Man City merged via alias
    assert en[0].away == "Spurs (England)"             # unknown -> suffixed


def test_align_cl_per_team_country_breaks_arsenal_belarus_tie():
    existing = ["Arsenal FC (England)", "Arsenal Tula (Russia)", "Arsenal (Belarus)"]
    tc = {"Arsenal FC": "England", "FC Bayern München": "Germany"}
    rows = [Result(date(2026, 9, 1), "Arsenal FC", "FC Bayern München", 1, 0, "UEFA CL")]
    aligned = align_to_state(rows, existing, team_country=tc)
    assert aligned[0].home == "Arsenal FC (England)"       # country routing breaks the tie
    assert aligned[0].away == "FC Bayern München (Germany)"  # not in state -> suffixed w/ country


#: football-data.org area.name -> schochastics seed country (mirrors the build).
_AREA_TO_SEED = {"Monaco": "France"}


def _seed_fixture():
    rows = json.loads((_REPO / "tests/data/seed_idents_sample.json").read_text())
    return [k for k, _g in rows], {k: g for k, g in rows}


def test_cl_and_domestic_names_fold_zero_new_idents():
    # The 7 clubs the review names MUST fold onto an EXISTING seed ident (NOT a
    # fresh "(Country)" phantom), via CL names (per-team area.name, Monaco
    # remapped to France) and via the football-data.co.uk domestic short names.
    seed_idents, seed_games = _seed_fixture()
    cases = [
        ("Eintracht Frankfurt", "Germany", "Eintracht Frankfurt (Germany)"),
        ("Ein Frankfurt", "Germany", "Eintracht Frankfurt (Germany)"),
        ("Sporting Clube de Portugal", "Portugal", "Sporting CP (Portugal)"),
        ("Sp Lisbon", "Portugal", "Sporting CP (Portugal)"),
        ("AS Monaco FC", "Monaco", "Monaco (France)"),      # area remapped
        ("AS Monaco", "France", "Monaco (France)"),
        ("Royale Union Saint-Gilloise", "Belgium", "Union Saint-Gilloise (Belgium)"),
        ("PAE AEK", "Greece", "AEK Athens (Greece)"),
        ("AEK", "Greece", "AEK Athens (Greece)"),
        ("Olympiakos", "Greece", "Olympiakos Piraeus (Greece)"),
        ("Sp Braga", "Portugal", "Sporting Braga (Portugal)"),
        ("Sporting Braga", "Portugal", "Sporting Braga (Portugal)"),
    ]
    tc = {name: _AREA_TO_SEED.get(area, area) for name, area, _exp in cases}
    rows = [Result(date(2026, 9, 1), name, "ZZ Dummy", 1, 0, "UEFA CL")
            for name, _a, _e in cases]
    aligned = align_to_state(rows, seed_idents, team_country=tc,
                             alias_map=_aliases(), games=seed_games)
    new_idents = []
    for (name, _a, expect), r in zip(cases, aligned):
        assert r.home == expect, f"{name} -> {r.home} (expected {expect})"
        if r.home not in seed_idents:
            new_idents.append(r.home)
    assert new_idents == [], f"created NEW idents: {new_idents}"


def test_no_alias_canonical_collision_within_country():
    # (1) Among the REAL seed idents, any same-country pair that folds to one
    # canonical must be the SAME club — i.e. one's token set ⊆ the other's.
    # Catches e.g. a mistaken Frankfurt += ["Bayern München"] (eintracht,
    # frankfurt vs bayern,munchen are not subsets).
    from betbot.data.selfbuilt_feed import AliasFolder, _ftokens, _translit, strip_country
    folder = AliasFolder(_aliases())
    seed = json.loads((_REPO / "tests/data/seed_idents_sample.json").read_text())
    by: dict = {}
    for ident, _g in seed:
        c = strip_country(ident)[1]
        by.setdefault(c, {}).setdefault(folder.canon_norm(ident), []).append(ident)
    for _c, folds in by.items():
        for _canon, idents in folds.items():
            if len(idents) < 2:
                continue
            tok = [_ftokens(i) for i in idents]
            for i in range(len(tok)):
                for j in range(i + 1, len(tok)):
                    assert tok[i] <= tok[j] or tok[j] <= tok[i], \
                        f"different clubs fold together: {idents}"
    # (2) No alt's _translit may equal a DIFFERENT canonical key's _translit.
    aliases = _aliases()
    canon_translits = {c: _translit(c) for c in aliases}
    canon_by_translit = {t: c for c, t in canon_translits.items()}
    for canon, alts in aliases.items():
        for alt in alts:
            at = _translit(alt)
            owner = canon_by_translit.get(at)
            assert owner in (None, canon), \
                f"alt {alt!r} of {canon!r} collides with canonical {owner!r}"
    # (3) Known-distinct clubs must stay distinct.
    for a, b in [("Sporting CP", "Sporting Braga"), ("Inter", "Milan"),
                 ("AEK Athens", "Olympiakos Piraeus"), ("Hellas Verona", "Chievo Verona"),
                 ("KSK Beveren", "Waasland Beveren")]:
        assert folder.canon_norm(a) != folder.canon_norm(b), f"{a} == {b}"


def test_aliasfolder_raises_on_conflicting_alt():
    from betbot.data.selfbuilt_feed import AliasFolder
    # "Shared Alt" folds to two different canonicals -> must raise, not last-wins.
    with pytest.raises(ValueError, match="fold conflict"):
        AliasFolder({"Club A": ["Shared Alt"], "Club B": ["Shared Alt"]})
    # The shipped table must itself be conflict-free (no raise).
    AliasFolder(_aliases())


# ---- snapshot round-trip --------------------------------------------
def test_write_snapshot_reads_back_with_from_date(tmp_path):
    from betbot.strategy.cl_engine import _load_snapshot
    out = tmp_path / "selfbuilt.csv"
    write_clubelo_snapshot(out, {"Man City": 1850.0, "Arsenal": 1800.0},
                           from_date=date(2026, 10, 1), country_of={"Man City": "ENG"})
    loaded, newest = _load_snapshot(Path(out))
    assert loaded["Man City"] == 1850.0 and loaded["Arsenal"] == 1800.0
    assert newest == date(2026, 10, 1)


# ---- F5: every alias target exists in ClubElo ------------------------
def test_every_alias_canonical_exists_in_clubelo():
    aliases = yaml.safe_load((_REPO / "config/selfbuilt_elo_aliases.yaml").read_text())["aliases"]
    # Prefer the live scrape; fall back to the committed fixture for CI.
    live = _REPO / "data/clubelo_scrape_latest.csv"
    fixture = _REPO / "tests/data/clubelo_scrape_sample.csv"
    src = live if live.exists() else fixture
    clubs = {club for club, _code in load_clubelo_rows(src)}
    missing = [k for k in aliases if k not in clubs]
    assert missing == [], f"alias keys not in ClubElo ({src.name}): {missing}"
