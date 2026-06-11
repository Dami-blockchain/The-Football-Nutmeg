"""Tests for the team-name matcher (Phase 2)."""

from __future__ import annotations

from betbot.data.models import MatchOutcome
from betbot.exchanges.matcher import (
    TeamAliasResolver,
    classify_binary_outcome,
    is_match_result_market,
    normalize,
    normalized_team_pair,
)


# ----------------------------------------------------------------------
# classify_binary_outcome (shared by Polymarket + Limitless adapters)
# ----------------------------------------------------------------------
def test_classify_binary_outcome():
    r = TeamAliasResolver()
    assert classify_binary_outcome("Arsenal", "Arsenal FC", "Chelsea FC", r) is MatchOutcome.HOME
    assert classify_binary_outcome("Chelsea", "Arsenal FC", "Chelsea FC", r) is MatchOutcome.AWAY
    assert classify_binary_outcome("Draw", "Arsenal FC", "Chelsea FC", r) is MatchOutcome.DRAW
    assert classify_binary_outcome("Spurs", "Arsenal FC", "Chelsea FC", r) is None
    assert classify_binary_outcome("", "Arsenal FC", "Chelsea FC", r) is None


# ----------------------------------------------------------------------
# normalize()
# ----------------------------------------------------------------------
def test_normalize_strips_diacritics_and_noise_tokens():
    assert normalize("FC Bayern München") == "bayern munchen"
    assert normalize("Paris Saint-Germain FC") == "paris saint germain"
    assert normalize("Arsenal FC") == "arsenal"
    assert normalize("AC Milan") == "milan"


def test_normalize_never_collapses_to_empty():
    # A name made entirely of noise tokens keeps its raw tokens.
    assert normalize("FC AC") != ""


# ----------------------------------------------------------------------
# fuzzy matching (no alias table)
# ----------------------------------------------------------------------
def test_fuzzy_match_handles_suffix_and_word_order():
    r = TeamAliasResolver()
    candidates = ["Manchester City", "Aston Villa", "Chelsea"]
    assert r.match("Manchester City FC", candidates) == "Manchester City"


def test_fuzzy_match_returns_none_below_threshold():
    r = TeamAliasResolver()
    # Nothing close to "Arsenal" in the candidate set.
    assert r.match("Arsenal FC", ["Aston Villa", "Chelsea"]) is None


def test_diacritics_match_without_alias():
    r = TeamAliasResolver()
    assert r.match("FC Bayern München", ["Bayern Munchen", "Dortmund"]) == "Bayern Munchen"


# ----------------------------------------------------------------------
# alias table
# ----------------------------------------------------------------------
def test_alias_resolves_abbreviation():
    r = TeamAliasResolver({"Paris Saint-Germain FC": ["PSG", "Paris SG"]})
    # "PSG" would never fuzzy-match "Paris Saint-Germain" — alias makes it exact.
    assert r.match("Paris Saint-Germain FC", ["PSG", "Marseille"]) == "PSG"


def test_alias_is_symmetric_via_canonical_fold():
    r = TeamAliasResolver({"Manchester City FC": ["Man City"]})
    # Looking up by the alias spelling still finds the canonical candidate.
    assert r.match("Man City", ["Manchester City FC", "Liverpool FC"]) == "Manchester City FC"


def test_same_team():
    r = TeamAliasResolver({"FC Bayern München": ["Bayern"]})
    assert r.same_team("Bayern", "FC Bayern München") is True
    assert r.same_team("Arsenal FC", "Chelsea FC") is False


# ----------------------------------------------------------------------
# from_yaml
# ----------------------------------------------------------------------
def test_from_yaml_missing_file_is_fuzzy_only(tmp_path):
    r = TeamAliasResolver.from_yaml(tmp_path / "does_not_exist.yaml")
    # No aliases, but fuzzy matching still works.
    assert r.match("Chelsea FC", ["Chelsea", "Arsenal"]) == "Chelsea"


def test_from_yaml_loads_aliases(tmp_path):
    p = tmp_path / "aliases.yaml"
    p.write_text('aliases:\n  "Real Madrid CF": ["Real Madrid", "Madrid"]\n')
    r = TeamAliasResolver.from_yaml(p)
    assert r.match("Real Madrid CF", ["Madrid", "Barcelona"]) == "Madrid"


# ----------------------------------------------------------------------
# international name variants (must still match via the alias table)
# ----------------------------------------------------------------------
def test_international_name_variants_match_via_aliases():
    r = TeamAliasResolver.from_yaml("config/team_aliases.yaml")
    assert r.same_team("Korea Republic", "South Korea")
    assert r.same_team("Türkiye", "Turkey")
    assert r.same_team("Czechia", "Czech Republic")
    assert r.same_team("United States", "USA")
    assert r.same_team("Côte d'Ivoire", "Ivory Coast")
    # ...without collapsing genuinely different nations.
    assert not r.same_team("Korea Republic", "Korea DPR")
    assert not r.same_team("Mexico", "South Africa")


# ----------------------------------------------------------------------
# is_match_result_market — props must NOT be 1X2 match-result markets
# ----------------------------------------------------------------------
def test_match_result_market_accepts_plain_1x2():
    assert is_match_result_market(None, "Mexico vs. South Africa")
    assert is_match_result_market({"marketType": "match_result"}, "Will Mexico win?")
    assert is_match_result_market({}, "Arsenal vs. Chelsea")


def test_prop_titles_are_not_match_result():
    props = [
        "Mexico to win by 2+ goals vs Czech Republic",
        "3+ total goals",
        "Over 2.5 goals",
        "Both teams to score",
        "BTTS",
        "Mexico -1.5 handicap",
        "Total cards over 4.5",
        "Corners over 9.5",
        "First goal scorer",
        "Anytime goalscorer",
        "Correct score 2-1",
        "Mexico to qualify",
        "Half-time result",
    ]
    for title in props:
        assert not is_match_result_market(None, title), title


def test_prop_threshold_metadata_marks_prop():
    # Limitless-style structured metadata: a non-zero threshold => prop.
    assert not is_match_result_market(
        {"marketType": "match_result", "goalsThreshold": 2.5}, "Mexico vs South Africa"
    )
    assert not is_match_result_market({"spreadThreshold": -1.5}, "Mexico vs South Africa")
    assert not is_match_result_market({"cardsThreshold": 4}, "Mexico vs South Africa")
    # A zero threshold is the 1X2 case and must NOT be rejected.
    assert is_match_result_market(
        {"marketType": "match_result", "goalsThreshold": 0}, "Mexico vs South Africa"
    )


def test_explicit_non_result_market_type_is_prop():
    assert not is_match_result_market({"marketType": "TOTALS"}, "Mexico vs South Africa")
    assert not is_match_result_market({"marketType": "spread"}, "Mexico vs South Africa")


def test_normalized_team_pair_is_orientation_agnostic():
    assert normalized_team_pair("Mexico", "South Africa") == normalized_team_pair(
        "South Africa", "Mexico"
    )
    assert normalized_team_pair("Mexico", "South Africa") != normalized_team_pair(
        "Mexico", "Czech Republic"
    )
