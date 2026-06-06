"""Tests for the team-name matcher (Phase 2)."""

from __future__ import annotations

from betbot.exchanges.matcher import TeamAliasResolver, normalize


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
