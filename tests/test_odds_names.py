"""OddsNameResolver — the safety net that stops a fixture being anchored to
another club's price.

The governing rule these tests encode: a MISSING resolution is fine (the
fixture ships unanchored, exactly as before the feature existed); a WRONG
resolution is a defect. So every "can't be sure" case must return None.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from betbot.data.odds_names import AliasTableError, OddsNameResolver

REPO_ROOT = Path(__file__).resolve().parents[1]
ALIAS_PATH = REPO_ROOT / "config" / "odds_team_aliases.yaml"

# The canonical dataset namespace (football-data.co.uk historical spellings,
# normalised) — the same keys our Glicko ratings and DC params use.
CANON = [
    "ath madrid", "ath bilbao", "barcelona", "espanol", "real madrid",
    "vallecano", "alaves", "sociedad", "celta", "betis",
    "man city", "man united", "nott m forest", "tottenham",
    "paris sg", "paris fc", "bayern munich", "m gladbach", "inter", "milan",
    "wolves", "burnley", "sunderland",
]

NAME_MAP = {
    "wolverhampton wanderers fc": "wolves",
    "atletico madrid": "ath madrid",
    "rayo vallecano madrid": "vallecano",
    "rcd espanyol de barcelona": "espanol",
    "fc barcelona": "barcelona",
    "manchester city fc": "man city",
    "paris saint germain fc": "paris sg",
}


def _resolver(**kw) -> OddsNameResolver:
    aliases = kw.pop("aliases", None)
    if aliases is None:
        aliases = yaml.safe_load(ALIAS_PATH.read_text())["aliases"]
    return OddsNameResolver(CANON, aliases=aliases, name_map=NAME_MAP, **kw)


# ---------------------------------------------------------------------------
# The known defect this feature exists to survive
# ---------------------------------------------------------------------------
def test_atl_madrid_fixtures_spelling_resolves_to_ath_madrid():
    """football-data.co.uk disagrees with ITSELF: fixtures.csv says
    'Atl. Madrid', every historical season file says 'Ath Madrid'."""
    r = _resolver()
    assert r.resolve("Atl. Madrid") == "ath madrid"
    assert r.resolve("Ath Madrid") == "ath madrid"
    # ...and the live football-data.org spelling lands on the same key, which
    # is the whole point: both sides must meet in one namespace.
    assert r.resolve("Atletico Madrid") == "ath madrid"


def test_rayo_vallecano_long_spelling_resolves():
    r = _resolver()
    assert r.resolve("Rayo Vallecano") == "vallecano"
    assert r.resolve("Vallecano") == "vallecano"


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("Man City", "man city"),
        ("Manchester City FC", "man city"),
        ("Paris SG", "paris sg"),
        ("Paris Saint-Germain FC", "paris sg"),
        ("Bayern Munich", "bayern munich"),
        ("Nott'm Forest", "nott m forest"),
        ("M'gladbach", "m gladbach"),
    ],
)
def test_ordinary_spellings_resolve(spelling, expected):
    assert _resolver().resolve(spelling) == expected


# ---------------------------------------------------------------------------
# The Espanyol/Barcelona failure mode
# ---------------------------------------------------------------------------
def test_espanyol_never_resolves_to_barcelona():
    """The incident: a fuzzy matcher folded 'RCD Espanyol de Barcelona' onto
    Barcelona and silently gave Espanyol another club's rating."""
    r = _resolver()
    assert r.resolve("Espanol") == "espanol"
    assert r.resolve("Espanyol") is None or r.resolve("Espanyol") == "espanol"
    assert r.resolve("RCD Espanyol de Barcelona") == "espanol"
    assert r.resolve("Barcelona") == "barcelona"
    assert r.resolve("RCD Espanyol de Barcelona") != r.resolve("Barcelona")


def test_paris_fc_never_resolves_to_paris_sg():
    """Two real, different Ligue 1 clubs whose names share a token. Note
    normalize() strips 'fc' as a noise token, so Paris FC's canonical key is
    plain 'paris' — still distinct from 'paris sg', which is the point."""
    r = _resolver()
    assert r.resolve("Paris FC") == "paris"
    assert r.resolve("Paris SG") == "paris sg"
    assert r.resolve("Paris FC") != r.resolve("Paris SG")


def test_ath_bilbao_never_resolves_to_ath_madrid():
    r = _resolver()
    assert r.resolve("Ath Bilbao") == "ath bilbao"
    assert r.resolve("Ath Bilbao") != r.resolve("Ath Madrid")


def test_inter_and_milan_stay_distinct():
    r = _resolver()
    assert r.resolve("Inter") == "inter"
    assert r.resolve("Milan") == "milan"
    assert r.resolve("Inter") != r.resolve("Milan")


# ---------------------------------------------------------------------------
# Unknown names are SKIPPED, never guessed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "unknown",
    ["Malaga", "Sheffield Wed", "Bradford City", "Some FC That Never Existed", "", "   "],
)
def test_unknown_names_return_none(unknown):
    """A club we have no ratings for must produce NO quote, not a near-miss.
    The fixture then ships unanchored — the pre-feature behaviour."""
    assert _resolver().resolve(unknown) is None


def test_near_miss_is_not_resolved():
    """'Real Sociedad' style near-misses must not silently snap to a
    lexically-similar canonical name. Explicit table or nothing."""
    r = _resolver()
    assert r.resolve("Real Madrid") == "real madrid"
    # 'Real Betis Balompie' is not in the table or the name map -> None.
    assert r.resolve("Real Betis Balompie Sevilla") is None


# ---------------------------------------------------------------------------
# The alias table itself must be safe by construction
# ---------------------------------------------------------------------------
def test_alias_table_targets_must_be_canonical():
    with pytest.raises(AliasTableError):
        OddsNameResolver(CANON, aliases={"Not A Real Club": ["Whatever"]})


def test_alias_table_rejects_one_spelling_pointing_at_two_clubs():
    with pytest.raises(AliasTableError):
        OddsNameResolver(
            CANON,
            aliases={"Barcelona": ["Barca"], "Espanol": ["Barca"]},
        )


def test_shipped_alias_file_loads_and_is_injective():
    """Guard on the real config/odds_team_aliases.yaml: every alias must point
    at a canonical club, and no two clubs may share an alias spelling."""
    aliases = yaml.safe_load(ALIAS_PATH.read_text())["aliases"]
    r = OddsNameResolver(CANON, aliases=aliases, name_map=NAME_MAP)
    pairs = r.alias_pairs
    assert pairs, "alias table should not be empty"
    for alt, canon in pairs.items():
        assert canon in r.canonical_names, f"{alt} -> {canon} is not canonical"


def test_missing_files_degrade_to_canonical_only():
    """A missing alias file must disable aliasing, not crash the daemon."""
    r = OddsNameResolver.from_files("/nonexistent/aliases.yaml", "/nonexistent/map.json")
    assert r.resolve("Atl. Madrid") is None
    assert r.alias_pairs == {}
