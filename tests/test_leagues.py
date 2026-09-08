"""The ONE canonical competition-code -> human league-name map."""

from __future__ import annotations

from betbot.leagues import LEAGUE_DISPLAY_NAMES, league_label


def test_known_codes_map_to_human_names():
    assert league_label("PL") == "Premier League"
    assert league_label("PD") == "La Liga"
    assert league_label("BL1") == "Bundesliga"
    assert league_label("SA") == "Serie A"
    assert league_label("FL1") == "Ligue 1"
    assert league_label("CL") == "Champions League"


def test_code_is_case_insensitive():
    assert league_label("pd") == "La Liga"
    assert league_label(" pl ") == "Premier League"


def test_unknown_code_degrades_to_raw_code():
    # A brand-new / historical competition still identifies the match.
    assert league_label("ZZ9") == "ZZ9"
    assert league_label("wc") == "World Cup"  # WC is mapped


def test_missing_code_yields_empty_string_never_none():
    assert league_label(None) == ""
    assert league_label("") == ""
    assert league_label("   ") == ""


def test_every_config_league_code_has_a_display_name():
    from betbot.config import LEAGUE_CODES

    for code in LEAGUE_CODES:
        assert code in LEAGUE_DISPLAY_NAMES, code
        assert league_label(code) != code  # a real human name, not the raw code
