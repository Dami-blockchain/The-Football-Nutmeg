"""Prediction delivery formatting — the tipster message shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from betbot.tips import (
    format_locked,
    format_prediction,
    format_prediction_with_lineup,
    format_result,
)


@dataclass
class _Bet:
    outcome: str
    market_price: float | None
    edge: float | None


@dataclass
class _Pred:
    home_team: str = "Man City"
    away_team: str = "Arsenal"
    p_home: float = 0.39
    p_draw: float = 0.31
    p_away: float = 0.30
    home_xg: float | None = 1.44
    away_xg: float | None = 1.17
    kickoff: datetime = datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc)
    competition_code: str = "PD"  # La Liga, to prove the human league label
    paper_bets: list = field(default_factory=list)


def test_format_prediction_has_home_away_tags_probs_and_no_bet_line():
    text = format_prediction(_Pred(), edge_threshold=0.05)
    assert "Man City (H)" in text
    assert "Arsenal (A)" in text
    # League shown by its human name (not the "PD" code), never "None".
    assert "La Liga" in text
    assert "PD" not in text
    assert "None" not in text
    # Model probabilities and kickoff are present.
    assert "H 39% / D 31% / A 30%" in text
    assert "22:30 EAT" in text  # 19:30 UTC rendered in EAT (UTC+3)
    # Pure tipster: NO bet / no-bet / market / edge language anywhere.
    assert "Bet" not in text
    assert "NO BET" not in text
    assert "Edge" not in text
    assert "Market" not in text


def test_bet_line_absent_even_when_a_paper_bet_exists():
    # A stored paper_bet (internal reco record) must NOT surface to the user.
    p = _Pred(paper_bets=[_Bet(outcome="HOME", market_price=0.45, edge=0.06)])
    text = format_prediction(p, edge_threshold=0.05)
    assert "Bet" not in text
    assert "Edge" not in text
    assert "Market" not in text
    # Prediction body still present.
    assert "Man City (H)" in text
    assert "H 39% / D 31% / A 30%" in text


def test_xg_shown_when_present_and_hidden_when_absent():
    with_xg = format_prediction(_Pred(), edge_threshold=0.05)
    assert "xG 1.44" in with_xg
    without = format_prediction(_Pred(home_xg=None, away_xg=None), edge_threshold=0.05)
    assert "xG" not in without


def test_lineup_variant_shows_xi_and_probs_but_no_bet_line():
    lineup = {
        "home": {"formation": "4-3-3", "xi": ["Ederson", "Walker", "Dias"]},
        "away": {"formation": "4-2-3-1", "xi": ["Raya", "White", "Saliba"]},
    }
    text = format_prediction_with_lineup(
        _Pred(paper_bets=[_Bet(outcome="HOME", market_price=0.45, edge=0.06)]),
        lineup,
        edge_threshold=0.05,
        adj_note="(lineup adjusted)",
    )
    # Confirmed XI block present.
    assert "Man City (H)* XI: [4-3-3] Ederson, Walker, Dias" in text
    assert "Arsenal (A)* XI: [4-2-3-1] Raya, White, Saliba" in text
    assert "(lineup adjusted)" in text
    # Prediction body present.
    assert "H 39% / D 31% / A 30%" in text
    assert "xG 1.44" in text
    # No bet-call language anywhere.
    assert "Bet" not in text
    assert "NO BET" not in text
    assert "Edge" not in text
    assert "Market" not in text


def test_format_locked_hides_probabilities():
    text = format_locked(_Pred())
    assert "Man City (H)" in text
    assert "Arsenal (A)" in text
    assert "La Liga" in text  # league label on the paywall teaser too
    assert "22:30 EAT" in text  # 19:30 UTC rendered in EAT (UTC+3)
    assert "🔒" in text
    assert "1 USDC" in text
    # No probabilities leak in the teaser.
    assert "%" not in text
    assert "Model" not in text
    assert "xG" not in text


@dataclass
class _Outcome:
    predicted_home: float
    predicted_draw: float
    predicted_away: float
    predicted_pick: str
    correct: bool
    home_goals: int
    away_goals: int
    competition_code: str = "PL"


def test_format_result_correct_pick():
    o = _Outcome(0.60, 0.25, 0.15, "HOME", True, 2, 0)
    text = format_result(o, "Arsenal", "Chelsea", competition_code="PL")
    assert "Full time: Arsenal 2-0 Chelsea" in text
    assert "Premier League" in text  # human league label on the result alert
    assert "Our call: Arsenal (H) — ✅ correct" in text
    assert "H 60% / D 25% / A 15%" in text


def test_format_result_unknown_code_degrades_and_omits_when_missing():
    o = _Outcome(0.60, 0.25, 0.15, "HOME", True, 2, 0)
    # Unknown code -> raw code, never a crash, never "None".
    raw = format_result(o, "A", "B", competition_code="WC")
    assert "World Cup" in raw  # WC is mapped
    other = format_result(o, "A", "B", competition_code="ZZ9")
    assert "ZZ9" in other and "None" not in other
    # No code at all (legacy caller) -> no league label, no "None".
    none = format_result(o, "A", "B")
    assert "None" not in none
    assert " · " not in none


def test_format_result_wrong_pick_away_label():
    o = _Outcome(0.20, 0.30, 0.50, "AWAY", False, 1, 1)
    text = format_result(o, "Home", "Away")
    assert "Full time: Home 1-1 Away" in text
    assert "Our call: Away (A) — ❌ wrong" in text


def test_format_prediction_states_predicted_winner():
    from types import SimpleNamespace
    from betbot.tips import format_prediction
    home_fav = SimpleNamespace(home_team="Man City", away_team="Arsenal",
                               p_home=0.52, p_draw=0.26, p_away=0.22,
                               home_xg=1.9, away_xg=1.1, competition_code="PL",
                               kickoff=None, paper_bets=[])
    out = format_prediction(home_fav)
    assert "Prediction:" in out and "Man City to win" in out and "52%" in out
    draw_top = SimpleNamespace(home_team="A", away_team="B",
                               p_home=0.30, p_draw=0.40, p_away=0.30,
                               home_xg=None, away_xg=None, competition_code="PL",
                               kickoff=None, paper_bets=[])
    assert "Prediction: Draw" in format_prediction(draw_top)
    away_fav = SimpleNamespace(home_team="A", away_team="B",
                               p_home=0.20, p_draw=0.25, p_away=0.55,
                               home_xg=None, away_xg=None, competition_code="PL",
                               kickoff=None, paper_bets=[])
    assert "B to win" in format_prediction(away_fav)


# ----------------------------------------------------------------------
# The BET / NO BET call is RETIRED (2026-09-08): it must never appear,
# even with the (now display-inert) confidence filter flag ON.
# ----------------------------------------------------------------------
from betbot.config import Settings  # noqa: E402


def _settings(**kw):
    base = dict(
        _env_file=None,
        FOOTBALL_DATA_API_KEY="fake-test-key",
        BETBOT_CONFIDENCE_FILTER="true",
        BETBOT_CONFIDENCE_THRESHOLD="0.60",
        BETBOT_CONFIDENCE_DRAW_MARGIN="0.05",
    )
    base.update(kw)
    return Settings(**base)


def test_no_bet_call_even_with_confidence_filter_on():
    # A strong favourite that once rendered "*BET: ... to win*" must now render
    # no call line at all — the flag no longer displays anything.
    pred = _Pred(p_home=0.72, p_draw=0.16, p_away=0.12)
    text = format_prediction(pred, settings=_settings())
    assert "BET" not in text          # neither BET nor NO BET
    assert "confidence bar" not in text
    # The named-club prediction and the triple still stand.
    assert "Man City to win" in text
    assert "H 72% / D 16% / A 12%" in text


def test_no_bet_call_below_threshold_either():
    pred = _Pred(p_home=0.55, p_draw=0.25, p_away=0.20)
    text = format_prediction(pred, settings=_settings())
    assert "BET" not in text


def test_lineup_wrapper_carries_no_bet_call():
    pred = _Pred(p_home=0.72, p_draw=0.16, p_away=0.12)
    lineup = {"home": {"formation": "4-3-3", "xi": ["A", "B"]}, "away": None}
    text = format_prediction_with_lineup(pred, lineup, settings=_settings())
    assert "BET" not in text
    assert "[4-3-3]" in text
