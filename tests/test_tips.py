"""Prediction delivery formatting — the tipster message shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from betbot.tips import format_locked, format_prediction, format_result


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
    paper_bets: list = field(default_factory=list)


def test_format_prediction_has_home_away_tags_and_bold_call():
    text = format_prediction(_Pred(), edge_threshold=0.05)
    assert "Man City (H)" in text
    assert "Arsenal (A)" in text
    # A bold bet/no-bet line is always present.
    assert "*Bet:" in text
    assert "19:30" in text


def test_no_bet_is_the_default_when_no_market():
    text = format_prediction(_Pred(paper_bets=[]), edge_threshold=0.05)
    assert "*Bet: NO BET*" in text
    assert "Market: n/a" in text


def test_xg_shown_when_present_and_hidden_when_absent():
    with_xg = format_prediction(_Pred(), edge_threshold=0.05)
    assert "xG 1.44" in with_xg
    without = format_prediction(_Pred(home_xg=None, away_xg=None), edge_threshold=0.05)
    assert "xG" not in without


def test_bet_call_when_edge_at_threshold():
    p = _Pred(paper_bets=[_Bet(outcome="HOME", market_price=0.45, edge=0.06)])
    text = format_prediction(p, edge_threshold=0.05)
    assert "Market: HOME 0.45" in text
    assert "Edge: +0.06" in text
    assert "*Bet: Man City (H)*" in text
    assert "NO BET" not in text


def test_priced_below_threshold_is_no_bet_but_shows_price():
    p = _Pred(paper_bets=[_Bet(outcome="HOME", market_price=0.45, edge=0.01)])
    text = format_prediction(p, edge_threshold=0.05)
    assert "*Bet: NO BET* (edge below threshold)" in text
    assert "Market: HOME 0.45" in text


def test_away_bet_uses_away_team_label():
    p = _Pred(paper_bets=[_Bet(outcome="AWAY", market_price=0.30, edge=0.08)])
    text = format_prediction(p, edge_threshold=0.05)
    assert "*Bet: Arsenal (A)*" in text


def test_format_locked_hides_probabilities():
    text = format_locked(_Pred())
    assert "Man City (H)" in text
    assert "Arsenal (A)" in text
    assert "19:30" in text
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


def test_format_result_correct_pick():
    o = _Outcome(0.60, 0.25, 0.15, "HOME", True, 2, 0)
    text = format_result(o, "Arsenal", "Chelsea")
    assert "Full time: Arsenal 2-0 Chelsea" in text
    assert "Our call: Arsenal (H) — ✅ correct" in text
    assert "H 60% / D 25% / A 15%" in text


def test_format_result_wrong_pick_away_label():
    o = _Outcome(0.20, 0.30, 0.50, "AWAY", False, 1, 1)
    text = format_result(o, "Home", "Away")
    assert "Full time: Home 1-1 Away" in text
    assert "Our call: Away (A) — ❌ wrong" in text
