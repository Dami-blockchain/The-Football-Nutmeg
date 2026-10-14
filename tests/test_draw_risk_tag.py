"""Branch B — draw-risk tag.

The tag is a single market-grounded line added to a high-confidence prediction
alert when the de-vigged MARKET draw probability clears the threshold. It
suggests the safer NO-on-the-opponent play. Covers the gating matrix, the
opponent/pick wording for both sides, and that it renders on both the revealed
prediction and the high-confidence alert surfaces — and nowhere when OFF.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


from betbot.config import Settings
from betbot.notify import format_high_conf_alert
from betbot.tips import draw_risk_line, format_prediction

KO = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc)


@dataclass
class Row:
    home_team: str = "Arsenal"
    away_team: str = "Chelsea"
    p_home: float = 0.70
    p_draw: float = 0.18
    p_away: float = 0.12
    home_xg: float | None = 1.8
    away_xg: float | None = 0.9
    competition_code: str = "PL"
    kickoff: datetime = KO
    anchor_source: str | None = "odds"
    stage: str | None = None
    market_p_draw: float | None = 0.22


def _settings(**env):
    import os
    for k, v in env.items():
        os.environ[k] = v
    try:
        return Settings()
    finally:
        for k in env:
            os.environ.pop(k, None)


def test_none_when_flag_off():
    assert draw_risk_line(Row(), Settings()) is None


def test_none_when_no_market_draw():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    assert draw_risk_line(Row(market_p_draw=None), s) is None


def test_none_below_threshold():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    assert draw_risk_line(Row(market_p_draw=0.19), s) is None


def test_none_when_low_confidence():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    # top pick below 0.65 -> never carries the tag
    assert draw_risk_line(Row(p_home=0.55, p_draw=0.25, p_away=0.20, market_p_draw=0.25), s) is None


def test_none_when_draw_is_top_pick():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    assert draw_risk_line(Row(p_home=0.30, p_draw=0.66, p_away=0.04, market_p_draw=0.30), s) is None


def test_home_pick_says_no_on_away():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    line = draw_risk_line(Row(), s)
    assert line is not None
    assert "22%" in line
    assert "NO on Chelsea" in line
    assert "Arsenal win or draw" in line


def test_away_pick_says_no_on_home():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    row = Row(p_home=0.12, p_draw=0.18, p_away=0.70, market_p_draw=0.21)
    line = draw_risk_line(row, s)
    assert "NO on Arsenal" in line
    assert "Chelsea win or draw" in line


def test_threshold_is_configurable():
    s = _settings(BETBOT_DRAW_RISK_TAG="true", BETBOT_DRAW_RISK_THRESHOLD="0.25")
    assert draw_risk_line(Row(market_p_draw=0.22), s) is None
    assert draw_risk_line(Row(market_p_draw=0.26), s) is not None


def test_format_prediction_includes_tag_when_on():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    body = format_prediction(Row(), settings=s)
    assert "Draw risk" in body
    assert "NO on Chelsea" in body


def test_format_prediction_excludes_tag_when_off():
    body = format_prediction(Row(), settings=Settings())
    assert "Draw risk" not in body


def test_high_conf_alert_includes_tag_when_on():
    s = _settings(BETBOT_DRAW_RISK_TAG="true")
    body = format_high_conf_alert(Row(), s)
    assert "Draw risk" in body
    assert "NO on Chelsea" in body


def test_high_conf_alert_excludes_tag_when_off():
    body = format_high_conf_alert(Row(), Settings())
    assert "Draw risk" not in body
