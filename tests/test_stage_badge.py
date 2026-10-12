"""feat/cl-stage-logging — the CL competition stage shows on every alert surface.

A friendly, EMOJI-FREE stage label ("League phase" / "Knockout · Round of 16"
...) is rendered INLINE in the header of every Telegram surface that shows a
single CL fixture call (after the closing ``*`` so bold is never broken), gated
by ``BETBOT_STAGE_BADGE`` (default ON). Facts under test:

* ``stage_label`` maps each football-data.org CL enum to its friendly label,
  prefixes the knockout rounds, and returns ``None`` for domestic (no stage),
  blank/NULL, unknown values, missing settings, and the flag off;
* the prediction, confirmed-XI, high-conf, result, matchday-notice and
  promotion-notice bodies all carry the label inline for a CL fixture;
* a domestic fixture, a CL fixture with NULL/unknown stage, and the flag off all
  render NOTHING (and never crash);
* no NO-BET / caution / accuracy copy is introduced.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from betbot.daily_jobs import (
    render_matchday_notice,
    render_morning_promotion_notice,
)
from betbot.notify import format_high_conf_alert
from betbot.tips import (
    format_prediction,
    format_prediction_with_lineup,
    format_result,
    stage_label,
)

KO = datetime(2026, 11, 4, 20, 0, tzinfo=timezone.utc)


def _pred(stage=None, code="CL", **kw):
    base = dict(
        fixture_id=1,
        home_team="Real Madrid",
        away_team="Liverpool",
        p_home=0.66,
        p_draw=0.20,
        p_away=0.14,
        home_xg=1.8,
        away_xg=0.9,
        kickoff=KO,
        competition_code=code,
        stage=stage,
        anchor_source="polymarket",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _outcome(stage=None, code="CL"):
    return SimpleNamespace(
        fixture_id=1,
        predicted_pick="HOME",
        correct=True,
        actual_outcome="HOME",
        home_goals=2,
        away_goals=1,
        predicted_home=0.66,
        predicted_draw=0.20,
        predicted_away=0.14,
        competition_code=code,
        anchor_source="polymarket",
        stage=stage,
    )


# ---- label mapping (emoji-free) -------------------------------------------


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("LEAGUE_STAGE", "League phase"),
        ("PLAYOFFS", "Knockout play-off"),
        ("LAST_16", "Knockout · Round of 16"),
        ("QUARTER_FINALS", "Knockout · Quarter-final"),
        ("SEMI_FINALS", "Knockout · Semi-final"),
        ("FINAL", "Knockout · Final"),
        ("last_16", "Knockout · Round of 16"),  # case-insensitive
        (" FINAL ", "Knockout · Final"),  # trimmed
    ],
)
def test_stage_label_mapping(settings, stage, expected):
    assert stage_label(stage, settings) == expected


def test_labels_carry_no_emoji(settings):
    for s in ("LEAGUE_STAGE", "PLAYOFFS", "LAST_16", "FINAL"):
        assert "🏆" not in stage_label(s, settings)


def test_knockout_labels_read_as_knockout(settings):
    for s in ("PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"):
        assert "Knockout" in stage_label(s, settings)
    assert "Knockout" not in stage_label("LEAGUE_STAGE", settings)


@pytest.mark.parametrize("stage", [None, "", "   ", "GROUP_B", "UNKNOWN"])
def test_stage_label_nothing_for_null_or_unknown(settings, stage):
    assert stage_label(stage, settings) is None


def test_stage_label_none_settings_and_flag_off(settings):
    assert stage_label("FINAL", None) is None
    off = settings.model_copy(update={"stage_badge": False})
    assert stage_label("FINAL", off) is None


# ---- format_prediction (inline in header) ---------------------------------


def test_prediction_cl_league_phase_inline(settings):
    text = format_prediction(_pred(stage="LEAGUE_STAGE"), settings=settings)
    first = text.splitlines()[0]
    assert "Real Madrid (H) v Liverpool (A)" in first
    assert "· Champions League · League phase —" in first  # inline, not own line
    assert "67%" not in text and "76%" not in text  # no accuracy figures
    assert "NO BET" not in text and "weaker" not in text


def test_prediction_cl_knockout_inline(settings):
    first = format_prediction(_pred(stage="LAST_16"), settings=settings).splitlines()[0]
    assert "· Champions League · Knockout · Round of 16 —" in first


def test_prediction_domestic_shows_no_stage(settings):
    text = format_prediction(_pred(stage=None, code="PL"), settings=settings)
    assert "League phase" not in text and "Knockout" not in text


def test_prediction_cl_null_stage_shows_nothing(settings):
    text = format_prediction(_pred(stage=None), settings=settings)
    assert "League phase" not in text and "Knockout" not in text


def test_prediction_flag_off_shows_nothing(settings):
    off = settings.model_copy(update={"stage_badge": False})
    text = format_prediction(_pred(stage="FINAL"), settings=off)
    assert "Knockout" not in text


def test_prediction_with_lineup_inherits_stage(settings):
    first = format_prediction_with_lineup(
        _pred(stage="SEMI_FINALS"), None, settings=settings
    ).splitlines()[0]
    assert "· Champions League · Knockout · Semi-final —" in first


# ---- format_result (inline in the Full-time header) -----------------------


def test_result_cl_knockout_inline(settings):
    text = format_result(
        _outcome(stage="QUARTER_FINALS"),
        "Real Madrid",
        "Liverpool",
        competition_code="CL",
        settings=settings,
    )
    first = text.splitlines()[0]
    assert first.startswith("*Full time: Real Madrid 2-1 Liverpool*")
    assert "· Champions League · Knockout · Quarter-final" in first


def test_result_null_stage_shows_nothing(settings):
    text = format_result(
        _outcome(stage=None),
        "Real Madrid",
        "Liverpool",
        competition_code="CL",
        settings=settings,
    )
    assert "League phase" not in text and "Knockout" not in text


# ---- format_high_conf_alert (inline in header) ----------------------------


def test_high_conf_alert_cl_stage_inline(settings):
    first = format_high_conf_alert(_pred(stage="FINAL"), settings).splitlines()[0]
    assert "Knockout · Final" in first


def test_high_conf_alert_domestic_no_stage(settings):
    text = format_high_conf_alert(_pred(stage=None, code="PL"), settings)
    assert "League phase" not in text and "Knockout" not in text


# ---- matchday notice -------------------------------------------------------


def test_matchday_notice_cl_stage(settings):
    body = render_matchday_notice(
        settings, [_pred(stage="LEAGUE_STAGE")], date(2026, 11, 4)
    )
    assert "· Champions League · League phase —" in body


def test_matchday_notice_flag_off(settings):
    off = settings.model_copy(update={"stage_badge": False})
    body = render_matchday_notice(off, [_pred(stage="LEAGUE_STAGE")], date(2026, 11, 4))
    assert "League phase" not in body


# ---- promotion notice ------------------------------------------------------


def test_promotion_notice_cl_stage(settings):
    body = render_morning_promotion_notice(_pred(stage="LAST_16"), settings)
    assert "· Champions League · Knockout · Round of 16" in body


def test_promotion_notice_domestic_no_stage(settings):
    body = render_morning_promotion_notice(_pred(stage=None, code="PL"), settings)
    assert "League phase" not in body and "Knockout" not in body


def test_promotion_notice_no_settings_backcompat():
    body = render_morning_promotion_notice(_pred(stage="FINAL"))
    assert "League phase" not in body and "Knockout" not in body
