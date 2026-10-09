"""Anchoring-coverage watchdog + unanchored display label.

Two guards against the SILENT odds-anchor failure mode (feed down / empty for
our leagues / names stop resolving): the run still writes rows, so nothing
complains, but every row is unanchored and users see RAW probabilities.

* :func:`betbot.main._tally_anchor_coverage` counts only ANCHORABLE (big-5)
  fixtures -- CL can never be anchored and must not drag the ratio down.
* :func:`betbot.main._anchor_coverage_guard` alerts the operator when the
  anchored FRACTION of a non-empty run falls below the floor, and stays silent
  on an empty day, a thin day, a disabled anchor, or a deliberate rollback.
* :func:`betbot.tips.anchor_note` lets a user-facing prediction say so: it now
  marks BOTH states (anchored / model-only) and is ON by default
  (``BETBOT_ANCHOR_BADGE``), so the badge tests below assert the default-ON
  behaviour and the one-env rollback.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from betbot import main as main_mod
from betbot.daily_jobs import render_user_predictions
from betbot.entitlement import Entitlement
from betbot.notify import format_high_conf_alert
from betbot.tips import anchor_note, format_prediction


def _pred(anchor_source=None, *, code="PL", fid=1):
    return SimpleNamespace(
        fixture_id=fid,
        competition_code=code,
        home_team="Arsenal",
        away_team="Leeds",
        p_home=0.71,
        p_draw=0.18,
        p_away=0.11,
        home_xg=1.6,
        away_xg=0.7,
        kickoff=None,
        anchor_source=anchor_source,
    )


def _cov_settings(settings, **over):
    base = {"odds_anchor_enabled": True, "odds_anchor_min_coverage": 0.5}
    base.update(over)
    return settings.model_copy(update=base)


def _run_guard(settings, coverage, monkeypatch):
    sent: list[dict] = []

    async def _fake_notify(s, text, **kwargs):
        sent.append({"text": text, **kwargs})
        return True

    monkeypatch.setattr(main_mod, "notify_operator", _fake_notify)
    asyncio.run(main_mod._anchor_coverage_guard(settings, coverage))
    return sent


# ---------------- _tally_anchor_coverage (item 1: CL excluded) ----------------
def test_tally_counts_only_anchorable_leagues():
    cov = {"scored": 0, "anchored": 0}
    for i in range(9):
        main_mod._tally_anchor_coverage(cov, "CL", _pred(anchor_source=None, code="CL", fid=i))
    main_mod._tally_anchor_coverage(cov, "PL", _pred(anchor_source="odds", code="PL", fid=99))
    # Only the single PL fixture is counted; the 9 CL fixtures are ignored.
    assert cov == {"scored": 1, "anchored": 1}


def test_tally_cl_only_run_counts_nothing():
    cov = {"scored": 0, "anchored": 0}
    for i in range(3):
        main_mod._tally_anchor_coverage(cov, "CL", _pred(anchor_source=None, code="CL", fid=i))
    assert cov == {"scored": 0, "anchored": 0}


def test_tally_none_coverage_is_noop():
    main_mod._tally_anchor_coverage(None, "PL", _pred())  # must not raise


def test_tally_unanchored_anchorable_counts_scored_not_anchored():
    cov = {"scored": 0, "anchored": 0}
    main_mod._tally_anchor_coverage(cov, "SA", _pred(anchor_source=None, code="SA"))
    assert cov == {"scored": 1, "anchored": 0}


# ---------------- guard ----------------
def test_guard_fires_when_coverage_below_floor(settings, monkeypatch):
    sent = _run_guard(_cov_settings(settings), {"scored": 10, "anchored": 2}, monkeypatch)
    assert len(sent) == 1
    assert sent[0]["kind"] == "anchor_coverage_low"
    assert "2/10" in sent[0]["text"]


def test_guard_silent_when_coverage_ok(settings, monkeypatch):
    sent = _run_guard(_cov_settings(settings), {"scored": 10, "anchored": 8}, monkeypatch)
    assert sent == []


def test_guard_silent_on_empty_day(settings, monkeypatch):
    sent = _run_guard(_cov_settings(settings), {"scored": 0, "anchored": 0}, monkeypatch)
    assert sent == []


def test_guard_silent_on_thin_day(settings, monkeypatch):
    # Item 5: fewer than 3 anchorable fixtures -> not meaningful, never page.
    sent = _run_guard(_cov_settings(settings), {"scored": 2, "anchored": 0}, monkeypatch)
    assert sent == []


def test_guard_silent_when_floor_zero(settings, monkeypatch):
    s = _cov_settings(settings, odds_anchor_min_coverage=0.0)
    sent = _run_guard(s, {"scored": 10, "anchored": 0}, monkeypatch)
    assert sent == []


def test_guard_silent_when_anchoring_disabled(settings, monkeypatch):
    # Item 3: a deliberate BETBOT_ODDS_ANCHOR=false rollback must not page daily.
    s = _cov_settings(settings, odds_anchor_enabled=False)
    sent = _run_guard(s, {"scored": 10, "anchored": 0}, monkeypatch)
    assert sent == []


def test_guard_fires_on_total_outage(settings, monkeypatch):
    sent = _run_guard(_cov_settings(settings), {"scored": 27, "anchored": 0}, monkeypatch)
    assert len(sent) == 1
    assert "0/27" in sent[0]["text"]


def test_guard_never_raises_when_notify_raises(settings, monkeypatch):
    async def _boom(s, text, **kwargs):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(main_mod, "notify_operator", _boom)
    # Must swallow the error -- a watchdog cannot break the scoring run.
    asyncio.run(main_mod._anchor_coverage_guard(_cov_settings(settings), {"scored": 10, "anchored": 0}))


# ---------------- anchor_note badge (default ON, BOTH states) ----------------
# The flag was repurposed 2026-10-09: BETBOT_ANCHOR_BADGE (default True) now
# marks BOTH states on every prediction-carrying message — "📊 Market-anchored"
# when a market line was used, "⚠️ Model-only" when none was. Off -> silent.
def test_anchor_note_marks_unanchored_by_default(settings):
    note = anchor_note(_pred(anchor_source=None), settings)
    assert note is not None and "Model-only" in note


def test_anchor_note_marks_anchored_by_default(settings):
    note = anchor_note(_pred(anchor_source="odds"), settings)
    assert note is not None and "Market-anchored" in note


def test_anchor_note_silent_when_badge_off(settings):
    s = settings.model_copy(update={"anchor_badge": False})
    assert anchor_note(_pred(anchor_source=None), s) is None
    assert anchor_note(_pred(anchor_source="odds"), s) is None


def test_anchor_note_none_without_settings():
    assert anchor_note(_pred(anchor_source=None), None) is None


def test_format_prediction_badges_unanchored_by_default(settings):
    assert "Model-only" in format_prediction(_pred(anchor_source=None), settings=settings)


def test_format_prediction_badges_anchored_by_default(settings):
    assert "Market-anchored" in format_prediction(_pred(anchor_source="odds"), settings=settings)


def test_format_prediction_silent_when_badge_off(settings):
    s = settings.model_copy(update={"anchor_badge": False})
    out = format_prediction(_pred(anchor_source="odds"), settings=s)
    assert "Market-anchored" not in out and "Model-only" not in out


# ---------------- format_high_conf_alert badge ----------------
def test_high_conf_alert_badges_unanchored_by_default(settings):
    assert "Model-only" in format_high_conf_alert(_pred(anchor_source=None), settings)


def test_high_conf_alert_badges_anchored_by_default(settings):
    assert "Market-anchored" in format_high_conf_alert(_pred(anchor_source="odds"), settings)


def test_high_conf_alert_silent_when_badge_off(settings):
    s = settings.model_copy(update={"anchor_badge": False})
    out = format_high_conf_alert(_pred(anchor_source=None), s)
    assert "Model-only" not in out and "Market-anchored" not in out


# ---------------- format_result badge ----------------
def test_format_result_badges_when_settings_passed(settings):
    from betbot.tips import format_result
    row = SimpleNamespace(
        predicted_pick="HOME", correct=True, actual_outcome="HOME",
        home_goals=2, away_goals=0,
        predicted_home=0.7, predicted_draw=0.2, predicted_away=0.1,
        anchor_source="odds",
    )
    body = format_result(row, "Arsenal", "Leeds", settings=settings)
    assert "Market-anchored" in body
    row.anchor_source = None
    assert "Model-only" in format_result(row, "Arsenal", "Leeds", settings=settings)


def test_format_result_silent_without_settings():
    from betbot.tips import format_result
    row = SimpleNamespace(
        predicted_pick="HOME", correct=True, actual_outcome="HOME",
        home_goals=2, away_goals=0,
        predicted_home=0.7, predicted_draw=0.2, predicted_away=0.1,
        anchor_source=None,
    )
    body = format_result(row, "Arsenal", "Leeds")
    assert "Model-only" not in body and "Market-anchored" not in body


def _outcome_row(anchor):
    return SimpleNamespace(
        predicted_pick="HOME", correct=True, actual_outcome="HOME",
        home_goals=2, away_goals=0,
        predicted_home=0.7, predicted_draw=0.2, predicted_away=0.1,
        anchor_source=anchor,
    )


def test_format_result_shown_anchor_overrides_outcome(settings):
    # The outcome row's FINAL state says anchored, but the call was SOLD
    # unanchored -> shown_anchor=None must win (badge model-only).
    from betbot.tips import format_result
    body = format_result(_outcome_row("odds"), "A", "B",
                         settings=settings, shown_anchor=None)
    assert "Model-only" in body and "Market-anchored" not in body


def test_format_result_show_badge_false_omits(settings):
    # Anchor state could not be established -> omit the badge entirely.
    from betbot.tips import format_result
    body = format_result(_outcome_row("odds"), "A", "B",
                         settings=settings, show_badge=False)
    assert "Model-only" not in body and "Market-anchored" not in body


# ---------------- render_user_predictions ----------------
def _ent(reason, *, trial=0, credits=0):
    allowed = reason in ("operator", "trial", "credit")
    return Entitlement(allowed=allowed, reason=reason,
                       trial_days_left=trial, credits_remaining=credits)


def test_render_user_predictions_badges_by_default(settings):
    text, _ = render_user_predictions(
        SimpleNamespace(telegram_user_id=111), [_pred(anchor_source=None)], settings,
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        already_revealed_fn=lambda uid, fid: False,
    )
    assert "Model-only" in text


def test_render_user_predictions_silent_when_badge_off(settings):
    s = settings.model_copy(update={"anchor_badge": False})
    text, _ = render_user_predictions(
        SimpleNamespace(telegram_user_id=111), [_pred(anchor_source=None)], s,
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        already_revealed_fn=lambda uid, fid: False,
    )
    assert "Model-only" not in text
