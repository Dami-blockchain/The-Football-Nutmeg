"""Anchoring-coverage watchdog + unanchored display label.

Two guards against the SILENT odds-anchor failure mode (feed down / empty for
our leagues / names stop resolving): the run still writes rows, so nothing
complains, but every row is unanchored and users see RAW probabilities.

* :func:`betbot.main._anchor_coverage_guard` alerts the operator when the
  anchored FRACTION of a non-empty run falls below the configured floor, and
  stays silent on a legitimately empty day.
* :func:`betbot.tips.anchor_note` lets a user-facing prediction say so, but only
  when the flag is on and the fixture really is unanchored.
"""

from __future__ import annotations

import asyncio
import dataclasses

from betbot import main as main_mod
from betbot.strategy.engine import Prediction
from betbot.tips import anchor_note, format_prediction


class _Settings:
    odds_anchor_min_coverage = 0.5
    anchor_label_unanchored = False


def _pred(anchor_source=None):
    p = Prediction(
        fixture_id=1,
        competition_code="PL",
        home_team="Arsenal",
        away_team="Leeds",
        p_home=0.71,
        p_draw=0.18,
        p_away=0.11,
        home_score=1500.0,
        away_score=1500.0,
        draw_score=0.0,
    )
    return dataclasses.replace(p, anchor_source=anchor_source)


def _run_guard(settings, coverage, monkeypatch):
    sent: list[dict] = []

    async def _fake_notify(s, text, **kwargs):
        sent.append({"text": text, **kwargs})
        return True

    monkeypatch.setattr(main_mod, "notify_operator", _fake_notify)
    asyncio.run(main_mod._anchor_coverage_guard(settings, coverage))
    return sent


# ---------------- guard ----------------
def test_guard_fires_when_coverage_below_floor(monkeypatch):
    sent = _run_guard(_Settings(), {"scored": 10, "anchored": 2}, monkeypatch)
    assert len(sent) == 1
    assert sent[0]["kind"] == "anchor_coverage_low"
    assert "2/10" in sent[0]["text"]


def test_guard_silent_when_coverage_ok(monkeypatch):
    sent = _run_guard(_Settings(), {"scored": 10, "anchored": 8}, monkeypatch)
    assert sent == []


def test_guard_silent_on_empty_day(monkeypatch):
    # 0 scored -> legitimately fixture-free day, never alert.
    sent = _run_guard(_Settings(), {"scored": 0, "anchored": 0}, monkeypatch)
    assert sent == []


def test_guard_disabled_when_floor_zero(monkeypatch):
    class S(_Settings):
        odds_anchor_min_coverage = 0.0

    sent = _run_guard(S(), {"scored": 10, "anchored": 0}, monkeypatch)
    assert sent == []


def test_guard_fires_on_total_outage(monkeypatch):
    # The live 10-08 shape: rows written, zero anchored.
    sent = _run_guard(_Settings(), {"scored": 27, "anchored": 0}, monkeypatch)
    assert len(sent) == 1
    assert "0/27" in sent[0]["text"]


# ---------------- label ----------------
def test_anchor_note_off_by_default():
    assert anchor_note(_pred(anchor_source=None), _Settings()) is None


def test_anchor_note_none_when_anchored_even_if_flag_on():
    class S(_Settings):
        anchor_label_unanchored = True

    assert anchor_note(_pred(anchor_source="odds"), S()) is None


def test_anchor_note_shown_when_flag_on_and_unanchored():
    class S(_Settings):
        anchor_label_unanchored = True

    note = anchor_note(_pred(anchor_source=None), S())
    assert note is not None and "Model-only" in note


def test_format_prediction_no_note_by_default():
    out = format_prediction(_pred(anchor_source=None), settings=_Settings())
    assert "Model-only" not in out


def test_format_prediction_adds_note_when_flag_on_and_unanchored():
    class S(_Settings):
        anchor_label_unanchored = True

    out = format_prediction(_pred(anchor_source=None), settings=S())
    assert "Model-only" in out


def test_format_prediction_no_note_when_anchored():
    class S(_Settings):
        anchor_label_unanchored = True

    out = format_prediction(_pred(anchor_source="odds"), settings=S())
    assert "Model-only" not in out
