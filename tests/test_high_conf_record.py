"""All-time high-confidence record (``/record all``).

Covers :func:`betbot.storage.repos.high_conf_record_all` (the sold-preferred
as-shown basis, anchored/unanchored split, bug-era start guard, club-only +
season scope inherited from ``prediction_outcomes_since``) and the Telegram
renderer :func:`betbot.telegram_bot._render_high_conf_record_body`.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionRow
from betbot.storage.repos import (
    HIGH_CONF_RECORD_START,
    high_conf_record_all,
    record_prediction_outcome,
    record_reveal,
)

SETTLED = datetime(2026, 9, 10, 20, 0, tzinfo=timezone.utc)
THIS_SEASON = datetime(2026, 8, 25, 19, 0, tzinfo=timezone.utc)
BUG_ERA = datetime(2026, 8, 10, 19, 0, tzinfo=timezone.utc)  # < start, >= season
WORLD_CUP = datetime(2026, 7, 18, 21, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path, monkeypatch):
    init_engine(tmp_path / "hc_record.sqlite")
    from betbot.config import get_settings

    monkeypatch.setenv("BETBOT_SEASON_START", "2026-08-01")
    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", "2026-08-17")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _pred_row(fid, home, away, *, code="PL", ko=THIS_SEASON):
    with session_scope() as s:
        s.add(PredictionRow(
            fixture_id=fid, competition_code=code, kickoff=ko,
            run_date=ko.date().isoformat(), home_team=home, away_team=away,
            p_home=0.5, p_draw=0.3, p_away=0.2,
            home_score=0.0, away_score=0.0, draw_score=0.0,
        ))


def _outcome(fid, ph, pd, pa, outcome, *, code="PL", ko=THIS_SEASON, anchor=None):
    record_prediction_outcome(
        fixture_id=fid, competition_code=code,
        p_home=ph, p_draw=pd, p_away=pa,
        actual_outcome=outcome, home_goals=2, away_goals=0,
        settled_at=SETTLED, kickoff=ko, anchor_source=anchor,
    )


def test_start_constant_is_operator_decision():
    # Operator decision 2026-10-09: all-time record begins when club matches did.
    assert HIGH_CONF_RECORD_START == date(2026, 8, 17)


def test_overall_and_anchor_split(db):
    _pred_row(1, "Arsenal", "Leeds")
    _outcome(1, 0.72, 0.18, 0.10, "HOME", anchor="odds")   # anchored, correct
    _pred_row(2, "Bayern", "Mainz")
    _outcome(2, 0.70, 0.20, 0.10, "AWAY", anchor="odds")   # anchored, wrong
    _pred_row(3, "Real", "Cadiz", code="CL")
    _outcome(3, 0.80, 0.12, 0.08, "HOME", code="CL")       # unanchored (CL), correct
    _pred_row(4, "Inter", "Lecce")
    _outcome(4, 0.55, 0.25, 0.20, "HOME")                  # below bar -> excluded
    _pred_row(5, "A", "B")
    _outcome(5, 0.20, 0.70, 0.10, "DRAW")                  # draw top -> excluded

    rec = high_conf_record_all(0.65)
    assert (rec["hits"], rec["n"]) == (2, 3)
    assert (rec["anchored"]["hits"], rec["anchored"]["n"]) == (1, 2)
    assert (rec["unanchored"]["hits"], rec["unanchored"]["n"]) == (1, 1)
    assert rec["bug_poisoned_excluded"] == 0
    # Per-call list carries teams + designation + pick + result.
    fids_home = {c["home"] for c in rec["calls"]}
    assert {"Arsenal", "Bayern", "Real"} == fids_home


def test_sold_triple_is_the_as_shown_basis(db):
    # Stored (post-rescore) triple dropped below the bar; the SOLD triple that
    # was shown cleared it -> the call still counts (judged at alert time).
    _pred_row(10, "Chelsea", "Burnley")
    _outcome(10, 0.50, 0.25, 0.25, "HOME")       # post-rescore, below bar
    record_reveal(1, 10, True, 0.72, 0.18, 0.10)  # sold above bar

    rec = high_conf_record_all(0.65)
    assert (rec["hits"], rec["n"]) == (1, 1)
    assert rec["calls"][0]["basis"] == "sold"
    assert rec["calls"][0]["conf"] == pytest.approx(0.72)


def test_bug_era_calls_excluded_and_counted(db):
    _pred_row(20, "Early", "Game", ko=BUG_ERA)
    _outcome(20, 0.80, 0.12, 0.08, "HOME", ko=BUG_ERA)   # pre-start -> excluded
    _pred_row(21, "Clean", "Game")
    _outcome(21, 0.75, 0.15, 0.10, "HOME")               # in scope -> counted

    rec = high_conf_record_all(0.65)
    assert (rec["hits"], rec["n"]) == (1, 1)
    assert rec["bug_poisoned_excluded"] == 1

    # start=None includes the bug-era call (nothing dropped).
    rec_all = high_conf_record_all(0.65, start=None)
    assert rec_all["n"] == 2
    assert rec_all["bug_poisoned_excluded"] == 0


def test_world_cup_excluded(db):
    _pred_row(30, "France", "England", code="WC", ko=WORLD_CUP)
    _outcome(30, 0.85, 0.10, 0.05, "HOME", code="WC", ko=WORLD_CUP)
    rec = high_conf_record_all(0.65)
    assert rec["n"] == 0


def test_empty_record(db):
    rec = high_conf_record_all(0.65)
    assert rec == {
        "n": 0, "hits": 0, "hit_rate": 0.0,
        "anchored": {"n": 0, "hits": 0, "hit_rate": 0.0},
        "unanchored": {"n": 0, "hits": 0, "hit_rate": 0.0},
        "bug_poisoned_excluded": 0,
        "start": HIGH_CONF_RECORD_START,
        "calls": [],
    }


def test_renderer_shows_split_and_calls(db):
    from betbot.telegram_bot import _render_high_conf_record_body

    _pred_row(1, "Arsenal", "Leeds")
    _outcome(1, 0.72, 0.18, 0.10, "HOME", anchor="odds")
    _pred_row(2, "Real", "Cadiz", code="CL")
    _outcome(2, 0.80, 0.12, 0.08, "AWAY", code="CL")  # unanchored, wrong

    body = _render_high_conf_record_body()
    assert "High-confidence record — all time" in body
    assert "Overall*: 1/2" in body
    assert "Market-anchored" in body and "Model-only" in body
    assert "Arsenal (H) v Leeds (A)" in body
    assert "✅" in body and "❌" in body


def test_renderer_empty(db):
    from betbot.telegram_bot import _render_high_conf_record_body

    body = _render_high_conf_record_body()
    assert "No settled high-confidence calls yet" in body
