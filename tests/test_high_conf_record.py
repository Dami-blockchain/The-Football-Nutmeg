"""All-time high-confidence record (``/record all``).

Covers :func:`betbot.storage.repos.high_conf_record_all` — the sold-preferred
as-shown basis, the anchored/unanchored split attributed to WHAT WAS SOLD
(stored reveal anchor, else recovered by matching the sold triple to its
predictions row), the stored-fallback reveal-row requirement, the bug-era start
guard, and the Telegram renderer
:func:`betbot.telegram_bot._render_high_conf_record_body`.
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


def _pred_row(fid, home, away, *, code="PL", ko=THIS_SEASON,
              p=(0.5, 0.3, 0.2), anchor=None, raw=None, run_date=None):
    rh, rd, ra = raw if raw is not None else (None, None, None)
    with session_scope() as s:
        s.add(PredictionRow(
            fixture_id=fid, competition_code=code, kickoff=ko,
            run_date=run_date or ko.date().isoformat(), home_team=home, away_team=away,
            p_home=p[0], p_draw=p[1], p_away=p[2],
            raw_p_home=rh, raw_p_draw=rd, raw_p_away=ra,
            anchor_source=anchor,
            home_score=0.0, away_score=0.0, draw_score=0.0,
        ))


def _outcome(fid, ph, pd, pa, outcome, *, code="PL", ko=THIS_SEASON, anchor=None):
    record_prediction_outcome(
        fixture_id=fid, competition_code=code,
        p_home=ph, p_draw=pd, p_away=pa,
        actual_outcome=outcome, home_goals=2, away_goals=0,
        settled_at=SETTLED, kickoff=ko, anchor_source=anchor,
    )


def _sold(fid, ph, pd, pa, *, anchor=None):
    # A reveal carrying the sold triple + its stored anchor state.
    record_reveal(1, fid, True, ph, pd, pa, anchor_source=anchor)


def test_start_constant_is_operator_decision():
    # Operator decision 2026-10-09: all-time record begins when club matches did.
    assert HIGH_CONF_RECORD_START == date(2026, 8, 17)


def test_overall_and_anchor_split(db):
    _pred_row(1, "Arsenal", "Leeds")
    _outcome(1, 0.72, 0.18, 0.10, "HOME", anchor="odds")
    _sold(1, 0.72, 0.18, 0.10, anchor="odds")   # anchored, correct
    _pred_row(2, "Bayern", "Mainz")
    _outcome(2, 0.70, 0.20, 0.10, "AWAY", anchor="odds")
    _sold(2, 0.70, 0.20, 0.10, anchor="odds")   # anchored, wrong
    # Model-only (CL, no market line): reveal carries NULL anchor, so the split
    # recovers "model-only (known)" by matching the sold triple to this row.
    _pred_row(3, "Real", "Cadiz", code="CL", p=(0.80, 0.12, 0.08), anchor=None)
    _outcome(3, 0.80, 0.12, 0.08, "HOME", code="CL")
    _sold(3, 0.80, 0.12, 0.08, anchor=None)     # model-only, correct
    _pred_row(4, "Inter", "Lecce")
    _outcome(4, 0.55, 0.25, 0.20, "HOME")
    _sold(4, 0.55, 0.25, 0.20)
    _pred_row(5, "A", "B")
    _outcome(5, 0.20, 0.70, 0.10, "DRAW")
    _sold(5, 0.20, 0.70, 0.10)

    rec = high_conf_record_all(0.65)
    assert (rec["hits"], rec["n"]) == (2, 3)
    assert (rec["anchored"]["hits"], rec["anchored"]["n"]) == (1, 2)
    assert (rec["unanchored"]["hits"], rec["unanchored"]["n"]) == (1, 1)
    assert rec["anchor_unknown"] == 0
    assert rec["pre_ledger"] == 0
    assert rec["bug_poisoned_excluded"] == 0
    assert {c["home"] for c in rec["calls"]} == {"Arsenal", "Bayern", "Real"}


def test_sold_triple_is_the_as_shown_basis(db):
    # Stored (post-rescore) triple dropped below the bar; the SOLD triple that
    # was shown cleared it -> the call still counts (judged at alert time).
    _pred_row(10, "Chelsea", "Burnley")
    _outcome(10, 0.50, 0.25, 0.25, "HOME")       # post-rescore, below bar
    _sold(10, 0.72, 0.18, 0.10, anchor="odds")   # sold above bar, anchored

    rec = high_conf_record_all(0.65)
    assert (rec["hits"], rec["n"]) == (1, 1)
    assert rec["calls"][0]["basis"] == "sold"
    assert rec["calls"][0]["conf"] == pytest.approx(0.72)
    assert rec["calls"][0]["anchored"] is True


def test_anchor_split_follows_sold_not_final_state(db):
    # Fable's 564681 case: sold on the RAW (unanchored) triple before a later
    # rescore anchored it. The outcome row reads 'odds' (final) but the call was
    # shown model-only. Reveal carries NULL anchor (legacy) -> recovered from the
    # predictions rows: the sold triple == an unanchored row's p_* / == the
    # anchored row's raw_p_*, so it is classified UNANCHORED.
    raw = (0.15, 0.16, 0.69)
    anch = (0.13, 0.15, 0.72)
    _pred_row(40, "PSV", "Shakhtar", p=raw, anchor=None,
              run_date="2026-08-24")                                 # early, raw
    _pred_row(40, "PSV", "Shakhtar", p=anch, anchor="odds", raw=raw,
              run_date="2026-08-25")                                 # later, anchored
    _outcome(40, anch[0], anch[1], anch[2], "AWAY", anchor="odds")   # final = anchored
    record_reveal(1, 40, True, raw[0], raw[1], raw[2])               # sold raw, NULL anchor

    rec = high_conf_record_all(0.65)
    assert (rec["hits"], rec["n"]) == (1, 1)   # AWAY correct
    assert rec["anchored"]["n"] == 0
    assert rec["unanchored"]["n"] == 1         # attributed to the SOLD (raw) state
    assert rec["calls"][0]["anchored"] is False


def test_stored_fallback_requires_a_reveal_row(db):
    # No sold triple on either; one fixture was revealed (NULL-triple legacy
    # row), the other never reached anyone -> only the revealed one counts, as a
    # pre-ledger (stored-basis) call.
    _pred_row(50, "Seen", "Team")
    _outcome(50, 0.75, 0.15, 0.10, "HOME", anchor="odds")
    record_reveal(1, 50, True)       # legacy NULL-triple reveal row
    _pred_row(51, "Unseen", "Team")
    _outcome(51, 0.80, 0.12, 0.08, "HOME", anchor="odds")
    # fid 51 has NO reveal row at all.

    rec = high_conf_record_all(0.65)
    assert rec["n"] == 1
    assert rec["calls"][0]["home"] == "Seen"
    assert rec["calls"][0]["basis"] == "stored"
    assert rec["pre_ledger"] == 1
    assert rec["anchored"]["n"] == 1   # stored basis uses the final anchor state


def test_bug_era_calls_excluded_and_counted(db):
    _pred_row(20, "Early", "Game", ko=BUG_ERA)
    _outcome(20, 0.80, 0.12, 0.08, "HOME", ko=BUG_ERA)   # pre-start -> excluded
    _sold(20, 0.80, 0.12, 0.08, anchor="odds")
    _pred_row(21, "Clean", "Game")
    _outcome(21, 0.75, 0.15, 0.10, "HOME")
    _sold(21, 0.75, 0.15, 0.10, anchor="odds")           # in scope -> counted

    rec = high_conf_record_all(0.65)
    assert (rec["hits"], rec["n"]) == (1, 1)
    assert rec["bug_poisoned_excluded"] == 1

    rec_all = high_conf_record_all(0.65, start=None)     # include everything
    assert rec_all["n"] == 2
    assert rec_all["bug_poisoned_excluded"] == 0


def test_world_cup_excluded(db):
    _pred_row(30, "France", "England", code="WC", ko=WORLD_CUP)
    _outcome(30, 0.85, 0.10, 0.05, "HOME", code="WC", ko=WORLD_CUP)
    _sold(30, 0.85, 0.10, 0.05, anchor="odds")
    assert high_conf_record_all(0.65)["n"] == 0


def test_empty_record(db):
    rec = high_conf_record_all(0.65)
    assert rec == {
        "n": 0, "hits": 0, "hit_rate": 0.0,
        "anchored": {"n": 0, "hits": 0, "hit_rate": 0.0},
        "unanchored": {"n": 0, "hits": 0, "hit_rate": 0.0},
        "anchor_unknown": 0,
        "pre_ledger": 0,
        "bug_poisoned_excluded": 0,
        "start": HIGH_CONF_RECORD_START,
        "calls": [],
    }


def test_renderer_shows_split_ci_and_calls(db):
    from betbot.telegram_bot import _render_high_conf_record_body

    _pred_row(1, "Arsenal", "Leeds")
    _outcome(1, 0.72, 0.18, 0.10, "HOME", anchor="odds")
    _sold(1, 0.72, 0.18, 0.10, anchor="odds")
    _pred_row(2, "Real", "Cadiz", code="CL")
    _outcome(2, 0.80, 0.12, 0.08, "AWAY", code="CL")
    _sold(2, 0.80, 0.12, 0.08, anchor=None)

    body = _render_high_conf_record_body()
    assert "High-confidence record — all time" in body
    assert "p≥0.65" in body               # >= rendered as the unicode sign
    assert "Overall*: 1/2" in body
    assert "95% CI" in body
    assert "Market-anchored" in body and "Model-only" in body
    assert "Arsenal (H) v Leeds (A)" in body
    assert "✅" in body and "❌" in body


def test_renderer_empty(db):
    from betbot.telegram_bot import _render_high_conf_record_body

    assert "No settled high-confidence calls yet" in _render_high_conf_record_body()


# ---------------- sold_call / anchor recovery ----------------
def test_sold_call_stored_anchor_wins(db):
    from betbot.storage.repos import sold_call
    _sold(60, 0.7, 0.2, 0.1, anchor="market")
    triple, anchor, known = sold_call(60)
    assert triple == pytest.approx((0.7, 0.2, 0.1))
    assert (anchor, known) == ("market", True)


def test_sold_call_recovers_when_reveal_anchor_null(db):
    from betbot.storage.repos import sold_call
    raw = (0.15, 0.16, 0.69)
    anch = (0.13, 0.15, 0.72)
    _pred_row(61, "A", "B", p=anch, anchor="odds", raw=raw)
    record_reveal(1, 61, True, raw[0], raw[1], raw[2])  # NULL anchor, sold raw
    triple, anchor, known = sold_call(61)
    assert triple == pytest.approx(raw)
    assert (anchor, known) == (None, True)   # known-unanchored via recovery


def test_sold_call_unknown_when_no_match(db):
    from betbot.storage.repos import sold_call
    record_reveal(1, 62, True, 0.9, 0.05, 0.05)  # no predictions row to match
    triple, anchor, known = sold_call(62)
    assert triple == pytest.approx((0.9, 0.05, 0.05))
    assert (anchor, known) == (None, False)


def test_sold_call_none_without_reveal(db):
    from betbot.storage.repos import sold_call
    assert sold_call(999) == (None, None, False)
