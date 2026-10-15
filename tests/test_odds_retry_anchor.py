"""Regression tests for the odds re-anchor fix (FIX 1) and the
anchor-coverage Markdown fix (FIX 2).

FIX 1 — the free odds card lands AFTER the 08:00 run, so a later lineup /
odds-retry rescore re-scores a fixture with no quote in hand. Three guarantees:

* ``upsert_prediction`` must NOT regress an already-anchored row to raw when the
  re-upsert carries no fresh anchor (this is what sent NULL-anchor reveals to
  paying users while the prediction row itself stayed 'odds');
* a GENUINE new quote still replaces the stored anchor;
* ``unanchored_anchorable_fixtures`` selects exactly the rows the retry pass
  should re-score (unanchored, not kicked off, big-5 only);
* ``OddsService.invalidate`` / ``has_rows_on_or_after`` behave.

FIX 2 — the ``anchor_coverage_low`` operator alert must be valid Telegram
legacy Markdown (a bare ``*`` in ``odds_refresh_*`` used to 400 every run).
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from betbot import main as main_mod
from betbot.data.odds import MatchOdds, OddsService
from betbot.data.odds_names import OddsNameResolver
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionRow
from betbot.storage.repos import (
    unanchored_anchorable_fixtures,
    upsert_prediction,
)
from betbot.strategy.engine import Prediction

NOW = datetime.now(timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "retry.sqlite")
    yield


def _raw(fixture_id: int, code: str = "PL") -> Prediction:
    return Prediction(
        fixture_id=fixture_id, competition_code=code, home_team="A", away_team="B",
        p_home=0.55, p_draw=0.25, p_away=0.20,
        home_score=1.0, away_score=0.0, draw_score=2.4,
    )


def _anchored(fixture_id: int, ph=0.48, pd=0.29, pa=0.23, code: str = "PL") -> Prediction:
    raw = _raw(fixture_id, code)
    return dataclasses.replace(
        raw, p_home=ph, p_draw=pd, p_away=pa,
        model_probs=(raw.p_home, raw.p_draw, raw.p_away),
        anchor_source="odds",
    )


def _row(fixture_id: int) -> PredictionRow:
    with session_scope() as s:
        row = s.query(PredictionRow).filter_by(fixture_id=fixture_id).one()
        s.expunge(row)
        return row


# ---------------------------------------------------------------------------
# upsert_prediction — anti-regression
# ---------------------------------------------------------------------------
def test_unanchored_rescore_does_not_regress_anchored_row(db):
    ko = NOW + timedelta(hours=5)
    upsert_prediction(_anchored(10), kickoff=ko)
    # A later no-quote rescore: raw triple, anchor_source None.
    upsert_prediction(_raw(10), kickoff=ko)
    row = _row(10)
    assert row.anchor_source == "odds"           # NOT blanked
    assert row.p_home == pytest.approx(0.48)       # anchored triple kept
    assert row.p_draw == pytest.approx(0.29)
    assert row.p_away == pytest.approx(0.23)
    assert row.raw_p_home == pytest.approx(0.55)   # raw snapshot intact


def test_genuine_new_quote_replaces_stored_anchor(db):
    ko = NOW + timedelta(hours=5)
    upsert_prediction(_anchored(11, ph=0.48), kickoff=ko)
    # A fresh anchor with a different line genuinely updates.
    upsert_prediction(_anchored(11, ph=0.40, pd=0.33, pa=0.27), kickoff=ko)
    row = _row(11)
    assert row.anchor_source == "odds"
    assert row.p_home == pytest.approx(0.40)


def test_first_anchor_on_a_previously_unanchored_row_applies(db):
    ko = NOW + timedelta(hours=5)
    upsert_prediction(_raw(12), kickoff=ko)        # unanchored baseline
    assert _row(12).anchor_source is None
    upsert_prediction(_anchored(12), kickoff=ko)   # card lands -> anchor
    row = _row(12)
    assert row.anchor_source == "odds"
    assert row.p_home == pytest.approx(0.48)


def test_unanchored_to_unanchored_refreshes_normally(db):
    ko = NOW + timedelta(hours=5)
    upsert_prediction(_raw(13), kickoff=ko)
    changed = dataclasses.replace(_raw(13), p_home=0.60, p_draw=0.22, p_away=0.18)
    upsert_prediction(changed, kickoff=ko)
    row = _row(13)
    assert row.anchor_source is None
    assert row.p_home == pytest.approx(0.60)       # no anchor to protect


# ---------------------------------------------------------------------------
# unanchored_anchorable_fixtures — retry-pass target selection
# ---------------------------------------------------------------------------
BIG5 = {"PL", "PD", "BL1", "SA", "FL1"}


def test_unanchored_anchorable_fixtures_selects_only_the_right_rows(db):
    future = NOW + timedelta(hours=6)
    past = NOW - timedelta(hours=6)
    upsert_prediction(_raw(1, "PL"), kickoff=future)        # want
    upsert_prediction(_anchored(2, code="PL"), kickoff=future)  # anchored -> skip
    upsert_prediction(_raw(3, "PL"), kickoff=past)          # kicked off -> skip
    upsert_prediction(_raw(4, "CL"), kickoff=future)        # not big-5 -> skip
    got = unanchored_anchorable_fixtures(NOW, BIG5)
    assert [fid for fid, _ in got] == [1]


def test_unanchored_anchorable_fixtures_empty_leagues_is_noop(db):
    upsert_prediction(_raw(5, "PL"), kickoff=NOW + timedelta(hours=6))
    assert unanchored_anchorable_fixtures(NOW, set()) == []


# ---------------------------------------------------------------------------
# OddsService.invalidate / has_rows_on_or_after
# ---------------------------------------------------------------------------
def _svc() -> OddsService:
    return OddsService(
        SimpleNamespace(),
        providers=[],
        resolver=OddsNameResolver(canonical=["a", "b"], name_map={}),
    )


def _match(d: date) -> MatchOdds:
    return MatchOdds(
        league="PL", match_date=d, home="a", away="b",
        price_home=2.0, price_draw=3.4, price_away=3.8, source="t", book="B365H",
    )


def test_invalidate_marks_cache_stale():
    svc = _svc()
    svc.load_rows([_match(date(2026, 10, 17))])
    assert not svc._is_stale()        # freshly loaded
    svc.invalidate()
    assert svc._is_stale()            # next prime will re-fetch


def test_has_rows_on_or_after_distinguishes_forward_from_past_card():
    svc = _svc()
    svc.load_rows([_match(date(2026, 10, 10))])   # a retained PAST card
    assert not svc.has_rows_on_or_after(date(2026, 10, 15))
    svc.load_rows([_match(date(2026, 10, 18))])   # a landed FORWARD card
    assert svc.has_rows_on_or_after(date(2026, 10, 15))


def test_has_rows_on_or_after_empty_index_is_false():
    assert not _svc().has_rows_on_or_after(date(2026, 1, 1))


# ---------------------------------------------------------------------------
# FIX 2 — anchor_coverage_low message is valid Telegram legacy Markdown
# ---------------------------------------------------------------------------
def _strip_code_spans(text: str) -> tuple[str, bool]:
    """Return (text outside backtick code spans, backticks_unbalanced)."""
    out: list[str] = []
    in_code = False
    for ch in text:
        if ch == "`":
            in_code = not in_code
            continue
        if not in_code:
            out.append(ch)
    return "".join(out), in_code


def test_anchor_coverage_low_message_is_valid_markdown():
    msg = main_mod._anchor_coverage_low_message(0, 25, 0.0, 0.65)
    # The log-key globs must be inside code spans.
    assert "`odds_refresh_*`" in msg
    assert "`odds_names_unresolved`" in msg
    stripped, unbalanced = _strip_code_spans(msg)
    assert not unbalanced                       # backticks balanced
    # Outside code spans legacy Markdown must have NO bare '*' (the 400 cause)
    # and balanced '_' (there are none, which is even).
    assert "*" not in stripped
    assert stripped.count("_") % 2 == 0
    # And it still carries the numbers so the operator can read it.
    assert "0/25" in msg and "65%" in msg


def test_guard_emits_valid_markdown_end_to_end(settings, monkeypatch):
    sent: list[str] = []

    async def _fake_notify(s, text, **kwargs):
        sent.append(text)
        return True

    monkeypatch.setattr(main_mod, "notify_operator", _fake_notify)
    s = settings.model_copy(
        update={"odds_anchor_enabled": True, "odds_anchor_min_coverage": 0.5}
    )
    asyncio.run(main_mod._anchor_coverage_guard(s, {"scored": 25, "anchored": 0}))
    assert len(sent) == 1
    stripped, unbalanced = _strip_code_spans(sent[0])
    assert not unbalanced and "*" not in stripped
