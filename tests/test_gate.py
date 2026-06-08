"""Tests for the live-readiness gate (Phase 5)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot.gate import evaluate_gate
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PaperBet, PredictionRow
from betbot.storage.repos import trip_kill_switch

NOW = datetime.now(timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "gate.sqlite")
    yield


def _insert_settled(fixture_id, outcome, won, price, stake, settled_at):
    pnl = stake * (1.0 / price - 1.0) if won else -stake
    with session_scope() as s:
        pred = PredictionRow(
            fixture_id=fixture_id, competition_code="PL", kickoff=settled_at,
            run_date="2026-01-01", home_team="A", away_team="B",
            p_home=0.5, p_draw=0.3, p_away=0.2,
            home_score=1.0, away_score=0.0, draw_score=2.4,
        )
        s.add(pred)
        s.flush()
        s.add(PaperBet(
            prediction_id=pred.id, fixture_id=fixture_id, outcome=outcome,
            our_probability=0.5, market_price=price, edge=0.1, stake_usd=stake,
            rationale="t", settled_at=settled_at,
            settled_outcome=outcome if won else "AWAY", pnl_usd=pnl,
        ))


def _seed_passing(n=25, win_frac=0.6):
    """n settled market bets spread over ~20 days, profitable, kill switch clear."""
    wins = int(n * win_frac)
    for i in range(n):
        settled_at = NOW - timedelta(days=20) + timedelta(days=20 * i / n)
        _insert_settled(1000 + i, "HOME", i < wins, 0.5, 10.0, settled_at)


def test_gate_fails_on_empty(db, settings):
    g = evaluate_gate(settings)
    assert g.passed is False
    assert any("settled market bets" in r for r in g.reasons)


def test_gate_passes_with_good_record(db, settings):
    _seed_passing()
    g = evaluate_gate(settings)
    assert g.passed is True, g.reasons
    assert g.result.n == 25
    assert g.window_days_observed >= settings.gate_min_window_days


def test_gate_fails_when_kill_switch_tripped(db, settings):
    _seed_passing()
    trip_kill_switch("test", -1.0, 1.0)
    g = evaluate_gate(settings)
    assert g.passed is False
    assert any("kill switch" in r.lower() for r in g.reasons)


def test_gate_fails_on_short_window(db, settings):
    # Plenty of bets, but all settled today -> window span ~0 < required.
    for i in range(25):
        _insert_settled(2000 + i, "HOME", i < 15, 0.5, 10.0, NOW - timedelta(hours=1))
    g = evaluate_gate(settings)
    assert g.passed is False
    assert any("spans" in r for r in g.reasons)
