"""Tests for settlement, P&L, and the drawdown kill switch (Phase 4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot.exchanges.base import Outcome
from betbot.settlement import SettlementWatcher, compute_pnl
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    insert_paper_bet,
    is_kill_switch_tripped,
    list_recent_paper_bets,
    upsert_prediction,
)
from betbot.strategy.engine import BetDecision, Prediction

NOW = datetime.now(timezone.utc)  # real-clock relative so drawdown-window tests never go stale


# ----------------------------------------------------------------------
# compute_pnl (pure)
# ----------------------------------------------------------------------
def test_pnl_win():
    # stake 10 at price 0.5 -> wins 10 (pays 1 per 0.5 share).
    assert compute_pnl(0.5, "HOME", "HOME", 10.0) == pytest.approx(10.0)


def test_pnl_loss():
    assert compute_pnl(0.5, "HOME", "AWAY", 10.0) == -10.0


def test_pnl_no_market_is_zero():
    assert compute_pnl(None, "HOME", "HOME", 10.0) == 0.0
    assert compute_pnl(None, "HOME", "AWAY", 10.0) == 0.0


def test_pnl_invalid_price_is_zero_on_win():
    assert compute_pnl(0.0, "HOME", "HOME", 10.0) == 0.0
    assert compute_pnl(1.0, "HOME", "HOME", 10.0) == 0.0
    assert compute_pnl(-0.2, "HOME", "HOME", 10.0) == 0.0


# ----------------------------------------------------------------------
# SettlementWatcher (DB-backed)
# ----------------------------------------------------------------------
@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "settle.sqlite")
    yield


class FakeFD:
    def __init__(self, results):
        self._results = results  # fixture_id -> match dict | None

    async def get_match(self, fixture_id):
        return self._results.get(fixture_id)


def _add_bet(fixture_id, outcome, market_price, stake, kickoff):
    pred = Prediction(
        fixture_id=fixture_id, competition_code="PL", home_team="A", away_team="B",
        p_home=0.5, p_draw=0.3, p_away=0.2, home_score=1.0, away_score=0.0, draw_score=2.4,
    )
    pid = upsert_prediction(pred, kickoff=kickoff)
    dec = BetDecision(
        fixture_id=fixture_id, competition_code="PL", home_team="A", away_team="B",
        outcome=outcome, our_probability=0.5, market_price=market_price, edge=0.1,
        stake_usd=stake, rationale="t",
    )
    insert_paper_bet(dec, pid)


def _finished(winner):
    return {"status": "FINISHED", "score": {"winner": winner}}


def _bet_by_fixture(fixture_id):
    return next(b for b in list_recent_paper_bets(days=30) if b.fixture_id == fixture_id)


async def test_settle_win(db, settings):
    past = NOW - timedelta(minutes=200)  # beyond the 150min grace
    _add_bet(101, Outcome.HOME, 0.5, 10.0, past)
    w = SettlementWatcher(FakeFD({101: _finished("HOME_TEAM")}), settings)
    summary = await w.settle_due(now=NOW)
    assert summary.settled == 1
    b = _bet_by_fixture(101)
    assert b.settled_outcome == "HOME"
    assert b.pnl_usd == pytest.approx(10.0)


async def test_settle_loss(db, settings):
    past = NOW - timedelta(minutes=200)
    _add_bet(102, Outcome.HOME, 0.5, 10.0, past)
    w = SettlementWatcher(FakeFD({102: _finished("AWAY_TEAM")}), settings)
    await w.settle_due(now=NOW)
    assert _bet_by_fixture(102).pnl_usd == -10.0


async def test_in_play_not_settled(db, settings):
    past = NOW - timedelta(minutes=200)
    _add_bet(103, Outcome.HOME, 0.5, 10.0, past)
    w = SettlementWatcher(FakeFD({103: {"status": "IN_PLAY"}}), settings)
    summary = await w.settle_due(now=NOW)
    assert summary.settled == 0 and summary.skipped_in_play == 1
    assert _bet_by_fixture(103).settled_at is None


async def test_too_recent_not_due(db, settings):
    recent = NOW - timedelta(minutes=30)  # inside the 150min grace
    _add_bet(104, Outcome.HOME, 0.5, 10.0, recent)
    w = SettlementWatcher(FakeFD({104: _finished("HOME_TEAM")}), settings)
    summary = await w.settle_due(now=NOW)
    assert summary.settled == 0
    assert _bet_by_fixture(104).settled_at is None


async def test_kill_switch_trips_on_drawdown(db, settings):
    past = NOW - timedelta(minutes=200)
    # 12 market bets x $10 = $120 staked, all lose -> pnl -120 < -0.20*120.
    results = {}
    for i in range(12):
        fid = 200 + i
        _add_bet(fid, Outcome.HOME, 0.5, 10.0, past)
        results[fid] = _finished("AWAY_TEAM")
    summary = await SettlementWatcher(FakeFD(results), settings).settle_due(now=NOW)
    assert summary.settled == 12
    assert summary.kill_switch_tripped is True
    assert is_kill_switch_tripped() is True


async def test_min_staked_floor_prevents_trip(db, settings):
    past = NOW - timedelta(minutes=200)
    # 5 bets x $10 = $50 staked (< $100 floor); all lose -> big % loss but no trip.
    results = {}
    for i in range(5):
        fid = 300 + i
        _add_bet(fid, Outcome.HOME, 0.5, 10.0, past)
        results[fid] = _finished("AWAY_TEAM")
    summary = await SettlementWatcher(FakeFD(results), settings).settle_due(now=NOW)
    assert summary.settled == 5
    assert summary.kill_switch_tripped is False
    assert is_kill_switch_tripped() is False
