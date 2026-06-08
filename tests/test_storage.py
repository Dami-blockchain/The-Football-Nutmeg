"""Storage-layer tests — KillSwitch CRUD (Phase 4)."""

from __future__ import annotations

import pytest

from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_kill_switch,
    is_kill_switch_tripped,
    reset_kill_switch,
    trip_kill_switch,
)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "ks.sqlite")
    yield


def test_get_kill_switch_creates_untripped(db):
    ks = get_kill_switch()
    assert ks.id == 1
    assert ks.tripped_at is None
    assert is_kill_switch_tripped() is False


def test_trip_and_reset(db):
    trip_kill_switch("test drawdown", -120.0, 120.0)
    assert is_kill_switch_tripped() is True
    ks = get_kill_switch()
    assert ks.tripped_at is not None
    assert ks.reason == "test drawdown"
    assert ks.realized_pnl_usd == -120.0
    assert ks.staked_usd == 120.0

    reset_kill_switch()
    assert is_kill_switch_tripped() is False
    assert get_kill_switch().tripped_at is None


def test_reason_is_truncated(db):
    trip_kill_switch("x" * 500, -1.0, 1.0)
    assert len(get_kill_switch().reason) <= 300
