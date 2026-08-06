"""Entitlement — the money core. Pure decision + the fail-closed wrapper.

``evaluate`` is exercised with zero I/O (all inputs injected). The impure
``entitlement_for`` wrapper is tested with a monkeypatched balance reader,
including the RPC-error path which must fail CLOSED (locked, not crash).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import betbot.entitlement as ent
from betbot.entitlement import TRIAL_DAYS, Entitlement, entitlement_for, evaluate

OPERATOR_ID = 111
NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


def _created(days_ago: float) -> datetime:
    return NOW - timedelta(days=days_ago)


def _eval(**over) -> Entitlement:
    base = dict(
        telegram_user_id=999,
        created_at=_created(10),
        predictions_consumed=0,
        usdc_balance=0.0,
        now=NOW,
        operator_id=OPERATOR_ID,
    )
    base.update(over)
    return evaluate(**base)


# ----------------------------------------------------------------------
# Operator — always allowed, no accounting
# ----------------------------------------------------------------------
def test_operator_always_allowed_regardless_of_balance_and_age():
    e = _eval(telegram_user_id=OPERATOR_ID, created_at=_created(999),
              usdc_balance=0.0, predictions_consumed=9999)
    assert e.allowed is True
    assert e.reason == "operator"
    assert e.trial_days_left == -1  # N/A sentinel
    assert e.credits_remaining == -1


# ----------------------------------------------------------------------
# Trial window (day 0..6 allowed; day 7+ ends)
# ----------------------------------------------------------------------
def test_trial_day_zero_allowed_full_window_left():
    e = _eval(created_at=_created(0), usdc_balance=0.0)
    assert e.allowed is True
    assert e.reason == "trial"
    assert e.trial_days_left == TRIAL_DAYS  # 7


def test_trial_days_one_through_six_allowed_with_correct_days_left():
    for d in range(0, TRIAL_DAYS):  # 0..6
        e = _eval(created_at=_created(d), usdc_balance=0.0)
        assert e.allowed is True, d
        assert e.reason == "trial"
        assert e.trial_days_left == TRIAL_DAYS - d


def test_day_seven_zero_balance_is_locked():
    e = _eval(created_at=_created(7), usdc_balance=0.0)
    assert e.allowed is False
    assert e.reason == "locked"
    assert e.trial_days_left == 0
    assert e.credits_remaining == 0


# ----------------------------------------------------------------------
# Paid credits (post-trial)
# ----------------------------------------------------------------------
def test_balance_three_consumed_one_gives_two_credits_allowed():
    e = _eval(created_at=_created(30), usdc_balance=3.0, predictions_consumed=1)
    assert e.allowed is True
    assert e.reason == "credit"
    assert e.credits_remaining == 2


def test_consumed_equals_floor_balance_is_locked():
    e = _eval(created_at=_created(30), usdc_balance=3.0, predictions_consumed=3)
    assert e.allowed is False
    assert e.reason == "locked"
    assert e.credits_remaining == 0


def test_fractional_balance_floors():
    e = _eval(created_at=_created(30), usdc_balance=2.9, predictions_consumed=0)
    assert e.reason == "credit"
    assert e.credits_remaining == 2  # floor(2.9) == 2


def test_consumed_above_floor_never_negative():
    e = _eval(created_at=_created(30), usdc_balance=2.0, predictions_consumed=5)
    assert e.allowed is False
    assert e.credits_remaining == 0


# ----------------------------------------------------------------------
# The impure wrapper — reads a live balance, fails CLOSED on RPC error
# ----------------------------------------------------------------------
@dataclass
class _FakeUser:
    telegram_user_id: int
    wallet_address: str
    created_at: datetime
    predictions_consumed: int


@dataclass
class _FakeSettings:
    telegram_allowed_user_id: int
    polygon_rpc_url: str = "https://rpc.example"


@dataclass
class _FakeCB:
    usdc: float
    ok: bool
    error: str | None = None


def test_wrapper_reads_balance_and_grants_credits(monkeypatch):
    import betbot.wallet as wallet

    monkeypatch.setattr(
        wallet, "usdc_balance", lambda *a, **k: _FakeCB(5.0, True)
    )
    u = _FakeUser(999, "0xabc", _created(30), 1)
    s = _FakeSettings(OPERATOR_ID)
    e = entitlement_for(u, s, now=NOW)
    assert e.allowed is True
    assert e.reason == "credit"
    assert e.credits_remaining == 4  # floor(5) - 1


def test_wrapper_rpc_error_fails_closed(monkeypatch):
    import betbot.wallet as wallet

    def _boom(*a, **k):
        raise RuntimeError("rpc down")

    monkeypatch.setattr(wallet, "usdc_balance", _boom)
    u = _FakeUser(999, "0xabc", _created(30), 0)  # past trial
    s = _FakeSettings(OPERATOR_ID)
    e = entitlement_for(u, s, now=NOW)
    assert e.allowed is False  # fail CLOSED — never reveal on an errored read
    assert e.reason == "locked"


def test_wrapper_balance_read_not_ok_treated_as_zero(monkeypatch):
    import betbot.wallet as wallet

    monkeypatch.setattr(
        wallet, "usdc_balance", lambda *a, **k: _FakeCB(0.0, False, "timeout")
    )
    u = _FakeUser(999, "0xabc", _created(30), 0)
    s = _FakeSettings(OPERATOR_ID)
    e = entitlement_for(u, s, now=NOW)
    assert e.allowed is False
    assert e.reason == "locked"


def test_wrapper_operator_skips_network(monkeypatch):
    import betbot.wallet as wallet

    def _should_not_be_called(*a, **k):
        raise AssertionError("operator must not hit the balance RPC")

    monkeypatch.setattr(wallet, "usdc_balance", _should_not_be_called)
    u = _FakeUser(OPERATOR_ID, "0xop", _created(999), 0)
    s = _FakeSettings(OPERATOR_ID)
    e = entitlement_for(u, s, now=NOW)
    assert e.allowed is True
    assert e.reason == "operator"


def test_wrapper_trial_user_still_allowed_on_rpc_error(monkeypatch):
    """A trial user is allowed even if the balance read explodes — trial does
    not depend on balance."""
    import betbot.wallet as wallet

    monkeypatch.setattr(
        wallet, "usdc_balance",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )
    u = _FakeUser(999, "0xabc", _created(2), 0)  # day 2 of trial
    s = _FakeSettings(OPERATOR_ID)
    e = entitlement_for(u, s, now=NOW)
    assert e.allowed is True
    assert e.reason == "trial"


def test_evaluate_is_referenced_module():
    # Trivial guard: the pure function is importable at module scope.
    assert callable(ent.evaluate)
