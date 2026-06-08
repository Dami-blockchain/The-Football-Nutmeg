"""Tests for pooled-fund share accounting — fairness is the whole point."""

from __future__ import annotations

import pytest

from betbot.pool import (
    nav_per_share,
    ownership_fraction,
    participant_value,
    shares_for_deposit,
    shares_for_withdrawal,
)


def test_first_deposit_gets_par_shares():
    nav = nav_per_share(0.0, 0.0)  # empty pool
    assert nav == 1.0
    assert shares_for_deposit(100.0, nav) == 100.0


def test_deposit_after_gain_is_fair():
    # pool worth $110 on 100 shares -> NAV 1.10
    nav = nav_per_share(110.0, 100.0)
    assert nav == pytest.approx(1.10)
    # a fresh $110 deposit gets 100 shares, NOT 110 — no free ride on prior gain
    assert shares_for_deposit(110.0, nav) == pytest.approx(100.0)
    # the original holder is still worth $110
    assert participant_value(100.0, nav) == pytest.approx(110.0)


def test_deposit_after_loss_is_fair():
    nav = nav_per_share(80.0, 100.0)  # down 20% -> NAV 0.80
    # $80 now buys 100 shares (you buy in cheaper after a loss)
    assert shares_for_deposit(80.0, nav) == pytest.approx(100.0)


def test_proportional_pnl_split():
    # two holders, 100 shares each (200 total); pool gains to $240 -> NAV 1.20
    nav = nav_per_share(240.0, 200.0)
    assert participant_value(100.0, nav) == pytest.approx(120.0)  # each shares the gain equally
    assert ownership_fraction(100.0, 200.0) == 0.5


def test_withdrawal_burns_correct_shares():
    nav = nav_per_share(120.0, 100.0)  # NAV 1.20
    assert shares_for_withdrawal(60.0, nav) == pytest.approx(50.0)  # $60 = 50 shares


def test_negative_rejected():
    with pytest.raises(ValueError):
        shares_for_deposit(-1.0, 1.0)
    with pytest.raises(ValueError):
        shares_for_withdrawal(-1.0, 1.0)
