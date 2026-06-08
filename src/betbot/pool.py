"""Pooled-fund share accounting for the shared (multi-user) betting pool.

A mutual-fund SHARE model so participants who join, add, or leave at different
times — and after the pool has gained or lost — are all treated fairly:

* The pool's net asset value (NAV) is its total value (cash + realised P&L).
* NAV-per-share = NAV / total_shares (starts at $1.00 when empty).
* Depositing $d mints ``d / nav_per_share`` shares.
* A participant's value is ``shares * nav_per_share`` — P&L accrues to the pool
  and is split in proportion to shares automatically.

Pure arithmetic, no DB. ⚠️ This pool is expected to LOSE money on these markets
(the strategy is −EV). Every participant must be told that; the bot + dashboard
surface each balance so losses are visible to all, by design.
"""

from __future__ import annotations

from dataclasses import dataclass

INITIAL_NAV_PER_SHARE = 1.0  # $1.00 per share at inception


@dataclass(frozen=True)
class Participant:
    user_id: int
    name: str
    shares: float
    deposited_usd: float   # lifetime gross deposits
    withdrawn_usd: float   # lifetime gross withdrawals


def nav_per_share(pool_value_usd: float, total_shares: float) -> float:
    if total_shares <= 0:
        return INITIAL_NAV_PER_SHARE
    return pool_value_usd / total_shares


def shares_for_deposit(amount_usd: float, nav: float) -> float:
    if amount_usd < 0:
        raise ValueError("deposit must be non-negative")
    if nav <= 0:
        nav = INITIAL_NAV_PER_SHARE
    return amount_usd / nav


def shares_for_withdrawal(amount_usd: float, nav: float) -> float:
    """Shares to burn to withdraw ``amount_usd`` at the current NAV."""
    if amount_usd < 0:
        raise ValueError("withdrawal must be non-negative")
    if nav <= 0:
        nav = INITIAL_NAV_PER_SHARE
    return amount_usd / nav


def participant_value(shares: float, nav: float) -> float:
    return shares * nav


def ownership_fraction(shares: float, total_shares: float) -> float:
    return shares / total_shares if total_shares > 0 else 0.0
