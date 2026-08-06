"""Prediction entitlement — the money core.

Every user gets a 7-day free trial from signup (``created_at``). After that,
each individual match prediction costs **1 USDC**, paid by sending USDC on
Polygon to the user's own deposit address; each 1 USDC = 1 prediction reveal.
The operator (``telegram_allowed_user_id``) is always free/unlimited.

:func:`evaluate` is deliberately PURE — no DB, no network, no clock. Every
input (balance, now, created_at) is injected, so it is unit-testable with zero
I/O and the reviewer can reason about the money decision in isolation. The
impure edge (reading the live on-chain USDC balance) lives in the thin
:func:`entitlement_for` wrapper, which fails CLOSED on an RPC error — a paying
user is never handed a reveal off an errored balance read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from betbot.logging import get_logger

log = get_logger(__name__)

TRIAL_DAYS = 7

# Sentinel for "not applicable" fields (operator has no trial/credit accounting).
_NA = -1


@dataclass(frozen=True)
class Entitlement:
    """The reveal decision for one user at one instant.

    ``reason`` is one of ``"operator" | "trial" | "credit" | "locked"``.
    ``trial_days_left`` and ``credits_remaining`` are ``-1`` (N/A) for the
    operator, who is unlimited.
    """

    allowed: bool
    reason: str
    trial_days_left: int
    credits_remaining: int


def _days_since(created_at: datetime, now: datetime) -> int:
    """Whole days elapsed between ``created_at`` and ``now`` (floored, >= 0).

    Both are coerced to UTC-aware so a naive ``created_at`` (SQLite can hand
    back naive datetimes) never raises on subtraction.
    """
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = now - created_at
    return max(0, delta.days)


def evaluate(
    *,
    telegram_user_id: int,
    created_at: datetime,
    predictions_consumed: int,
    usdc_balance: float,
    now: datetime,
    operator_id: int,
) -> Entitlement:
    """Decide whether this user may reveal one more prediction. Pure.

    Precedence: operator -> trial -> paid credits -> locked. Does NOT mutate
    anything — the caller increments ``predictions_consumed`` only when it
    actually REVEALS a prediction.
    """
    if operator_id and telegram_user_id == operator_id:
        return Entitlement(allowed=True, reason="operator",
                           trial_days_left=_NA, credits_remaining=_NA)

    trial_days_left = max(0, TRIAL_DAYS - _days_since(created_at, now))
    if trial_days_left > 0:
        return Entitlement(allowed=True, reason="trial",
                           trial_days_left=trial_days_left, credits_remaining=0)

    # Past the trial: 1 paid credit per whole USDC held, minus what's consumed.
    credits_remaining = max(0, math.floor(usdc_balance) - predictions_consumed)
    if credits_remaining >= 1:
        return Entitlement(allowed=True, reason="credit",
                           trial_days_left=0, credits_remaining=credits_remaining)
    return Entitlement(allowed=False, reason="locked",
                       trial_days_left=0, credits_remaining=0)


def entitlement_for(user, settings, *, now: datetime | None = None) -> Entitlement:
    """Impure wrapper: read the live Polygon USDC balance, then :func:`evaluate`.

    ``betbot.wallet`` (web3) is imported lazily so importing this module never
    drags in web3. On any RPC failure the balance is treated as 0.0 and the
    failure is LOGGED — this fails CLOSED (a paying user is never revealed a
    prediction off an errored read).
    """
    now = now or datetime.now(timezone.utc)
    operator_id = settings.telegram_allowed_user_id

    balance = 0.0
    # Skip the network entirely for the operator — they're unlimited regardless.
    if not (operator_id and user.telegram_user_id == operator_id):
        try:
            from betbot.wallet import usdc_balance

            cb = usdc_balance(
                user.wallet_address, "polygon", settings.polygon_rpc_url
            )
            if cb.ok:
                balance = float(cb.usdc)
            else:
                log.warning(
                    "entitlement_balance_read_failed",
                    telegram_user_id=user.telegram_user_id,
                    error=cb.error,
                )
        except Exception as e:  # noqa: BLE001 — fail closed on any RPC error
            log.warning(
                "entitlement_balance_read_error",
                telegram_user_id=user.telegram_user_id,
                error=str(e),
            )
            balance = 0.0

    return evaluate(
        telegram_user_id=user.telegram_user_id,
        created_at=user.created_at,
        predictions_consumed=user.predictions_consumed,
        usdc_balance=balance,
        now=now,
        operator_id=operator_id,
    )
