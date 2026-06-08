"""Live-readiness gate (Phase 5).

Before the bot is allowed to place real-money orders, ``evaluate_gate`` checks
that we have *earned* the right to: enough settled market bets, over a long
enough window, with an acceptable hit rate and non-negative ROI, and the kill
switch clear. The live pre-flight in ``main.py`` refuses to start in live mode
unless this passes. Favourite-only (no-market) bets are excluded throughout —
they carry no real-money signal.

This is a gate, not a guarantee: a passing gate means "the paper record isn't
obviously bad", not "this will be profitable". The operator decides to fund.
"""

from __future__ import annotations

from dataclasses import dataclass

from betbot.backtest import BacktestResult, backtest_stored
from betbot.storage.repos import is_kill_switch_tripped, list_settled_market_bets


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]
    result: BacktestResult
    window_days_observed: float
    kill_switch_tripped: bool


def evaluate_gate(settings) -> GateResult:
    bets = list_settled_market_bets()
    result = backtest_stored()

    if len(bets) >= 2 and bets[0].settled_at and bets[-1].settled_at:
        span_days = (bets[-1].settled_at - bets[0].settled_at).total_seconds() / 86400.0
    else:
        span_days = 0.0

    tripped = is_kill_switch_tripped()
    reasons: list[str] = []

    if result.n < settings.gate_min_bets:
        reasons.append(
            f"only {result.n} settled market bets (need {settings.gate_min_bets})"
        )
    if span_days < settings.gate_min_window_days:
        reasons.append(
            f"data spans {span_days:.1f}d (need {settings.gate_min_window_days:.0f}d)"
        )
    if result.hit_rate < settings.gate_min_hit_rate:
        reasons.append(
            f"hit rate {result.hit_rate:.1%} (need ≥{settings.gate_min_hit_rate:.0%})"
        )
    if result.roi < settings.gate_min_roi:
        reasons.append(
            f"ROI {result.roi:+.1%} (need ≥{settings.gate_min_roi:+.0%})"
        )
    if tripped:
        reasons.append("kill switch is TRIPPED")

    return GateResult(
        passed=not reasons,
        reasons=reasons,
        result=result,
        window_days_observed=span_days,
        kill_switch_tripped=tripped,
    )
