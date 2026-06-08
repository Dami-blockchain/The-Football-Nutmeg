"""Backtest harness (Phase 5).

Two modes:

* ``backtest_stored`` — replay our own settled, market-priced bets and report
  hit rate, ROI, Brier score, and a per-outcome breakdown. This is the honest
  measure of how the strategy has actually done.
* ``backtest_mock`` — a synthetic diagnostic against a *fair* market (market
  price == true probability). It exists to confirm a sanity property: with no
  informational edge, edge-filtered betting nets ≈0 ROI minus noise. If mock
  ROI is reliably positive, the edge filter has a bug.

Brier score = mean((p_forecast − outcome)²) over the bet's chosen outcome
(outcome = 1 if the bet won, else 0). Lower is better; 0.25 is the coin-flip
baseline. Calibration, not P&L, is the real signal for a weak model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from betbot.storage.repos import list_settled_market_bets


@dataclass(frozen=True)
class OutcomeStat:
    outcome: str
    n: int
    wins: int
    pnl_usd: float
    staked_usd: float

    @property
    def hit_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def roi(self) -> float:
        return self.pnl_usd / self.staked_usd if self.staked_usd else 0.0


@dataclass(frozen=True)
class BacktestResult:
    n: int
    wins: int
    pnl_usd: float
    staked_usd: float
    brier: float
    per_outcome: dict[str, OutcomeStat]

    @property
    def hit_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def roi(self) -> float:
        return self.pnl_usd / self.staked_usd if self.staked_usd else 0.0


def _won(outcome: str, settled_outcome: str | None) -> bool:
    return settled_outcome is not None and outcome == settled_outcome


def compute_stats(bets) -> BacktestResult:
    """Aggregate stats over any objects exposing the PaperBet fields used here.

    Accepts real ORM rows or synthetic stand-ins (mock mode).
    """
    n = len(bets)
    wins = 0
    pnl = 0.0
    staked = 0.0
    brier_sum = 0.0
    buckets: dict[str, list[tuple[bool, float, float]]] = {}
    for b in bets:
        won = _won(b.outcome, b.settled_outcome)
        wins += int(won)
        pnl += b.pnl_usd or 0.0
        staked += b.stake_usd
        brier_sum += (b.our_probability - (1.0 if won else 0.0)) ** 2
        buckets.setdefault(b.outcome, []).append((won, b.pnl_usd or 0.0, b.stake_usd))

    per_outcome: dict[str, OutcomeStat] = {}
    for outcome, items in buckets.items():
        per_outcome[outcome] = OutcomeStat(
            outcome=outcome,
            n=len(items),
            wins=sum(int(w) for w, _, _ in items),
            pnl_usd=sum(p for _, p, _ in items),
            staked_usd=sum(s for _, _, s in items),
        )

    return BacktestResult(
        n=n,
        wins=wins,
        pnl_usd=pnl,
        staked_usd=staked,
        brier=(brier_sum / n if n else 0.0),
        per_outcome=per_outcome,
    )


def backtest_stored(window_days: int | None = None) -> BacktestResult:
    return compute_stats(list_settled_market_bets(window_days))


@dataclass
class _SynthBet:
    outcome: str
    settled_outcome: str | None
    our_probability: float
    market_price: float
    stake_usd: float
    pnl_usd: float


def backtest_mock(
    n: int = 1000,
    *,
    edge_threshold: float = 0.05,
    noise: float = 0.08,
    stake: float = 10.0,
    seed: int = 1234,
) -> BacktestResult:
    """Synthetic fair-market diagnostic.

    For each trial: draw a true probability ``p``; the market prices it fairly
    (``market = p``); our model sees a noisy estimate. We bet only when our
    estimate beats the market by ``edge_threshold`` — but since the market is
    fair, that "edge" is pure noise, so realized ROI should hover near 0.
    """
    rng = random.Random(seed)
    bets: list[_SynthBet] = []
    for _ in range(n):
        p = rng.uniform(0.1, 0.9)
        our_p = min(0.99, max(0.01, p + rng.gauss(0.0, noise)))
        if our_p - p < edge_threshold:
            continue
        won = rng.random() < p  # reality follows the TRUE prob, not ours
        pnl = stake * (1.0 / p - 1.0) if won else -stake
        bets.append(_SynthBet(
            outcome="HOME",
            settled_outcome="HOME" if won else "AWAY",
            our_probability=our_p,
            market_price=p,
            stake_usd=stake,
            pnl_usd=pnl,
        ))
    return compute_stats(bets)
