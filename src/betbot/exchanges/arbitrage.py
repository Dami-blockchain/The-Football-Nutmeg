"""Cross-venue arbitrage detection (and gated execution).

A 3-way match has outcomes HOME / DRAW / AWAY. If you buy the YES share of each
outcome — taking the *cheapest* price for each across Polymarket and Limitless —
and the three prices sum to less than 1 (net of fees), you have locked a profit:
exactly one outcome resolves to $1, and you paid less than $1 total.

    arb_margin = 1 - (best_HOME + best_DRAW + best_AWAY) - fees

This needs NO prediction skill — it's market-neutral. Caveats are real and the
caller must respect them:
  * You need ALL THREE outcomes available somewhere (Limitless often has no
    DRAW market, so a draw can only be covered on Polymarket).
  * Settlement risk: Polymarket and Limitless resolve independently; a match
    *could* resolve differently between them (rare, not zero).
  * Execution risk: prices move between legs; partial fills leave you exposed.
  * Gas + taker fees on two chains can erase a thin margin.

Detection is read-only and safe. Execution is heavily gated and OFF by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from betbot.exchanges.base import ExchangeAdapter, ExchangeName, MarketRef, Outcome
from betbot.logging import get_logger

log = get_logger(__name__)

_OUTCOMES = (Outcome.HOME, Outcome.DRAW, Outcome.AWAY)


@dataclass(frozen=True)
class Leg:
    outcome: Outcome
    exchange: ExchangeName
    price: float
    size: float
    market: MarketRef


@dataclass(frozen=True)
class ArbOpportunity:
    home_team: str
    away_team: str
    legs: dict[str, Leg]          # outcome.value -> cheapest Leg
    price_sum: float
    fees: float

    @property
    def margin(self) -> float:
        """Locked profit fraction per $1 of guaranteed payout (>0 = arb)."""
        return 1.0 - self.price_sum - self.fees

    @property
    def complete(self) -> bool:
        """All three outcomes covered — required for a true lock."""
        return all(o.value in self.legs for o in _OUTCOMES)


class ArbScanner:
    def __init__(self, adapters: list[ExchangeAdapter], *, fee_per_leg: float = 0.0) -> None:
        self._adapters = adapters
        self._fee_per_leg = fee_per_leg

    async def scan(self, home_team: str, away_team: str, kickoff: datetime) -> ArbOpportunity | None:
        """Best cross-venue prices per outcome; returns the opportunity or None.

        Returns None only if fewer than two outcomes are quoted anywhere (no
        meaningful comparison). A returned opportunity may still have margin<=0
        (no arb) — the caller decides via ``.margin`` and ``.complete``.
        """
        # outcome.value -> list of candidate Legs across venues
        candidates: dict[str, list[Leg]] = {}
        for adapter in self._adapters:
            try:
                market = await adapter.find_market(home_team, away_team, kickoff)
            except Exception as e:  # noqa: BLE001
                log.warning("arb_find_market_failed", exchange=str(getattr(adapter, "name", "?")), error=str(e))
                continue
            if market is None:
                continue
            for outcome in _OUTCOMES:
                try:
                    q = await adapter.get_orderbook(market, outcome)
                except Exception:  # noqa: BLE001
                    continue
                if q is not None and 0.0 < q.yes_price < 1.0:
                    candidates.setdefault(outcome.value, []).append(
                        Leg(outcome, q.exchange, q.yes_price, q.yes_size, market)
                    )

        if len(candidates) < 2:
            return None

        best = {ov: min(legs, key=lambda lg: lg.price) for ov, legs in candidates.items()}
        price_sum = sum(lg.price for lg in best.values())
        fees = self._fee_per_leg * len(best)
        return ArbOpportunity(home_team, away_team, best, price_sum, fees)


def size_legs(opp: ArbOpportunity, budget_usd: float) -> dict[str, float]:
    """Stake per outcome to spend ~``budget_usd`` with equal payout across legs.

    For a guaranteed payout P: stake_o = P * price_o, total = P * price_sum.
    So P = budget / price_sum, and stake_o = budget * price_o / price_sum.
    """
    if opp.price_sum <= 0:
        return {}
    return {ov: budget_usd * lg.price / opp.price_sum for ov, lg in opp.legs.items()}
