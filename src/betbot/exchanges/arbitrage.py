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
from betbot.exchanges.matcher import is_match_result_market, normalized_team_pair
from betbot.logging import get_logger

log = get_logger(__name__)

_OUTCOMES = (Outcome.HOME, Outcome.DRAW, Outcome.AWAY)


def _market_is_match_result(market: MarketRef) -> bool:
    """A leg may only enter an opportunity from a 1X2 match-result market."""
    return is_match_result_market(market.metadata, market.title)


def markets_compatible(a: MarketRef, b: MarketRef) -> tuple[bool, str]:
    """Are two venues' markets the SAME underlying 1X2 match?

    Two legs may only be paired into an opportunity (and ultimately "locked" by
    the executor) when they refer to the same fixture AND the same outcome
    semantics. We compare, in order:

      1. both must be match-result markets (never a prop vs 1X2 mismatch);
      2. fixture id when BOTH expose one (authoritative);
      3. otherwise the normalised team pair (orientation-agnostic).

    Returns ``(True, "")`` when compatible, else ``(False, reason)``.
    """
    if not _market_is_match_result(a) or not _market_is_match_result(b):
        return False, "one leg is not a 1X2 match-result market"

    fa = a.metadata.get("fixture_id")
    fb = b.metadata.get("fixture_id")
    if fa is not None and fb is not None:
        if str(fa) != str(fb):
            return False, f"fixture id mismatch ({fa!r} != {fb!r})"
        return True, ""

    pair_a = normalized_team_pair(
        str(a.metadata.get("home_team") or ""), str(a.metadata.get("away_team") or "")
    )
    pair_b = normalized_team_pair(
        str(b.metadata.get("home_team") or ""), str(b.metadata.get("away_team") or "")
    )
    if not pair_a or not pair_b:
        return False, "missing team-pair metadata on a leg — cannot verify identity"
    if pair_a != pair_b:
        return False, f"team-pair mismatch ({sorted(pair_a)} != {sorted(pair_b)})"
    return True, ""


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
            # Cross-venue identity guard (part 1): a leg may only enter the
            # candidate pool from a real 1X2 match-result market. A prop (exact
            # scoreline, totals, spread, cards, …) can never be a 1X2 leg, so we
            # exclude it here — BEFORE picking the cheapest per outcome — so a
            # tempting cheap prop can't crowd out the genuine 1X2 quote.
            if not _market_is_match_result(market):
                log.warning(
                    "arb_leg_incompatible_dropped",
                    match=f"{home_team} v {away_team}",
                    exchange=str(getattr(adapter, "name", "?")),
                    market_id=market.market_id,
                    reason="market is not a 1X2 match-result market",
                )
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

        kept = {ov: min(legs, key=lambda lg: lg.price) for ov, legs in candidates.items()}

        # Cross-venue identity guard (part 2): the cheapest legs may come from
        # different venues, so before treating them as ONE lockable opportunity
        # we require the chosen legs to be MUTUALLY compatible (same fixture /
        # same team pair). If any two disagree, the scanner cannot tell which
        # venue is the real market, so it refuses to pair ANY of them — the
        # executor must never "lock" two markets it can't prove are identical.
        ordered = [kept[ov] for ov in ("HOME", "DRAW", "AWAY") if ov in kept]
        ref = ordered[0].market if ordered else None
        for leg in ordered[1:]:
            ok, reason = markets_compatible(ref, leg.market)
            if not ok:
                # Identity conflict among the chosen legs — abort the whole
                # opportunity rather than guess which venue is the real market.
                log.warning(
                    "arb_leg_incompatible_dropped",
                    match=f"{home_team} v {away_team}",
                    outcome=leg.outcome.value,
                    exchange=str(getattr(leg.exchange, "value", leg.exchange)),
                    market_id=leg.market.market_id,
                    reason=reason,
                    note="conflicting market identity across chosen legs — no lock",
                )
                return None

        if len(kept) < 2:
            return None

        price_sum = sum(lg.price for lg in kept.values())
        fees = self._fee_per_leg * len(kept)
        return ArbOpportunity(home_team, away_team, kept, price_sum, fees)


def size_legs(opp: ArbOpportunity, budget_usd: float) -> dict[str, float]:
    """Stake per outcome to spend ~``budget_usd`` with equal payout across legs.

    For a guaranteed payout P: stake_o = P * price_o, total = P * price_sum.
    So P = budget / price_sum, and stake_o = budget * price_o / price_sum.
    """
    if opp.price_sum <= 0:
        return {}
    return {ov: budget_usd * lg.price / opp.price_sum for ov, lg in opp.legs.items()}
