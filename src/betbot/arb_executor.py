"""Cross-venue arbitrage EXECUTOR — heavily gated, OFF by default.

Takes the opportunities the :class:`~betbot.exchanges.arbitrage.ArbScanner`
finds and — only when every gate clears — buys all three outcome legs at their
cheapest venue, locking ``margin`` per $1 of guaranteed payout.

Gates (ALL must pass; the executor re-checks them itself on every opportunity):
  * ``BETBOT_ARB_EXECUTE=true`` (default false),
  * ``mode=live``,
  * drawdown kill switch clear,
  * ``margin >= BETBOT_ARB_EXECUTE_MIN_MARGIN`` (default 0.02 — deliberately
    stricter than the 0.01 Telegram-alert threshold),
  * every leg on an executable venue (Polymarket / Limitless — SX Bet is
    scan-only),
  * sizing fits ``BETBOT_ARB_MAX_STAKE_USD`` *including* Limitless min-share
    floors and the Polymarket order minimum,
  * today's arb stake stays under ``BETBOT_ARB_DAILY_CAP_USD``,
  * the agent wallet covers each venue's legs on that venue's chain.

Execution order: Limitless legs FIRST (the less liquid venue), and only on a
confirmed fill do the Polymarket legs go out (FOK-style market buys). If a
later leg fails after earlier fills, we log loudly and notify the operator with
the exact exposure left open — we never auto-chase. Every attempt by the armed
executor is persisted to ``arb_executions``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from betbot.exchanges.arbitrage import ArbOpportunity, Leg
from betbot.exchanges.base import ExchangeAdapter, ExchangeName
from betbot.logging import get_logger
from betbot.notify import send_telegram
from betbot.storage.repos import (
    arb_staked_today_usd,
    insert_arb_execution,
    is_kill_switch_tripped,
    update_arb_execution,
)

log = get_logger(__name__)

# Venues the executor can actually trade. SX Bet legs are scan-only.
EXECUTABLE_VENUES = frozenset({ExchangeName.POLYMARKET, ExchangeName.LIMITLESS})
# Polymarket marketable-order floor (USDC).
POLYMARKET_MIN_ORDER_USD = 1.0
# Agent wallet chain per venue (one EVM address, two chains).
VENUE_CHAIN = {ExchangeName.POLYMARKET: "polygon", ExchangeName.LIMITLESS: "base"}

_FAILED_STATUSES = frozenset(
    {"failed", "rejected", "canceled", "cancelled", "expired", "unmatched", "error"}
)

# Terminal row statuses (see storage.models.ArbExecution docstring).
STATUS_FILLED = "filled"
STATUS_ABORTED = "aborted_no_exposure"
STATUS_PARTIAL = "partial_exposure_open"


@dataclass
class PlannedLeg:
    leg: Leg
    stake_usd: float
    status: str = "planned"
    fill_price: float | None = None
    order_id: str = ""
    error: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "outcome": self.leg.outcome.value,
            "exchange": self.leg.exchange.value,
            "quoted_price": self.leg.price,
            "quoted_size": self.leg.size,
            "market_id": self.leg.market.market_id,
            "stake_usd": self.stake_usd,
            "status": self.status,
            "fill_price": self.fill_price,
            "order_id": self.order_id,
            "error": self.error,
        }


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    detail: str = ""
    row_id: int | None = None
    legs: list[PlannedLeg] = field(default_factory=list)


def _floor_cents(x: float) -> float:
    """Round a dollar amount DOWN to whole cents (never overshoot caps)."""
    return math.floor(x * 100.0) / 100.0


def _limitless_min_shares(leg: Leg) -> float:
    """Per-market min order size in shares, from the scanned market metadata."""
    child = leg.market.metadata.get("outcome_markets", {}).get(leg.outcome.value) or {}
    try:
        return float(child.get("min_shares") or 0.0)
    except (TypeError, ValueError):
        return 0.0


class ArbExecutor:
    """Self-gating executor. ``execute()`` is safe to call on every scan tick."""

    def __init__(
        self,
        settings,
        adapters: dict[ExchangeName, ExchangeAdapter],
        *,
        notify: Callable[..., Awaitable[bool]] | None = None,
        balance_usd: Callable[[str], float] | None = None,
    ) -> None:
        self._settings = settings
        self._adapters = adapters
        self._notify = notify or send_telegram
        self._balance_usd = balance_usd or self._agent_balance

    # ------------------------------------------------------------------
    def _agent_balance(self, chain: str) -> float:
        """Agent-wallet USDC balance on ``chain`` (0.0 on RPC failure)."""
        # Lazy import: betbot.wallet pulls in web3 (the `api` extra).
        from betbot.wallet import get_or_create_address, usdc_balance

        s = self._settings
        address = get_or_create_address(s.wallet_keyfile)
        rpc = s.polygon_rpc_url if chain == "polygon" else s.base_rpc_url
        bal = usdc_balance(address, chain, rpc)
        return bal.usdc if bal.ok else 0.0

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------
    def plan_legs(self, opp: ArbOpportunity) -> tuple[list[PlannedLeg] | None, str]:
        """Size each leg for an equal guaranteed payout within the stake cap.

        Payout target P = budget / price_sum; stake_o = P * price_o (rounded
        down to cents). Limitless legs buy P shares, so the market's min-share
        floor binds directly on P; Polymarket legs must clear the $1 order
        minimum. Returns ``(legs, "")`` or ``(None, reason)``.
        """
        budget = self._settings.arb_max_stake_usd
        if opp.price_sum <= 0:
            return None, "non-positive price sum"
        payout = budget / opp.price_sum
        for ov, leg in opp.legs.items():
            if leg.exchange == ExchangeName.LIMITLESS:
                min_shares = _limitless_min_shares(leg)
                if payout < min_shares:
                    return None, (
                        f"{ov} leg on LIMITLESS needs >= {min_shares:g} shares "
                        f"(${min_shares * leg.price:.2f} @ {leg.price:.3f}) but the "
                        f"${budget:.2f} stake cap only affords {payout:.1f} shares"
                    )
        planned = [
            PlannedLeg(leg=leg, stake_usd=_floor_cents(payout * leg.price))
            for leg in opp.legs.values()
        ]
        for pl in planned:
            if (
                pl.leg.exchange == ExchangeName.POLYMARKET
                and pl.stake_usd < POLYMARKET_MIN_ORDER_USD
            ):
                return None, (
                    f"{pl.leg.outcome.value} leg on POLYMARKET sized at "
                    f"${pl.stake_usd:.2f} — below the "
                    f"${POLYMARKET_MIN_ORDER_USD:.2f} order minimum"
                )
        return planned, ""

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _legs_json(legs: list[PlannedLeg]) -> str:
        return json.dumps([pl.snapshot() for pl in legs], default=str)

    def _persist(
        self,
        opp: ArbOpportunity,
        status: str,
        legs: list[PlannedLeg],
        *,
        net: float | None = None,
        error: str | None = None,
    ) -> int:
        return insert_arb_execution(
            home_team=opp.home_team,
            away_team=opp.away_team,
            margin=opp.margin,
            price_sum=opp.price_sum,
            stake_usd=round(sum(pl.stake_usd for pl in legs), 2),
            status=status,
            legs_json=self._legs_json(legs),
            net_expected_usd=net,
            error=error,
        )

    @staticmethod
    def _snapshot_only(opp: ArbOpportunity) -> list[PlannedLeg]:
        return [PlannedLeg(leg=leg, stake_usd=0.0) for leg in opp.legs.values()]

    def _reject(
        self, opp: ArbOpportunity, status: str, reason: str,
        legs: list[PlannedLeg] | None = None,
    ) -> ExecutionResult:
        """Persist + log a gated rejection (armed executor, no money moved)."""
        row_id = self._persist(opp, status, legs or self._snapshot_only(opp), error=reason)
        log.info(
            "arb_execute_rejected",
            match=f"{opp.home_team} v {opp.away_team}",
            status=status,
            reason=reason,
            margin=round(opp.margin, 4),
        )
        return ExecutionResult(status, reason, row_id)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def execute(self, opp: ArbOpportunity) -> ExecutionResult:
        s = self._settings

        # -- static gates: not armed -> no orders, no rows, no noise -------
        if not s.arb_execute:
            return ExecutionResult("skipped_disabled", "BETBOT_ARB_EXECUTE is false")
        if s.mode != "live":
            return ExecutionResult("skipped_paper", f"mode={s.mode!r} is not live")
        if is_kill_switch_tripped():
            log.warning("arb_execute_blocked_kill_switch",
                        match=f"{opp.home_team} v {opp.away_team}")
            return ExecutionResult("skipped_kill_switch", "drawdown kill switch tripped")

        # -- armed: every decision from here is persisted -------------------
        if not opp.complete:
            return self._reject(opp, "rejected_incomplete",
                                "not all three outcomes covered — no true lock")
        if opp.margin < s.arb_execute_min_margin:
            return self._reject(
                opp, "rejected_thin_margin",
                f"margin {opp.margin:.4f} < execute threshold "
                f"{s.arb_execute_min_margin:.4f}",
            )
        unsupported = [
            leg.exchange.value for leg in opp.legs.values()
            if leg.exchange not in EXECUTABLE_VENUES or leg.exchange not in self._adapters
        ]
        if unsupported:
            return self._reject(opp, "rejected_unsupported_venue",
                                f"cannot execute on: {', '.join(sorted(set(unsupported)))}")

        planned, why = self.plan_legs(opp)
        if planned is None:
            return self._reject(opp, "rejected_min_size", why)

        total = round(sum(pl.stake_usd for pl in planned), 2)
        staked_today = arb_staked_today_usd()
        if staked_today + total > s.arb_daily_cap_usd:
            return self._reject(
                opp, "rejected_daily_cap",
                f"${staked_today:.2f} staked today + ${total:.2f} would exceed "
                f"the ${s.arb_daily_cap_usd:.2f} daily cap",
                planned,
            )

        need: dict[str, float] = {}
        for pl in planned:
            chain = VENUE_CHAIN[pl.leg.exchange]
            need[chain] = need.get(chain, 0.0) + pl.stake_usd
        for chain, amount in sorted(need.items()):
            have = self._balance_usd(chain)
            if have < amount:
                return self._reject(
                    opp, "rejected_insufficient_balance",
                    f"{chain} balance ${have:.2f} < ${amount:.2f} needed for its leg(s)",
                    planned,
                )

        return await self._place_legs(opp, planned, total)

    async def _place_legs(
        self, opp: ArbOpportunity, planned: list[PlannedLeg], total: float
    ) -> ExecutionResult:
        """Place legs least-liquid-first; stop (don't chase) on any failure."""
        # Limitless (thin) first — only on its confirmed fill do the deeper
        # Polymarket legs go out.
        ordered = sorted(
            planned, key=lambda pl: 0 if pl.leg.exchange == ExchangeName.LIMITLESS else 1
        )
        row_id = self._persist(opp, "executing", ordered)
        match = f"{opp.home_team} v {opp.away_team}"
        filled: list[PlannedLeg] = []

        for i, pl in enumerate(ordered):
            adapter = self._adapters[pl.leg.exchange]
            try:
                result = await adapter.place_order(
                    pl.leg.market, pl.leg.outcome, pl.stake_usd, pl.leg.price
                )
                status = str(getattr(result, "status", "")).lower()
                if status in _FAILED_STATUSES:
                    raise RuntimeError(f"order not filled (status={status!r})")
                pl.status = "filled"
                pl.fill_price = float(getattr(result, "avg_price", pl.leg.price))
                pl.order_id = str(getattr(result, "order_id", ""))
                filled.append(pl)
                log.info(
                    "arb_leg_filled", match=match, leg=i + 1, of=len(ordered),
                    exchange=pl.leg.exchange.value, outcome=pl.leg.outcome.value,
                    stake_usd=pl.stake_usd, fill_price=pl.fill_price,
                )
            except Exception as e:  # noqa: BLE001 — venue errors must not escape
                pl.status = "failed"
                pl.error = str(e)[:300]
                for rest in ordered[i + 1:]:
                    rest.status = "not_attempted"
                return await self._handle_leg_failure(
                    opp, ordered, filled, pl, row_id, match
                )

        # All legs filled — the lock is on.
        payout = min(pl.stake_usd / pl.leg.price for pl in filled)
        net = round(payout - total, 2)
        update_arb_execution(
            row_id, status=STATUS_FILLED, legs_json=self._legs_json(ordered),
            net_expected_usd=net,
        )
        log.info(
            "arb_executed", match=match, margin=round(opp.margin, 4),
            staked_usd=total, net_expected_usd=net, row_id=row_id,
        )
        legs_txt = "\n".join(
            f"  {pl.leg.outcome.value}: {pl.leg.exchange.value} "
            f"@ {pl.fill_price:.3f} (${pl.stake_usd:.2f})"
            for pl in ordered
        )
        await self._notify(
            self._settings,
            f"✅ *Arb executed*\n*{match}*\n"
            f"margin {opp.margin:+.1%}, staked ${total:.2f}, "
            f"locked ~${net:+.2f}\n{legs_txt}",
        )
        return ExecutionResult(STATUS_FILLED, f"net ~${net:+.2f}", row_id, ordered)

    async def _handle_leg_failure(
        self,
        opp: ArbOpportunity,
        ordered: list[PlannedLeg],
        filled: list[PlannedLeg],
        failed: PlannedLeg,
        row_id: int,
        match: str,
    ) -> ExecutionResult:
        if not filled:
            # First leg failed — nothing filled, no exposure. Clean abort.
            update_arb_execution(
                row_id, status=STATUS_ABORTED, legs_json=self._legs_json(ordered),
                error=failed.error,
            )
            log.warning(
                "arb_aborted_no_exposure", match=match,
                exchange=failed.leg.exchange.value, error=failed.error, row_id=row_id,
            )
            await self._notify(
                self._settings,
                f"⚠️ *Arb aborted* (no exposure)\n*{match}*\n"
                f"first leg {failed.leg.outcome.value} on "
                f"{failed.leg.exchange.value} failed: {failed.error}",
            )
            return ExecutionResult(STATUS_ABORTED, failed.error, row_id, ordered)

        # A later leg failed AFTER fills — the lock is broken and real exposure
        # is open. Log loudly, tell the operator EXACTLY what is open, and do
        # NOT auto-chase.
        exposure = round(sum(pl.stake_usd for pl in filled), 2)
        update_arb_execution(
            row_id, status=STATUS_PARTIAL, legs_json=self._legs_json(ordered),
            error=failed.error,
        )
        log.error(
            "ARB_LEG_FAILED_EXPOSURE_OPEN", match=match,
            failed_exchange=failed.leg.exchange.value,
            failed_outcome=failed.leg.outcome.value,
            error=failed.error,
            open_exposure_usd=exposure,
            open_legs=[
                f"{pl.leg.outcome.value}@{pl.leg.exchange.value}:{pl.stake_usd}"
                for pl in filled
            ],
            row_id=row_id,
            note="NOT auto-chasing — operator must close or hold manually",
        )
        filled_txt = "\n".join(
            f"  OPEN {pl.leg.outcome.value}: {pl.leg.exchange.value} "
            f"@ {(pl.fill_price if pl.fill_price is not None else pl.leg.price):.3f} "
            f"(${pl.stake_usd:.2f})"
            for pl in filled
        )
        await self._notify(
            self._settings,
            f"🚨 *ARB LEG FAILED — exposure open*\n*{match}*\n"
            f"failed leg: {failed.leg.outcome.value} on "
            f"{failed.leg.exchange.value} (${failed.stake_usd:.2f})"
            f" — {failed.error}\n"
            f"exposure left open: ${exposure:.2f}\n{filled_txt}\n"
            f"NOT auto-chasing. Close or hold manually (db row {row_id}).",
        )
        return ExecutionResult(STATUS_PARTIAL, failed.error, row_id, ordered)

    async def execute_all(self, opportunities: list[ArbOpportunity]) -> list[ExecutionResult]:
        results = []
        for opp in opportunities:
            try:
                results.append(await self.execute(opp))
            except Exception as e:  # noqa: BLE001 — one bad opp must not kill the tick
                log.error(
                    "arb_execute_unexpected",
                    match=f"{opp.home_team} v {opp.away_team}", error=str(e),
                )
        return results
