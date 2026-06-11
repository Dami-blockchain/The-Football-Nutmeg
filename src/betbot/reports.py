"""Daily Telegram report formatting — pure data shapes + formatters.

Formatting is split from scanning / storage / sending so the exact message
bodies are unit-testable with fixture data and zero network. Messages use
Telegram Markdown (``parse_mode="Markdown"``, same as :mod:`betbot.notify`):
*bold* headers and triple-backtick monospace blocks for columnar tables —
Telegram renders those fixed-width, which is what keeps columns aligned on a
phone screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

_MATCH_WIDTH = 24  # truncate long "Home v Away" strings so tables fit a phone


@dataclass(frozen=True)
class ArbLine:
    """One arb opportunity, reduced to exactly what the digest message needs."""

    match: str                # "Arsenal v Chelsea"
    margin: float             # locked profit fraction; >0 = real arb
    price_sum: float          # total cost of the three YES legs
    legs: tuple[str, ...]     # e.g. ("HOME: POLYMARKET @ 0.450", ...)


@dataclass(frozen=True)
class BetLine:
    """One bet row for the daily report (a trade placed or settled today)."""

    match: str
    outcome: str
    stake_usd: float
    market_price: float | None = None   # None = favourite-only (no market)
    settled_outcome: str | None = None
    pnl_usd: float | None = None


@dataclass(frozen=True)
class BalanceLine:
    """USDC balances for one wallet. ``None`` means the RPC read failed —
    shown as ``err``, never silently as 0 (a zero and a failure are different
    facts when real funds are involved)."""

    owner: str
    address: str
    polygon_usdc: float | None
    base_usdc: float | None


@dataclass(frozen=True)
class DailyReport:
    """Everything the 21:00 report needs, gathered before formatting."""

    day: date
    trades: tuple[BetLine, ...]
    settlements: tuple[BetLine, ...]
    realised_today_usd: float
    realised_cumulative_usd: float
    arb_count_today: int
    balances: tuple[BalanceLine, ...]


# ----------------------------------------------------------------------
# Formatters (pure)
# ----------------------------------------------------------------------
def format_arb_digest(lines: Sequence[ArbLine], day: date) -> str:
    """The 09:00 arb digest body. Clean 'no arb' message when nothing found."""
    header = f"*Arb digest — {day.isoformat()}*"
    if not lines:
        return (
            f"{header}\n\n"
            "No arb found today — cross-venue prices never summed below $1."
        )
    parts = [header, "", f"Found *{len(lines)}* opportunity(ies):"]
    for ln in lines:
        parts.append("")
        parts.append(
            f"*{ln.match}* — margin *{ln.margin:+.1%}* (cost {ln.price_sum:.3f})"
        )
        parts.extend(f"  {leg}" for leg in ln.legs)
    return "\n".join(parts)


def format_daily_report(report: DailyReport) -> str:
    """The 21:00 full daily report body."""
    parts = [f"*Daily report — {report.day.isoformat()}*", ""]

    if report.trades:
        parts.append(f"*Trades placed today ({len(report.trades)})*")
        parts.append(_mono(_trades_table(report.trades)))
    else:
        parts.append("*Trades placed today:* none")

    if report.settlements:
        parts.append(f"*Settled today ({len(report.settlements)})*")
        parts.append(_mono(_settlements_table(report.settlements)))
    else:
        parts.append("*Settled today:* none")

    parts.append(
        f"*Realised P&L:* today {report.realised_today_usd:+.2f} USD"
        f" | cumulative {report.realised_cumulative_usd:+.2f} USD"
    )
    parts.append(f"*Arb opportunities today:* {report.arb_count_today}")

    if report.balances:
        parts.append("*Balances (USDC)*")
        parts.append(_mono(_balances_table(report.balances)))
    else:
        parts.append("*Balances:* unavailable")
    return "\n".join(parts)


def _mono(text: str) -> str:
    return f"```\n{text}\n```"


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    right: frozenset[int] = frozenset(),
) -> str:
    """Plain-text aligned table; ``right`` holds indices of numeric columns."""
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(
            c.rjust(widths[i]) if i in right else c.ljust(widths[i])
            for i, c in enumerate(cells)
        ).rstrip()

    return "\n".join([fmt(headers)] + [fmt(r) for r in rows])


def _trades_table(bets: Sequence[BetLine]) -> str:
    rows = [
        (
            b.match[:_MATCH_WIDTH],
            b.outcome,
            f"{b.stake_usd:.2f}",
            f"{b.market_price:.2f}" if b.market_price is not None else "-",
        )
        for b in bets
    ]
    return _table(("match", "out", "stake", "price"), rows, right=frozenset({2, 3}))


def _settlements_table(bets: Sequence[BetLine]) -> str:
    rows = [
        (
            b.match[:_MATCH_WIDTH],
            b.outcome,
            b.settled_outcome or "-",
            f"{b.pnl_usd:+.2f}" if b.pnl_usd is not None else "-",
        )
        for b in bets
    ]
    return _table(("match", "out", "result", "pnl"), rows, right=frozenset({3}))


def _balances_table(balances: Sequence[BalanceLine]) -> str:
    def cell(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "err"

    rows = []
    for b in balances:
        total = (b.polygon_usdc or 0.0) + (b.base_usdc or 0.0)
        rows.append(
            (b.owner[:16], cell(b.polygon_usdc), cell(b.base_usdc), f"{total:.2f}")
        )
    return _table(
        ("owner", "polygon", "base", "total"), rows, right=frozenset({1, 2, 3})
    )
