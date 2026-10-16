"""Read-only client for the public Polymarket Data-API (``data-api.polymarket.com``).

This module NEVER signs, places, cancels or settles anything. It makes
authenticated-by-nobody GET requests against a PUBLIC address and returns the
JSON verbatim. It exists so the bot can mirror the operator's OWN on-chain
Polymarket trading (his proxy wallet) into a local read-only ledger — the same
way :mod:`betbot.exchanges.polymarket_gamma` reads the public catalogue.

Field semantics — VERIFIED live 2026-10-17 against the operator's wallet by
reconciling three known fills (not guessed; see ``docs``/the PR notes):

``/activity`` rows (one per on-chain event)::

    type            TRADE | REDEEM | YIELD | TAKER_REBATE | MAKER_REBATE | ...
    side            BUY | SELL      (empty "" for non-TRADE)
    size            number of outcome SHARES (not USDC)
    price           fill price in [0,1]  (0 for REDEEM/YIELD)
    usdcSize        the ACTUAL USDC that moved, INCLUSIVE of the sports taker
                    fee. This is the single most important field and the reason
                    we do not have to reconstruct cash flows:
                        BUY  -> usdcSize = size*price + fee   (you pay more)
                        SELL -> usdcSize = size*price - fee   (you receive less)
                        REDEEM -> usdcSize = winning shares * $1 (fee-free payout)
    asset           the ERC1155 CLOB token id of the outcome held
    conditionId     the market's condition id (-> Gamma event)
    outcome         "Yes" | "No"      (the binary leg)
    outcomeIndex    0 | 1
    transactionHash on-chain tx hash
    timestamp       unix seconds
    title/slug/eventSlug  human + url-slug market identifiers

The sports TAKER FEE is ``size * 0.05 * price * (1 - price)`` — Polymarket's
5% * p(1-p) curve. Reconciled to the cent against the operator's real fills:
a BUY of 600.08 Stuttgart shares @0.71 shows usdcSize 432.2346 == 600.08*0.71
+ 600.08*0.05*0.71*0.29. ``usdcSize`` already bakes it in, so P&L built from
``usdcSize`` is net-of-fee WITHOUT us re-deriving anything; :func:`taker_fee`
is provided only to REPORT the fee component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from betbot.logging import get_logger

log = get_logger(__name__)

DATA_API_BASE = "https://data-api.polymarket.com"

# Cloudflare fronts the data-api; a real browser User-Agent avoids the 403 the
# default httpx UA draws (same lesson as the Highlightly / odds clients).
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# The /activity endpoint caps a single page well above the operator's lifetime
# trade count, but we paginate defensively so this keeps working as the ledger
# grows. 500 is the largest page the API honours.
_PAGE_LIMIT = 500
# Hard stop so a misbehaving endpoint can never spin forever.
_MAX_PAGES = 200


class DataApiError(RuntimeError):
    """A data-api request failed or returned an unusable shape."""


def taker_fee(size: float, price: float) -> float:
    """Polymarket sports taker fee for a fill: ``size * 0.05 * p * (1-p)``.

    Returns 0 for a degenerate price (0/1) or non-trade (size 0). This is the
    fee ALREADY embedded in ``usdcSize``; we compute it only to show the fee
    split in reports, never to adjust cash flows.
    """
    p = float(price)
    if p <= 0.0 or p >= 1.0 or size <= 0.0:
        return 0.0
    return float(size) * 0.05 * p * (1.0 - p)


@dataclass(frozen=True)
class PmActivity:
    """One normalised /activity row. Mirrors the API fields we persist."""

    transaction_hash: str
    condition_id: str
    asset: str
    activity_type: str
    side: str
    size: float
    price: float
    usdc_size: float
    outcome: str
    outcome_index: int
    title: str
    slug: str
    event_slug: str
    timestamp: int

    @property
    def fee_usdc(self) -> float:
        if self.activity_type != "TRADE":
            return 0.0
        return taker_fee(self.size, self.price)

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "PmActivity":
        return cls(
            transaction_hash=str(row.get("transactionHash") or ""),
            condition_id=str(row.get("conditionId") or ""),
            asset=str(row.get("asset") or ""),
            activity_type=str(row.get("type") or ""),
            side=str(row.get("side") or ""),
            size=float(row.get("size") or 0.0),
            price=float(row.get("price") or 0.0),
            usdc_size=float(row.get("usdcSize") or 0.0),
            outcome=str(row.get("outcome") or ""),
            outcome_index=int(row.get("outcomeIndex") or 0),
            title=str(row.get("title") or ""),
            slug=str(row.get("slug") or ""),
            event_slug=str(row.get("eventSlug") or ""),
            timestamp=int(row.get("timestamp") or 0),
        )


@dataclass(frozen=True)
class PmPosition:
    """One normalised /positions row — an open or settled token holding."""

    condition_id: str
    asset: str
    size: float
    avg_price: float
    initial_value: float
    gross_initial_value: float
    entry_fees_usdc: float
    current_value: float
    cash_pnl: float
    cur_price: float
    redeemable: bool
    title: str
    slug: str
    event_slug: str
    outcome: str
    outcome_index: int
    end_date: str

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "PmPosition":
        return cls(
            condition_id=str(row.get("conditionId") or ""),
            asset=str(row.get("asset") or ""),
            size=float(row.get("size") or 0.0),
            avg_price=float(row.get("avgPrice") or 0.0),
            initial_value=float(row.get("initialValue") or 0.0),
            gross_initial_value=float(row.get("grossInitialValue") or 0.0),
            entry_fees_usdc=float(row.get("entryFeesUsdc") or 0.0),
            current_value=float(row.get("currentValue") or 0.0),
            cash_pnl=float(row.get("cashPnl") or 0.0),
            cur_price=float(row.get("curPrice") or 0.0),
            redeemable=bool(row.get("redeemable") or False),
            title=str(row.get("title") or ""),
            slug=str(row.get("slug") or ""),
            event_slug=str(row.get("eventSlug") or ""),
            outcome=str(row.get("outcome") or ""),
            outcome_index=int(row.get("outcomeIndex") or 0),
            end_date=str(row.get("endDate") or ""),
        )


class PmDataClient:
    """Thin synchronous read-only client. Context-managed."""

    def __init__(self, base_url: str = DATA_API_BASE, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"},
            timeout=timeout,
        )

    def __enter__(self) -> "PmDataClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        try:
            resp = self._client.get(path, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            # Never log the wallet in a way that would reach a user channel;
            # this is a server-side log line only.
            raise DataApiError(f"{path} request failed: {type(e).__name__}") from e
        except ValueError as e:  # JSON decode (a Cloudflare HTML shield 200)
            raise DataApiError(f"{path} returned non-JSON body") from e
        return data

    def fetch_activity(
        self,
        address: str,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> list[PmActivity]:
        """All /activity rows for ``address`` (paginated), newest-first.

        ``start``/``end`` are unix-second bounds passed through to the API when
        supported; we ALSO filter client-side so a backfill from a fixed date
        is correct regardless of whether the server honours the window.
        """
        out: list[PmActivity] = []
        offset = 0
        for _ in range(_MAX_PAGES):
            params: dict[str, Any] = {
                "user": address,
                "limit": _PAGE_LIMIT,
                "offset": offset,
            }
            if start is not None:
                params["start"] = start
            if end is not None:
                params["end"] = end
            batch = self._get_json("/activity", params)
            if not isinstance(batch, list):
                raise DataApiError("/activity did not return a list")
            if not batch:
                break
            for row in batch:
                if not isinstance(row, dict):
                    continue
                act = PmActivity.from_api(row)
                if start is not None and act.timestamp < start:
                    continue
                if end is not None and act.timestamp > end:
                    continue
                out.append(act)
            if len(batch) < _PAGE_LIMIT:
                break
            offset += _PAGE_LIMIT
        return out

    def fetch_positions(self, address: str) -> list[PmPosition]:
        """Current /positions rows for ``address`` (open + settled-unredeemed)."""
        data = self._get_json("/positions", {"user": address})
        if not isinstance(data, list):
            raise DataApiError("/positions did not return a list")
        return [PmPosition.from_api(r) for r in data if isinstance(r, dict)]
