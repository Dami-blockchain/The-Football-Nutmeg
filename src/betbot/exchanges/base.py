"""ExchangeAdapter Protocol + shared types.

Phase 1 ships no concrete adapters — these types exist so the strategy
layer can be written against a stable interface that Polymarket/Limitless
implementations slot into later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from betbot.data.models import MatchOutcome


class ExchangeName(str, Enum):
    POLYMARKET = "POLYMARKET"
    LIMITLESS = "LIMITLESS"


Outcome = MatchOutcome


@dataclass(frozen=True, slots=True)
class MarketRef:
    exchange: ExchangeName
    market_id: str
    title: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OrderbookQuote:
    exchange: ExchangeName
    market_id: str
    outcome: Outcome
    yes_price: float
    yes_size: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class OrderResult:
    exchange: ExchangeName
    order_id: str
    market_id: str
    outcome: Outcome
    filled_size: float
    avg_price: float
    status: str
    raw_response: dict[str, Any]


@runtime_checkable
class ExchangeAdapter(Protocol):
    name: ExchangeName

    async def find_market(
        self, home_team: str, away_team: str, kickoff: datetime
    ) -> MarketRef | None: ...

    async def get_orderbook(
        self, market: MarketRef, outcome: Outcome
    ) -> OrderbookQuote | None: ...

    async def place_order(
        self,
        market: MarketRef,
        outcome: Outcome,
        size_usd: float,
        max_price: float,
    ) -> OrderResult: ...

    async def get_position(self, market: MarketRef) -> float: ...

    async def claim_winnings(self, market: MarketRef) -> bool: ...
