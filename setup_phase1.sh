#!/usr/bin/env bash
# setup_phase1.sh — bootstraps Phase 1 of The Football Smart Manager.
#
# Idempotent: re-running it is safe. It will overwrite source files with
# the canonical content but leaves your data/ directory, .env, and any
# git history alone.
#
# Run from the project root (where pyproject.toml lives).
#
# Usage:
#     chmod +x setup_phase1.sh
#     ./setup_phase1.sh

set -euo pipefail

# Sanity check: we should be in a directory that already has pyproject.toml
# (created in batch 1) and a venv. If not, bail loudly.
if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: pyproject.toml not found. Run this from your project root (~/tfsm)."
    exit 1
fi

echo "==> Creating source directory tree..."
mkdir -p src/betbot/data
mkdir -p src/betbot/strategy
mkdir -p src/betbot/storage
mkdir -p src/betbot/exchanges
mkdir -p src/betbot/utils
mkdir -p tests
mkdir -p data

# Empty __init__.py for every package directory (safe to re-touch).
for d in src/betbot src/betbot/data src/betbot/strategy src/betbot/storage \
         src/betbot/exchanges src/betbot/utils tests; do
    touch "$d/__init__.py"
done

echo "==> Writing src/betbot/config.py..."
cat > src/betbot/config.py <<'PYEOF'
"""Centralised configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Top-5 European leagues + UCL. Codes are football-data.org competition IDs.
LEAGUE_CODES: tuple[str, ...] = ("PL", "PD", "BL1", "SA", "FL1", "CL")


class Settings(BaseSettings):
    """All runtime knobs. Loaded from .env at process start.

    Mutable by tests (we don't ``frozen=True``), but the production daemon
    only reads it once at startup via :func:`get_settings`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Mode ----------------------------------------------------------
    mode: Literal["paper", "live"] = Field(default="paper", alias="BETBOT_MODE")

    # ---- Football-data.org --------------------------------------------
    football_data_api_key: str = Field(default="", alias="FOOTBALL_DATA_API_KEY")
    football_data_base_url: str = Field(
        default="https://api.football-data.org/v4",
        alias="FOOTBALL_DATA_BASE_URL",
    )
    football_data_rate_limit_per_min: int = Field(
        default=10, alias="FOOTBALL_DATA_RATE_LIMIT_PER_MIN"
    )

    # League scope (immutable for v1).
    leagues: tuple[str, ...] = LEAGUE_CODES

    # ---- Strategy knobs -----------------------------------------------
    home_advantage: float = Field(default=0.3, alias="BETBOT_HOME_ADVANTAGE")
    draw_score: float = Field(default=2.4, alias="BETBOT_DRAW_SCORE")
    softmax_temp: float = Field(default=1.0, alias="BETBOT_SOFTMAX_TEMP")
    opp_strength_weight: float = Field(
        default=0.5, alias="BETBOT_OPP_STRENGTH_WEIGHT"
    )

    # ---- Risk controls ------------------------------------------------
    fixed_stake_usd: float = Field(default=10.0, alias="BETBOT_FIXED_STAKE_USD")
    max_bet_usd: float = Field(default=50.0, alias="BETBOT_MAX_BET_USD")
    daily_exposure_cap_usd: float = Field(
        default=200.0, alias="BETBOT_DAILY_EXPOSURE_CAP_USD"
    )
    edge_threshold: float = Field(default=0.05, alias="BETBOT_EDGE_THRESHOLD")

    # ---- Storage ------------------------------------------------------
    db_path: Path = Field(
        default=Path("./data/betbot.sqlite"), alias="BETBOT_DB_PATH"
    )

    # ---- Logging ------------------------------------------------------
    log_level: str = Field(default="INFO", alias="BETBOT_LOG_LEVEL")

    # ---- Scheduler ----------------------------------------------------
    daemon_cron: str = Field(default="0 8 * * *", alias="BETBOT_DAEMON_CRON")

    # ---- Convenience properties ---------------------------------------
    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Re-read by clearing the cache."""
    return Settings()
PYEOF

echo "==> Writing src/betbot/logging.py..."
cat > src/betbot/logging.py <<'PYEOF'
"""structlog setup. Console-renderer in dev, JSON in production later."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Idempotent — safe to call multiple times."""
    level_num = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level_num,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
PYEOF

echo "==> Writing src/betbot/utils/cache.py..."
cat > src/betbot/utils/cache.py <<'PYEOF'
"""Tiny TTL cache. Used by the football-data client.

Not threadsafe; we run a single asyncio loop, which is enough.
"""

from __future__ import annotations

import time
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: float, max_size: int = 1024) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[object, tuple[float, T]] = {}

    def get(self, key: object) -> T | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: object, value: T) -> None:
        if len(self._store) >= self._max_size:
            # Evict the oldest entry. O(n) but n is bounded.
            oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            del self._store[oldest_key]
        self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
PYEOF

echo "==> Writing src/betbot/data/models.py..."
cat > src/betbot/data/models.py <<'PYEOF'
"""Pure domain models — frozen dataclasses, no ORM, no IO."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MatchOutcome(str, Enum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"


class Side(str, Enum):
    HOME = "HOME"
    AWAY = "AWAY"


@dataclass(frozen=True, slots=True)
class Team:
    id: int
    name: str
    short_name: str | None = None
    tla: str | None = None  # three-letter abbreviation


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One historical match used in form calculation."""
    home_team: Team
    away_team: Team
    home_goals: int
    away_goals: int
    kickoff: datetime
    competition_code: str

    @property
    def outcome(self) -> MatchOutcome:
        if self.home_goals > self.away_goals:
            return MatchOutcome.HOME
        if self.away_goals > self.home_goals:
            return MatchOutcome.AWAY
        return MatchOutcome.DRAW

    def points_for(self, side: Side) -> int:
        outcome = self.outcome
        if outcome is MatchOutcome.DRAW:
            return 1
        if side is Side.HOME and outcome is MatchOutcome.HOME:
            return 3
        if side is Side.AWAY and outcome is MatchOutcome.AWAY:
            return 3
        return 0


@dataclass(frozen=True, slots=True)
class Fixture:
    """An upcoming match we need to score."""
    id: int
    home_team: Team
    away_team: Team
    kickoff: datetime
    competition_code: str


@dataclass(frozen=True, slots=True)
class FormSnapshot:
    """One team's last-5 form. Used by FormService output."""
    team: Team
    weighted_points: float
    raw_points: int
    matches_considered: int


@dataclass(frozen=True, slots=True)
class FixtureForm:
    """Both teams' form alongside the fixture they apply to."""
    fixture: Fixture
    home_form: FormSnapshot
    away_form: FormSnapshot
PYEOF

echo "==> Writing src/betbot/data/football_data.py..."
cat > src/betbot/data/football_data.py <<'PYEOF'
"""Async client for football-data.org v4 API.

- Sliding-window rate limiter (10 req/min on free tier by default).
- TTL cache to amortise repeated calls within a scoring run.
- 429 backoff that respects Retry-After when present.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import httpx

from betbot.logging import get_logger
from betbot.utils.cache import TTLCache

log = get_logger(__name__)


class FootballDataError(RuntimeError):
    """Raised on transport / API errors talking to football-data.org."""


class _SlidingWindow:
    """N requests per 60s, blocking when full."""

    def __init__(self, limit_per_min: int) -> None:
        self._limit = max(1, limit_per_min)
        self._timestamps: deque[float] = deque(maxlen=self._limit)

    async def acquire(self) -> None:
        loop = asyncio.get_event_loop()
        now = loop.time()
        while len(self._timestamps) >= self._limit:
            oldest = self._timestamps[0]
            wait = 60.0 - (now - oldest)
            if wait <= 0:
                self._timestamps.popleft()
                break
            log.debug("rate_limit_wait", seconds=round(wait, 2))
            await asyncio.sleep(wait + 0.05)
            now = loop.time()
        self._timestamps.append(now)


class FootballDataClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.football-data.org/v4",
        *,
        rate_limit_per_min: int = 10,
        cache_ttl_seconds: float = 60.0,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            log.warning("football_data_no_api_key", note="Public endpoints only.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._limiter = _SlidingWindow(rate_limit_per_min)
        self._cache: TTLCache[Any] = TTLCache(ttl_seconds=cache_ttl_seconds)
        self._owns_client = client is None
        headers = {"X-Auth-Token": api_key} if api_key else {}
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, headers=headers
        )

    async def __aenter__(self) -> "FootballDataClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> Any:
        cache_key = (path, tuple(sorted((params or {}).items())))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._limiter.acquire()
        url = f"{self._base_url}{path}"
        log.debug("football_data_request", path=path, params=params)
        try:
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as e:
            raise FootballDataError(f"network error calling {path}: {e}") from e

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "10"))
            log.warning("football_data_rate_limited", retry_after=retry_after)
            await asyncio.sleep(min(retry_after, 60.0))
            return await self._get(path, params)

        if resp.status_code >= 400:
            raise FootballDataError(
                f"{path} -> {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise FootballDataError(f"non-JSON response from {path}") from e

        self._cache.set(cache_key, data)
        return data

    # ------------------------------------------------------------------
    async def list_scheduled_matches(
        self, competition_code: str, date_from: str, date_to: str
    ) -> list[dict[str, Any]]:
        data = await self._get(
            f"/competitions/{competition_code}/matches",
            params={
                "dateFrom": date_from,
                "dateTo": date_to,
                "status": "SCHEDULED",
            },
        )
        matches = data.get("matches") or []
        return [m for m in matches if isinstance(m, dict)]

    async def list_team_recent_matches(
        self, team_id: int, limit: int = 5, before: str | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"status": "FINISHED", "limit": limit}
        if before:
            params["dateTo"] = before
        data = await self._get(f"/teams/{team_id}/matches", params=params)
        matches = data.get("matches") or []
        sorted_m = sorted(
            (m for m in matches if isinstance(m, dict)),
            key=lambda m: m.get("utcDate", ""),
            reverse=True,
        )
        return sorted_m[:limit]

    async def get_standings(
        self, competition_code: str
    ) -> list[dict[str, Any]]:
        data = await self._get(f"/competitions/{competition_code}/standings")
        standings = data.get("standings") or []
        for s in standings:
            if s.get("type") == "TOTAL":
                table = s.get("table") or []
                return [row for row in table if isinstance(row, dict)]
        return []

    async def get_team(self, team_id: int) -> dict[str, Any]:
        return await self._get(f"/teams/{team_id}")

    async def list_competition_teams(
        self, competition_code: str
    ) -> list[dict[str, Any]]:
        data = await self._get(f"/competitions/{competition_code}/teams")
        return data.get("teams") or []

    async def get_match(self, match_id: int) -> dict[str, Any] | None:
        try:
            return await self._get(f"/matches/{match_id}")
        except FootballDataError as e:
            if "404" in str(e):
                return None
            raise
PYEOF

echo "==> Writing src/betbot/exchanges/base.py..."
cat > src/betbot/exchanges/base.py <<'PYEOF'
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
PYEOF

echo "==> Writing src/betbot/strategy/probabilities.py..."
cat > src/betbot/strategy/probabilities.py <<'PYEOF'
"""Pure-math helpers — no IO, no settings dependency.

Importable in any environment, even without the rest of the bot's deps.
"""

from __future__ import annotations

import math


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically-stable softmax.

    The max-subtraction trick keeps overflow at bay even when scores are
    in the thousands — handy when callers pass already-scaled values.
    """
    if not scores:
        return []
    t = max(temperature, 1e-9)
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    if total <= 0:
        # Degenerate case — return uniform.
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


def opponent_strength_factor(
    opponent_position: float | None,
    league_size: int,
    weight: float,
) -> float:
    """Scale an opponent's threat by their current league position.

    Higher league position (smaller number) = stronger opponent = higher
    factor. Position 1 in a 20-team league with weight 0.5 yields 1.5;
    last place yields 0.5; missing data yields 1.0.
    """
    if opponent_position is None or league_size <= 1:
        return 1.0
    # Normalise to [0, 1] where 0 = top, 1 = bottom.
    norm = (opponent_position - 1) / (league_size - 1)
    norm = max(0.0, min(1.0, norm))
    # Convert to a scaling factor centred at 1.0:
    #   top (norm=0) -> 1 + weight
    #   bottom (norm=1) -> 1 - weight
    return 1.0 + weight * (1.0 - 2.0 * norm)


def edge(our_probability: float, market_price: float) -> float:
    """Edge in probability units. Positive = we think it's underpriced."""
    return our_probability - market_price


def implied_probability(market_price: float) -> float:
    """Clip a market price into [0, 1] before treating it as a probability."""
    return max(0.0, min(1.0, market_price))
PYEOF

echo "==> Writing src/betbot/strategy/engine.py..."
cat > src/betbot/strategy/engine.py <<'PYEOF'
"""StrategyEngine — converts FixtureForm into probabilities + bet decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from betbot.data.models import FixtureForm, MatchOutcome
from betbot.logging import get_logger
from betbot.strategy.probabilities import edge, softmax

if TYPE_CHECKING:
    from betbot.config import Settings

log = get_logger(__name__)


# Re-export so adapter code can import a single Outcome name.
Outcome = MatchOutcome


@dataclass(frozen=True, slots=True)
class Prediction:
    """The model's view of one fixture."""

    fixture_id: int
    competition_code: str
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    home_score: float
    away_score: float
    draw_score: float

    @property
    def best_outcome(self) -> Outcome:
        triples = [
            (Outcome.HOME, self.p_home),
            (Outcome.DRAW, self.p_draw),
            (Outcome.AWAY, self.p_away),
        ]
        return max(triples, key=lambda kv: kv[1])[0]


@dataclass(frozen=True, slots=True)
class BetDecision:
    """Edge-filtered decision suitable for logging or live placement."""

    fixture_id: int
    competition_code: str
    home_team: str
    away_team: str
    outcome: Outcome
    our_probability: float
    market_price: float
    edge: float
    stake_usd: float
    rationale: str


class StrategyEngine:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    def predict(self, fixture_form: FixtureForm) -> Prediction:
        s = self._settings
        home_score = fixture_form.home_form.weighted_points + s.home_advantage
        away_score = fixture_form.away_form.weighted_points
        draw_score = s.draw_score

        probs = softmax([home_score, draw_score, away_score], s.softmax_temp)
        return Prediction(
            fixture_id=fixture_form.fixture.id,
            competition_code=fixture_form.fixture.competition_code,
            home_team=fixture_form.fixture.home_team.name,
            away_team=fixture_form.fixture.away_team.name,
            p_home=probs[0],
            p_draw=probs[1],
            p_away=probs[2],
            home_score=home_score,
            away_score=away_score,
            draw_score=draw_score,
        )

    # ------------------------------------------------------------------
    def decide_with_market(
        self,
        prediction: Prediction,
        outcome: Outcome,
        market_price: float,
    ) -> BetDecision | None:
        """Apply the edge filter; return a BetDecision or None.

        ``None`` means the market quote vetoed the bet — don't fall back
        to favourite-only logging in this case.
        """
        s = self._settings
        our_p = {
            Outcome.HOME: prediction.p_home,
            Outcome.DRAW: prediction.p_draw,
            Outcome.AWAY: prediction.p_away,
        }[outcome]
        e = edge(our_p, market_price)
        if e < s.edge_threshold:
            return None
        stake = min(s.fixed_stake_usd, s.max_bet_usd)
        rationale = (
            f"edge {e:+.3f} ({our_p:.3f} - {market_price:.3f}) at "
            f"≥{s.edge_threshold:.3f} threshold; stake ${stake:.0f}"
        )
        return BetDecision(
            fixture_id=prediction.fixture_id,
            competition_code=prediction.competition_code,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            outcome=outcome,
            our_probability=our_p,
            market_price=market_price,
            edge=e,
            stake_usd=stake,
            rationale=rationale,
        )
PYEOF

echo "==> Writing src/betbot/data/form.py..."
cat > src/betbot/data/form.py <<'PYEOF'
"""FormService — pulls last-5 form from football-data.org and weights it."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from betbot.data.football_data import FootballDataClient
from betbot.data.models import (
    Fixture,
    FixtureForm,
    FormSnapshot,
    MatchResult,
    Side,
    Team,
)
from betbot.logging import get_logger

if TYPE_CHECKING:
    from betbot.config import Settings

log = get_logger(__name__)


# Recency weights for the last 5 matches, most-recent first.
_RECENCY_WEIGHTS: tuple[float, ...] = (1.5, 1.3, 1.1, 1.0, 0.9)


def _parse_kickoff(s: str) -> datetime:
    """Parse an ISO-8601 string from football-data into an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _parse_team(raw: dict[str, Any] | None) -> Team:
    if raw is None:
        return Team(id=0, name="Unknown")
    return Team(
        id=int(raw.get("id") or 0),
        name=str(raw.get("name") or "Unknown"),
        short_name=raw.get("shortName"),
        tla=raw.get("tla"),
    )


def _parse_match_result(raw: dict[str, Any]) -> MatchResult | None:
    home = _parse_team(raw.get("homeTeam"))
    away = _parse_team(raw.get("awayTeam"))
    score = (raw.get("score") or {}).get("fullTime") or {}
    home_goals = score.get("home")
    away_goals = score.get("away")
    if home_goals is None or away_goals is None:
        return None
    return MatchResult(
        home_team=home,
        away_team=away,
        home_goals=int(home_goals),
        away_goals=int(away_goals),
        kickoff=_parse_kickoff(raw.get("utcDate") or ""),
        competition_code=str((raw.get("competition") or {}).get("code") or ""),
    )


class FormService:
    """Computes weighted-points form for a team over its last 5 finished matches.

    Opponent-strength adjustment: each match's points contribution is
    multiplied by an opponent-strength factor derived from the opponent's
    current league position (top of table => higher factor, bottom =>
    lower).
    """

    def __init__(self, client: FootballDataClient, settings: "Settings") -> None:
        self._client = client
        self._settings = settings
        # Per-competition cache of {team_id: position} to avoid repeated
        # standings calls within a single scoring run.
        self._standings_cache: dict[str, dict[int, int]] = {}
        self._standings_size: dict[str, int] = {}

    # ------------------------------------------------------------------
    async def _get_positions(self, competition_code: str) -> tuple[dict[int, int], int]:
        if competition_code in self._standings_cache:
            return (
                self._standings_cache[competition_code],
                self._standings_size[competition_code],
            )
        table = await self._client.get_standings(competition_code)
        positions: dict[int, int] = {}
        for row in table:
            tid = (row.get("team") or {}).get("id")
            pos = row.get("position")
            if isinstance(tid, int) and isinstance(pos, int):
                positions[tid] = pos
        size = len(positions) or 20  # sane default for top-5 leagues
        self._standings_cache[competition_code] = positions
        self._standings_size[competition_code] = size
        return positions, size

    # ------------------------------------------------------------------
    async def _team_snapshot(
        self,
        team: Team,
        before_kickoff: datetime,
        competition_code: str,
    ) -> FormSnapshot:
        # Pull a few extra so we can drop matches that haven't happened yet.
        raw = await self._client.list_team_recent_matches(
            team.id, limit=10, before=before_kickoff.date().isoformat()
        )
        positions, league_size = await self._get_positions(competition_code)

        weight_w = self._settings.opp_strength_weight
        results: list[tuple[MatchResult, Side]] = []
        for m in raw:
            mr = _parse_match_result(m)
            if mr is None or mr.kickoff >= before_kickoff:
                continue
            if mr.home_team.id == team.id:
                side = Side.HOME
            elif mr.away_team.id == team.id:
                side = Side.AWAY
            else:
                continue
            results.append((mr, side))
            if len(results) >= 5:
                break

        weighted = 0.0
        raw_pts = 0
        for i, (mr, side) in enumerate(results):
            w = _RECENCY_WEIGHTS[i] if i < len(_RECENCY_WEIGHTS) else 1.0
            opponent = mr.away_team if side is Side.HOME else mr.home_team
            opp_pos = positions.get(opponent.id)
            factor = _opp_strength_factor(opp_pos, league_size, weight_w)
            pts = mr.points_for(side)
            weighted += w * pts * factor
            raw_pts += pts

        return FormSnapshot(
            team=team,
            weighted_points=weighted,
            raw_points=raw_pts,
            matches_considered=len(results),
        )

    # ------------------------------------------------------------------
    async def fixture_form(
        self,
        fixture_id: int,
        competition_code: str,
        kickoff: datetime,
        home_team: Team,
        away_team: Team,
    ) -> FixtureForm:
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        home_form = await self._team_snapshot(home_team, kickoff, competition_code)
        away_form = await self._team_snapshot(away_team, kickoff, competition_code)
        return FixtureForm(
            fixture=Fixture(
                id=fixture_id,
                home_team=home_team,
                away_team=away_team,
                kickoff=kickoff,
                competition_code=competition_code,
            ),
            home_form=home_form,
            away_form=away_form,
        )


def _opp_strength_factor(
    opp_position: int | None, league_size: int, weight: float
) -> float:
    # Local re-implementation to avoid circular import with probabilities.
    if opp_position is None or league_size <= 1:
        return 1.0
    norm = (opp_position - 1) / (league_size - 1)
    norm = max(0.0, min(1.0, norm))
    return 1.0 + weight * (1.0 - 2.0 * norm)
PYEOF

echo "==> Writing src/betbot/storage/db.py..."
cat > src/betbot/storage/db.py <<'PYEOF'
"""SQLAlchemy engine + session helpers.

Single SQLite file. The engine is process-global; ``init_engine`` is
idempotent. Tests reset the globals via ``monkeypatch``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from betbot.storage.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(db_path: Path) -> Engine:
    """Create the engine + schema. Idempotent."""
    global _engine, _SessionLocal
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path.absolute()}"
    engine = create_engine(
        url,
        future=True,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session. Commits on success, rolls back on error."""
    if _SessionLocal is None:
        raise RuntimeError("init_engine() must be called before session_scope().")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
PYEOF

echo "==> Writing src/betbot/storage/models.py..."
cat > src/betbot/storage/models.py <<'PYEOF'
"""ORM models. Kept narrow — only the tables we actually use."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    """One row per (fixture_id, run_date) — what the model thought today."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("fixture_id", "run_date", name="uq_predictions_fix_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, index=True)
    competition_code: Mapped[str] = mapped_column(String(8))
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    run_date: Mapped[str] = mapped_column(String(10))  # ISO date string

    home_team: Mapped[str] = mapped_column(String(80))
    away_team: Mapped[str] = mapped_column(String(80))

    p_home: Mapped[float] = mapped_column(Float)
    p_draw: Mapped[float] = mapped_column(Float)
    p_away: Mapped[float] = mapped_column(Float)

    home_score: Mapped[float] = mapped_column(Float)
    away_score: Mapped[float] = mapped_column(Float)
    draw_score: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    paper_bets: Mapped[list["PaperBet"]] = relationship(back_populates="prediction")


class PaperBet(Base):
    """A logged bet. ``market_price`` is None for Phase-1 favourite-only bets."""

    __tablename__ = "paper_bets"
    __table_args__ = (
        UniqueConstraint("fixture_id", "outcome", name="uq_paper_bets_fix_out"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id"), index=True
    )
    fixture_id: Mapped[int] = mapped_column(Integer, index=True)

    outcome: Mapped[str] = mapped_column(String(4))  # HOME / DRAW / AWAY
    our_probability: Mapped[float] = mapped_column(Float)
    market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge: Mapped[float | None] = mapped_column(Float, nullable=True)
    stake_usd: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Settlement fields — filled in by SettlementWatcher in Phase 4.
    settled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    settled_outcome: Mapped[str | None] = mapped_column(String(4), nullable=True)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    prediction: Mapped["PredictionRow"] = relationship(back_populates="paper_bets")
PYEOF

echo "==> Writing src/betbot/storage/repos.py..."
cat > src/betbot/storage/repos.py <<'PYEOF'
"""Repository helpers — narrow, intention-revealing CRUD."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from betbot.logging import get_logger
from betbot.storage.db import session_scope
from betbot.storage.models import PaperBet, PredictionRow
from betbot.strategy.engine import BetDecision, Outcome, Prediction

log = get_logger(__name__)


def upsert_prediction(
    prediction: Prediction, *, kickoff: datetime
) -> int:
    """Insert a prediction row (or update if (fixture_id, run_date) exists).

    Returns the row id.
    """
    run_date = date.today().isoformat()
    with session_scope() as s:
        existing = s.execute(
            select(PredictionRow)
            .where(PredictionRow.fixture_id == prediction.fixture_id)
            .where(PredictionRow.run_date == run_date)
        ).scalar_one_or_none()
        if existing is not None:
            existing.p_home = prediction.p_home
            existing.p_draw = prediction.p_draw
            existing.p_away = prediction.p_away
            existing.home_score = prediction.home_score
            existing.away_score = prediction.away_score
            existing.draw_score = prediction.draw_score
            s.flush()
            return existing.id
        row = PredictionRow(
            fixture_id=prediction.fixture_id,
            competition_code=prediction.competition_code,
            kickoff=kickoff,
            run_date=run_date,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            p_home=prediction.p_home,
            p_draw=prediction.p_draw,
            p_away=prediction.p_away,
            home_score=prediction.home_score,
            away_score=prediction.away_score,
            draw_score=prediction.draw_score,
        )
        s.add(row)
        s.flush()
        return row.id


def insert_paper_bet(decision: BetDecision, prediction_id: int) -> bool:
    """Insert an edge-filtered paper bet. Returns False if a duplicate exists."""
    with session_scope() as s:
        existing = s.execute(
            select(PaperBet)
            .where(PaperBet.fixture_id == decision.fixture_id)
            .where(PaperBet.outcome == decision.outcome.value)
        ).scalar_one_or_none()
        if existing is not None:
            return False
        s.add(
            PaperBet(
                prediction_id=prediction_id,
                fixture_id=decision.fixture_id,
                outcome=decision.outcome.value,
                our_probability=decision.our_probability,
                market_price=decision.market_price,
                edge=decision.edge,
                stake_usd=decision.stake_usd,
                rationale=decision.rationale,
            )
        )
        return True


def insert_paper_bet_no_market(
    prediction: Prediction,
    prediction_id: int,
    outcome: Outcome,
    *,
    stake_usd: float,
    rationale: str,
) -> bool:
    """Insert a Phase-1-style favourite-only paper bet (no market price)."""
    our_p = {
        Outcome.HOME: prediction.p_home,
        Outcome.DRAW: prediction.p_draw,
        Outcome.AWAY: prediction.p_away,
    }[outcome]
    with session_scope() as s:
        existing = s.execute(
            select(PaperBet)
            .where(PaperBet.fixture_id == prediction.fixture_id)
            .where(PaperBet.outcome == outcome.value)
        ).scalar_one_or_none()
        if existing is not None:
            return False
        s.add(
            PaperBet(
                prediction_id=prediction_id,
                fixture_id=prediction.fixture_id,
                outcome=outcome.value,
                our_probability=our_p,
                market_price=None,
                edge=None,
                stake_usd=stake_usd,
                rationale=rationale,
            )
        )
        return True


def list_recent_paper_bets(days: int = 7) -> list[PaperBet]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet)
                .where(PaperBet.created_at >= cutoff)
                .order_by(PaperBet.created_at.desc())
            ).scalars()
        )
        # Detach so attribute access works after the session closes.
        s.expunge_all()
        return rows


def daily_paper_exposure_usd() -> float:
    """Sum of stakes placed today (used by the risk control)."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    with session_scope() as s:
        rows = s.execute(
            select(PaperBet.stake_usd).where(PaperBet.created_at >= today_start)
        ).scalars()
        return float(sum(rows))
PYEOF

echo "==> Writing src/betbot/main.py..."
cat > src/betbot/main.py <<'PYEOF'
"""The Football Smart Manager — CLI entrypoint.

Commands (run as ``tfsm <command>`` or ``betbot <command>``):
    tfsm run-once      Score the next 48h of fixtures and log paper bets.
    tfsm run-daemon    Schedule run-once daily at 08:00 UTC.
    tfsm bets list     Print recent paper bets to stdout.
    tfsm init-db       Create the SQLite schema (called automatically).
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta, timezone
from typing import Annotated

import typer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient, FootballDataError
from betbot.data.form import FormService, _parse_kickoff, _parse_team
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    daily_paper_exposure_usd,
    insert_paper_bet_no_market,
    list_recent_paper_bets,
    upsert_prediction,
)
from betbot.strategy.engine import StrategyEngine

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
bets_app = typer.Typer(help="Inspect logged paper bets.")
app.add_typer(bets_app, name="bets")


# ----------------------------------------------------------------------
# Scoring run
# ----------------------------------------------------------------------
async def _score_once() -> int:
    """Pull fixtures in the next 48h, score each, log paper bets."""
    settings = get_settings()
    log = get_logger(__name__)
    init_engine(settings.db_path)

    log.info(
        "starting_scoring_run",
        mode=settings.mode,
        leagues=list(settings.leagues),
    )

    today = date.today()
    date_from = today.isoformat()
    # +2: football-data dateTo is exclusive, so +2 = include tomorrow.
    date_to = (today + timedelta(days=2)).isoformat()

    paper_bets_logged = 0

    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        form_service = FormService(client, settings)
        engine = StrategyEngine(settings)

        for league in settings.leagues:
            try:
                matches = await client.list_scheduled_matches(
                    league, date_from, date_to
                )
            except FootballDataError as e:
                log.warning("league_fetch_failed", league=league, error=str(e))
                continue

            log.info(
                "league_fetched",
                league=league,
                upcoming=len(matches),
                window=f"{date_from}..{date_to}",
            )
            for m in matches:
                try:
                    bets = await _score_and_log_one(
                        m, league, form_service, engine, settings
                    )
                    paper_bets_logged += bets
                except FootballDataError as e:
                    log.warning(
                        "fixture_score_failed",
                        league=league,
                        match_id=m.get("id"),
                        error=str(e),
                    )
                except Exception as e:  # noqa: BLE001 — defensive
                    log.error(
                        "fixture_score_unexpected",
                        league=league,
                        match_id=m.get("id"),
                        error=str(e),
                    )

    log.info(
        "scoring_run_done",
        paper_bets=paper_bets_logged,
        daily_exposure_usd=round(daily_paper_exposure_usd(), 2),
    )
    return paper_bets_logged


async def _score_and_log_one(
    match: dict,
    league: str,
    form_service: FormService,
    engine: StrategyEngine,
    settings,
) -> int:
    """Score one fixture, log a Phase-1 favourite-only paper bet."""
    log = get_logger(__name__)
    fixture_id = int(match["id"])
    kickoff = _parse_kickoff(match["utcDate"])
    home = _parse_team(match["homeTeam"])
    away = _parse_team(match["awayTeam"])

    fixture_form = await form_service.fixture_form(
        fixture_id=fixture_id,
        competition_code=league,
        kickoff=kickoff,
        home_team=home,
        away_team=away,
    )

    prediction = engine.predict(fixture_form)
    log.info(
        "prediction",
        fixture_id=fixture_id,
        league=league,
        home=prediction.home_team,
        away=prediction.away_team,
        kickoff=kickoff.isoformat(),
        p_home=round(prediction.p_home, 3),
        p_draw=round(prediction.p_draw, 3),
        p_away=round(prediction.p_away, 3),
        home_form=fixture_form.home_form.weighted_points,
        away_form=fixture_form.away_form.weighted_points,
    )

    pred_id = upsert_prediction(prediction, kickoff=kickoff)

    # Risk gate: stop if today's exposure has already hit the cap.
    if (
        daily_paper_exposure_usd() + settings.fixed_stake_usd
        > settings.daily_exposure_cap_usd
    ):
        log.warning(
            "exposure_cap_reached",
            cap_usd=settings.daily_exposure_cap_usd,
        )
        return 0

    favourite = prediction.best_outcome
    p = {
        favourite: max(prediction.p_home, prediction.p_draw, prediction.p_away)
    }[favourite]
    rationale = (
        f"Phase-1 paper bet on model favourite: "
        f"P({favourite.value})={p:.3f}; "
        f"home_form_w={fixture_form.home_form.weighted_points:.2f}, "
        f"away_form_w={fixture_form.away_form.weighted_points:.2f}"
    )
    inserted = insert_paper_bet_no_market(
        prediction=prediction,
        prediction_id=pred_id,
        outcome=favourite,
        stake_usd=settings.fixed_stake_usd,
        rationale=rationale,
    )
    if inserted:
        log.info(
            "paper_bet_logged",
            fixture_id=fixture_id,
            outcome=favourite.value,
            stake_usd=settings.fixed_stake_usd,
        )
        return 1
    log.debug("paper_bet_already_logged", fixture_id=fixture_id)
    return 0


# ----------------------------------------------------------------------
# CLI commands
# ----------------------------------------------------------------------
@app.command("run-once")
def run_once() -> None:
    """Score the next 48h of fixtures and log paper bets."""
    settings = get_settings()
    configure_logging(settings.log_level)
    n = asyncio.run(_score_once())
    typer.echo(f"Logged {n} paper bet(s).")


@app.command("run-daemon")
def run_daemon(
    cron: Annotated[
        str | None,
        typer.Option(help="Cron expression (UTC). Defaults to settings."),
    ] = None,
) -> None:
    """Run the scoring job on a schedule (default 08:00 UTC daily)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)

    cron_expr = cron or settings.daemon_cron
    trigger = CronTrigger.from_crontab(cron_expr, timezone=timezone.utc)

    async def _main() -> None:
        scheduler = AsyncIOScheduler(timezone=timezone.utc)
        scheduler.add_job(_score_once, trigger=trigger, id="daily_score")
        scheduler.start()
        log.info("daemon_started", cron=cron_expr)
        await _score_once()  # immediate first run
        try:
            await asyncio.Event().wait()
        finally:
            scheduler.shutdown(wait=False)

    asyncio.run(_main())


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the SQLite schema."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    typer.echo(f"DB initialized at {settings.db_path.resolve()}")


@bets_app.command("list")
def bets_list(
    since: Annotated[
        str,
        typer.Option(help="How far back to look, e.g. '7d', '24h'."),
    ] = "7d",
) -> None:
    """List recent paper bets."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)

    days = _parse_since(since)
    rows = list_recent_paper_bets(days=days)
    if not rows:
        typer.echo(f"No paper bets in the last {since}.")
        return
    typer.echo(
        f"{'created_at':<20}  {'fixture':>8}  {'outcome':<5}  "
        f"{'p':>5}  {'stake':>6}  rationale"
    )
    for b in rows:
        ts = b.created_at.strftime("%Y-%m-%d %H:%M") if b.created_at else "?"
        typer.echo(
            f"{ts:<20}  {b.fixture_id:>8}  {b.outcome:<5}  "
            f"{b.our_probability:>5.2f}  ${b.stake_usd:>5.0f}  "
            f"{b.rationale[:80]}"
        )


def _parse_since(s: str) -> int:
    """Convert '7d' or '24h' to a number of days (rounded up)."""
    s = s.strip().lower()
    if s.endswith("d"):
        return max(1, int(s[:-1]))
    if s.endswith("h"):
        return max(1, (int(s[:-1]) + 23) // 24)
    return max(1, int(s))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
PYEOF

echo "==> Writing tests/conftest.py..."
cat > tests/conftest.py <<'PYEOF'
"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from betbot.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Default Settings instance with deterministic knobs for tests."""
    return Settings(
        BETBOT_MODE="paper",
        FOOTBALL_DATA_API_KEY="fake-test-key",
        BETBOT_FIXED_STAKE_USD=10,
        BETBOT_MAX_BET_USD=50,
        BETBOT_DAILY_EXPOSURE_CAP_USD=200,
        BETBOT_EDGE_THRESHOLD=0.05,
        BETBOT_HOME_ADVANTAGE=0.3,
        BETBOT_DRAW_SCORE=2.4,
        BETBOT_SOFTMAX_TEMP=1.0,
        BETBOT_OPP_STRENGTH_WEIGHT=0.5,
    )
PYEOF

echo "==> Writing tests/test_probabilities.py..."
cat > tests/test_probabilities.py <<'PYEOF'
"""Tests for the pure math helpers."""

from __future__ import annotations

import math

import pytest

from betbot.strategy.probabilities import (
    edge,
    implied_probability,
    opponent_strength_factor,
    softmax,
)


class TestSoftmax:
    def test_uniform_input(self) -> None:
        p = softmax([1.0, 1.0, 1.0])
        assert all(math.isclose(x, 1 / 3, abs_tol=1e-9) for x in p)

    def test_sums_to_one(self) -> None:
        p = softmax([1.0, 2.0, 3.0])
        assert math.isclose(sum(p), 1.0, abs_tol=1e-9)

    def test_monotonic(self) -> None:
        p = softmax([1.0, 2.0, 3.0])
        assert p[0] < p[1] < p[2]

    def test_handles_large_scores(self) -> None:
        # The max-subtraction trick should prevent overflow.
        p = softmax([1000.0, 1001.0, 1002.0])
        assert math.isclose(sum(p), 1.0, abs_tol=1e-9)

    def test_empty(self) -> None:
        assert softmax([]) == []


class TestOpponentStrength:
    def test_top_team(self) -> None:
        assert opponent_strength_factor(1.0, 20, 0.5) == pytest.approx(1.5)

    def test_bottom_team(self) -> None:
        assert opponent_strength_factor(20.0, 20, 0.5) == pytest.approx(0.5)

    def test_unknown_position(self) -> None:
        assert opponent_strength_factor(None, 20, 0.5) == 1.0


class TestEdgeAndImplied:
    def test_edge(self) -> None:
        assert edge(0.6, 0.5) == pytest.approx(0.1)
        assert edge(0.4, 0.5) == pytest.approx(-0.1)

    def test_implied_clipping(self) -> None:
        assert implied_probability(-0.1) == 0.0
        assert implied_probability(1.5) == 1.0
        assert implied_probability(0.4) == 0.4
PYEOF

echo "==> Writing tests/test_cache.py..."
cat > tests/test_cache.py <<'PYEOF'
"""Tests for the TTL cache."""

from __future__ import annotations

import time

from betbot.utils.cache import TTLCache


def test_set_and_get() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    assert c.get("a") == 1


def test_expiry() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=0.05)
    c.set("a", 1)
    time.sleep(0.1)
    assert c.get("a") is None


def test_eviction_at_max_size() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=60, max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # should evict "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_clear() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.clear()
    assert c.get("a") is None
    assert len(c) == 0
PYEOF

echo ""
echo "==> Done writing files."
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  pip install -e '.[dev]'"
echo "  python -m compileall -q src tests"
echo "  pytest"
echo "  cp .env.example .env   # then edit .env: set FOOTBALL_DATA_API_KEY"
echo "  tfsm init-db"
echo "  tfsm run-once"
