"""Free pre-match odds providers + the shared service that anchors on them.

WHY THIS EXISTS
---------------
``anchor_to_market`` only ever fired when Polymarket happened to list the
fixture, so most big-5 matches shipped raw, unanchored model output. The
literature (and our own held-out season) is unambiguous that the bookmaker
line is the strongest single public forecaster, so a fixture with no market
anchor is a fixture priced worse than it needs to be.

WHAT THIS IS NOT
----------------
Anchoring moves us TOWARD market-level accuracy. It cannot beat the market —
by construction it shrinks us toward the price. Nothing here is an edge and
nothing here should ever be described as one.

DESIGN
------
* :class:`OddsProvider` is a small protocol so a second free source can be
  dropped in later without touching callers.
* :class:`FootballDataCoUkProvider` is the first implementation:
  football-data.co.uk publishes ONE free CSV (``fixtures.csv``) covering every
  division's upcoming fixtures with several books' 1X2 prices. No API key, no
  signup, no quota, no cost. One HTTP GET serves a whole Saturday's card.
* :class:`OddsService` owns the shared TTL cache + minimum request interval,
  so a 20-fixture matchday is ONE request, not twenty (the Highlightly
  lesson: share the service across the alert batch).
* Team names are resolved through :class:`OddsNameResolver` — an EXPLICIT
  table. Unresolvable names are skipped and logged, never guessed.

Every failure mode degrades to "no quote", and a "no quote" fixture takes the
existing unanchored path. Nothing here can raise into the alert path.
"""

from __future__ import annotations

import asyncio
import csv
import io
import time
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from betbot.data.odds_names import OddsNameResolver
from betbot.logging import get_logger
from betbot.strategy.ensemble import de_vig

log = get_logger(__name__)

# football-data.co.uk division code -> our football-data.org competition code.
# Mirrors scripts/fetch_club_results.py so the odds land in the same namespace
# as the results that trained the model.
DIV_TO_LEAGUE: dict[str, str] = {
    "E0": "PL",
    "SP1": "PD",
    "D1": "BL1",
    "I1": "SA",
    "F1": "FL1",
}
LEAGUE_TO_DIV: dict[str, str] = {v: k for k, v in DIV_TO_LEAGUE.items()}

# Book preference for the 1X2 triple. Pinnacle first (sharpest, lowest margin),
# then Bet365, then the market average, then the max. All three columns of a
# book must be present or we move to the next book — mixing books across
# outcomes would fabricate an overround that is not any real market's.
PREMATCH_BOOKS: tuple[tuple[str, str, str], ...] = (
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
    ("MaxH", "MaxD", "MaxA"),
)
# Closing columns exist only in the historical season files. They are NOT
# available pre-match and must never be used by the live path — a backtest
# that anchors on them is optimistic versus what we could actually get.
CLOSING_BOOKS: tuple[tuple[str, str, str], ...] = (
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("MaxCH", "MaxCD", "MaxCA"),
)

# football-data.co.uk serves plain files but fronts them with a CDN that 403s
# some default agents (same lesson as the Highlightly client).
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


# ----------------------------------------------------------------------
# Value types
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MatchOdds:
    """One fixture's 1X2 decimal prices from one provider."""

    league: str          # our competition code (PL/PD/BL1/SA/FL1)
    match_date: date
    home: str            # canonical dataset name (normalised)
    away: str
    price_home: float    # decimal odds
    price_draw: float
    price_away: float
    source: str          # provider name
    book: str            # which book's columns were used
    kind: str = "prematch"  # "prematch" | "closing"

    def implied(self) -> tuple[float, float, float]:
        """Raw implied probabilities — these SUM ABOVE 1 (the overround)."""
        return (1.0 / self.price_home, 1.0 / self.price_draw, 1.0 / self.price_away)

    def probabilities(self) -> tuple[float, float, float]:
        """De-vigged 1X2 probabilities (overround stripped, sums to 1)."""
        p = de_vig(list(self.implied()))
        return (p[0], p[1], p[2])

    @property
    def overround(self) -> float:
        return sum(self.implied())


@dataclass(frozen=True)
class OddsQuote:
    """What the anchoring path consumes: de-vigged probs + full provenance."""

    odds: MatchOdds

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return self.odds.probabilities()

    @property
    def source(self) -> str:
        return self.odds.source


class OddsProvider(Protocol):
    """Minimal contract for a free pre-match 1X2 odds source."""

    name: str

    def fetch(self, leagues: Sequence[str]) -> list[MatchOdds]:
        """Blocking fetch of upcoming fixtures' prices for ``leagues``.

        MUST NOT raise: a provider that cannot be reached returns ``[]``.
        """
        ...


# ----------------------------------------------------------------------
# Provider #1 — football-data.co.uk (free, no key, no signup)
# ----------------------------------------------------------------------
def _parse_price(row: dict, keys: tuple[str, str, str]) -> tuple[float, float, float] | None:
    out = []
    for k in keys:
        v = (row.get(k) or "").strip()
        if not v:
            return None
        try:
            f = float(v)
        except ValueError:
            return None
        if f <= 1.0 or f > 1000.0:  # decimal odds must exceed 1.0
            return None
        out.append(f)
    return (out[0], out[1], out[2])


def _parse_date(raw: str) -> date | None:
    """football-data.co.uk uses dd/mm/yyyy (older files dd/mm/yy)."""
    parts = (raw or "").strip().split("/")
    if len(parts) != 3:
        return None
    d, m, y = parts
    if not (d.isdigit() and m.isdigit() and y.isdigit()):
        return None
    year = int(y) + 2000 if len(y) == 2 else int(y)
    try:
        return date(year, int(m), int(d))
    except ValueError:
        return None


def _http_get(url: str, timeout: float) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 — a dead feed must never break alerts
        log.warning("odds_fetch_failed", url=url, error=str(e))
        return None


class FootballDataCoUkProvider:
    """Free upcoming-fixture 1X2 prices from football-data.co.uk.

    ONE request covers every division, so a full matchday costs a single GET.
    """

    name = "football-data.co.uk"
    FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
    SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

    def __init__(
        self,
        resolver: OddsNameResolver,
        *,
        timeout: float = 30.0,
        fixtures_url: str | None = None,
        fetcher=None,
    ) -> None:
        self._resolver = resolver
        self._timeout = timeout
        self._fixtures_url = fixtures_url or self.FIXTURES_URL
        self._fetcher = fetcher or _http_get
        # Spellings the feed used that we refused to guess at. Reported by the
        # audit script and logged once per run — this is how the alias table
        # grows, deliberately.
        self.unresolved: dict[str, int] = {}
        # Exact coverage counters (in-scope fixtures seen / skipped on a name).
        self.attempted_fixtures = 0
        self.skipped_fixtures = 0

    # -- public ---------------------------------------------------------
    def fetch(self, leagues: Sequence[str]) -> list[MatchOdds]:
        raw = self._fetcher(self._fixtures_url, self._timeout)
        if not raw:
            return []
        return self.parse(raw, leagues, kind="prematch")

    def fetch_season(self, season: str, league: str, *, kind: str = "prematch") -> list[MatchOdds]:
        """Historical season file (backtest only — never the live path).

        ``kind="closing"`` reads the C-columns, which are CLOSING prices and
        therefore NOT available before kickoff.
        """
        div = LEAGUE_TO_DIV.get(league)
        if div is None:
            return []
        raw = self._fetcher(self.SEASON_URL.format(season=season, div=div), self._timeout)
        if not raw:
            return []
        return self.parse(raw, [league], kind=kind, force_league=league)

    # -- parsing (pure, unit-testable) ----------------------------------
    def parse(
        self,
        raw: str,
        leagues: Sequence[str],
        *,
        kind: str = "prematch",
        force_league: str | None = None,
    ) -> list[MatchOdds]:
        wanted = {lg for lg in leagues if lg in LEAGUE_TO_DIV}
        books = PREMATCH_BOOKS if kind == "prematch" else CLOSING_BOOKS
        out: list[MatchOdds] = []
        # Strip a UTF-8 BOM: football-data.co.uk's fixtures.csv carries one, and
        # it otherwise glues itself to the first column name ("﻿Div").
        text = raw.lstrip("﻿")
        for row in csv.DictReader(io.StringIO(text)):
            row = {(k or "").lstrip("﻿").strip(): v for k, v in row.items()}
            league = force_league or DIV_TO_LEAGUE.get((row.get("Div") or "").strip())
            if league is None or (wanted and league not in wanted):
                continue
            d = _parse_date(row.get("Date", ""))
            if d is None:
                continue
            raw_home = (row.get("HomeTeam") or "").strip()
            raw_away = (row.get("AwayTeam") or "").strip()
            if not (raw_home and raw_away):
                continue
            self.attempted_fixtures += 1
            home = self._resolver.resolve(raw_home)
            away = self._resolver.resolve(raw_away)
            if home is None or away is None:
                self.skipped_fixtures += 1
                for nm, res in ((raw_home, home), (raw_away, away)):
                    if res is None and nm:
                        self.unresolved[nm] = self.unresolved.get(nm, 0) + 1
                continue
            if home == away:
                # Two different spellings collapsing onto one club is exactly
                # the Espanyol/Barcelona defect. Refuse the row loudly.
                log.error(
                    "odds_name_collision",
                    provider=self.name,
                    home=raw_home,
                    away=raw_away,
                    resolved=home,
                )
                continue
            prices = None
            book = ""
            for cols in books:
                prices = _parse_price(row, cols)
                if prices is not None:
                    book = cols[0]
                    break
            if prices is None:
                continue
            out.append(
                MatchOdds(
                    league=league,
                    match_date=d,
                    home=home,
                    away=away,
                    price_home=prices[0],
                    price_draw=prices[1],
                    price_away=prices[2],
                    source=self.name,
                    book=book,
                    kind=kind,
                )
            )
        return out


# ----------------------------------------------------------------------
# Shared service (one cache + one rate limiter for the whole batch)
# ----------------------------------------------------------------------
class OddsService:
    """Shared, rate-limited, cached access to the configured odds providers.

    Call :meth:`prime` once per batch (it is idempotent and TTL-guarded, so
    calling it per fixture is cheap and still results in ONE HTTP GET), then
    :meth:`quote` per fixture — a pure dict lookup.
    """

    def __init__(
        self,
        settings,
        providers: Iterable[OddsProvider] | None = None,
        *,
        resolver: OddsNameResolver | None = None,
        clock=time.monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        given = list(providers) if providers is not None else None
        if resolver is None and given:
            for p in given:
                r = getattr(p, "_resolver", None)
                if r is not None:
                    resolver = r
                    break
        self._resolver = resolver or _build_resolver(settings)
        # providers=[] is a deliberate OFFLINE service (rows injected via
        # load_rows: backtests, replays, tests). Only ``None`` means "give me
        # the default live provider".
        self._providers: list[OddsProvider] = (
            given if given is not None else [_default_provider(settings, self._resolver)]
        )
        self._index: dict[tuple[str, str, str], list[MatchOdds]] = {}
        self._loaded_at: float | None = None
        self._last_request_at: float | None = None
        self._lock = asyncio.Lock()

    # -- loading --------------------------------------------------------
    async def prime(self, leagues: Sequence[str]) -> int:
        """Refresh the shared cache if stale. Returns #fixtures indexed."""
        async with self._lock:
            if not self._is_stale():
                return sum(len(v) for v in self._index.values())
            if not self._may_request():
                log.debug("odds_rate_limited", provider_count=len(self._providers))
                return sum(len(v) for v in self._index.values())
            self._last_request_at = self._clock()
            rows: list[MatchOdds] = []
            for provider in self._providers:
                try:
                    got = await asyncio.to_thread(provider.fetch, list(leagues))
                except Exception as e:  # noqa: BLE001 — never break the alert path
                    log.warning("odds_provider_error", provider=provider.name, error=str(e))
                    continue
                log.info("odds_provider_fetched", provider=provider.name, fixtures=len(got))
                rows.extend(got)
                unresolved = getattr(provider, "unresolved", None)
                if unresolved:
                    log.warning(
                        "odds_names_unresolved",
                        provider=provider.name,
                        names=sorted(unresolved),
                    )
            self._reindex(rows)
            self._loaded_at = self._clock()
            return len(rows)

    def load_rows(self, rows: Iterable[MatchOdds]) -> None:
        """Inject rows directly (tests, backtests, offline replays)."""
        self._reindex(list(rows))
        self._loaded_at = self._clock()

    def _reindex(self, rows: list[MatchOdds]) -> None:
        idx: dict[tuple[str, str, str], list[MatchOdds]] = {}
        for r in rows:
            # First provider in the list wins a duplicate fixture; later
            # providers only fill gaps.
            idx.setdefault((r.league, r.home, r.away), []).append(r)
        self._index = idx

    def _is_stale(self) -> bool:
        if self._loaded_at is None:
            return True
        return (self._clock() - self._loaded_at) >= float(
            getattr(self._settings, "odds_cache_ttl_seconds", 21600.0)
        )

    def _may_request(self) -> bool:
        if self._last_request_at is None:
            return True
        gap = float(getattr(self._settings, "odds_min_request_interval_seconds", 60.0))
        return (self._clock() - self._last_request_at) >= gap

    # -- lookup ---------------------------------------------------------
    def quote(
        self, league: str, match_date: date, home_name: str, away_name: str
    ) -> OddsQuote | None:
        """De-vigged market quote for one fixture, or ``None`` (skip + log).

        ``home_name``/``away_name`` are LIVE football-data.org names; they go
        through the same explicit resolver as the feed's own spellings, so
        both sides meet in the canonical dataset namespace.
        """
        resolver = self._resolver
        home = resolver.resolve(home_name)
        away = resolver.resolve(away_name)
        if home is None or away is None:
            log.info(
                "odds_fixture_name_unresolved",
                league=league,
                home=home_name,
                away=away_name,
                resolved_home=home,
                resolved_away=away,
            )
            return None
        candidates = self._index.get((league, home, away)) or []
        if not candidates:
            return None
        slack = int(getattr(self._settings, "odds_max_date_slack_days", 3))
        best: MatchOdds | None = None
        best_gap = None
        for c in candidates:
            gap = abs((c.match_date - match_date).days)
            if gap > slack:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = c, gap
        if best is None:
            return None
        return OddsQuote(odds=best)

    @property
    def resolver(self) -> OddsNameResolver:
        return self._resolver

    @property
    def providers(self) -> list[OddsProvider]:
        return list(self._providers)

    def __len__(self) -> int:
        return sum(len(v) for v in self._index.values())


def _build_resolver(settings) -> OddsNameResolver:
    return OddsNameResolver.from_files(
        getattr(settings, "odds_team_alias_path", Path("./config/odds_team_aliases.yaml")),
        getattr(settings, "club_name_map_path", Path("./data/club_name_map.json")),
    )


def _default_provider(settings, resolver: OddsNameResolver | None = None) -> OddsProvider:
    return FootballDataCoUkProvider(
        resolver or _build_resolver(settings),
        timeout=float(getattr(settings, "odds_http_timeout_seconds", 30.0)),
    )


def build_odds_service(settings) -> OddsService | None:
    """Construct the shared service, or ``None`` when the flag is off.

    Never raises: a broken alias table or missing name map disables anchoring
    rather than taking the daemon down.
    """
    if not getattr(settings, "odds_anchor_enabled", False):
        return None
    try:
        return OddsService(settings)
    except Exception as e:  # noqa: BLE001
        log.error("odds_service_init_failed", error=str(e))
        return None


# One process-wide service so the per-fixture pre-match RE-SCORE jobs (two per
# fixture) share the same cache and rate limiter as the daily scoring run.
# Without this a 10-fixture day would fire 20+ separate HTTP GETs.
_SHARED_SERVICE: OddsService | None = None


def shared_odds_service(settings) -> OddsService | None:
    """The process-wide :class:`OddsService` (``None`` when the flag is off)."""
    global _SHARED_SERVICE
    if not getattr(settings, "odds_anchor_enabled", False):
        return None
    if _SHARED_SERVICE is None:
        _SHARED_SERVICE = build_odds_service(settings)
    return _SHARED_SERVICE


def reset_shared_odds_service() -> None:
    """Drop the process-wide service (tests / config reloads)."""
    global _SHARED_SERVICE
    _SHARED_SERVICE = None
