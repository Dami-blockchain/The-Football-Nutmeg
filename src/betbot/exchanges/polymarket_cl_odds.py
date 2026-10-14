"""Polymarket per-match Champions League 1X2 odds — orderbook-midpoint pricer.

The gap this closes: the free bookmaker CSV feed (``betbot.data.odds``) covers
the big-5 domestic leagues only, so every CL fixture shipped as raw model
output — no market anchor, and therefore no de-vigged market draw probability
for the draw-risk tag. Polymarket lists per-match CL 1X2 markets, but the live
matcher never found them: ``polymarket_gamma`` discovers UCL under tag ``1234``
(``_LEAGUE_TAG_FALLBACK``), and since Oct 2025 per-match CL events moved to tag
``100977`` (slug ``ucl``) / ``106197``. This provider rediscovers them by the
``ucl`` tag SLUG and prices them correctly.

What it is NOT: it does not place orders and it does not touch the exchange
decision path. It produces a de-vigged (H, D, A) probability triple that
:mod:`betbot.strategy.cl_pm_anchor` feeds to the SAME ``anchor_triple`` the
domestic odds anchor uses. Flag-gated (``BETBOT_CL_PM_ANCHOR``), default OFF.

Pricing discipline (Fable scope, 2026-10-14):

* Price = **orderbook midpoint** of each YES token (CLOB ``/midpoint``), NOT the
  last trade. 9/400 last fills in the research harvest were garbage (a draw at
  0.996, an away at 0.999, triples summing to 0.89); midpoints are quote-driven
  and far cleaner.
* Reject a triple if any leg is missing or one-sided (a midpoint at/near 0 or 1
  means an empty/stale book side), or the raw sum of the three mids is outside
  ``[0.95, 1.08]`` — a well-formed 1X2 book carries a small overround, not a
  90%/110% distortion. A rejected fixture ships UNANCHORED (logged), never on a
  bad quote.
* De-vig by normalising the three accepted mids to sum 1.

Team matching reuses the shared :class:`TeamAliasResolver` (exact alias table
first, fuzzy fallback that REFUSES ambiguous ties — the mechanism that already
stops PSG→"Paris FC" and "Athletic Club"→"Atletico"). CL-specific spellings are
layered in from ``config/cl_pm_aliases.yaml``.

Every failure mode returns ``None`` (no quote) rather than raising: an anchor
must never break an alert.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from betbot.exchanges.matcher import TeamAliasResolver, normalize
from betbot.logging import get_logger

log = get_logger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Per-match CL events live under the ``ucl`` tag slug (numeric 100977, with a
# secondary 106197 seen for some rounds). We discover by SLUG because the slug
# listing reliably returns the per-match events the numeric legacy tag (1234)
# no longer does. A browser User-Agent is mandatory — Cloudflare 1010s a bare
# urllib agent, exactly as ``highlightly`` and the ClubElo scrape already note.
CL_TAG_SLUG = "ucl"
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# A midpoint this close to 0 or 1 means one side of the book is empty/stale —
# treat the leg as missing rather than trust a one-sided quote.
_ONE_SIDED_EPS = 0.02

GammaGetter = Callable[[str, dict[str, Any]], Awaitable[Any]]
ClobGetter = Callable[[str], Awaitable[float | None]]


@dataclass(frozen=True)
class CLQuote:
    """A de-vigged per-match CL 1X2 quote from Polymarket midpoints."""

    probabilities: tuple[float, float, float]  # (home, draw, away), sums to 1
    raw_sum: float  # sum of the three raw midpoints before de-vig
    source: str = "pm"


@dataclass(frozen=True)
class _CLEvent:
    """A parsed per-match CL event: the three YES token ids + resolved teams."""

    home: str
    away: str
    match_date: str  # ISO yyyy-mm-dd (kickoff date, UTC)
    tok_home: str
    tok_draw: str
    tok_away: str


# --------------------------------------------------------------------------
# Default network getters (urllib + browser UA). Injectable for tests.
# --------------------------------------------------------------------------
def _urlopen_json(url: str, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.load(r)


async def _default_gamma_get(path: str, params: dict[str, Any]) -> Any:
    from urllib.parse import urlencode

    qs = urlencode(params)
    url = f"{GAMMA_BASE}{path}?{qs}"
    return await asyncio.to_thread(_urlopen_json, url)


async def _default_clob_midpoint(token_id: str) -> float | None:
    url = f"{CLOB_BASE}/midpoint?token_id={token_id}"
    try:
        data = await asyncio.to_thread(_urlopen_json, url)
    except Exception:  # noqa: BLE001
        return None
    try:
        return float(data["mid"])
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
def _decode_tokens(market: dict[str, Any]) -> list[str]:
    ct = market.get("clobTokenIds")
    if isinstance(ct, str):
        try:
            ct = json.loads(ct)
        except (ValueError, TypeError):
            return []
    return list(ct or [])


def _parse_event(ev: dict[str, Any]) -> _CLEvent | None:
    """Turn a Gamma event into a :class:`_CLEvent`, or ``None`` if not a clean
    per-match 1X2 event.

    The real 1X2 event has EXACTLY three markets, one of which is the draw
    ("... end in a draw?"); derivative events (first-to-score, halftime,
    player props, to-qualify) either have a different market count or a draw
    question that is NOT "end in a draw", so this filter excludes them.
    """
    markets = ev.get("markets") or []
    if len(markets) != 3:
        return None
    title = (ev.get("title") or "").strip()
    parts = re.split(r"\s+vs\.?\s+", title)
    if len(parts) != 2:
        return None
    home_name, away_name = parts[0].strip(), parts[1].strip()
    questions = [(m.get("question") or "") for m in markets]
    draw_idx = [i for i, q in enumerate(questions) if "end in a draw" in q.lower()]
    if len(draw_idx) != 1:
        return None
    di = draw_idx[0]
    win_idx = [i for i in range(3) if i != di]

    # Map the two win markets to home/away. groupItemTitle equality first
    # (Gamma fills it with the exact team name), then the win question text.
    hi = ai = None
    for i in win_idx:
        git = (markets[i].get("groupItemTitle") or "").strip()
        if normalize(git) == normalize(home_name):
            hi = i
        elif normalize(git) == normalize(away_name):
            ai = i
    if hi is None or ai is None:
        hi = ai = None
        for i in win_idx:
            ql = questions[i].lower()
            if home_name.lower() in ql:
                hi = i
            elif away_name.lower() in ql:
                ai = i
    if hi is None or ai is None or hi == ai:
        return None

    toks = {i: _decode_tokens(markets[i]) for i in range(3)}
    if any(not toks[i] for i in range(3)):
        return None

    gst = markets[di].get("gameStartTime") or markets[hi].get("gameStartTime")
    if not gst:
        return None
    match_date = str(gst)[:10]

    return _CLEvent(
        home=home_name,
        away=away_name,
        match_date=match_date,
        tok_home=toks[hi][0],
        tok_draw=toks[di][0],
        tok_away=toks[ai][0],
    )


class PolymarketCLOddsProvider:
    """Discover + price per-match CL 1X2 markets on Polymarket (midpoints)."""

    def __init__(
        self,
        *,
        resolver: TeamAliasResolver,
        gamma_get: GammaGetter | None = None,
        clob_midpoint: ClobGetter | None = None,
        min_raw_sum: float = 0.95,
        max_raw_sum: float = 1.08,
        ttl_seconds: float = 21600.0,
        clock=time.monotonic,
    ) -> None:
        self._resolver = resolver
        self._gamma_get = gamma_get or _default_gamma_get
        self._clob_midpoint = clob_midpoint or _default_clob_midpoint
        self._min_sum = min_raw_sum
        self._max_sum = max_raw_sum
        # Cache TTL, mirroring the domestic odds-anchor convention
        # (BETBOT_ODDS_TTL_SEC, 6h). This daemon is a long-lived
        # AsyncIOScheduler: without a TTL the event index AND — critically —
        # the per-token MIDPOINT prices would be frozen at first-prime forever,
        # anchoring day-2 fixtures to day-1 prices. On expiry both caches are
        # dropped and rebuilt, so every scoring run prices off fresh mids.
        self._ttl = ttl_seconds
        self._clock = clock
        self._primed_at: float | None = None
        # Serialise concurrent prime() refreshes (the daemon scores fixtures
        # concurrently): without this, N fixtures racing a cold cache each fire
        # a full discovery. The lock is created lazily so the provider stays
        # usable outside a running loop (tests build it bare).
        self._lock: asyncio.Lock | None = None
        # After a discovery failure, don't re-hammer the feed for every fixture
        # in the same run: back off for _fail_backoff seconds (short, so a
        # recovered feed is retried the next matchday run, not this one).
        self._fail_backoff = 60.0
        self._last_fail_at: float | None = None
        self._events: list[_CLEvent] | None = None
        # (canon_home, canon_away, date) -> _CLEvent
        self._index: dict[tuple[str, str, str], _CLEvent] = {}
        self._mid_cache: dict[str, float | None] = {}

    # ------------------------------------------------------------------
    async def prime(self) -> int:
        """Discover + parse open CL per-match events. Idempotent per instance.

        Returns the number of clean 1X2 events indexed. Never raises: on a feed
        failure it indexes nothing and the fixture ships unanchored.
        """
        if self._events is not None and self._primed_at is not None and (
            self._clock() - self._primed_at
        ) < self._ttl:
            return len(self._events)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            # Re-check under the lock: a concurrent caller may have just primed.
            if self._events is not None and self._primed_at is not None and (
                self._clock() - self._primed_at
            ) < self._ttl:
                return len(self._events)
            # Fresh discovery failed recently? Degrade to the last-good index
            # (or empty) without re-hitting the feed for every fixture in a run.
            if self._last_fail_at is not None and (
                self._clock() - self._last_fail_at
            ) < self._fail_backoff:
                return len(self._events or [])
            return await self._refresh()

    async def _refresh(self) -> int:
        raw: list[dict[str, Any]] = []
        try:
            raw = await self._list_events(CL_TAG_SLUG)
        except Exception as e:  # noqa: BLE001
            log.warning("cl_pm_discovery_failed", error=str(e))
            self._last_fail_at = self._clock()
            # On a discovery failure keep any last-good index (it degrades
            # safely as the calendar rolls past it — quote() matches on date)
            # and do NOT stamp _primed_at, so a recovered feed re-primes next
            # tick. Only index-and-stamp an EMPTY result on the first prime.
            if self._events is None:
                self._events = []
            return len(self._events)
        # Fresh discovery: replace both caches wholesale and (re)stamp.
        self._index = {}
        self._mid_cache = {}
        parsed: list[_CLEvent] = []
        seen: set[tuple[str, str, str]] = set()
        for ev in raw:
            p = _parse_event(ev)
            if p is None:
                continue
            key = self._key(p.home, p.away, p.match_date)
            if key is None or key in seen:
                continue
            seen.add(key)
            self._index[key] = p
            parsed.append(p)
        self._events = parsed
        self._primed_at = self._clock()
        log.info("cl_pm_primed", raw_events=len(raw), parsed=len(parsed))
        return len(parsed)

    async def _list_events(self, tag_slug: str) -> list[dict[str, Any]]:
        """Page open events for a tag slug (Gamma caps at 100/page)."""
        out: list[dict[str, Any]] = []
        offset = 0
        page = 100
        while True:
            batch = await self._gamma_get(
                "/events",
                {
                    "tag_slug": tag_slug,
                    "closed": "false",
                    "limit": page,
                    "offset": offset,
                    "order": "startDate",
                    "ascending": "true",
                },
            )
            if not isinstance(batch, list) or not batch:
                break
            out.extend(batch)
            if len(batch) < page:
                break
            offset += page
            if offset > 2000:
                break
        return out

    def _key(self, home: str, away: str, match_date: str) -> tuple[str, str, str] | None:
        ch = self._resolver._canonical_norm(home)
        ca = self._resolver._canonical_norm(away)
        if not ch or not ca:
            return None
        return (ch, ca, match_date)

    # ------------------------------------------------------------------
    async def quote(
        self, home: str, away: str, match_date: str
    ) -> CLQuote | None:
        """De-vigged (H, D, A) midpoint quote for a fixture, or ``None``.

        ``None`` (with a logged reason) when: the fixture is not listed, a leg's
        book is missing/one-sided, or the raw midpoint sum is outside the
        overround band. Never raises.
        """
        await self.prime()
        ev = self._lookup(home, away, match_date)
        if ev is None:
            log.info(
                "odds_anchor_skipped",
                league="CL",
                home=home,
                away=away,
                reason="pm_not_listed",
            )
            return None
        mids = []
        for tok in (ev.tok_home, ev.tok_draw, ev.tok_away):
            m = await self._midpoint(tok)
            mids.append(m)
        if any(m is None for m in mids):
            log.info(
                "odds_anchor_skipped",
                league="CL",
                home=home,
                away=away,
                reason="pm_leg_missing",
            )
            return None
        if any(m <= _ONE_SIDED_EPS or m >= 1.0 - _ONE_SIDED_EPS for m in mids):
            log.info(
                "odds_anchor_skipped",
                league="CL",
                home=home,
                away=away,
                reason="pm_leg_one_sided",
                mids=[round(m, 4) for m in mids],
            )
            return None
        raw_sum = sum(mids)
        if not (self._min_sum <= raw_sum <= self._max_sum):
            log.info(
                "odds_anchor_skipped",
                league="CL",
                home=home,
                away=away,
                reason="pm_sum_out_of_band",
                raw_sum=round(raw_sum, 4),
            )
            return None
        devig = tuple(m / raw_sum for m in mids)
        return CLQuote(probabilities=devig, raw_sum=raw_sum, source="pm")

    def _lookup(self, home: str, away: str, match_date: str) -> _CLEvent | None:
        key = self._key(home, away, match_date)
        if key is None:
            return None
        return self._index.get(key)

    async def _midpoint(self, token_id: str) -> float | None:
        if token_id in self._mid_cache:
            return self._mid_cache[token_id]
        m = await self._clob_midpoint(token_id)
        self._mid_cache[token_id] = m
        return m


# --------------------------------------------------------------------------
# Alias wiring + process-wide shared instance
# --------------------------------------------------------------------------
_DEF_ALIASES = Path("config/team_aliases.yaml")
_CL_ALIASES = Path("config/cl_pm_aliases.yaml")


def _load_merged_aliases() -> dict[str, list[str]]:
    """Domestic alias table + CL-specific overlay (overlay wins on a key)."""
    merged: dict[str, list[str]] = {}
    for path in (_DEF_ALIASES, _CL_ALIASES):
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for canon, alts in (data.get("aliases") or {}).items():
            merged.setdefault(canon, [])
            merged[canon].extend(alts or [])
    return merged


def build_cl_pm_provider(settings) -> PolymarketCLOddsProvider:
    resolver = TeamAliasResolver(aliases=_load_merged_aliases())
    return PolymarketCLOddsProvider(resolver=resolver)


_SHARED: PolymarketCLOddsProvider | None = None


def shared_cl_pm_provider(settings) -> PolymarketCLOddsProvider | None:
    """Process-wide provider, or ``None`` when ``BETBOT_CL_PM_ANCHOR`` is off."""
    global _SHARED
    if not getattr(settings, "cl_pm_anchor_enabled", False):
        return None
    if _SHARED is None:
        _SHARED = build_cl_pm_provider(settings)
    return _SHARED


def reset_shared_cl_pm_provider() -> None:
    """Test hook: drop the cached provider."""
    global _SHARED
    _SHARED = None
