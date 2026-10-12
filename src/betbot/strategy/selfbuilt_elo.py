"""Self-built cross-league Elo FALLBACK support for the CL engine.

Pure, dependency-light helpers (no DB, no network) so the staleness /
both-stale decision is cheaply unit-testable. The engine
(:class:`betbot.strategy.cl_engine.EuropeanStrategyEngine`) loads the
self-built snapshot through here and asks :func:`decide_active` which snapshot
to price off.

Design (FALLBACK_SCOPE.md rev 2):

* The self-built snapshot is written in the SAME ClubElo CSV schema
  (``Rank,Club,Country,Level,Elo,From,To``) with ``Club``/``Country`` already
  bridged to ClubElo canonical keys (see ``scripts/build_selfbuilt_elo.py``),
  so the engine can load it with the exact same ``_load_snapshot`` reader and
  resolve fixture names against it identically.
* Option (a): ratings are served RAW and priced with the LIVE site constants
  (home_adv 51 / scale 312) — the setup the 22-season backtest validated. There
  is no OLS / affine map. The only guard is a per-build spread sanity band:
  reject a build whose Elo SD falls outside ``[SD_MIN, SD_MAX]`` (catches a
  corrupted / partial build).
* The self-built snapshot has its OWN staleness contract
  (``SELFBUILT_STALE_AFTER_DAYS``, kept in the 14-21 band because the free CL
  feed is per-matchday — roughly weekly in the league phase, with longer gaps
  between matchdays — so a 10-day threshold would cry wolf on a healthy feed).
* ``decide_active`` encodes the flag gate + the both-stale rule and never
  presents a doubly-stale rating as if fresh.
"""

from __future__ import annotations

import statistics as _st
from dataclasses import dataclass, field
from datetime import date

#: Self-built snapshot staleness threshold (days), measured off its own ``From``
#: date. Kept in the scope's 14-21 band (NOT 10): the free CL feed is
#: per-matchday (~weekly in the league phase, longer gaps between rounds), so a
#: 10-day threshold would trip on a healthy feed.
SELFBUILT_STALE_AFTER_DAYS = 17

#: Hard upper bound (days): a served rating older than this by its own ``From``
#: is too stale to trust at all, so the engine drops to the naive form engine
#: rather than price off it (§7 optional upper bound). Comfortably above both
#: staleness thresholds so it only catches a genuinely abandoned feed.
HARD_NAIVE_AFTER_DAYS = 90

#: Per-build Elo-SD sanity band (option (a), §4). A healthy self-built snapshot
#: measured SD ~147; a corrupted / partial build collapses or explodes the
#: spread. Reject outside this band so a bad build is treated as unusable
#: rather than priced.
SELFBUILT_SD_MIN = 120.0
SELFBUILT_SD_MAX = 180.0


def elo_sd(clubs: dict[str, float]) -> float | None:
    """Population SD of a snapshot's Elo values (None if < 2 clubs)."""
    if len(clubs) < 2:
        return None
    return _st.pstdev(clubs.values())


@dataclass(frozen=True)
class SnapshotHealth:
    """Load + freshness verdict for one rating snapshot.

    ``usable`` means "has enough credible ratings to price off". ``stale``
    means "older than its own threshold / undated / empty". A snapshot can be
    usable but stale (aged but credible), which is exactly the both-stale case.
    """

    clubs: dict[str, float] = field(default_factory=dict)
    from_date: date | None = None
    usable: bool = False
    stale: bool = True
    sd: float | None = None
    age_days: int | None = None
    reason: str = "empty"


def assess_site(
    clubs: dict[str, float],
    from_date: date | None,
    *,
    today: date,
    stale_after: int,
) -> SnapshotHealth:
    """Freshness verdict for the live site scrape.

    The site scrape has NO SD band (it is the real distribution, already
    sanity-filtered + degenerate-country-guarded by ``_load_snapshot``): it is
    usable whenever it is non-empty. Staleness mirrors the engine's existing
    ``_check_freshness`` (age from ``From``; undated -> stale).
    """
    if not clubs:
        return SnapshotHealth(reason="missing_or_empty")
    sd = elo_sd(clubs)
    if from_date is None:
        return SnapshotHealth(
            clubs=clubs, from_date=None, usable=True, stale=True, sd=sd,
            age_days=None, reason="undated",
        )
    age = (today - from_date).days
    stale = age > stale_after
    return SnapshotHealth(
        clubs=clubs, from_date=from_date, usable=True, stale=stale, sd=sd,
        age_days=age, reason=(f"stale_{age}d" if stale else "fresh"),
    )


def assess_selfbuilt(
    clubs: dict[str, float],
    from_date: date | None,
    *,
    today: date,
    stale_after: int = SELFBUILT_STALE_AFTER_DAYS,
    sd_min: float = SELFBUILT_SD_MIN,
    sd_max: float = SELFBUILT_SD_MAX,
) -> SnapshotHealth:
    """Freshness + SD-band verdict for the self-built snapshot.

    Unusable (``usable=False``) when empty OR the per-build Elo SD is outside
    ``[sd_min, sd_max]`` (the corrupted/partial-build guard, §4). An unusable
    snapshot is also stale by construction. A usable snapshot is stale when its
    own ``From`` is older than ``stale_after`` (or undated).
    """
    if not clubs:
        return SnapshotHealth(reason="missing_or_empty")
    sd = elo_sd(clubs)
    if sd is None or not (sd_min <= sd <= sd_max):
        return SnapshotHealth(
            clubs=clubs, from_date=from_date, usable=False, stale=True, sd=sd,
            reason=f"sd_out_of_band_{'' if sd is None else round(sd, 1)}",
        )
    if from_date is None:
        return SnapshotHealth(
            clubs=clubs, from_date=None, usable=True, stale=True, sd=sd,
            age_days=None, reason="undated",
        )
    age = (today - from_date).days
    stale = age > stale_after
    return SnapshotHealth(
        clubs=clubs, from_date=from_date, usable=True, stale=stale, sd=sd,
        age_days=age, reason=(f"stale_{age}d" if stale else "fresh"),
    )


@dataclass(frozen=True)
class FallbackDecision:
    """Which snapshot the engine should price off, and whether we are in a
    degraded fallback regime (for the operator alert + dual-log flag)."""

    source: str            # "site_scrape" | "self_built" | "naive"
    clubs: dict[str, float]
    engaged: bool          # True whenever we are NOT serving a fresh site scrape
    reason: str


def _fresher(a: SnapshotHealth, b: SnapshotHealth) -> SnapshotHealth:
    """Return the usable snapshot with the more-recent ``From`` date.

    A dated snapshot always beats an undated one; between two dated ones the
    later ``From`` wins; ties favour ``a`` (the site scrape, by call order).
    """
    if a.from_date is None and b.from_date is not None:
        return b
    if b.from_date is None and a.from_date is not None:
        return a
    if a.from_date is None and b.from_date is None:
        return a
    return a if a.from_date >= b.from_date else b


def decide_active(
    site: SnapshotHealth,
    selfbuilt: SnapshotHealth,
    *,
    flag_on: bool,
) -> FallbackDecision:
    """Pick the snapshot the engine prices off.

    * Flag OFF -> ALWAYS the site scrape, exactly as today (the regression
      guard: present-but-stale keeps serving aged ratings, missing -> empty ->
      naive via the engine's unresolved-team guard). ``engaged=False`` always.
    * Flag ON + fresh site -> site scrape, ``engaged=False``.
    * Flag ON + stale site:
        - self-built usable AND fresh -> serve self-built (the primary
          fallback). ``engaged=True``.
        - otherwise BOTH-STALE: serve whichever usable snapshot is fresher by
          ``From``; if NEITHER is usable, fall through to naive (empty clubs).
          ``engaged=True`` in every branch (we are degraded and must alert).
    """
    if not flag_on:
        return FallbackDecision(
            source="site_scrape", clubs=site.clubs, engaged=False,
            reason="flag_off",
        )

    if not site.stale and site.usable:
        return FallbackDecision(
            source="site_scrape", clubs=site.clubs, engaged=False,
            reason="site_fresh",
        )

    # Site is stale/missing and the flag is ON: enter the fallback regime.
    if selfbuilt.usable and not selfbuilt.stale:
        return FallbackDecision(
            source="self_built", clubs=selfbuilt.clubs, engaged=True,
            reason="site_stale_selfbuilt_fresh",
        )

    # Both-stale: serve the fresher usable snapshot, else naive.
    usable = [s for s in (site, selfbuilt) if s.usable]
    if not usable:
        return FallbackDecision(
            source="naive", clubs={}, engaged=True,
            reason="both_unusable_naive",
        )
    winner = usable[0] if len(usable) == 1 else _fresher(site, selfbuilt)
    # Hard upper bound: even the fresher snapshot can be too old to trust.
    if winner.age_days is not None and winner.age_days > HARD_NAIVE_AFTER_DAYS:
        return FallbackDecision(
            source="naive", clubs={}, engaged=True,
            reason=f"both_stale_over_{HARD_NAIVE_AFTER_DAYS}d_naive",
        )
    source = "self_built" if winner is selfbuilt else "site_scrape"
    return FallbackDecision(
        source=source, clubs=winner.clubs, engaged=True,
        reason=f"both_stale_served_fresher_{source}",
    )
