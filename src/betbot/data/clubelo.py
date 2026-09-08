"""ClubElo snapshot refresh — the live cross-league Elo source for the CL engine.

``EuropeanStrategyEngine`` reads ``data/clubelo_latest.csv``; this keeps that
file fresh. Used by the daemon's daily tick (so a CL fixture is always priced
off today's ratings) and by ``scripts/fetch_clubelo.py --latest``.

Network failures are non-fatal: the caller logs and carries on with the last
snapshot (and the engine falls back to naive for any club it can't resolve).
But "carries on with the last snapshot" is only safe while that snapshot is
*recent*, so a failed refresh now ends by checking the on-disk file's age and
logging at **ERROR** when it has gone stale. That is the seam the operator
notifier hangs off: it does not send anything itself, it just makes the
degraded state loud and machine-readable via :func:`snapshot_status`.

Three hardening measures, all learned from real failures:

* **Bounded retry with exponential backoff.** ``api.clubelo.com`` accepts the
  TCP connection and then sends nothing at all when its origin is unwell, so a
  single attempt with a long timeout just hangs and gives up. Short per-attempt
  timeout, a few attempts, jittered backoff — and no hammering a free source.
* **Payload validation.** A short body, a missing header or an out-of-range Elo
  means we did not get a ratings CSV; refuse it rather than overwrite a good
  snapshot with a bad one.
* **Atomic write.** The old code wrote straight to ``dest``. A process killed
  (or a disk filled) mid-write leaves a truncated CSV whose last row parses as
  a *plausible* club with a nonsense rating — e.g. ``2,Bayern,GER,1,20`` loads
  as Bayern at Elo 20.0 and silently prices Bayern as the worst team in Europe.
  We write to a sibling temp file and ``os.replace`` it into place, so ``dest``
  is only ever a complete snapshot.
"""

from __future__ import annotations

import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from betbot.logging import get_logger

log = get_logger(__name__)

CLUBELO_URL = "http://api.clubelo.com/{d}"

#: Per-attempt socket timeout. Short on purpose: the observed failure mode is a
#: connection that is accepted and then silent, so waiting longer buys nothing.
DEFAULT_TIMEOUT = 20
#: Total attempts (1 initial + 2 retries). Bounded — this is a free source.
DEFAULT_RETRIES = 3
#: Backoff is ``BACKOFF_BASE * 2**attempt`` seconds, plus jitter.
BACKOFF_BASE = 2.0
BACKOFF_CAP = 30.0

#: A snapshot older than this is loud. ClubElo republishes daily, and the CL
#: engine's own hard cutoff is 14 days, so 3 days is an early warning that
#: still leaves a week and a half of runway to fix the feed.
STALE_AFTER_DAYS = 3

#: Sanity band for a ClubElo rating. Real values sit ~1000-2100; anything
#: outside this is a parse artefact (truncation, shifted columns), not a club.
MIN_ELO = 500.0
MAX_ELO = 2600.0
#: A real snapshot lists every ranked club in Europe (~600 rows).
MIN_ROWS = 50
#: Tolerance for unusable rows before the whole payload is refused.
BAD_ROW_FRACTION = 0.05
BAD_ROW_FLOOR = 5

EXPECTED_HEADER = "Rank,Club,Country,Level,Elo,From,To"


@dataclass(frozen=True)
class SnapshotStatus:
    """Machine-readable freshness of the on-disk ClubElo snapshot.

    The seam for the operator notifier: it can call :func:`snapshot_status` and
    decide whether to page, without this module knowing that Telegram exists.
    """

    path: Path
    exists: bool
    age_days: float | None
    stale: bool
    reason: str
    snapshot_date: date | None
    clubs: int

    def as_log_fields(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "age_days": None if self.age_days is None else round(self.age_days, 2),
            "reason": self.reason,
            "snapshot_date": None if self.snapshot_date is None else self.snapshot_date.isoformat(),
            "clubs": self.clubs,
        }


def _parse_snapshot_date(text: str) -> tuple[date | None, int]:
    """Newest ``From`` date in the CSV, plus the club-row count.

    ``From`` — NOT ``To``. ClubElo's ``To`` column is the end of a rating's
    validity window and is therefore in the *future* for a current snapshot, so
    any age computed from it is negative and no staleness check built on it can
    ever fire. ``From`` is when the rating was last recomputed, which is the
    honest measure of how old the data is.
    """
    newest: date | None = None
    rows = 0
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        rows += 1
        try:
            dt = date.fromisoformat(parts[5].strip())
        except ValueError:
            continue
        if newest is None or dt > newest:
            newest = dt
    return newest, rows


def snapshot_status(path: Path, *, stale_after_days: int = STALE_AFTER_DAYS) -> SnapshotStatus:
    """Inspect the on-disk snapshot without fetching anything.

    Age is taken from the CSV's own newest ``From`` date when it parses, and
    falls back to file mtime otherwise — never from ``To`` (see above).
    """
    path = Path(path)
    if not path.exists():
        return SnapshotStatus(path, False, None, True, "missing", None, 0)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return SnapshotStatus(path, True, None, True, f"unreadable:{e.__class__.__name__}", None, 0)

    snap_date, rows = _parse_snapshot_date(text)
    if not text.startswith("Rank,") or rows == 0:
        return SnapshotStatus(path, True, None, True, "unparseable", None, rows)

    if snap_date is not None:
        age = float((date.today() - snap_date).days)
    else:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age = (datetime.now(timezone.utc) - mtime).total_seconds() / 86400.0

    stale = age > stale_after_days
    return SnapshotStatus(
        path, True, age, stale, f"stale_{age:.1f}d" if stale else "fresh", snap_date, rows
    )


def check_snapshot_freshness(
    path: Path, *, stale_after_days: int = STALE_AFTER_DAYS
) -> SnapshotStatus:
    """Log the snapshot's freshness — **ERROR** when stale — and return it.

    ERROR (not warning) is deliberate, but note what actually happens: a
    stale-but-present snapshot is NOT dropped to naive — the CL engine prices
    every resolvable club verbatim off those aged ratings (only a *missing* or
    unparseable file leaves the club list empty and forces the naive path). So
    the real degradation is "CL priced off N-day-old ratings", which grows worse
    the older the file gets; that is what the operator needs to hear. There is
    deliberately no hard staleness cutoff: a month-old API-scale file is a
    smaller error than a fresh but mis-scaled one. Routing ERROR to the operator
    is the notifier's job, not this module's.
    """
    st = snapshot_status(path, stale_after_days=stale_after_days)
    if st.stale:
        log.error("clubelo_snapshot_stale", **st.as_log_fields())
    else:
        log.debug("clubelo_snapshot_fresh", **st.as_log_fields())
    return st


def _validate(text: str) -> str | None:
    """Return a rejection reason, or None when the payload is a real snapshot.

    Deliberately tolerant of a *stray* odd row (ClubElo's tail is full of tiny
    clubs and the band below is set from 17.6k observed rows, min 666 / max
    2085 — but a new minnow should not cost us the whole snapshot) and strict
    about a *systematically* wrong body: bad header, wrong columns, or a large
    share of unusable rows means we did not get ratings and must not clobber a
    good file with them.
    """
    if not text.startswith("Rank,"):
        return "bad_header"
    lines = text.splitlines()
    if not lines or lines[0].strip() != EXPECTED_HEADER:
        return "unexpected_columns"

    good = 0
    bad = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 7 or not parts[1].strip():
            bad += 1
            continue
        try:
            elo = float(parts[4])
        except ValueError:
            bad += 1
            continue
        if MIN_ELO <= elo <= MAX_ELO:
            good += 1
        else:
            bad += 1

    if good < MIN_ROWS:
        return f"too_few_rows:{good}"
    if bad > max(BAD_ROW_FLOOR, good * BAD_ROW_FRACTION):
        return f"too_many_bad_rows:{bad}/{good + bad}"
    return None


def _write_atomic(dest: Path, text: str) -> None:
    """Write via a sibling temp file + ``os.replace`` so ``dest`` is never partial."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


# ============================================================================
# HTML scrape fallback (clubelo.com website)
# ----------------------------------------------------------------------------
# The machine-readable CSV API (api.clubelo.com) was deactivated upstream
# (``/Fixtures`` -> "Fixtures API deactivated"; dated endpoints -> 502). The
# public website https://clubelo.com/ still serves ratings, fresh, as HTML, but
# on a DIFFERENT numeric scale from the API CSV the CL engine was
# tuned on (see SCRAPE_MONITOR_NAME). So this is NOT an engine fallback: it does
# not feed clubelo_latest.csv and never lets the engine price off site numbers.
# It fetches the page once per refresh, parses the ranking table, and writes a
# labelled, non-authoritative MONITORING snapshot (SCRAPE_MONITOR_NAME) for
# coverage tracking and a future re-tune corpus. The engine keeps pricing off
# the last real API snapshot, and the staleness alarm on clubelo_latest.csv is
# deliberately left to fire — a fresh-but-mis-scaled file must never silence it.
#
# Two scrape-specific realities the API did not have:
#  * The website ranking is now WORLDWIDE, while the API CSV was Europe-only.
#    We filter to the country set of the previous (last-known-good) snapshot so
#    the scope, and the Ecuador-"Barcelona" / Uruguay-"Liverpool" name
#    collisions that come with non-European clubs, are excluded.
#  * Website display names differ from the API's short names ("Internazionale"
#    vs "Inter", "Bayern München" vs "Bayern"). Rather than invent a second
#    matcher, we CANONICALISE every scraped club back to the previous snapshot's
#    name via the SAME TeamAliasResolver the CL engine resolves fixtures with,
#    so the emitted Club column is identical to what the API produced. Clubs in
#    the previous snapshot we cannot refresh, and new top-flight clubs we cannot
#    canonicalise, are surfaced in the returned report — never silently dropped.

SCRAPE_URL = "https://clubelo.com/"
#: A stale free site behind Cloudflare-style filters 403s a default urllib UA;
#: the codebase already needs a realistic browser UA for Highlightly.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

#: Sibling filename the scrape writes to. The scrape is NOT the engine's tuned
#: input — the clubelo.com website's table Elo is on a DIFFERENT scale from the
#: ``api.clubelo.com`` CSV the CL engine was tuned on. CAUSE UNKNOWN, and it is
#: NOT an "Elo +/- Golo" composite (an earlier guess since disproved: the table
#: cell equals the site's own "Elo" field to within rounding; the "+/-" column
#: is the since-yesterday delta, and Golo is a separate, uncorrelated metric).
#: The residuals cluster by COUNTRY (e.g. ENG +49..+74, RUS -97, TUR -79),
#: consistent with a re-calibrated site model rather than a fixed rescale.
#: Feeding raw site ratings to the API-tuned engine would silently mis-price
#: every tie, and the best bridge measured (a global slope + per-country offset)
#: still leaves ~20.9 API pts/club residual and ~29.5 pts on the match
#: difference d — comparable to the ~60-pt rating *differences* that decide
#: matches. So the scrape does NOT overwrite ``clubelo_latest.csv``; it lands
#: here as a labelled, non-authoritative monitoring/coverage snapshot (site
#: scale) while the engine keeps pricing off the last real API snapshot, and the
#: staleness alarm is deliberately left to fire — a fresh-but-mis-scaled file
#: must never silence it.
#:
#: This is NOT an argument for permanent inaction: a pinned API-scale file only
#: beats the best bridge while it is young. Measured drift of the pinned file:
#: 15.4 pts/club @30d, 22.7 @60d, 27.0 @90d, 29.0 @120d (21.8/32.0/38.3/40.7 on
#: d). It crosses the best bridge (~20.9/club, 29.5 on d) at ~90-100 days
#: in-season — roughly EARLY DECEMBER 2026 for a 2026-08-31 pin; past that the
#: bridged site feed is the smaller error and the re-tune should ship. For
#: whoever builds it: the bridge is global slope + per-country offset (Golo does
#: NOT help), and Approach B (Glicko) is out on coverage —
#: scripts/fetch_club_results.py seeds only PL/PD/BL1/SA/FL1, so Glicko has NO
#: rating at all for Sporting, PSV, Galatasaray, Bodø/Glimt or Shakhtar.
SCRAPE_MONITOR_NAME = "clubelo_scrape_latest.csv"

#: clubelo.com's website and the api.clubelo.com CSV disagree on the country
#: code for nine UEFA nations (the site uses the modern ISO-ish code, the API
#: the older one). Left unmapped, every club in these nations — including recent
#: CL qualifiers Slovan Bratislava (SVK) and FCSB (ROU) — silently fails the
#: scope/country-agreement test and is dropped. Canonicalise both sides through
#: this map (API code -> site code) before any country comparison.
COUNTRY_CODE_MAP = {
    "ROM": "ROU",  # Romania
    "SLK": "SVK",  # Slovakia
    "BHZ": "BIH",  # Bosnia and Herzegovina
    "MAC": "MKD",  # North Macedonia
    "MOL": "MDA",  # Moldova
    "LAT": "LVA",  # Latvia
    "LIT": "LTU",  # Lithuania
    "FAR": "FRO",  # Faroe Islands
    "MNT": "MNE",  # Montenegro
}


def _canon_country(code: str | None) -> str | None:
    """Fold a country code to its canonical form for cross-source comparison."""
    if not code:
        return None
    code = code.strip()
    return COUNTRY_CODE_MAP.get(code, code)

_RE_TR = re.compile(r"(?=<tr)")
_RE_FLAG = re.compile(r'<img alt="([A-Z]{3})"')
_RE_LEVEL = re.compile(r"Level (\d+) \(\d+ teams\)")
_RE_RANK = re.compile(r"<small>\s*(\d+)\s*</small>")
_RE_SLUG = re.compile(r'<a href="/([^"]+)"><span class="(?:NonAst|Ast)">')
_RE_NAME = re.compile(r'<span class="Ast">([^<]+)</span>')
_RE_RATING = re.compile(r'<td class="r">(-?\d+)</td>')
_RE_SNAPDATE = re.compile(r'<h1><a href="/(\d{4}-\d{2}-\d{2})/">')


@dataclass(frozen=True)
class ScrapedClub:
    rank: int | None
    name: str
    country: str | None
    level: int | None
    elo: int


def _parse_clubelo_html(html: str) -> tuple[list[ScrapedClub], date | None]:
    """Parse the clubelo.com ranking page into rows + the snapshot date.

    The ranking is a sequence of per-country ``<table class="ast">`` blocks.
    Every ranked club row carries its OWN country flag (``<img alt="XXX">``); the
    country is read per-row and NOT carried forward. Carrying it forward is
    unsafe: the page's "Calculation" section lists ~18 flagless rows (African /
    Oceanian clubs) that would otherwise inherit the previous European block's
    country and be mis-tagged — harmless while they are filtered out, dangerous
    the day that preceding block is European. A row with no flag gets
    ``country=None`` and is dropped downstream by the scope filter. The Level
    header IS a section header (``<i>Level N (M teams)</i>``, no club cell) and
    legitimately applies to the rows beneath it, so that one carries forward.

    Each row has a global rank, an optional club link (top clubs are linked,
    deep lower-division clubs are plain ``<span class="Ast">`` text) and an
    integer Elo in ``<td class="r">``. The eloData JS widget at the top has no
    ``<span class="Ast">`` cell, so requiring that span cleanly excludes it.

    The snapshot date is the ``/YYYY-MM-DD/`` in the page ``<h1>`` — the date
    the ratings are effective, which is what the staleness check needs.
    """
    md = _RE_SNAPDATE.search(html)
    snap_date: date | None = None
    if md:
        try:
            snap_date = date.fromisoformat(md.group(1))
        except ValueError:
            snap_date = None

    rows: list[ScrapedClub] = []
    level: int | None = None
    for tr in _RE_TR.split(html):
        ml = _RE_LEVEL.search(tr)
        if ml:
            level = int(ml.group(1))
        # Country is per-row: read this row's own flag, else None (see docstring).
        fl = _RE_FLAG.search(tr)
        country = fl.group(1) if fl else None
        nm = _RE_NAME.search(tr)
        rt = _RE_RATING.search(tr)
        if not (nm and rt):
            continue
        rk = _RE_RANK.search(tr)
        name = nm.group(1).strip().replace(",", " ")
        try:
            elo = int(rt.group(1))
        except ValueError:
            continue
        rows.append(
            ScrapedClub(
                rank=int(rk.group(1)) if rk else None,
                name=name,
                country=country,
                level=level,
                elo=elo,
            )
        )
    return rows, snap_date


def _read_reference_clubs(path: Path) -> list[dict[str, str]]:
    """Rows of the pinned API-scale reference CSV as dicts (name + country + scope).

    This is the *reference* the scrape canonicalises against and takes its
    European scope from — it must be a real API snapshot, NEVER the scrape's own
    output. Reading the scrape output back would let a single transient miss
    delete a club permanently (it never returns to the reference, so it is never
    looked for again) and coverage would decay monotonically. Returns [] if
    unreadable/absent (cold start).
    """
    try:
        import csv as _csv

        with path.open(encoding="utf-8") as fh:
            return [dict(r) for r in _csv.DictReader(fh)]
    except (OSError, ValueError):
        return []


def _build_csv_from_scrape(
    scraped: list[ScrapedClub],
    reference: list[dict[str, str]],
    *,
    snap_date: date,
    resolver=None,
) -> tuple[str, dict[str, object]]:
    """Render a *site-scale* monitoring CSV from scraped rows, keyed to the
    reference snapshot's names.

    This is NOT the engine's tuned input (see ``SCRAPE_MONITOR_NAME``): the
    website's table Elo is on a different scale from the API CSV (cause unknown,
    country-dependent — not a Golo composite), so this file exists only for
    coverage monitoring and to accumulate a
    site-scale history for a possible future re-tune. It still canonicalises to
    the reference names so the two are directly comparable.

    Matching is deliberate, to avoid rating theft (a fuzzy match stealing a
    rating from an unrelated club) and silently dropping whole nations:

    * **Country agreement is required for every hit.** Codes are folded through
      :data:`COUNTRY_CODE_MAP` first, so the nine UEFA nations the two sources
      code differently (Slovakia, Romania, ...) still match.
    * **Two passes over the whole reference set:** exact/alias first (so an exact
      name can never be stolen by an earlier fuzzy match), fuzzy second. Every
      fuzzy hit is logged at INFO with its score. A cross-country fuzzy hit is
      impossible by construction, so such a club is simply left unrefreshed.

    Returns ``(csv_text, report)``; the report surfaces coverage:

    * ``unrefreshed`` — reference clubs with no in-country scraped match (kept
      OUT of the file);
    * ``new_top_flight`` — scraped Level-1 clubs not in the reference, emitted
      under their display name.
    """
    from collections import defaultdict

    from betbot.exchanges.matcher import DEFAULT_THRESHOLD, TeamAliasResolver, normalize

    try:
        from rapidfuzz import fuzz
    except Exception:  # noqa: BLE001 — score logging is best-effort
        fuzz = None

    if resolver is None:
        try:
            resolver = TeamAliasResolver.from_yaml("config/team_aliases.yaml")
        except (OSError, ValueError):
            resolver = TeamAliasResolver()

    # European scope = the reference's country set, folded to canonical codes so
    # SVK/ROU/... are recognised. Empty => cold start (keep every scraped row).
    euro = {_canon_country(r.get("Country")) for r in reference if r.get("Country")}
    euro.discard(None)
    pool = [c for c in scraped if (not euro) or (_canon_country(c.country) in euro)]

    # Log (name, country) duplicates with differing Elo — otherwise last-wins
    # would silently pick one (B5). Bucket scraped rows by canonical country;
    # first in page (rank) order wins within a country.
    scraped_by_cc: dict[str | None, dict[str, ScrapedClub]] = defaultdict(dict)
    dups: list[str] = []
    for c in pool:
        cc = _canon_country(c.country)
        bucket = scraped_by_cc[cc]
        if c.name in bucket and bucket[c.name].elo != c.elo:
            dups.append(f"{c.name}/{c.country}:{bucket[c.name].elo}!={c.elo}")
            continue
        bucket.setdefault(c.name, c)
    if dups:
        log.info("clubelo_scrape_duplicate_names", count=len(dups), sample=dups[:20])

    header = "Rank,Club,Country,Level,Elo,From,To"
    frm = snap_date.isoformat()
    lines: list[str] = [header]
    claimed: set[tuple[str | None, str]] = set()
    unrefreshed: list[str] = []
    fuzzy_hits: list[str] = []

    def _emit(prow: dict[str, str], sc: ScrapedClub) -> None:
        club = (prow.get("Club") or "").strip()
        rank = sc.rank if sc.rank is not None else (prow.get("Rank") or "0")
        country = (prow.get("Country") or sc.country or "").strip()
        level = (prow.get("Level") or (str(sc.level) if sc.level else "1")).strip()
        lines.append(f"{rank},{club},{country},{level},{sc.elo},{frm},{frm}")

    if reference and pool:
        pending = [
            p for p in reference if (p.get("Club") or "").strip()
        ]
        # Pass 1: exact / alias, in-country only (threshold above 100 => the
        # fuzzy stage can never fire, so only an exact/alias hit returns).
        # Pass 2: fuzzy, in-country only, over rows still unmatched.
        still: list[dict[str, str]] = []
        for fuzzy in (False, True):
            src = pending if not fuzzy else still
            still = []
            for prow in src:
                club = (prow.get("Club") or "").strip()
                cc = _canon_country(prow.get("Country"))
                cand = scraped_by_cc.get(cc, {})
                names = [n for n in cand if (cc, n) not in claimed]
                if not names:
                    still.append(prow)
                    continue
                hit = resolver.match(
                    club, names, threshold=DEFAULT_THRESHOLD if fuzzy else 100.5
                )
                if hit is None:
                    still.append(prow)
                    continue
                claimed.add((cc, hit))
                sc = cand[hit]
                if fuzzy:
                    score = (
                        fuzz.token_set_ratio(normalize(club), normalize(hit))
                        if fuzz is not None
                        else -1.0
                    )
                    fuzzy_hits.append(f"{club}->{hit} ({cc}) score={score:.0f}")
                _emit(prow, sc)
        unrefreshed = [
            (p.get("Club") or "").strip() for p in still if (p.get("Club") or "").strip()
        ]
    else:
        # Cold start: emit scraped rows directly under display names.
        for cc, bucket in scraped_by_cc.items():
            for sc in bucket.values():
                claimed.add((cc, sc.name))
                lines.append(
                    f"{sc.rank or 0},{sc.name},{sc.country or ''},"
                    f"{sc.level or 1},{sc.elo},{frm},{frm}"
                )

    # New top-flight clubs the reference did not have.
    new_top: list[str] = []
    if reference:
        for cc, bucket in scraped_by_cc.items():
            for sc in bucket.values():
                if (cc, sc.name) in claimed or sc.level != 1:
                    continue
                new_top.append(sc.name)
                country = (sc.country or "").strip()
                lines.append(
                    f"{sc.rank or 0},{sc.name},{country},"
                    f"{sc.level or 1},{sc.elo},{frm},{frm}"
                )

    if fuzzy_hits:
        log.info(
            "clubelo_scrape_fuzzy_hits", count=len(fuzzy_hits), hits=fuzzy_hits[:40]
        )

    report: dict[str, object] = {
        "scraped_total": len(scraped),
        "in_scope": len(pool),
        "emitted": len(lines) - 1,
        "unrefreshed_count": len(unrefreshed),
        "unrefreshed": unrefreshed[:40],
        "fuzzy_hit_count": len(fuzzy_hits),
        "duplicate_count": len(dups),
        "new_top_flight_count": len(new_top),
        "new_top_flight": new_top[:40],
        "scale": "site",  # NOT the API scale the CL engine is tuned on
    }
    return "\n".join(lines) + "\n", report


def scrape_latest(
    dest: Path,
    *,
    reference: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep=time.sleep,
    resolver=None,
    html: str | None = None,
) -> bool:
    """Scrape clubelo.com into ``dest`` as a SITE-SCALE monitoring snapshot.

    ``dest`` is the monitoring file (:data:`SCRAPE_MONITOR_NAME`), NOT the CL
    engine's tuned ``clubelo_latest.csv``: the website's table Elo is on a
    different scale (cause unknown, country-dependent — not a Golo composite),
    and writing it into the engine's input would
    silently mis-price every tie AND — because a scrape stamps today's ``From``
    date — quietly silence the staleness alarm on a mis-scaled file. So the
    engine keeps reading the last real API snapshot; this file is only for
    coverage monitoring and a future re-tune. A dated copy of each successful
    scrape is also written under ``data/clubelo_site/YYYY-MM-DD.csv`` so a
    site-scale history actually accrues (``dest`` itself is overwritten daily).

    ``reference`` is the pinned API snapshot the scrape canonicalises names and
    takes its European scope from (defaults to ``clubelo_latest.csv`` beside
    ``dest``). It is read, never written — see :func:`_read_reference_clubs`.

    Fetches the ranking page once (polite: realistic UA, short timeout, bounded
    jittered-backoff retry, never hammered), parses it, validates the rebuilt
    CSV with the same ``_validate`` gate the API path uses, and writes it
    atomically. Returns True only when a valid snapshot was written. ``html`` is
    injectable so tests parse a captured fixture instead of hitting the network.
    """
    dest = Path(dest)
    ref_path = Path(reference) if reference is not None else dest.with_name(
        "clubelo_latest.csv"
    )
    previous = _read_reference_clubs(ref_path)

    if html is None:
        attempts = max(1, retries)
        last_error = "unknown"
        for attempt in range(attempts):
            try:
                req = urllib.request.Request(SCRAPE_URL, headers={"User-Agent": BROWSER_UA})
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                    html = resp.read().decode("utf-8", errors="replace")
                break
            except Exception as e:  # noqa: BLE001 — fallback must never crash the caller
                last_error = f"{e.__class__.__name__}: {e}"
                if attempt + 1 < attempts:
                    delay = min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt)) * (
                        1.0 + random.random() * 0.25
                    )
                    log.warning(
                        "clubelo_scrape_attempt_failed",
                        attempt=attempt + 1, attempts=attempts,
                        error=last_error, retry_in_s=round(delay, 1),
                    )
                    sleep(delay)
                    continue
        if html is None:
            log.error("clubelo_scrape_fetch_failed", error=last_error, url=SCRAPE_URL)
            return False

    scraped, snap_date = _parse_clubelo_html(html)
    if not scraped or snap_date is None:
        log.error(
            "clubelo_scrape_unparseable",
            rows=len(scraped), have_date=snap_date is not None,
        )
        return False

    csv_text, report = _build_csv_from_scrape(
        scraped, previous, snap_date=snap_date, resolver=resolver
    )
    reason = _validate(csv_text)
    if reason is not None:
        log.error("clubelo_scrape_bad_payload", reason=reason, **report)
        return False

    _write_atomic(dest, csv_text)
    # Dated site-scale archive. ``dest`` is overwritten every run, so on its own
    # NO history accrues (the old docstring claim was false). Also drop an
    # immutable dated copy under ``data/clubelo_site/`` (~33 KB/day) so a genuine
    # site-scale corpus builds up over time for a future re-tune. Best-effort:
    # a failure here must never fail the scrape or reset any staleness clock.
    archived: str | None = None
    try:
        archive_dir = dest.parent / "clubelo_site"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{snap_date.isoformat()}.csv"
        _write_atomic(archive_path, csv_text)
        archived = str(archive_path)
    except OSError as e:  # archiving must never crash the caller
        log.warning(
            "clubelo_scrape_archive_failed", error=f"{e.__class__.__name__}: {e}"
        )
    log.info(
        "clubelo_scrape_monitor_written",
        source="scrape", dest=str(dest), archived=archived,
        snapshot=snap_date.isoformat(), clubs=report["emitted"],
        note="site_scale_monitoring_only_not_engine_input", **report,
    )
    if report["unrefreshed_count"]:
        log.warning(
            "clubelo_scrape_coverage_gap",
            unrefreshed_count=report["unrefreshed_count"],
            unrefreshed=report["unrefreshed"],
        )
    return True



def _scrape_monitor(engine_dest: Path, *, timeout: int, retries: int, sleep) -> None:
    """Best-effort: refresh the site-scale monitoring file beside ``engine_dest``.

    Runs when the API path has failed. Deliberately does NOT return whether it
    succeeded and NEVER touches ``engine_dest``: the site is a different rating
    scale, so it cannot serve as the CL engine's tuned input, and it must not
    reset the staleness clock on the engine snapshot. It exists only so the
    operator can see live site coverage and so a site-scale history accrues (as
    dated per-run copies under ``data/clubelo_site/``, written by
    :func:`scrape_latest`) for a possible future re-tune. ``engine_dest`` is
    passed only as the reference
    (names + scope) — it is read, never written.
    """
    monitor = engine_dest.with_name(SCRAPE_MONITOR_NAME)
    try:
        ok = scrape_latest(
            monitor, reference=engine_dest, timeout=timeout, retries=retries, sleep=sleep
        )
    except Exception as e:  # noqa: BLE001 — monitoring must never crash the caller
        log.warning("clubelo_scrape_monitor_failed", error=f"{e.__class__.__name__}: {e}")
        return
    if not ok:
        log.warning("clubelo_scrape_monitor_unavailable", monitor=str(monitor))


def refresh_latest(
    dest: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep=time.sleep,
    stale_after_days: int = STALE_AFTER_DAYS,
    snapshot_date: date | None = None,
    scrape_fallback: bool = True,
) -> bool:
    """Fetch a real API-scale ClubElo snapshot to ``dest`` (today's unless overridden).

    Retries transient network failures with jittered exponential backoff. When
    every attempt fails — or the API returns a bad/deactivated payload — the
    existing snapshot is left UNTOUCHED, its age is checked (a stale one logs at
    ERROR so the degradation is visible), and this returns ``False``.

    The clubelo.com website is NOT used to refresh ``dest``. Its table Elo is on
    a different scale from this API CSV (cause unknown, country-dependent — not a
    Golo composite), so substituting it
    would silently mis-price the API-tuned CL engine and — by stamping today's
    date — hide the fact that the real feed is down. Instead, on an API failure
    a separate site-scale *monitoring* file is refreshed beside ``dest`` (see
    :func:`_scrape_monitor`) and the engine keeps pricing off the last real API
    snapshot; the staleness alarm is intentionally allowed to fire.

    ``snapshot_date`` pins a historical snapshot (the backtest cache under
    ``data/clubelo/``). Those files are old *by design*, so the staleness check
    (and the monitoring scrape) is skipped for them — this only applies to the
    live ``clubelo_latest.csv``.
    """
    dest = Path(dest)
    historical = snapshot_date is not None
    d = (snapshot_date or date.today()).isoformat()
    url = CLUBELO_URL.format(d=d)
    attempts = max(1, retries)
    last_error = "unknown"

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read()
        except Exception as e:  # noqa: BLE001 — a failed refresh must never crash the caller
            last_error = f"{e.__class__.__name__}: {e}"
            if attempt + 1 < attempts:
                delay = min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt)) * (
                    1.0 + random.random() * 0.25
                )
                log.warning(
                    "clubelo_refresh_attempt_failed",
                    attempt=attempt + 1, attempts=attempts,
                    error=last_error, retry_in_s=round(delay, 1),
                )
                sleep(delay)
                continue
            break

        text = raw.decode("utf-8", errors="replace")
        reason = _validate(text)
        if reason is not None:
            # A bad payload is the server's answer, not a transport blip:
            # retrying will not change it, and we must not overwrite a good
            # snapshot with it.
            log.error("clubelo_refresh_bad_payload", reason=reason, head=text[:60], snapshot=d)
            if not historical:
                if scrape_fallback:
                    _scrape_monitor(dest, timeout=timeout, retries=retries, sleep=sleep)
                check_snapshot_freshness(dest, stale_after_days=stale_after_days)
            return False

        _write_atomic(dest, text)
        clubs = max(text.count("\n") - 1, 0)
        log.info(
            "clubelo_refreshed",
            clubs=clubs, dest=str(dest), snapshot=d, attempts=attempt + 1,
        )
        return True

    log.error(
        "clubelo_refresh_failed",
        error=last_error, attempts=attempts, timeout_s=timeout, url=url,
    )
    if not historical:
        if scrape_fallback:
            _scrape_monitor(dest, timeout=timeout, retries=retries, sleep=sleep)
        check_snapshot_freshness(dest, stale_after_days=stale_after_days)
    return False
