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

    ERROR (not warning) is deliberate: a stale ClubElo file silently degrades
    every Champions League prediction to the naive form engine, which is the
    single largest accuracy regression the system can suffer without crashing.
    Routing ERROR to the operator is the notifier's job, not this module's.
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


def refresh_latest(
    dest: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep=time.sleep,
    stale_after_days: int = STALE_AFTER_DAYS,
) -> bool:
    """Fetch today's ClubElo snapshot to ``dest``. Returns True on success.

    Retries transient network failures with jittered exponential backoff. When
    every attempt fails the existing snapshot is left untouched and its age is
    checked: a stale one logs at ERROR so the degradation is visible.
    """
    dest = Path(dest)
    d = date.today().isoformat()
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
    check_snapshot_freshness(dest, stale_after_days=stale_after_days)
    return False
