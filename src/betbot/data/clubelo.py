"""ClubElo snapshot refresh — the live cross-league Elo source for the CL engine.

``EuropeanStrategyEngine`` reads ``data/clubelo_latest.csv``; this keeps that
file fresh. Used by the daemon's daily tick (so a CL fixture is always priced
off today's ratings) and by ``scripts/fetch_clubelo.py --latest``. Network
failures are non-fatal: the caller logs and carries on with the last snapshot
(and the engine falls back to naive for any club it can't resolve).
"""

from __future__ import annotations

import urllib.request
from datetime import date
from pathlib import Path

from betbot.logging import get_logger

log = get_logger(__name__)

CLUBELO_URL = "http://api.clubelo.com/{d}"


def refresh_latest(dest: Path, *, timeout: int = 30) -> bool:
    """Fetch today's ClubElo snapshot to ``dest``. Returns True on success."""
    d = date.today().isoformat()
    try:
        raw = urllib.request.urlopen(CLUBELO_URL.format(d=d), timeout=timeout).read()  # noqa: S310
    except Exception as e:  # noqa: BLE001 — a failed refresh must never crash the caller
        log.warning("clubelo_refresh_failed", error=str(e))
        return False
    text = raw.decode("utf-8", errors="replace")
    if not text.startswith("Rank,"):
        log.warning("clubelo_refresh_bad_payload", head=text[:40])
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    clubs = max(text.count("\n") - 1, 0)
    log.info("clubelo_refreshed", clubs=clubs, dest=str(dest), snapshot=d)
    return True
