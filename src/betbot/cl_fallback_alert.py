"""Operator alert for the CL self-built-Elo fallback (engage / disengage).

Separated from the engine so the transition logic is pure and unit-testable and
the IO (state file + Telegram) is a thin, fully failure-isolated wrapper.

Contract (FALLBACK_SCOPE.md §7 / §7a):

* Fire ONCE per staleness EPISODE (keyed on the stale scrape's ``From`` date)
  so a multi-day outage does not spam; send a single "disengaged" notice when
  a fresh scrape returns.
* MUST ``await notify_operator`` from the async scheduler job — the sync
  wrapper refuses inside a running loop (notify.py) and would silently no-op.
* Observability, never the prediction: the whole thing is wrapped so a state-
  file error or a Telegram outage can NEVER break a scoring run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from betbot.logging import get_logger
from betbot.notify import notify_operator

log = get_logger(__name__)

#: Default on-disk episode marker. A tiny JSON file (NOT a DB table — the scope
#: keeps the only net-new table to ``cl_rating_shadow``) that remembers the
#: current fallback episode across scoring runs so a disengage transition can
#: be detected.
DEFAULT_STATE_PATH = Path("data/cl_fallback_state.json")


@dataclass(frozen=True)
class FallbackEpisode:
    engaged: bool = False
    source: str = "site_scrape"
    snapshot_from: str | None = None   # site scrape ``From`` iso — the episode key


def load_state(path: Path) -> FallbackEpisode:
    """Read the last-seen episode. Any error -> a clean disengaged state."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return FallbackEpisode(
            engaged=bool(d.get("engaged", False)),
            source=str(d.get("source", "site_scrape")),
            snapshot_from=d.get("snapshot_from"),
        )
    except (OSError, ValueError, TypeError):
        return FallbackEpisode()


def save_state(path: Path, episode: FallbackEpisode) -> None:
    """Persist the current episode. Best-effort; never raises to the caller."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({
            "engaged": episode.engaged,
            "source": episode.source,
            "snapshot_from": episode.snapshot_from,
        }),
        encoding="utf-8",
    )


def decide_alert(
    prev: FallbackEpisode,
    cur: FallbackEpisode,
    *,
    site_age_days: int | None,
) -> tuple[str, str] | None:
    """Pure transition -> ``(dedupe_key, message)`` to send, or ``None``.

    * ENGAGE when we are now engaged and were not (or the episode key — the
      stale scrape's ``From`` — changed, i.e. a NEW outage).
    * DISENGAGE when we were engaged and now are not.
    * No message while steady (engaged across the same episode, or healthy).
    """
    age = "unknown" if site_age_days is None else f"{site_age_days}d"
    if cur.engaged and (
        not prev.engaged or prev.snapshot_from != cur.snapshot_from
    ):
        key = f"cl_fallback_{cur.snapshot_from}"
        if cur.source == "naive":
            body = (
                f"CL fallback ENGAGED: ClubElo site scrape is {age} stale "
                f"(From {cur.snapshot_from}) AND the self-built snapshot is also "
                f"stale/unusable. CL is now on the NAIVE form engine (~50.7% vs "
                f"~58.7% held-out). Operator action needed."
            )
        else:
            body = (
                f"CL fallback ENGAGED: ClubElo site scrape is {age} stale "
                f"(From {cur.snapshot_from}). CL is now priced off the "
                f"{cur.source} ratings. This is a DEGRADED fallback — the "
                f"self-built ratings disagree with real ClubElo on the "
                f"favourite for ~1 in 5 CL fixtures. Refresh the scrape."
            )
        return key, body
    if prev.engaged and not cur.engaged:
        key = f"cl_fallback_clear_{prev.snapshot_from}"
        body = (
            "CL fallback DISENGAGED: a fresh ClubElo site scrape has returned. "
            "The CL engine is back on the live site ratings."
        )
        return key, body
    return None


async def maybe_alert_cl_fallback(
    settings,
    cl_engine,
    *,
    state_path: Path | None = None,
    notify=notify_operator,
    now: float | None = None,
) -> bool:
    """Fire the engage/disengage alert if the fallback state changed.

    Fully failure-isolated: reads the engine's read-only seams, compares to the
    persisted episode, awaits ``notify_operator`` (kind ``cl_fallback``,
    per-episode dedupe) and persists the new state. Returns True iff a message
    was sent. ANY error is logged and swallowed — this must never break a run.
    """
    try:
        path = state_path or DEFAULT_STATE_PATH
        engaged = bool(getattr(cl_engine, "fallback_engaged", False))
        source = str(getattr(cl_engine, "fallback_source", "site_scrape"))
        snap_date = getattr(cl_engine, "_snapshot_date", None)
        snap_from = None if snap_date is None else str(snap_date)
        age = getattr(cl_engine, "snapshot_age_days", None)

        cur = FallbackEpisode(engaged=engaged, source=source, snapshot_from=snap_from)
        prev = load_state(path)
        decision = decide_alert(prev, cur, site_age_days=age)
        if decision is None:
            # Keep the marker current even with no message (e.g. source drift
            # inside the same episode) so a later disengage still resolves.
            if (prev.engaged, prev.snapshot_from) != (cur.engaged, cur.snapshot_from):
                save_state(path, cur)
            return False
        key, body = decision
        sent = await notify(
            settings, body, kind="cl_fallback", dedupe_key=key, now=now,
        )
        save_state(path, cur)
        return bool(sent)
    except Exception as e:  # noqa: BLE001 — alert must never break a scoring run
        log.warning("cl_fallback_alert_failed", error=str(e))
        return False
