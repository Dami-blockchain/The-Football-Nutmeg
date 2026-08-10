"""Lineup-adjusted rating shift (PURE — no network, no DB).

Given a team's confirmed starting XI and its season player-minutes, compute a
**negative** Glicko-point adjustment proportional to how much of the team's
expected first XI is absent (benched / injured / rotated out). Full expected XI
present -> 0.0; entire expected XI missing -> ``-max_penalty``.

Name matching caveat
--------------------
api-football's ``startXI`` player names and its ``/players`` (minutes) names are
BOTH full names ("Kevin De Bruyne"), so a straight ``matcher.normalize`` on both
sides matches them exactly in the common case. To be robust to the occasional
abbreviated startXI surname ("De Bruyne" vs "Kevin De Bruyne"), we ALSO index
each expected regular by its normalized *last token* (surname) and treat a
surname hit in the confirmed XI as "present". This is a deliberate,
last-name-only fallback: two players sharing a surname in the same XI could
collide, but that is rare and only ever makes a team look MORE complete (never
falsely penalised). See ``tests/test_lineup.py`` for the pinned behaviour.

The network side (fetching lineups / minutes / injuries) lives in
``betbot.data.lineup_service``; this module is only the math so it stays
trivially unit-testable.
"""

from __future__ import annotations

from betbot.exchanges.matcher import normalize


def _last_token(norm_name: str) -> str:
    """Last whitespace token of an already-normalized name ('' if empty)."""
    parts = norm_name.split()
    return parts[-1] if parts else ""


def lineup_rating_adjustment(
    confirmed_xi: set[str],
    player_minutes: dict[str, int],
    *,
    max_penalty: float,
    top_n: int = 11,
) -> float:
    """Negative Glicko-point shift for a team's confirmed XI vs its regulars.

    * Take the team's ``top_n`` highest-minutes players (its "expected
      regulars") from ``player_minutes``; weight each by its share of those
      players' total minutes.
    * For every expected regular whose (normalized) name is NOT found in the
      confirmed XI, add its weight -> ``absent_fraction`` in ``[0, 1]``.
    * Return ``-absent_fraction * max_penalty``, clamped to ``[-max_penalty, 0]``.

    No-data safety: empty ``confirmed_xi`` (lineup not out yet) OR empty
    ``player_minutes`` -> ``0.0`` (the caller then uses the baseline prediction).
    """
    if not confirmed_xi or not player_minutes:
        return 0.0

    # Top-N expected regulars by minutes (positive minutes only — a 0-minute
    # player is not an "expected regular").
    regulars = sorted(
        ((name, mins) for name, mins in player_minutes.items() if mins > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_n]
    total = sum(mins for _, mins in regulars)
    if total <= 0:
        return 0.0

    # Normalized confirmed XI: full form + surname index for the abbreviated-
    # startXI fallback documented in the module docstring.
    confirmed_norm: set[str] = set()
    confirmed_surnames: set[str] = set()
    for raw in confirmed_xi:
        n = normalize(raw)
        if n:
            confirmed_norm.add(n)
            confirmed_surnames.add(_last_token(n))

    absent_fraction = 0.0
    for name, mins in regulars:
        n = normalize(name)
        present = n in confirmed_norm or (
            _last_token(n) != "" and _last_token(n) in confirmed_surnames
        )
        if not present:
            absent_fraction += mins / total

    adj = -absent_fraction * max_penalty
    # Clamp defensively (float summation can drift a hair past 1.0).
    if adj < -max_penalty:
        adj = -max_penalty
    if adj > 0.0:
        adj = 0.0
    return adj
