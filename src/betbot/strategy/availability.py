"""Player-availability (injury) -> Glicko-rating adjustment. PURE, no IO.

v1 is a deliberately COARSE signal: it counts how many players are unavailable
(injured / suspended) for a team and shifts that team's Glicko rating DOWN by a
small, capped amount. It does **not** weight by player importance — losing your
star striker and losing your fourth-choice full-back count the same here. That
is the honest limitation of v1.

Magnitudes are conservative on purpose. Glicko points are small (SCALE =
173.7178; a ~170-point gap is a ~1-sigma strength difference), so the defaults
(8 pts/player, capped at 40) move a 3-way probability by only a few points even
for a heavily depleted side — directionally right, not a sledgehammer. The
feature is OFF by default until validated on the calibration report.

v2 ideas (out of scope here): weight each absence by the player's market value
/ recent minutes; consume the confirmed starting XI / late team-sheet rather
than season injury lists; learn per-position weights from settled outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AvailabilityAdjustment:
    """Result of an availability assessment for one team.

    ``rating_delta`` is the NEGATIVE Glicko-point shift to add to the team's
    rating before computing probabilities (0.0 or below).
    """

    n_unavailable: int
    rating_delta: float


def availability_penalty(
    n_unavailable: int, *, per_player: float, cap: float
) -> float:
    """A NEGATIVE Glicko-rating adjustment for ``n_unavailable`` absent players.

    Returns ``-min(cap, n_unavailable * per_player)``:

    * ``0`` unavailable -> ``0.0`` (no adjustment, exact current behaviour);
    * grows linearly at ``per_player`` Glicko points per absent player;
    * floored at ``-cap`` so even a wholesale absence list can't tank the
      rating into nonsense.

    This is a COARSE absence-count signal — it does NOT weight by player
    importance. Keep ``per_player`` / ``cap`` conservative.
    """
    n = max(0, int(n_unavailable))
    per = max(0.0, float(per_player))
    cap_abs = abs(float(cap))
    return -min(cap_abs, n * per)


def assess(
    n_unavailable: int, *, per_player: float, cap: float
) -> AvailabilityAdjustment:
    """Convenience wrapper returning a small result dataclass."""
    return AvailabilityAdjustment(
        n_unavailable=max(0, int(n_unavailable)),
        rating_delta=availability_penalty(
            n_unavailable, per_player=per_player, cap=cap
        ),
    )
