"""Pre-registered confidence filter on the BET / NO BET call.

The model always produces a most-likely outcome (the argmax of H/D/A). That is
a PREDICTION. Whether we *call* it — i.e. put it forward as a bet — is a
separate, deliberately conservative decision made here.

Two pre-registered rules, both fixed BEFORE looking at any live result:

1. **Confidence threshold.** Call only when the favourite's probability clears
   ``club_confidence_threshold`` (default 0.60). Measured on
   ``data/club_results.csv`` (n=10,734, Pinnacle CLOSING odds) the favourite's
   hit rate by bucket was: p>=0.55 -> 68.1% (38% of fixtures), p>=0.60 -> 72.2%
   (27%), p>=0.65 -> 75.0% (19%). Those are CLOSING-odds numbers and therefore
   optimistic relative to the T-24h prices we would actually have live.

2. **Draw-aware abstention.** Never call when ``p_draw`` is within
   ``club_confidence_draw_margin`` (default 0.05) of the favourite. Draws are
   ~25.4% of outcomes and are effectively unpickable, so that category is
   abstained from outright. A draw is also never itself a callable pick.

   MEASURED CAVEAT (be honest about this): at the shipped 0.60 threshold this
   rule is a **no-op**. Because ``p_fav + p_draw <= 1``, a gap below ``m``
   requires ``p_fav < (1 + m) / 2`` — i.e. below 0.525 for m = 0.05 — so the
   rule cannot fire at any threshold at or above that. Replayed over the
   held-out 2025-26 season (n=1752) it blocked exactly 0 calls at every
   threshold from 0.40 to 0.60. It is kept as a safety belt that becomes live
   if the threshold is ever lowered, NOT as something currently doing work. Do
   not attribute any part of the called-pick hit rate to it.

HONESTY (non-negotiable, applies to every use of this module):
    The filtered ~70% figure is an **accuracy KPI**. It is NOT edge, NOT +EV,
    and NOT evidence of beating the market. Backing favourites at a fair market
    price is ~0 EV by construction — a high hit rate on short-priced selections
    is exactly what an efficient market produces. Nothing here may be
    presented, labelled, or phrased as +EV or as beating the closing line.

The filter reads whatever final blended probability triple it is handed. When
the market-anchoring work lands, the anchored probabilities are what the caller
will pass, and the threshold then applies to the anchored favourite with no
change needed here.

Pure functions — no DB, no network, no config import at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass

Probs = tuple[float, float, float]  # (p_home, p_draw, p_away)

_LABELS = ("HOME", "DRAW", "AWAY")


@dataclass(frozen=True)
class ConfidenceCall:
    """The outcome of the filter for one fixture.

    ``pick`` is always the model's argmax (unchanged — the prediction stands
    even when we decline to call it). ``called`` is True only when every
    pre-registered rule passes. ``reason`` is a short machine-stable code:
    ``"called"``, ``"disabled"``, ``"below_threshold"``, ``"draw_favourite"``,
    or ``"draw_too_close"``.
    """

    pick: str
    p_pick: float
    p_draw: float
    called: bool
    reason: str

    @property
    def is_no_bet(self) -> bool:
        return not self.called


def favourite(probs: Probs) -> tuple[str, float]:
    """``(pick, p)`` for the argmax of the triple. Ties resolve H > D > A."""
    idx = max(range(3), key=lambda i: probs[i])
    return _LABELS[idx], float(probs[idx])


def evaluate(
    probs: Probs,
    *,
    enabled: bool,
    threshold: float,
    draw_margin: float,
) -> ConfidenceCall:
    """Apply the pre-registered filter to one final probability triple.

    ``probs`` must be the FINAL blended (and, once anchoring lands, anchored)
    probabilities — whatever the user is shown.

    When ``enabled`` is False the filter is inert: ``called`` is False and
    ``reason`` is ``"disabled"``. That is the shipped default, so the live
    behaviour of the bot is unchanged until the flag is turned on deliberately.
    """
    pick, p_pick = favourite(probs)
    p_draw = float(probs[1])

    if not enabled:
        return ConfidenceCall(pick, p_pick, p_draw, False, "disabled")
    if pick == "DRAW":
        # The draw is never a callable pick (rule 2, degenerate case).
        return ConfidenceCall(pick, p_pick, p_draw, False, "draw_favourite")
    if p_pick < threshold:
        return ConfidenceCall(pick, p_pick, p_draw, False, "below_threshold")
    if p_pick - p_draw < draw_margin:
        return ConfidenceCall(pick, p_pick, p_draw, False, "draw_too_close")
    return ConfidenceCall(pick, p_pick, p_draw, True, "called")


def evaluate_settings(probs: Probs, settings) -> ConfidenceCall:
    """:func:`evaluate` wired to a :class:`betbot.config.Settings` instance."""
    return evaluate(
        probs,
        enabled=bool(settings.club_confidence_filter),
        threshold=float(settings.club_confidence_threshold),
        draw_margin=float(settings.club_confidence_draw_margin),
    )


# --- Reporting helpers -------------------------------------------------

def wilson_interval(hits: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. ``(lo, hi)``.

    Preferred over the normal approximation at the sample sizes we actually
    have (tens, not thousands) — it stays inside [0, 1] and does not collapse
    to a zero-width interval at 0% or 100%.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - half), min(1.0, centre + half))


def call_stats(records: list[tuple[Probs, str]], **kw) -> dict:
    """Split a list of ``(probs, actual_outcome)`` into the TWO metrics.

    Returns ``{"all": {...}, "called": {...}}`` where each block is
    ``{n, hits, hit_rate, ci_lo, ci_hi}``, plus ``call_rate`` on the called
    block. These two numbers measure different things and must never be merged
    or reported as one figure:

    * **all** — 3-way accuracy over every fixture (model skill; the market
      closing line sits at roughly 53-54% on our data, which is the ceiling a
      sane all-match number lives under);
    * **called** — hit rate on the subset the filter actually calls (a
      selection KPI on short-priced favourites; NOT edge).
    """
    all_n = all_hits = 0
    call_n = call_hits = 0
    for probs, actual in records:
        pick, _ = favourite(probs)
        all_n += 1
        all_hits += int(pick == actual)
        decision = evaluate(probs, **kw)
        if decision.called:
            call_n += 1
            call_hits += int(decision.pick == actual)
    a_lo, a_hi = wilson_interval(all_hits, all_n)
    c_lo, c_hi = wilson_interval(call_hits, call_n)
    return {
        "all": {
            "n": all_n,
            "hits": all_hits,
            "hit_rate": (all_hits / all_n) if all_n else 0.0,
            "ci_lo": a_lo,
            "ci_hi": a_hi,
        },
        "called": {
            "n": call_n,
            "hits": call_hits,
            "hit_rate": (call_hits / call_n) if call_n else 0.0,
            "ci_lo": c_lo,
            "ci_hi": c_hi,
            "call_rate": (call_n / all_n) if all_n else 0.0,
        },
    }
