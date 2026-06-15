"""Dispersion challenger (flag-gated).

A single coefficient ``kappa`` that *sharpens the home/away decisive split* of a
finished 1X2 probability triple, leaving the **draw probability untouched**. It
exists to correct the model's measured under-dispersion: against a sharp market,
the ensemble systematically over-rates weak sides (e.g. Tunisia 53% to advance
vs market 8%, Iran 40% to beat Belgium vs market 12%). Widening the
favourite-vs-underdog gap pulls those minnow probabilities down.

Mechanism — operate on ``r = p_home / (p_home + p_away)``, the share of the
*decisive* (non-draw) mass that goes to the home side, in logit space:

    r' = sigmoid(kappa * logit(r))

``kappa = 1`` is the EXACT identity (no-op). ``kappa > 1`` pushes ``r`` toward
0/1 — favourites get a larger share, underdogs a smaller one. The draw mass
``p_draw`` is preserved exactly, so this never disturbs the (separately
calibrated) draw model. Pure math, no DB, no network.
"""

from __future__ import annotations

import math

_EPS = 1e-9


def apply_dispersion(
    probs: tuple[float, float, float], kappa: float
) -> tuple[float, float, float]:
    """Sharpen the home/away split of ``(p_home, p_draw, p_away)`` by ``kappa``.

    ``kappa == 1.0`` (or a degenerate input) returns the triple unchanged. The
    draw probability is always preserved; only the home/away split is rescaled,
    then the result is renormalised to keep the original draw mass.
    """
    p_home, p_draw, p_away = probs
    if kappa == 1.0 or kappa <= 0.0:
        return (p_home, p_draw, p_away)
    decisive = p_home + p_away
    if decisive <= _EPS:
        return (p_home, p_draw, p_away)
    r = min(1.0 - _EPS, max(_EPS, p_home / decisive))
    logit = math.log(r / (1.0 - r))
    r2 = 1.0 / (1.0 + math.exp(-kappa * logit))
    return (decisive * r2, p_draw, decisive * (1.0 - r2))
