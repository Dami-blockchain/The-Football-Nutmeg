"""Online model selection — Hedge (exponential weights) over two experts.

"Pick the right model for every match" is unknowable in advance; the honest
version is to let accumulating evidence pick it: log BOTH the pure-Glicko and
the ensemble prediction for every fixture, score each by RPS as results
settle, and weight the live prediction by exp(-eta * cumulative_loss). The
Hedge guarantee: total loss approaches the better expert's loss, so if one
model really is better this tournament, the bot converges onto it within a
handful of matchdays — and if they're equivalent it sits safely in between.

The Qatar-2022 backtest motivated this: the ensemble beat Glicko on 1,600+
ordinary internationals but not (inconclusively) on the 64-match WC sample.
Rather than betting real money on either side of that ambiguity, the bot
runs the comparison live and follows the winner.

Pure math — storage lives in betbot.storage.repos (model_predictions table).
"""

from __future__ import annotations

import math


def hedge_weights(
    loss_a: float, loss_b: float, *, eta: float = 2.0
) -> tuple[float, float]:
    """Exponential weights for two experts given cumulative losses.

    ``eta`` is the learning rate: 0 = ignore evidence (50/50 forever);
    higher = faster convergence onto the lower-loss expert. RPS per match is
    in [0, 1] and a real skill gap is worth ~0.01-0.03 RPS/match, so the
    default eta=2.0 shifts weight meaningfully only once a gap persists
    across many matches — one fluky upset barely moves it.
    """
    m = min(loss_a, loss_b)  # subtract min: exp() stays in (0, 1]
    ea = math.exp(-eta * (loss_a - m))
    eb = math.exp(-eta * (loss_b - m))
    total = ea + eb
    return ea / total, eb / total
