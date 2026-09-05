"""Prediction delivery formatting — the tipster message bodies.

Pure formatters over a stored :class:`~betbot.storage.models.PredictionRow`
(and its linked ``paper_bet``, which IS our recommendation). Kept separate from
storage/network so the exact message shapes are unit-testable with fixture data.

The bot is a pure tipster: every match prediction carries a home/away
designation, the model H/D/A probabilities, and the xG readout when present.
The old bet field (stake / market price / edge) stays removed by operator
directive — the internal paper_bet record is logged for our own accuracy
tracking but never rendered to users.

When the flag-gated confidence filter is ON (``BETBOT_CONFIDENCE_FILTER``,
default OFF), a BOLD BET / NO BET call is appended, defaulting to NO BET: a BET
is issued only when the FINAL blended favourite clears the pre-registered
threshold and the draw is far enough away (see
:mod:`betbot.strategy.confidence`). With the flag OFF the output is
byte-identical to the pure-tipster format above, so live behaviour is unchanged
until the operator turns it on.

The call is a SELECTION rule, not a value claim. Called picks are short-priced
favourites, so their higher hit rate is an ACCURACY KPI — never label it
+EV, edge, or beating the market anywhere in this copy.

Messages use Telegram Markdown (``parse_mode="Markdown"``), matching
:mod:`betbot.reports`.
"""

from __future__ import annotations

from betbot.timefmt import eat_time


def _kickoff_str(pred) -> str:
    """Kickoff as ``HH:MM EAT`` (Africa/Nairobi); empty if absent.

    The stored kickoff is UTC; every user-facing surface shows EAT via the
    shared :func:`betbot.timefmt.eat_time` helper.
    """
    return eat_time(getattr(pred, "kickoff", None))


def _confidence_line(pred, settings=None) -> str:
    """BOLD BET / NO BET call for one prediction, or '' when the flag is OFF.

    Reads the FINAL blended probabilities carried on ``pred`` — whatever the
    user is shown — so once market anchoring lands the call is made on the
    ANCHORED favourite with no change needed here.
    """
    from betbot.strategy.confidence import evaluate_settings

    if settings is None:
        from betbot.config import get_settings

        settings = get_settings()
    if not getattr(settings, "club_confidence_filter", False):
        return ""
    call = evaluate_settings((pred.p_home, pred.p_draw, pred.p_away), settings)
    if not call.called:
        return "*NO BET* — below our confidence bar"
    team = pred.home_team if call.pick == "HOME" else pred.away_team
    side = "H" if call.pick == "HOME" else "A"
    return f"*BET: {team} ({side}) to win* ({call.p_pick:.0%} confidence)"


def format_prediction(
    pred, *, edge_threshold: float | None = None, settings=None
) -> str:
    """Full revealed prediction: teams (H/A), model triple + xG.

    The BOLD BET / NO BET call is appended only when the confidence filter flag
    is ON; at its shipped default (OFF) the output carries no bet language at
    all. Pure tipster output — the bot no longer emits a bet / no-bet call or the
    market/edge line that supported it. ``edge_threshold`` is accepted (and
    ignored) for backwards-compatible call sites.
    """
    home, away = pred.home_team, pred.away_team
    ko = _kickoff_str(pred)
    header = f"*{home} (H) v {away} (A)*"
    if ko:
        header += f" — {ko}"

    # Predicted winner = the model's most likely outcome (argmax of H/D/A),
    # stated plainly with its probability (favourites often sit below 50% once
    # the draw is in play — the % keeps it honest).
    _pick, _p = max(
        [("home", pred.p_home), ("draw", pred.p_draw), ("away", pred.p_away)],
        key=lambda kv: kv[1],
    )
    if _pick == "draw":
        winner = f"🏆 *Prediction: Draw* ({_p:.0%})"
    else:
        _team = pred.home_team if _pick == "home" else pred.away_team
        winner = f"🏆 *Prediction: {_team} to win* ({_p:.0%})"

    model = (
        f"Model: H {pred.p_home:.0%} / D {pred.p_draw:.0%} / A {pred.p_away:.0%}"
    )
    if pred.home_xg is not None and pred.away_xg is not None:
        model += f"   (xG {pred.home_xg:.2f}–{pred.away_xg:.2f})"

    parts = [header, winner, model]
    call = _confidence_line(pred, settings)
    if call:
        parts.append(call)
    return "\n".join(parts)


def _format_xi(side: dict | None) -> str:
    """One team's confirmed XI + formation as ``[4-3-3] Name, Name, …`` or ''."""
    if not side:
        return ""
    xi = list(side.get("xi") or [])
    formation = (side.get("formation") or "").strip()
    if not xi:
        return ""
    prefix = f"[{formation}] " if formation else ""
    return prefix + ", ".join(xi)


def format_prediction_with_lineup(
    pred,
    lineup: dict | None,
    *,
    edge_threshold: float | None = None,
    adj_note: str | None = None,
    absences: str | None = None,
    settings=None,
) -> str:
    """Full revealed prediction PREFIXED with the confirmed XIs (pre-match alert).

    ``lineup`` is ``{"home": {"formation", "xi"}, "away": {...}}`` (from
    :meth:`ApiFootballClient.get_lineups`) or ``None``. When present, both XIs
    (with formation) are shown above the standing prediction block; an optional
    ``absences`` line flags the key regulars who are OUT (only when the lineup
    adjustment is nonzero), and ``adj_note`` carries a caveat (e.g. lineup not
    yet confirmed). The prediction body itself is the UNCHANGED
    :func:`format_prediction` output, so the standing format rule is preserved.
    """
    home, away = pred.home_team, pred.away_team
    parts: list[str] = []
    if lineup:
        home_xi = _format_xi(lineup.get("home"))
        away_xi = _format_xi(lineup.get("away"))
        if home_xi:
            parts.append(f"*{home} (H)* XI: {home_xi}")
        if away_xi:
            parts.append(f"*{away} (A)* XI: {away_xi}")
        if absences:
            parts.append(f"⚠️ Key absences: {absences}")
    if adj_note:
        parts.append(adj_note)
    body = format_prediction(
        pred, edge_threshold=edge_threshold, settings=settings
    )
    if parts:
        return "\n".join(parts) + "\n\n" + body
    return body


def _pick_label(pick: str, home: str, away: str) -> str:
    if pick == "HOME":
        return f"{home} (H)"
    if pick == "AWAY":
        return f"{away} (A)"
    return "the draw"


def format_result(outcome_row, home_team: str, away_team: str) -> str:
    """End-of-match RESULT ALERT body for one settled fixture.

    Shows the final score, whether OUR pick was right, and the model's original
    triple — no new probabilities are gated (the user already saw/paid for the
    prediction). ``outcome_row`` is a
    :class:`~betbot.storage.models.PredictionOutcome`.
    """
    verdict = "✅ correct" if outcome_row.correct else "❌ wrong"
    pick = _pick_label(outcome_row.predicted_pick, home_team, away_team)
    return "\n".join([
        f"*Full time: {home_team} {outcome_row.home_goals}-"
        f"{outcome_row.away_goals} {away_team}*",
        f"Our call: {pick} — {verdict}",
        f"Model had H {outcome_row.predicted_home:.0%} / "
        f"D {outcome_row.predicted_draw:.0%} / A {outcome_row.predicted_away:.0%}",
    ])


def format_locked(pred) -> str:
    """Teaser for a locked prediction — teams + kickoff only, NO probabilities."""
    home, away = pred.home_team, pred.away_team
    ko = _kickoff_str(pred)
    header = f"*{home} (H) v {away} (A)*"
    if ko:
        header += f" — {ko}"
    return f"{header}\n🔒 send 1 USDC (Polygon) to unlock this prediction"
