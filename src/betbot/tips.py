"""Prediction delivery formatting — the tipster message bodies.

Pure formatters over a stored :class:`~betbot.storage.models.PredictionRow`
(and its linked ``paper_bet``, which IS our recommendation). Kept separate from
storage/network so the exact message shapes are unit-testable with fixture data.

The bot is a pure tipster: every match prediction carries a home/away
designation, the model H/D/A probabilities, and the xG readout when present.
There is NO user-facing bet / no-bet call (removed by operator directive) — the
internal paper_bet / edge record is still logged for our own accuracy tracking
but never rendered to users.

Messages use Telegram Markdown (``parse_mode="Markdown"``), matching
:mod:`betbot.reports`.
"""

from __future__ import annotations


def _kickoff_str(pred) -> str:
    """Kickoff as ``HH:MM`` UTC; empty if absent."""
    ko = getattr(pred, "kickoff", None)
    if ko is None:
        return ""
    return ko.strftime("%H:%M")


def format_prediction(pred, *, edge_threshold: float | None = None) -> str:
    """Full revealed prediction: teams (H/A), model triple + xG.

    Pure tipster output — the bot no longer emits a bet / no-bet call or the
    market/edge line that supported it. ``edge_threshold`` is accepted (and
    ignored) for backwards-compatible call sites.
    """
    home, away = pred.home_team, pred.away_team
    ko = _kickoff_str(pred)
    header = f"*{home} (H) v {away} (A)*"
    if ko:
        header += f" — {ko}"

    model = (
        f"Model: H {pred.p_home:.0%} / D {pred.p_draw:.0%} / A {pred.p_away:.0%}"
    )
    if pred.home_xg is not None and pred.away_xg is not None:
        model += f"   (xG {pred.home_xg:.2f}–{pred.away_xg:.2f})"

    return "\n".join([header, model])


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
    body = format_prediction(pred, edge_threshold=edge_threshold)
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
