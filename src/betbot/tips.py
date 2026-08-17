"""Prediction delivery formatting — the tipster message bodies.

Pure formatters over a stored :class:`~betbot.storage.models.PredictionRow`
(and its linked ``paper_bet``, which IS our recommendation). Kept separate from
storage/network so the exact message shapes are unit-testable with fixture data.

The STANDING format rule (a memory-pinned product requirement): every match
prediction carries a home/away designation, market anchoring, the xG readout
when present, and a BOLD bet/no-bet call — defaulting to NO BET. A prediction is
a BET only when a stored paper_bet exists with a market edge at/above threshold;
otherwise (no market, or edge below threshold) it is NO BET.

Messages use Telegram Markdown (``parse_mode="Markdown"``), matching
:mod:`betbot.reports`.
"""

from __future__ import annotations

from betbot.config import get_settings


def _kickoff_str(pred) -> str:
    """Kickoff as ``HH:MM`` UTC; empty if absent."""
    ko = getattr(pred, "kickoff", None)
    if ko is None:
        return ""
    return ko.strftime("%H:%M")


def _reco_bet(pred, *, edge_threshold: float):
    """The stored recommendation for this prediction, or ``None`` = NO BET.

    Our recommendation is the linked paper_bet. A paper_bet with a market price
    and an edge >= threshold is a BET on ``outcome``; anything else (no
    paper_bet, no market price, or edge below threshold) is NO BET.
    """
    bets = list(getattr(pred, "paper_bets", None) or [])
    for b in bets:
        if b.market_price is not None and b.edge is not None and b.edge >= edge_threshold:
            return b
    return None


def _outcome_label(outcome: str, home: str, away: str) -> str:
    if outcome == "HOME":
        return f"{home} (H)"
    if outcome == "AWAY":
        return f"{away} (A)"
    return "the draw"


def format_prediction(pred, *, edge_threshold: float | None = None) -> str:
    """Full revealed prediction: teams (H/A), model triple + xG, market, call."""
    if edge_threshold is None:
        edge_threshold = get_settings().edge_threshold

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

    reco = _reco_bet(pred, edge_threshold=edge_threshold)
    if reco is not None:
        market = f"Market: {reco.outcome} {reco.market_price:.2f}   Edge: {reco.edge:+.2f}"
        call = f"*Bet: {_outcome_label(reco.outcome, home, away)}* (edge {reco.edge:+.2f})"
    else:
        # Any priced-but-below-threshold reco still shows the price if we have it.
        priced = next(
            (b for b in (getattr(pred, "paper_bets", None) or [])
             if b.market_price is not None),
            None,
        )
        if priced is not None:
            market = f"Market: {priced.outcome} {priced.market_price:.2f}"
            call = "*Bet: NO BET* (edge below threshold)"
        else:
            market = "Market: n/a"
            call = "*Bet: NO BET* (no market)"

    return "\n".join([header, model, market, call])


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
