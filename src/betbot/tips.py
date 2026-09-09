"""Prediction delivery formatting — the tipster message bodies.

Pure formatters over a stored :class:`~betbot.storage.models.PredictionRow`
(and its linked ``paper_bet``, which IS our recommendation). Kept separate from
storage/network so the exact message shapes are unit-testable with fixture data.

The bot is a pure tipster: every match prediction carries a home/away
designation, the model H/D/A probabilities, and the xG readout when present.
The old bet field (stake / market price / edge) stays removed by operator
directive — the internal paper_bet record is logged for our own accuracy
tracking but never rendered to users.

RETIRED (2026-09-08, operator directive): the bot no longer emits ANY BET /
NO BET call on the prediction surface. The long-standing format rule that
"every prediction carries a BOLD bet/no-bet call defaulting to NO BET" is
PERMANENTLY WITHDRAWN — do NOT reinstate it. Next to a model with no
demonstrated edge it read as a betting recommendation. The
``BETBOT_CONFIDENCE_FILTER`` flag and :mod:`betbot.strategy.confidence` survive
ONLY as an internal SELECTION metric (used by the backtests); they render
nothing to users. The honesty guard is now a plain non-advice caveat on the
user-facing surfaces instead of a NO BET default.

Messages use Telegram Markdown (``parse_mode="Markdown"``), matching
:mod:`betbot.reports`.
"""

from __future__ import annotations

from betbot.leagues import league_label
from betbot.timefmt import eat_time

#: Honest non-advice caveat shown on every FORWARD-LOOKING prediction surface
#: (the revealed prediction, the confirmed-XI alert, the high-confidence alert).
#: It replaces the retired NO BET default (2026-09-08) as the honesty guard, and
#: MUST stay anti-value: the pick is an accuracy signal, explicitly NOT +EV or
#: betting advice. Kept in one place so both formatters render identical copy.
NON_ADVICE_CAVEAT = "_Model call, not betting advice — accuracy signal, not +EV._"


def _kickoff_str(pred) -> str:
    """Kickoff as ``HH:MM EAT`` (Africa/Nairobi); empty if absent.

    The stored kickoff is UTC; every user-facing surface shows EAT via the
    shared :func:`betbot.timefmt.eat_time` helper.
    """
    return eat_time(getattr(pred, "kickoff", None))


def format_prediction(
    pred, *, edge_threshold: float | None = None, settings=None
) -> str:
    """Full revealed prediction: teams (H/A), model triple + xG.

    Pure tipster output: teams, the named-club prediction, the model triple and
    xG. The bot emits NO bet/no-bet call — the standing rule was retired
    2026-09-08 (see the module docstring); do not reinstate it. ``edge_threshold``
    and ``settings`` are accepted (and ignored) for backwards-compatible callers.
    """
    home, away = pred.home_team, pred.away_team
    ko = _kickoff_str(pred)
    header = f"*{home} (H) v {away} (A)*"
    league = league_label(getattr(pred, "competition_code", None))
    if league:
        header += f" · {league}"
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

    parts = [header, winner, model, NON_ADVICE_CAVEAT]
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


def format_result(
    outcome_row,
    home_team: str,
    away_team: str,
    *,
    competition_code: str | None = None,
    sold_triple: tuple[float, float, float] | None = None,
) -> str:
    """End-of-match RESULT ALERT body for one settled fixture.

    Shows the final score, whether OUR pick was right, and the model's triple —
    no new probabilities are gated (the user already saw/paid for the
    prediction). ``outcome_row`` is a
    :class:`~betbot.storage.models.PredictionOutcome`.

    ``sold_triple`` is the ``(p_home, p_draw, p_away)`` ACTUALLY SOLD to the
    user (from the reveal ledger). When present it is quoted verbatim as "what
    you were shown", because ``outcome_row``'s triple is the POST-rescore one
    and can differ from what the user actually paid for. When ``None`` (legacy
    reveal rows, or the fixture was never revealed) only the model triple is
    shown, exactly as before.

    CONSISTENCY: when a sold triple is present and its argmax differs from the
    post-rescore ``predicted_pick`` (a pick-flip between sale and settlement),
    the "Our call" line AND its correct/wrong verdict are derived from the SOLD
    pick — scored against ``actual_outcome`` — so the message can never say
    "Our call: X — correct" while telling the user they were sold Y.
    """
    picked = outcome_row.predicted_pick
    was_correct = outcome_row.correct
    if sold_triple is not None:
        sh, sd, sa = sold_triple
        sold_pick = max(
            (("HOME", sh), ("DRAW", sd), ("AWAY", sa)), key=lambda kv: kv[1]
        )[0]
        if sold_pick != picked:
            picked = sold_pick
            was_correct = sold_pick == outcome_row.actual_outcome
    verdict = "✅ correct" if was_correct else "❌ wrong"
    pick = _pick_label(picked, home_team, away_team)
    league = league_label(competition_code)
    ft = (
        f"*Full time: {home_team} {outcome_row.home_goals}-"
        f"{outcome_row.away_goals} {away_team}*"
    )
    if league:
        ft += f" · {league}"
    lines = [
        ft,
        f"Our call: {pick} — {verdict}",
        f"Model had H {outcome_row.predicted_home:.0%} / "
        f"D {outcome_row.predicted_draw:.0%} / A {outcome_row.predicted_away:.0%}",
    ]
    if sold_triple is not None:
        sh, sd, sa = sold_triple
        lines.append(
            f"As shown to you: H {sh:.0%} / D {sd:.0%} / A {sa:.0%}"
        )
    return "\n".join(lines)


def format_locked(pred) -> str:
    """Teaser for a locked prediction — teams + kickoff only, NO probabilities."""
    home, away = pred.home_team, pred.away_team
    ko = _kickoff_str(pred)
    header = f"*{home} (H) v {away} (A)*"
    league = league_label(getattr(pred, "competition_code", None))
    if league:
        header += f" · {league}"
    if ko:
        header += f" — {ko}"
    return f"{header}\n🔒 send 1 USDC (Polygon) to unlock this prediction"


def render_result_correction(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    *,
    competition_code: str | None = None,
) -> str:
    """Short, plain, non-alarming note that a full-time score we PUBLISHED has
    been corrected.

    Only goals-only corrections reach a user (a winner change is never
    auto-applied — see SettlementWatcher.reverify_recent_scores), so the winner
    — and therefore whether our call was right — is unchanged, and the copy says
    so explicitly. Tone matches render_morning_drop_notice: an info line, the
    corrected score in bold, no probabilities or alarm.
    """
    league = league_label(competition_code)
    league_tag = f" \u00b7 {league}" if league else ""
    return (
        "*\u26bd Result correction*\n\n"
        f"*{home_team} {home_goals}-{away_goals} {away_team}*{league_tag}\n\n"
        "\u2139\ufe0f An earlier result feed had this match\u2019s score "
        "wrong \u2014 it has now been corrected to the score above. The winner "
        "is unchanged, so our call is unaffected."
    )
