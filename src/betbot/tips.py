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

#: Sentinel: ``format_result(shown_anchor=...)`` default meaning "badge off the
#: outcome row's own anchor_source". A caller that knows the SOLD triple's
#: anchor state (which can differ from the final post-rescore state) passes it
#: explicitly instead.
_USE_OUTCOME_ANCHOR = object()


def anchor_badge(anchor_source, settings) -> str | None:
    """One-line market-provenance badge for a shown probability triple.

    The operator asked (2026-10-09) that EVERY prediction-carrying message state
    plainly whether the shown ``p_*`` triple was anchored to a market line or is
    the raw model. Returns:

    * ``"📊 _Market-anchored_"`` when ``anchor_source`` is set (the triple was
      shrunk toward a bookmaker / exchange line), or
    * ``"⚠️ _Model-only — no market odds for this fixture_"`` when it is NULL
      (no market line was available, so the shown triple is the raw model).

    Gated by ``BETBOT_ANCHOR_BADGE`` (default ON). ``settings is None`` or the
    flag off -> ``None`` (append nothing), so callers that pass no settings
    (and the one env rollback) are byte-identical to the pre-badge output.
    """
    if settings is None or not getattr(settings, "anchor_badge", True):
        return None
    if anchor_source:
        return "📊 _Market-anchored_"
    return "⚠️ _Model-only — no market odds for this fixture_"


def anchor_note(pred, settings) -> str | None:
    """Back-compat shim: the badge for a prediction row's ``anchor_source``.

    Kept so existing call sites (``format_prediction``,
    ``format_high_conf_alert``) need no change; delegates to
    :func:`anchor_badge`.
    """
    return anchor_badge(getattr(pred, "anchor_source", None), settings)


def draw_risk_line(pred, settings) -> str | None:
    """One-line DRAW-RISK tag for a high-confidence call, or ``None``.

    Operator directive (2026-10-15): when a fixture's DE-VIGGED MARKET draw
    probability is unusually high, flag it and name the safer play — buy NO on
    the opponent, which pays if our pick wins OR the match draws. This is the
    sole sanctioned exception to the 2026-09-08 "no bet/no-bet copy" rule.

    Returns the line ONLY when all of these hold (else ``None`` -> nothing
    appended, byte-identical to before):

    * ``BETBOT_DRAW_RISK_TAG`` is on (``settings`` present);
    * a MARKET draw prob was recorded on the row (``market_p_draw`` not NULL) —
      we NEVER guess it from the model's own draw prob;
    * that prob is >= ``draw_risk_threshold`` (default 0.20);
    * the call is high-confidence (top pick >= ``high_conf_alert_min_p``) and is
      NOT the draw itself (a draw pick has no single opponent, and is never a
      high-confidence pick anyway).
    """
    if settings is None or not getattr(settings, "draw_risk_tag_enabled", False):
        return None
    mpd = getattr(pred, "market_p_draw", None)
    if mpd is None:
        return None
    if float(mpd) < float(getattr(settings, "draw_risk_threshold", 0.20)):
        return None
    home, away = pred.home_team, pred.away_team
    pick, p = max(
        (("HOME", pred.p_home), ("DRAW", pred.p_draw), ("AWAY", pred.p_away)),
        key=lambda kv: kv[1],
    )
    if pick == "DRAW":
        return None
    if p < float(getattr(settings, "high_conf_alert_min_p", 0.65)):
        return None
    pick_team, opponent = (home, away) if pick == "HOME" else (away, home)
    return (
        f"⚠️ Draw risk: market rates a draw at {mpd:.0%}. "
        f"Safer play: NO on {opponent} (pays if {pick_team} win or draw)."
    )


#: football-data.org CL ``stage`` enum -> friendly, operator-facing label.
#: Knockout rounds carry a "Knockout" prefix so the operator reads them AS
#: knockouts (measured weaker: ~67% vs ~76% at 0.65). PLAYOFFS already says
#: "Knockout", so it is not double-prefixed. Any value not in this map (and a
#: blank / NULL / unknown stage) renders NOTHING.
_STAGE_FRIENDLY = {
    "LEAGUE_STAGE": "League phase",
    "PLAYOFFS": "Knockout play-off",
    "LAST_16": "Knockout · Round of 16",
    "QUARTER_FINALS": "Knockout · Quarter-final",
    "SEMI_FINALS": "Knockout · Semi-final",
    "FINAL": "Knockout · Final",
}


def stage_label(stage, settings) -> str | None:
    """One-line CL competition-stage badge for a shown fixture call.

    ``stage`` is football-data.org's raw CL enum (stored on the prediction /
    outcome row; CL-only, so domestic rows carry ``None``). Returns the friendly
    label ("League phase", "Knockout · Round of 16", ...) or
    ``None`` when there is nothing honest to show: domestic/absent stage
    (``None`` / blank), an unrecognised value, no ``settings``, or the flag off.
    Gated by ``BETBOT_STAGE_BADGE`` (default ON) so one env line rolls it back
    and callers that pass no settings are byte-identical to the pre-badge
    output.
    """
    if settings is None or not getattr(settings, "stage_badge", True):
        return None
    if not stage:
        return None
    return _STAGE_FRIENDLY.get(str(stage).strip().upper())


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
    _stage = stage_label(getattr(pred, "stage", None), settings)
    if _stage:
        header += f" · {_stage}"
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
    _note = anchor_note(pred, settings)
    if _note:
        parts.insert(len(parts) - 1, _note)
    _risk = draw_risk_line(pred, settings)
    if _risk:
        parts.insert(len(parts) - 1, _risk)
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
    settings=None,
    shown_anchor=_USE_OUTCOME_ANCHOR,
    show_badge: bool = True,
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
    _stage = stage_label(getattr(outcome_row, "stage", None), settings)
    if _stage:
        ft += f" · {_stage}"
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
    # Market-provenance badge, matching the prediction surfaces. The badge must
    # describe WHAT WAS SOLD, which can differ from the fixture's final
    # post-rescore ``anchor_source`` when an early alert shipped before the
    # kickoff-60 rescore anchored it. Callers that know the sold triple's anchor
    # state pass it via ``shown_anchor`` (and ``show_badge=False`` to omit the
    # badge when that state cannot be established); the default sentinel falls
    # back to the outcome row's own anchor_source. ``settings is None`` -> no
    # badge (legacy callers / unit tests unchanged).
    if show_badge:
        anchor = (
            getattr(outcome_row, "anchor_source", None)
            if shown_anchor is _USE_OUTCOME_ANCHOR
            else shown_anchor
        )
        badge = anchor_badge(anchor, settings)
        if badge:
            lines.append(badge)
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
        "\u2139\ufe0f The full-time score we sent earlier was wrong "
        "(our results source has since corrected it). The winner is "
        "unchanged, so our call is unaffected."
    )
