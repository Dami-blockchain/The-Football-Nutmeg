"""Repository helpers — narrow, intention-revealing CRUD."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from betbot.logging import get_logger
from betbot.storage.db import session_scope
from betbot.storage.models import (
    ArbExecution,
    Deposit,
    GasTopup,
    GlickoRating,
    KillSwitch,
    ModelPrediction,
    PaperBet,
    PredictionOutcome,
    PredictionReveal,
    PredictionRow,
    TreasuryBridge,
    User,
)
from betbot.strategy.engine import BetDecision, Outcome, Prediction
from betbot.strategy.ensemble import ranked_probability_score
from betbot.strategy.glicko import Glicko2Rating, update_rating

log = get_logger(__name__)


def upsert_prediction(
    prediction: Prediction, *, kickoff: datetime
) -> int:
    """Insert a prediction row (or update if (fixture_id, run_date) exists).

    Returns the row id.
    """
    run_date = date.today().isoformat()
    with session_scope() as s:
        existing = s.execute(
            select(PredictionRow)
            .where(PredictionRow.fixture_id == prediction.fixture_id)
            .where(PredictionRow.run_date == run_date)
        ).scalar_one_or_none()
        if existing is not None:
            # Kickoff too: football-data rewrites ``utcDate`` when a fixture is
            # moved, and every caller passes the kickoff it just read upstream.
            # Skipping it here is what let a rescheduled match keep its ORIGINAL
            # time forever — which pinned its pre-match alerts to a dead slot.
            existing.kickoff = kickoff
            existing.p_home = prediction.p_home
            existing.p_draw = prediction.p_draw
            existing.p_away = prediction.p_away
            existing.home_score = prediction.home_score
            existing.away_score = prediction.away_score
            existing.draw_score = prediction.draw_score
            existing.home_xg = prediction.home_xg
            existing.away_xg = prediction.away_xg
            s.flush()
            return existing.id
        row = PredictionRow(
            fixture_id=prediction.fixture_id,
            competition_code=prediction.competition_code,
            kickoff=kickoff,
            run_date=run_date,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            p_home=prediction.p_home,
            p_draw=prediction.p_draw,
            p_away=prediction.p_away,
            home_score=prediction.home_score,
            away_score=prediction.away_score,
            draw_score=prediction.draw_score,
            home_xg=prediction.home_xg,
            away_xg=prediction.away_xg,
        )
        s.add(row)
        s.flush()
        return row.id


def insert_paper_bet(decision: BetDecision, prediction_id: int) -> bool:
    """Insert an edge-filtered paper bet. Returns False if a duplicate exists."""
    with session_scope() as s:
        existing = s.execute(
            select(PaperBet)
            .where(PaperBet.fixture_id == decision.fixture_id)
            .where(PaperBet.outcome == decision.outcome.value)
        ).scalar_one_or_none()
        if existing is not None:
            return False
        s.add(
            PaperBet(
                prediction_id=prediction_id,
                fixture_id=decision.fixture_id,
                outcome=decision.outcome.value,
                our_probability=decision.our_probability,
                market_price=decision.market_price,
                edge=decision.edge,
                stake_usd=decision.stake_usd,
                rationale=decision.rationale,
            )
        )
        return True


def insert_paper_bet_no_market(
    prediction: Prediction,
    prediction_id: int,
    outcome: Outcome,
    *,
    stake_usd: float,
    rationale: str,
) -> bool:
    """Insert a Phase-1-style favourite-only paper bet (no market price)."""
    our_p = {
        Outcome.HOME: prediction.p_home,
        Outcome.DRAW: prediction.p_draw,
        Outcome.AWAY: prediction.p_away,
    }[outcome]
    with session_scope() as s:
        existing = s.execute(
            select(PaperBet)
            .where(PaperBet.fixture_id == prediction.fixture_id)
            .where(PaperBet.outcome == outcome.value)
        ).scalar_one_or_none()
        if existing is not None:
            return False
        s.add(
            PaperBet(
                prediction_id=prediction_id,
                fixture_id=prediction.fixture_id,
                outcome=outcome.value,
                our_probability=our_p,
                market_price=None,
                edge=None,
                stake_usd=stake_usd,
                rationale=rationale,
            )
        )
        return True


def list_recent_paper_bets(days: int = 7) -> list[PaperBet]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet)
                .where(PaperBet.created_at >= cutoff)
                .order_by(PaperBet.created_at.desc())
            ).scalars()
        )
        # Detach so attribute access works after the session closes.
        s.expunge_all()
        return rows


def daily_paper_exposure_usd() -> float:
    """Sum of stakes placed today (used by the risk control)."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    with session_scope() as s:
        rows = s.execute(
            select(PaperBet.stake_usd).where(PaperBet.created_at >= today_start)
        ).scalars()
        return float(sum(rows))


# ----------------------------------------------------------------------
# Settlement (Phase 4)
# ----------------------------------------------------------------------
def list_unsettled_bets_due(now: datetime, grace_minutes: int) -> list[PaperBet]:
    """Unsettled bets whose kickoff + grace has passed — ready to settle."""
    cutoff = now - timedelta(minutes=grace_minutes)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet)
                .join(PredictionRow, PaperBet.prediction_id == PredictionRow.id)
                .where(PaperBet.settled_at.is_(None))
                .where(PredictionRow.kickoff <= cutoff)
                .order_by(PaperBet.fixture_id)
            ).scalars()
        )
        # Detach so attribute access works after the session closes.
        s.expunge_all()
        return rows


def record_settlement(
    bet_id: int, settled_outcome: str, pnl_usd: float, settled_at: datetime
) -> None:
    with session_scope() as s:
        bet = s.get(PaperBet, bet_id)
        if bet is None:
            return
        bet.settled_at = settled_at
        bet.settled_outcome = settled_outcome
        bet.pnl_usd = pnl_usd


def settled_pnl_window(days: int) -> tuple[float, float]:
    """``(realized_pnl, staked)`` over settled MARKET bets in the trailing window.

    No-market (favourite-only) bets are excluded — they have no real-money
    equivalent and must not pollute the kill-switch / gate signal.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PaperBet.pnl_usd, PaperBet.stake_usd)
                .where(PaperBet.settled_at.is_not(None))
                .where(PaperBet.settled_at >= cutoff)
                .where(PaperBet.market_price.is_not(None))
            )
        )
    pnl = float(sum((r[0] or 0.0) for r in rows))
    staked = float(sum((r[1] or 0.0) for r in rows))
    return pnl, staked


# ----------------------------------------------------------------------
# Prediction-vs-reality outcome ledger (R10a — accuracy tracking)
# ----------------------------------------------------------------------
_OUTCOME_INDEX = {"HOME": 0, "DRAW": 1, "AWAY": 2}


def score_prediction(
    p_home: float, p_draw: float, p_away: float, actual_outcome: str
) -> tuple[str, bool, float, float, float]:
    """Pure scoring: ``(pick, correct, brier, rps, log_loss)`` for one prediction.

    ``pick`` is the argmax outcome (HOME/DRAW/AWAY). ``brier`` is the multi-class
    Brier over the 3 outcomes; ``rps`` reuses the ensemble ranked-probability
    score; ``log_loss`` is ``-log(p_actual)`` (clipped so a zero probability
    can't produce an infinite loss).
    """
    probs = (float(p_home), float(p_draw), float(p_away))
    pick = ("HOME", "DRAW", "AWAY")[max(range(3), key=lambda i: probs[i])]
    idx = _OUTCOME_INDEX[actual_outcome]
    y = [1.0 if i == idx else 0.0 for i in range(3)]
    brier = sum((probs[i] - y[i]) ** 2 for i in range(3))
    rps = ranked_probability_score(probs, idx)
    p_actual = min(1.0, max(1e-12, probs[idx]))
    log_loss = -math.log(p_actual)
    return pick, pick == actual_outcome, brier, rps, log_loss


def record_prediction_outcome(
    *,
    fixture_id: int,
    competition_code: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    actual_outcome: str,
    home_goals: int,
    away_goals: int,
    settled_at: datetime,
    result_notified: bool = False,
    kickoff: datetime | None = None,
) -> bool:
    """INSERT-OR-IGNORE one scored prediction. Returns True iff NEWLY inserted.

    The unique constraint on ``fixture_id`` makes this idempotent: a re-run of
    settlement over the same finished fixture returns False and writes nothing,
    which is what gates the one-shot per-match rating update + result alert.

    ``result_notified=True`` records the row PRE-notified — used for STALE
    backfill (fixtures that finished long ago), which must enter the accuracy
    ledger without ever triggering an end-of-match RESULT ALERT.

    ``kickoff`` is when the match was PLAYED. The record is scoped by season and
    ``settled_at`` cannot express that — a July fixture settled in August is
    still a July fixture — so a row without it is excluded from the record.
    """
    pick, correct, brier, rps, log_loss = score_prediction(
        p_home, p_draw, p_away, actual_outcome
    )
    try:
        with session_scope() as s:
            s.add(
                PredictionOutcome(
                    fixture_id=fixture_id,
                    competition_code=competition_code,
                    predicted_home=float(p_home),
                    predicted_draw=float(p_draw),
                    predicted_away=float(p_away),
                    predicted_pick=pick,
                    actual_outcome=actual_outcome,
                    correct=correct,
                    brier=brier,
                    rps=rps,
                    log_loss=log_loss,
                    kickoff=kickoff,
                    home_goals=int(home_goals),
                    away_goals=int(away_goals),
                    result_notified=result_notified,
                    settled_at=settled_at,
                )
            )
        return True
    except IntegrityError:
        # Already scored (uq_prediction_outcomes_fixture) — safe no-op.
        return False


def _ledger_epoch() -> datetime | None:
    """Earliest settlement instant an accuracy read is allowed to include.

    Outcomes settled before BETBOT_ACCURACY_LEDGER_EPOCH come from the
    degenerate 0/0/100-AWAY rating bug and are NOT representative of the model
    we ship, so quoting them to a user would be dishonest. Returns None when
    the setting is empty/unparseable (cutoff disabled).
    """
    from betbot.config import get_settings

    raw = (get_settings().accuracy_ledger_epoch or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except ValueError:
        log.warning("bad_accuracy_ledger_epoch", value=raw)
        return None


# A real 1X2 forecast never assigns a component below this. Anything that does
# is the degenerate 0/0/100-AWAY rating bug — those rows are excluded from every
# accuracy read no matter when they settled. The date epoch alone is NOT enough:
# stale fixtures are backfilled with settled_at = now, so a poisoned prediction
# made before the fix can land in the ledger dated after it (verified on the
# live ledger 2026-08-18: a 0.000/0.000/1.000 row carried settled_at 2026-08-17
# 18:00, i.e. inside the epoch).
def _current_season_start(now: datetime | None = None) -> datetime:
    """1 August of the football year containing ``now``.

    European club seasons run August-May, so the year flips on 1 August: in
    July 2027 the current season is still 2026/27, and from 1 August 2027 it is
    2027/28. Derived rather than hardcoded because a fixed date silently
    expires — next August a stale-backfilled May fixture would pass a
    2026-08-01 boundary and re-enter a record claiming to be "this season",
    which is the exact bug this scoping exists to prevent.
    """
    now = now or datetime.now(timezone.utc)
    year = now.year if now.month >= 8 else now.year - 1
    return datetime(year, 8, 1, tzinfo=timezone.utc)


def _season_start() -> datetime | None:
    """Start of the current club season, or None when scoping is disabled.

    ``BETBOT_SEASON_START`` is an OVERRIDE, not the source of truth: set it to
    pin a boundary (the test suite sets it empty to disable scoping entirely),
    leave it at its default and the boundary rolls forward on its own.
    """
    from betbot.config import get_settings

    raw = (get_settings().season_start or "").strip()
    if not raw:
        return None
    if raw.lower() == "auto":
        return _current_season_start()
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except ValueError:
        log.warning("bad_season_start", value=raw, note="falling back to auto")
        return _current_season_start()


def _is_club_competition(code: str) -> bool:
    """True for the club competitions the record covers.

    An ALLOWLIST, deliberately: a denylist of {"WC"} would silently readmit the
    next international tournament (EURO, Nations League, friendlies), which is
    the same class of fixture for the same reason — a different engine, neutral
    venues, and no bearing on club-season form.

    The allowlist is the CONFIGURED league set union the built-in codes, not the
    constant alone. Every other stage of the pipeline drives off
    ``settings.leagues``, so an operator who adds a competition there would
    otherwise have it fetched, predicted, bet on and settled — and then silently
    dropped from the record, which would quietly under-count.

    Compared upper-case: ``data/form.py`` stores whatever the API returns
    verbatim, and settlement/lineups already normalise before their own lookups.
    """
    from betbot.config import LEAGUE_CODES, get_settings

    if not code:
        return False
    allowed = {c.upper() for c in LEAGUE_CODES}
    try:
        allowed |= {c.upper() for c in get_settings().leagues}
    except Exception:  # noqa: BLE001 — settings must never break a read
        pass
    return code.upper() in allowed


_DEGENERATE_P = 1e-4


def _is_degenerate(row: PredictionOutcome) -> bool:
    return min(row.predicted_home, row.predicted_draw, row.predicted_away) < _DEGENERATE_P


def prediction_outcomes_since(days: int) -> list[PredictionOutcome]:
    """Scored predictions settled within the trailing window (newest first).

    Four filters, ALL of which must hold for a row to be reported:

    * settled on/after the accuracy-ledger epoch
      (``BETBOT_ACCURACY_LEDGER_EPOCH``),
    * a non-degenerate probability triple (see ``_DEGENERATE_P``),
    * a CLUB competition (see :func:`_is_club_competition`), and
    * kicked off on/after ``BETBOT_SEASON_START``.

    The first two keep the 0/0/100-AWAY-bug era out of every user-facing
    accuracy figure, including rows the bug era backfilled under a recent
    ``settled_at``. The last two keep the record to what it claims to be —
    this season's club football. Both are needed and neither implies the
    other: the July World Cup ties passed the epoch filter (settled 17 August)
    and were non-degenerate, so only an explicit competition + kickoff test
    excludes them.

    While a season scope is active, a row with no ``kickoff`` is EXCLUDED — it
    cannot be placed in a season, and a record that guesses is worse than one
    that abstains. The startup backfill fills these from ``predictions``, so it
    should be empty in practice. With ``BETBOT_SEASON_START`` unset there is no
    season to place a row in, so undated rows are kept.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    epoch = _ledger_epoch()
    if epoch is not None and epoch > cutoff:
        cutoff = epoch
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PredictionOutcome)
                .where(PredictionOutcome.settled_at >= cutoff)
                .order_by(PredictionOutcome.settled_at.desc())
            ).scalars()
        )
        s.expunge_all()
        clean = [r for r in rows if not _is_degenerate(r)]
        # Captured BEFORE the scope filter below reassigns ``clean``, so the
        # degenerate log reports degenerate rows only. Comparing the final list
        # against ``rows`` would bill every out-of-scope row as ledger poison
        # and mislead the one log line that exists to trace it.
        degenerate = len(rows) - len(clean)
        season = _season_start()
        kept = []
        non_club = undated = pre_season = 0
        for r in clean:
            if not _is_club_competition(r.competition_code):
                non_club += 1
                continue
            if season is not None:
                # Undated rows are dropped ONLY while a season scope is active:
                # the exclusion exists so an undateable row cannot be assigned
                # to a season it may not belong to. With no season configured
                # there is nothing to place it in and nothing to get wrong, so
                # dropping it would just discard a real result.
                ko = r.kickoff
                if ko is None:
                    undated += 1
                    continue
                if ko.tzinfo is None:
                    ko = ko.replace(tzinfo=timezone.utc)
                if ko < season:
                    pre_season += 1
                    continue
            kept.append(r)
        if non_club or undated or pre_season:
            log.info(
                "accuracy_ledger_rows_out_of_scope",
                non_club=non_club,
                undated=undated,
                pre_season=pre_season,
            )
        clean = kept
        if degenerate:
            log.info(
                "accuracy_ledger_degenerate_rows_excluded",
                excluded=degenerate,
            )
        return clean


def track_record(days: int = 30) -> dict:
    """Rolling accuracy over scored predictions in the trailing window.

    Returns TWO metrics that measure different things and must NEVER be merged
    into a single figure by any caller:

    * top level {n, hits, hit_rate, mean_brier, mean_rps, mean_logloss} —
      ALL-MATCH 3-way accuracy over every scored prediction. This is model
      skill; the market closing line sits at ~53-54% on our own data, which is
      the ceiling a sane all-match number lives under.
    * ["called"] {n, hits, hit_rate, ci_lo, ci_hi, call_rate, enabled}
      — hit rate on ONLY the picks the confidence filter actually calls as a
      BET. This is a SELECTION KPI on short-priced favourites. It is NOT edge,
      NOT +EV and NOT evidence of beating the market: backing favourites at a
      fair price is ~0 EV by construction. With the filter flag off, n is 0
      and enabled is False.

    With no data the rates are 0.0 — the caller is responsible for saying
    "sample too small" honestly. Neither figure is CLV or profit.
    """
    from betbot.config import get_settings
    from betbot.strategy.confidence import call_stats

    settings = get_settings()
    rows = prediction_outcomes_since(days)
    n = len(rows)
    stats = call_stats(
        [
            ((r.predicted_home, r.predicted_draw, r.predicted_away), r.actual_outcome)
            for r in rows
        ],
        enabled=bool(settings.club_confidence_filter),
        threshold=float(settings.club_confidence_threshold),
        draw_margin=float(settings.club_confidence_draw_margin),
    )
    called = dict(stats["called"])
    called["enabled"] = bool(settings.club_confidence_filter)
    if n == 0:
        return {
            "n": 0, "hits": 0, "hit_rate": 0.0,
            "mean_brier": 0.0, "mean_rps": 0.0, "mean_logloss": 0.0,
            "called": called,
        }
    hits = sum(1 for r in rows if r.correct)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / n,
        "mean_brier": sum(r.brier for r in rows) / n,
        "mean_rps": sum(r.rps for r in rows) / n,
        "mean_logloss": sum(r.log_loss for r in rows) / n,
        "called": called,
    }


def high_conf_band_tally(min_p: float, *, days: int = 366) -> tuple[int, int]:
    """``(hits, n)`` for settled fixtures that would have cleared the high-conf
    alert gate this season.

    Reuses :func:`prediction_outcomes_since`, so the SAME four scope filters
    apply — ledger epoch, non-degenerate triple, CLUB competition, current
    season kickoff — which is exactly "club-only, current season, World Cup
    excluded, bug-era excluded". A row counts when its stored model top-pick
    probability ``max(predicted_home, predicted_draw, predicted_away)`` is at
    least ``min_p`` AND the top pick is NOT the draw, mirroring
    :func:`betbot.main.high_conf_alert_passes` so the live tally measures the
    same population the band statistics describe.

    ``days`` is a generous trailing window (default a full year); the season /
    epoch filters inside ``prediction_outcomes_since`` do the real scoping, so
    the number is "this season" regardless of the window as long as it spans
    the season start. This is a HIT-RATE tally, never edge/ROI — the caller
    presents it honestly and flags a small sample.
    """
    rows = prediction_outcomes_since(days)
    hits = n = 0
    for r in rows:
        triple = [
            ("HOME", r.predicted_home),
            ("DRAW", r.predicted_draw),
            ("AWAY", r.predicted_away),
        ]
        top_pick, top_p = max(triple, key=lambda kv: kv[1])
        if top_p < min_p or top_pick == "DRAW":
            continue
        n += 1
        if r.correct:
            hits += 1
    return hits, n


def outcome_result_notified(fixture_id: int) -> bool:
    """True once the RESULT ALERT has been broadcast for this fixture."""
    with session_scope() as s:
        row = s.execute(
            select(PredictionOutcome.result_notified)
            .where(PredictionOutcome.fixture_id == fixture_id)
            .limit(1)
        ).scalar_one_or_none()
        return bool(row)


def mark_result_notified(fixture_id: int) -> None:
    """Flag a fixture's result as broadcast so it is never re-sent."""
    with session_scope() as s:
        row = s.execute(
            select(PredictionOutcome)
            .where(PredictionOutcome.fixture_id == fixture_id)
        ).scalar_one_or_none()
        if row is not None:
            row.result_notified = True


def outcomes_pending_result_alert(days: int = 3) -> list[PredictionOutcome]:
    """Recently-settled fixtures whose RESULT ALERT hasn't been sent yet."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PredictionOutcome)
                .where(PredictionOutcome.settled_at >= cutoff)
                .where(PredictionOutcome.result_notified.is_(False))
                .order_by(PredictionOutcome.settled_at.asc())
            ).scalars()
        )
        s.expunge_all()
        return rows


# Outcome scoring only looks back this far. Bounds the every-tick /matches
# re-fetch of fixtures that never reach a final status (postponed/abandoned)
# AND stops a first deploy from backfilling months of pre-outcome-loop history
# (109 fixtures in prod at review time) in one settle run.
_OUTCOME_LOOKBACK_DAYS = 30


def list_unsettled_predictions_due(
    now: datetime, grace_minutes: int
) -> list[PredictionRow]:
    """Freshest prediction per fixture whose kickoff+grace has passed AND which
    has NOT yet been scored into the outcome ledger.

    Drives outcome scoring for EVERY prediction (not just those carrying a
    paper bet). One row per fixture (latest run_date wins) so a fixture is
    scored once. Kickoffs older than ``_OUTCOME_LOOKBACK_DAYS`` are ignored
    forever (never-final fixtures must not be re-fetched every tick for
    eternity, and ancient history must not flood a first deploy).
    """
    cutoff = now - timedelta(minutes=grace_minutes)
    floor = now - timedelta(days=_OUTCOME_LOOKBACK_DAYS)
    with session_scope() as s:
        scored = set(
            s.execute(select(PredictionOutcome.fixture_id)).scalars()
        )
        rows = list(
            s.execute(
                select(PredictionRow)
                .where(PredictionRow.kickoff <= cutoff)
                .where(PredictionRow.kickoff >= floor)
                .order_by(PredictionRow.kickoff.asc())
            ).scalars()
        )
        s.expunge_all()
    best: dict[int, PredictionRow] = {}
    for r in rows:
        if r.fixture_id in scored:
            continue
        cur = best.get(r.fixture_id)
        if cur is None or (r.run_date, r.id) > (cur.run_date, cur.id):
            best[r.fixture_id] = r
    return list(best.values())


def list_recent_predictions(days: int = 7) -> list[PredictionRow]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PredictionRow)
                .where(PredictionRow.created_at >= cutoff)
                .order_by(PredictionRow.kickoff.asc())
            ).scalars()
        )
        s.expunge_all()
        return rows


def list_settled_market_bets(window_days: int | None = None) -> list[PaperBet]:
    """Settled bets that carried a market price (favourite-only bets excluded).

    Ordered oldest-first by settlement time so callers can measure the window
    span. Used by the backtest + gate.
    """
    with session_scope() as s:
        stmt = (
            select(PaperBet)
            .where(PaperBet.settled_at.is_not(None))
            .where(PaperBet.market_price.is_not(None))
            .order_by(PaperBet.settled_at.asc())
        )
        if window_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
            stmt = stmt.where(PaperBet.settled_at >= cutoff)
        rows = list(s.execute(stmt).scalars())
        s.expunge_all()
        return rows


# ----------------------------------------------------------------------
# Daily report queries (scheduled Telegram jobs)
# ----------------------------------------------------------------------
def list_bets_created_between(
    start: datetime, end: datetime
) -> list[tuple[PaperBet, str, str, float | None, float | None]]:
    """``(bet, home_team, away_team, home_xg, away_xg)`` for bets logged in
    ``[start, end)``.

    Team names + xG are joined in here (rather than via the lazy relationship)
    because rows are detached before returning — relationship access on a
    detached row raises DetachedInstanceError.
    """
    with session_scope() as s:
        rows = list(
            s.execute(
                select(
                    PaperBet, PredictionRow.home_team, PredictionRow.away_team,
                    PredictionRow.home_xg, PredictionRow.away_xg,
                )
                .join(PredictionRow, PaperBet.prediction_id == PredictionRow.id)
                .where(PaperBet.created_at >= start)
                .where(PaperBet.created_at < end)
                .order_by(PaperBet.created_at.asc())
            )
        )
        s.expunge_all()
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]


def list_bets_settled_between(
    start: datetime, end: datetime
) -> list[tuple[PaperBet, str, str, float | None, float | None]]:
    """``(bet, home_team, away_team, home_xg, away_xg)`` for bets settled in
    ``[start, end)``."""
    with session_scope() as s:
        rows = list(
            s.execute(
                select(
                    PaperBet, PredictionRow.home_team, PredictionRow.away_team,
                    PredictionRow.home_xg, PredictionRow.away_xg,
                )
                .join(PredictionRow, PaperBet.prediction_id == PredictionRow.id)
                .where(PaperBet.settled_at.is_not(None))
                .where(PaperBet.settled_at >= start)
                .where(PaperBet.settled_at < end)
                .order_by(PaperBet.settled_at.asc())
            )
        )
        s.expunge_all()
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]


def cumulative_realized_pnl_usd() -> float:
    """All-time realised P&L over settled bets.

    No-market (favourite-only) bets settle at 0 by design, so including them
    changes nothing — this is the honest "since inception" number.
    """
    with session_scope() as s:
        rows = s.execute(
            select(PaperBet.pnl_usd).where(PaperBet.settled_at.is_not(None))
        ).scalars()
        return float(sum(r or 0.0 for r in rows))


# ----------------------------------------------------------------------
# Kill switch (Phase 4)
# ----------------------------------------------------------------------
def get_kill_switch() -> KillSwitch:
    """Return the single kill-switch row, creating it (untripped) if absent."""
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        if ks is None:
            ks = KillSwitch(id=1)
            s.add(ks)
            s.flush()
        s.expunge_all()
        return ks


def is_kill_switch_tripped() -> bool:
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        return ks is not None and ks.tripped_at is not None


def trip_kill_switch(
    reason: str, realized_pnl_usd: float, staked_usd: float
) -> None:
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        if ks is None:
            ks = KillSwitch(id=1)
            s.add(ks)
        ks.tripped_at = datetime.now(timezone.utc)
        ks.reason = reason[:300]
        ks.realized_pnl_usd = realized_pnl_usd
        ks.staked_usd = staked_usd


def reset_kill_switch() -> None:
    with session_scope() as s:
        ks = s.get(KillSwitch, 1)
        if ks is None:
            return
        ks.tripped_at = None
        ks.reason = None
        ks.realized_pnl_usd = None
        ks.staked_usd = None


# ----------------------------------------------------------------------
# Glicko ratings (Phase 5.5)
# ----------------------------------------------------------------------
def get_rating(
    team_name: str,
    *,
    default_rating: float = 1500.0,
    default_rd: float = 200.0,
    default_vol: float = 0.06,
) -> Glicko2Rating:
    with session_scope() as s:
        row = s.execute(
            select(GlickoRating).where(GlickoRating.team_name == team_name)
        ).scalar_one_or_none()
        if row is None:
            return Glicko2Rating(default_rating, default_rd, default_vol)
        return Glicko2Rating(row.rating, row.rd, row.volatility, row.last_period)


def rating_exists(team_name: str) -> bool:
    """True iff a REAL Glicko row exists for this team.

    Settlement's per-match nudge checks this so it never fabricates a
    near-default rating row for a team the weekly re-seed doesn't know (e.g. a
    just-promoted club) — such a row would defeat the club engine's
    unknown-team fallback to the form engine.
    """
    with session_scope() as s:
        return (
            s.execute(
                select(GlickoRating.id)
                .where(GlickoRating.team_name == team_name)
                .limit(1)
            ).scalar_one_or_none()
            is not None
        )


def upsert_rating(team_name: str, rating: Glicko2Rating, *, team_id: int | None = None) -> None:
    with session_scope() as s:
        row = s.execute(
            select(GlickoRating).where(GlickoRating.team_name == team_name)
        ).scalar_one_or_none()
        if row is None:
            s.add(GlickoRating(
                team_name=team_name, team_id=team_id, rating=rating.rating,
                rd=rating.rd, volatility=rating.volatility, last_period=rating.last_period,
            ))
        else:
            row.rating, row.rd, row.volatility = rating.rating, rating.rd, rating.volatility
            row.last_period = rating.last_period
            if team_id is not None:
                row.team_id = team_id


def upsert_rating_if_fresher(
    team_name: str, rating: Glicko2Rating, *, team_id: int | None = None
) -> bool:
    """Upsert a rating ONLY when it is at least as fresh as the stored row.

    The weekly re-seed (``scripts/seed_glicko_club.py``) rebuilds every rating
    from the football-data.co.uk history CSV and would otherwise SET each row
    back to the CSV's end-state — clobbering the in-season Glicko nudges that
    settlement applies between re-seeds whenever the CSV lags the live results
    (football-data.co.uk publishes only a few times a week, so mid-week it is
    always behind the DB).

    Freshness is the ISO ``last_period`` date both writers stamp: the re-seed
    stamps the last CSV match date per team; the settlement nudge stamps the
    settle date. We overwrite the rating fields only when the incoming
    ``last_period >=`` the stored one (or the stored row has no period yet), so

    * a re-seed whose history has caught up to (or past) the DB takes over
      cleanly — it is a full-history replay, so there is no double counting; and
    * a re-seed whose CSV still lags a nudged row leaves that row's RATING
      untouched, preserving the in-season learning.

    Creation is never blocked, so the re-seed remains the only writer that
    CREATES ratings (settlement's unknown-team guard relies on that). ``team_id``
    metadata is always refreshed when supplied, even when the rating is kept.

    Returns True iff the rating fields were written.
    """
    with session_scope() as s:
        row = s.execute(
            select(GlickoRating).where(GlickoRating.team_name == team_name)
        ).scalar_one_or_none()
        if row is None:
            s.add(GlickoRating(
                team_name=team_name, team_id=team_id, rating=rating.rating,
                rd=rating.rd, volatility=rating.volatility,
                last_period=rating.last_period,
            ))
            return True
        incoming, existing = rating.last_period, row.last_period
        fresher = existing is None or (incoming is not None and incoming >= existing)
        if fresher:
            row.rating, row.rd, row.volatility = (
                rating.rating, rating.rd, rating.volatility
            )
            row.last_period = incoming
        if team_id is not None:
            row.team_id = team_id
        return fresher


def all_ratings() -> list[tuple[str, Glicko2Rating]]:
    with session_scope() as s:
        rows = list(
            s.execute(select(GlickoRating).order_by(GlickoRating.rating.desc())).scalars()
        )
        return [(r.team_name, Glicko2Rating(r.rating, r.rd, r.volatility, r.last_period))
                for r in rows]


def apply_rating_period(
    matches: list[tuple[str, str, str]],
    period: str,
    *,
    tau: float = 0.5,
    default_rating: float = 1500.0,
    default_rd: float = 200.0,
    default_vol: float = 0.06,
) -> int:
    """Apply one Glicko-2 rating period from ``(home, away, outcome)`` results.

    ``outcome`` is "HOME"/"AWAY"/"DRAW". All updates use opponents' PRE-period
    ratings (correct Glicko-2 semantics), then persist. Returns teams updated.
    """
    teams: set[str] = set()
    for home, away, _ in matches:
        teams.add(home)
        teams.add(away)
    current = {
        t: get_rating(t, default_rating=default_rating, default_rd=default_rd,
                      default_vol=default_vol)
        for t in teams
    }
    per_team: dict[str, list[tuple[float, float, float]]] = {t: [] for t in teams}
    for home, away, outcome in matches:
        sh = 1.0 if outcome == "HOME" else (0.5 if outcome == "DRAW" else 0.0)
        sa = 1.0 if outcome == "AWAY" else (0.5 if outcome == "DRAW" else 0.0)
        per_team[home].append((current[away].rating, current[away].rd, sh))
        per_team[away].append((current[home].rating, current[home].rd, sa))
    for t in teams:
        upsert_rating(t, update_rating(current[t], per_team[t], tau=tau, period=period))
    return len(teams)


# ----------------------------------------------------------------------
# Model dual-log (model_predictions) — passive head-to-head ledger
# ----------------------------------------------------------------------
# Re-armed for the CLUB engine (the World Cup Hedge writer was removed in
# 6abc132). This is a PASSIVE, decision-neutral ledger: it records the pure
# Glicko challenger vs the RAW pre-calibration ensemble the club engine serves,
# and scores each with RPS at settlement. It changes NOTHING about what the
# daemon predicts or trades. It is also the raw-triple store the calibration
# fit trains on (scripts/fit_ensemble_calibration_club.py), which is why the
# ensemble triple stored here is the log-pool BEFORE calibration.
def upsert_model_prediction(
    *,
    fixture_id: int,
    home_team: str,
    away_team: str,
    glicko: tuple[float, float, float],
    ensemble: tuple[float, float, float],
    w_glicko: float,
    w_ensemble: float,
) -> None:
    """Store/refresh one fixture's (pure-Glicko, RAW-ensemble) pre-match pair.

    Upserts on ``fixture_id`` (unique). While the row is unsettled the triples
    are refreshed each scoring tick — mirroring the main prediction path's
    always-fresh rescore. Once ``outcome`` is filled the row is frozen so the
    RPS is scored against the pre-match snapshot that stood at settlement.
    """
    gh, gd, ga = glicko
    eh, ed, ea = ensemble
    with session_scope() as s:
        row = s.execute(
            select(ModelPrediction).where(ModelPrediction.fixture_id == fixture_id)
        ).scalar_one_or_none()
        if row is not None and row.outcome is not None:
            return  # settled — frozen snapshot
        if row is None:
            s.add(ModelPrediction(
                fixture_id=fixture_id, home_team=home_team, away_team=away_team,
                g_home=gh, g_draw=gd, g_away=ga,
                e_home=eh, e_draw=ed, e_away=ea,
                w_glicko=w_glicko, w_ensemble=w_ensemble,
            ))
        else:
            row.home_team, row.away_team = home_team, away_team
            row.g_home, row.g_draw, row.g_away = gh, gd, ga
            row.e_home, row.e_draw, row.e_away = eh, ed, ea
            row.w_glicko, row.w_ensemble = w_glicko, w_ensemble


def score_model_prediction(*, fixture_id: int, actual_outcome: str) -> bool:
    """Fill a dual-log row's outcome + RPS for both models. Idempotent.

    Returns True iff a row was newly scored (present and previously unsettled).
    """
    idx = {"HOME": 0, "DRAW": 1, "AWAY": 2}.get(actual_outcome)
    if idx is None:
        return False
    with session_scope() as s:
        row = s.execute(
            select(ModelPrediction).where(ModelPrediction.fixture_id == fixture_id)
        ).scalar_one_or_none()
        if row is None or row.outcome is not None:
            return False
        row.outcome = actual_outcome
        row.rps_glicko = ranked_probability_score(
            (row.g_home, row.g_draw, row.g_away), idx
        )
        row.rps_ensemble = ranked_probability_score(
            (row.e_home, row.e_draw, row.e_away), idx
        )
        return True


# ----------------------------------------------------------------------
# Deposit pipeline (CCTP bridging — betbot/bridge.py)
# ----------------------------------------------------------------------
# Terminal status: a leg whose funds have been delivered AND venue approvals
# have run. Everything else is "active" and (a) blocks re-detection on its
# source chain and (b) gets resumed by every scan tick.
DEPOSIT_DONE = "done"


def create_deposit(
    *,
    user_id: int,
    wallet_address: str,
    source_chain: str,
    dest_chain: str,
    amount_usdc: float,
    balance_snapshot: float,
    status: str,
) -> int:
    with session_scope() as s:
        row = Deposit(
            user_id=user_id,
            wallet_address=wallet_address,
            source_chain=source_chain,
            dest_chain=dest_chain,
            amount_usdc=amount_usdc,
            balance_snapshot=balance_snapshot,
            status=status,
        )
        s.add(row)
        s.flush()
        return row.id


def has_active_source_deposit(wallet_address: str, source_chain: str) -> bool:
    """True while ANY leg sourced from this (wallet, chain) is unfinished.

    This is the first idempotency guard: a balance seen twice while its
    pipeline is in flight must not create a second deposit record.
    """
    with session_scope() as s:
        row = s.execute(
            select(Deposit.id)
            .where(Deposit.wallet_address == wallet_address)
            .where(Deposit.source_chain == source_chain)
            .where(Deposit.status != DEPOSIT_DONE)
            .limit(1)
        ).scalar_one_or_none()
        return row is not None


def has_active_dest_deposit(wallet_address: str, dest_chain: str) -> bool:
    """True while ANY unfinished leg is bridging TOWARD this (wallet, chain).

    Detection guard: an in-flight leg's mint can land on the destination
    chain before its status persists; detecting balances there while the leg
    is active would record the bridged funds as a phantom new deposit and
    double-count the delivered baseline.
    """
    with session_scope() as s:
        row = s.execute(
            select(Deposit.id)
            .where(Deposit.wallet_address == wallet_address)
            .where(Deposit.dest_chain == dest_chain)
            .where(Deposit.status != DEPOSIT_DONE)
            .limit(1)
        ).scalar_one_or_none()
        return row is not None


def list_active_deposits() -> list[Deposit]:
    """Unfinished legs, oldest first — the scan tick resumes each of these."""
    with session_scope() as s:
        rows = list(
            s.execute(
                select(Deposit)
                .where(Deposit.status != DEPOSIT_DONE)
                .order_by(Deposit.created_at.asc())
            ).scalars()
        )
        s.expunge_all()
        return rows


def delivered_to_chain_usdc(wallet_address: str, dest_chain: str) -> float:
    """USDC already delivered to ``dest_chain`` for this wallet by past legs.

    Used as the detection baseline on TRADING chains (where delivered funds
    stay in the wallet): a new deposit is only the balance ABOVE this number.
    Local legs (source == dest) count from creation — the funds never left;
    bridged legs count once minted. Trading spend pushes the real balance
    below this baseline, which only makes detection more conservative (we
    under-detect rather than ever double-bridge).
    """
    with session_scope() as s:
        rows = list(
            s.execute(
                select(Deposit.amount_usdc, Deposit.source_chain, Deposit.status)
                .where(Deposit.wallet_address == wallet_address)
                .where(Deposit.dest_chain == dest_chain)
            )
        )
    total = 0.0
    for amount, source_chain, status in rows:
        if source_chain == dest_chain or status in ("minted", DEPOSIT_DONE):
            total += amount or 0.0
    return total


def update_deposit(
    deposit_id: int,
    *,
    status: str | None = None,
    burn_tx: str | None = None,
    mint_tx: str | None = None,
    error: str | None = None,
) -> None:
    with session_scope() as s:
        row = s.get(Deposit, deposit_id)
        if row is None:
            return
        if status is not None:
            row.status = status
            row.error = None  # a successful step clears the previous error
        if burn_tx is not None:
            row.burn_tx = burn_tx
        if mint_tx is not None:
            row.mint_tx = mint_tx
        if error is not None:
            row.error = error[:300]


# ----------------------------------------------------------------------
# Treasury rebalancer (agent-owned float; betbot/bridge.py TreasuryRebalancer)
# ----------------------------------------------------------------------
def create_treasury_bridge(
    *, source_chain: str, dest_chain: str, amount_usdc: float, status: str
) -> int:
    with session_scope() as s:
        row = TreasuryBridge(
            source_chain=source_chain,
            dest_chain=dest_chain,
            amount_usdc=amount_usdc,
            status=status,
        )
        s.add(row)
        s.flush()
        return row.id


def active_treasury_bridge() -> TreasuryBridge | None:
    """The single in-flight treasury leg, if any (oldest non-``done`` row).

    This IS the rebalancer's idempotency lock: a non-None result means a
    bridge is already mid-flight, so no new one may start. Detached for use
    after the session closes.
    """
    with session_scope() as s:
        row = s.execute(
            select(TreasuryBridge)
            .where(TreasuryBridge.status != DEPOSIT_DONE)
            .order_by(TreasuryBridge.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()
        if row is not None:
            s.expunge_all()
        return row


def update_treasury_bridge(
    bridge_id: int,
    *,
    status: str | None = None,
    burn_tx: str | None = None,
    mint_tx: str | None = None,
    error: str | None = None,
) -> None:
    with session_scope() as s:
        row = s.get(TreasuryBridge, bridge_id)
        if row is None:
            return
        if status is not None:
            row.status = status
            row.error = None  # a successful step clears the previous error
        if burn_tx is not None:
            row.burn_tx = burn_tx
        if mint_tx is not None:
            row.mint_tx = mint_tx
        if error is not None:
            row.error = error[:300]


# ----------------------------------------------------------------------
# Agent gas spend audit (deposit pipeline abuse guard)
# ----------------------------------------------------------------------
def record_gas_topup(
    *, wallet_address: str, chain: str, amount: float, tx: str | None = None
) -> None:
    """Persist one agent-funded native-gas top-up (see bridge._ensure_gas)."""
    with session_scope() as s:
        s.add(
            GasTopup(
                wallet_address=wallet_address, chain=chain, amount=amount, tx=tx
            )
        )


def count_gas_topups_since(wallet_address: str, since: datetime) -> int:
    """Top-ups this wallet received since ``since`` — feeds the daily cap.

    Persisted (not in-memory) ON PURPOSE: a daemon restart must not reset
    the cap, or an attacker could trigger unlimited agent gas spend by
    timing deposits around restarts.
    """
    with session_scope() as s:
        n = s.execute(
            select(func.count())
            .select_from(GasTopup)
            .where(GasTopup.wallet_address == wallet_address)
            .where(GasTopup.created_at >= since)
        ).scalar_one()
        return int(n)


# Arbitrage executions
# ----------------------------------------------------------------------
def insert_arb_execution(
    *,
    home_team: str,
    away_team: str,
    margin: float,
    price_sum: float,
    stake_usd: float,
    status: str,
    legs_json: str,
    net_expected_usd: float | None = None,
    error: str | None = None,
) -> int:
    """Persist one arb execution attempt; returns the row id."""
    with session_scope() as s:
        row = ArbExecution(
            home_team=home_team[:80],
            away_team=away_team[:80],
            margin=margin,
            price_sum=price_sum,
            stake_usd=stake_usd,
            status=status,
            legs_json=legs_json,
            net_expected_usd=net_expected_usd,
            error=error[:500] if error else None,
        )
        s.add(row)
        s.flush()
        return row.id


def update_arb_execution(
    exec_id: int,
    *,
    status: str | None = None,
    legs_json: str | None = None,
    net_expected_usd: float | None = None,
    error: str | None = None,
) -> None:
    with session_scope() as s:
        row = s.get(ArbExecution, exec_id)
        if row is None:
            return
        if status is not None:
            row.status = status
        if legs_json is not None:
            row.legs_json = legs_json
        if net_expected_usd is not None:
            row.net_expected_usd = net_expected_usd
        if error is not None:
            row.error = error[:500]


def arb_staked_today_usd() -> float:
    """Today's total arb stake over rows where money moved (or may have).

    ``rejected_*`` rows were gated before any order and don't count; everything
    else (executing / aborted / partial / filled) counts conservatively toward
    the BETBOT_ARB_DAILY_CAP_USD daily cap.
    """
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    with session_scope() as s:
        rows = s.execute(
            select(ArbExecution.stake_usd)
            .where(ArbExecution.created_at >= today_start)
            .where(ArbExecution.status.not_like("rejected%"))
        ).scalars()
        return float(sum(rows))


def list_recent_arb_executions(days: int = 7) -> list[ArbExecution]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with session_scope() as s:
        rows = list(
            s.execute(
                select(ArbExecution)
                .where(ArbExecution.created_at >= cutoff)
                .order_by(ArbExecution.created_at.desc())
            ).scalars()
        )
        s.expunge_all()
        return rows


# ----------------------------------------------------------------------
# Multi-user accounts (each user has their OWN isolated wallet)
# ----------------------------------------------------------------------
def get_or_create_user(
    telegram_user_id: int, name: str, *, secrets_dir: str, keyfile: str | None = None
) -> User:
    """Return the user, generating a fresh per-user wallet on first contact.

    The wallet key lives at ``<secrets_dir>/users/<telegram_id>.key`` (0600,
    gitignored) unless ``keyfile`` overrides it — used to map the operator onto
    the existing agent wallet so their prior deposit isn't stranded. Funds stay
    in each user's own wallet — never pooled.
    """
    from pathlib import Path

    from betbot.wallet import get_or_create_address

    keyfile = Path(keyfile) if keyfile else Path(secrets_dir) / "users" / f"{telegram_user_id}.key"
    with session_scope() as s:
        u = s.execute(
            select(User).where(User.telegram_user_id == telegram_user_id)
        ).scalar_one_or_none()
        if u is None:
            address = get_or_create_address(keyfile)
            u = User(
                telegram_user_id=telegram_user_id, name=name[:80],
                wallet_address=address, wallet_keyfile=str(keyfile), active=True,
            )
            s.add(u)
            s.flush()
        s.expunge_all()
        return u


def get_user(telegram_user_id: int) -> User | None:
    with session_scope() as s:
        u = s.execute(
            select(User).where(User.telegram_user_id == telegram_user_id)
        ).scalar_one_or_none()
        if u is not None:
            s.expunge_all()
        return u


def list_users() -> list[User]:
    with session_scope() as s:
        rows = list(
            s.execute(select(User).where(User.active.is_(True))
                      .order_by(User.created_at.asc())).scalars()
        )
        s.expunge_all()
        return rows


def increment_predictions_consumed(telegram_user_id: int) -> None:
    """Charge one paid prediction reveal to this user (post-trial billing).

    Called by the caller ONLY after a prediction is actually revealed — the
    entitlement decision itself never mutates. No-op if the user is unknown.
    """
    with session_scope() as s:
        u = s.execute(
            select(User).where(User.telegram_user_id == telegram_user_id)
        ).scalar_one_or_none()
        if u is None:
            return
        u.predictions_consumed = (u.predictions_consumed or 0) + 1


# ----------------------------------------------------------------------
# Reveal ledger (per-(user, fixture) money idempotency)
# ----------------------------------------------------------------------
def has_revealed(telegram_user_id: int, fixture_id: int) -> bool:
    """True if this fixture has already been revealed to this user.

    An already-revealed fixture is re-shown FREE and never re-charged — this is
    the guard that stops the matchday alert, the kickoff alert, and every
    repeat of ``/predictions`` from billing the same fixture more than once.
    """
    with session_scope() as s:
        row = s.execute(
            select(PredictionReveal.id)
            .where(PredictionReveal.telegram_user_id == telegram_user_id)
            .where(PredictionReveal.fixture_id == fixture_id)
            .limit(1)
        ).scalar_one_or_none()
        return row is not None


def fixture_was_ever_revealed(fixture_id: int) -> bool:
    """True if ANY user has a reveal-ledger row for this fixture.

    Read-only, user-AGNOSTIC mirror of :func:`has_revealed` (which is scoped to
    a single ``telegram_user_id``). The result path uses it to honour the
    product promise "high confidence AT ALERT TIME": ``upsert_prediction``
    overwrites the stored ``p_*`` triple in place on every rescore, so a fixture
    that cleared the confidence bar when we alerted can read below it by
    settlement. If it was ever revealed to anyone, its outcome is still owed.
    """
    with session_scope() as s:
        row = s.execute(
            select(PredictionReveal.id)
            .where(PredictionReveal.fixture_id == fixture_id)
            .limit(1)
        ).scalar_one_or_none()
        return row is not None


def record_reveal(
    telegram_user_id: int, fixture_id: int, charged: bool
) -> bool:
    """Record that this fixture was revealed to this user. Idempotent.

    Returns ``True`` if a NEW ledger row was inserted, ``False`` if one already
    existed (the unique constraint fired). Callers increment the paid-credit
    counter only when this returns True AND ``charged`` is set, so committing a
    reveal twice (e.g. a retried Telegram send) can never double-charge.
    """
    try:
        with session_scope() as s:
            s.add(
                PredictionReveal(
                    telegram_user_id=telegram_user_id,
                    fixture_id=fixture_id,
                    charged=charged,
                )
            )
        return True
    except IntegrityError:
        # Row already exists (uq_reveal_user_fixture) — safe no-op.
        return False


# ----------------------------------------------------------------------
# Prediction delivery queries (tipster alerts + /predictions)
# ----------------------------------------------------------------------
def predictions_for_kickoff_range(
    start_dt: datetime, end_dt: datetime
) -> list[PredictionRow]:
    """Predictions whose kickoff is in ``[start_dt, end_dt)``, earliest first.

    ONE row per fixture — the latest ``run_date`` wins. The scoring window is
    48h, so a fixture is re-scored on consecutive daily runs and the table
    holds 2-3 rows per fixture (unique on (fixture_id, run_date)); without this
    dedup the tipster message would list the same fixture multiple times AND
    (money bug) charge a paying user one credit per duplicate row in a single
    render.

    Rows are detached; the linked paper_bet (our recommendation) is eager-loaded
    so :func:`betbot.tips.format_prediction` can read it after the session
    closes without a DetachedInstanceError.
    """
    from sqlalchemy.orm import selectinload

    with session_scope() as s:
        rows = list(
            s.execute(
                select(PredictionRow)
                .where(PredictionRow.kickoff >= start_dt)
                .where(PredictionRow.kickoff < end_dt)
                .order_by(PredictionRow.kickoff.asc())
                .options(selectinload(PredictionRow.paper_bets))
            ).scalars()
        )
        s.expunge_all()
    # Keep only the freshest row per fixture (max run_date; id breaks ties),
    # preserving kickoff order.
    best: dict[int, PredictionRow] = {}
    for r in rows:
        cur = best.get(r.fixture_id)
        if cur is None or (r.run_date, r.id) > (cur.run_date, cur.id):
            best[r.fixture_id] = r
    return [r for r in rows if best[r.fixture_id] is r]


def prediction_for_fixture(fixture_id: int) -> PredictionRow | None:
    """The most recent prediction for ``fixture_id`` (latest run_date wins).

    Detached, with its paper_bets eager-loaded — used by the per-fixture
    kickoff-60m alert.
    """
    from sqlalchemy.orm import selectinload

    with session_scope() as s:
        row = s.execute(
            select(PredictionRow)
            .where(PredictionRow.fixture_id == fixture_id)
            .order_by(PredictionRow.run_date.desc())
            .options(selectinload(PredictionRow.paper_bets))
            .limit(1)
        ).scalar_one_or_none()
        if row is not None:
            s.expunge_all()
        return row




def upcoming_prediction_fixtures(
    start_dt: datetime, end_dt: datetime, *, exclude_settled: bool = True
) -> dict[int, tuple[datetime, str]]:
    """``{fixture_id: (stored_kickoff, competition_code)}`` over a kickoff range.

    The competition code rides along because the kickoff re-sync needs to know
    which league window a fixture belongs to: when a league's window fetch
    fails, its fixtures must NOT be mistaken for "moved out of the window" and
    sent down the per-fixture fallback.

    The freshest row per fixture wins (max ``run_date``, id breaks ties), so the
    map holds exactly the kickoff the alert scheduler would plan against. Used
    by the kickoff re-sync to spot fixtures upstream has since moved.

    ``exclude_settled`` drops fixtures already in the outcome ledger. The range
    reaches into the PAST — a match moved forward looks long overdue on our
    clock, which is precisely the row that needs correcting — and without this
    every finished fixture in that window would be re-compared for nothing.
    """
    with session_scope() as s:
        settled: set[int] = (
            set(s.execute(select(PredictionOutcome.fixture_id)).scalars())
            if exclude_settled
            else set()
        )
        rows = list(
            s.execute(
                select(
                    PredictionRow.fixture_id,
                    PredictionRow.kickoff,
                    PredictionRow.run_date,
                    PredictionRow.id,
                    PredictionRow.competition_code,
                )
                .where(PredictionRow.kickoff >= start_dt)
                .where(PredictionRow.kickoff < end_dt)
            )
        )
    best: dict[int, tuple[str, int, datetime, str]] = {}
    for fixture_id, kickoff, run_date, row_id, code in rows:
        if fixture_id in settled:
            continue
        cur = best.get(fixture_id)
        if cur is None or (run_date, row_id) > (cur[0], cur[1]):
            best[fixture_id] = (run_date, row_id, kickoff, code)
    # SQLite hands back naive datetimes; every caller compares these against
    # aware upstream kickoffs, and a tz mismatch there reads as "moved".
    return {
        fid: (
            ko.replace(tzinfo=timezone.utc) if ko.tzinfo is None else ko,
            code,
        )
        for fid, (_rd, _id, ko, code) in best.items()
    }


def update_prediction_kickoff(fixture_id: int, kickoff: datetime) -> int:
    """Point EVERY stored row for ``fixture_id`` at ``kickoff``. Returns rows hit.

    All rows, not just the freshest: a fixture is re-scored on consecutive daily
    runs, so it carries 2-3 rows, and the stale ones drive
    :func:`list_unsettled_predictions_due` (settlement would keep asking for a
    result at a time the match no longer kicks off at). Rows already on the new
    time are left alone so the count reports real changes.
    """
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PredictionRow).where(
                    PredictionRow.fixture_id == fixture_id
                )
            ).scalars()
        )
        changed = 0
        for row in rows:
            stored = row.kickoff
            if stored is not None and stored.tzinfo is None:
                stored = stored.replace(tzinfo=timezone.utc)
            if stored == kickoff:
                continue
            row.kickoff = kickoff
            changed += 1
        return changed
