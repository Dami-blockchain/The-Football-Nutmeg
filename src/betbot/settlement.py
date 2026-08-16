"""Settlement + P&L + drawdown kill switch (Phase 4).

After matches finish, :class:`SettlementWatcher` walks unsettled bets whose
kickoff + grace has passed, fetches the result from football-data, computes
realized P&L, and writes it back. It then evaluates the trailing-window
drawdown and trips the kill switch if losses breach the configured threshold
(with a minimum-staked floor so a tiny-but-unlucky sample can't trip it).

P&L convention (per $1 staked on a YES share priced ``p``):
    win  -> stake * (1/p - 1)     (you paid p, it pays out 1)
    loss -> -stake
No-market (favourite-only) bets settle at **0** — they have no real-money
equivalent and are excluded from the kill-switch signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from betbot.data.football_data import FootballDataClient
from betbot.logging import get_logger
from betbot.storage.repos import (
    get_rating,
    is_kill_switch_tripped,
    list_unsettled_bets_due,
    list_unsettled_predictions_due,
    record_prediction_outcome,
    record_settlement,
    settled_pnl_window,
    trip_kill_switch,
    upsert_rating,
)
from betbot.strategy.engine import Outcome
from betbot.strategy.glicko import update_rating

log = get_logger(__name__)

# football-data statuses that mean the result is final.
SETTLED_STATUSES = frozenset({"FINISHED", "AWARDED"})

# score.winner -> our Outcome.
_WINNER_TO_OUTCOME = {
    "HOME_TEAM": Outcome.HOME,
    "AWAY_TEAM": Outcome.AWAY,
    "DRAW": Outcome.DRAW,
}

# Competitions whose per-match Glicko ratings we nudge from the result between
# the weekly full re-seed: the five domestic top leagues + the Champions League.
# (The weekly re-seed rebuilds from full history and stays authoritative; these
# per-match updates just keep ratings responsive in between.)
_RATED_COMPETITIONS = frozenset({"PL", "PD", "BL1", "SA", "FL1", "CL"})


def _final_goals(match: dict) -> tuple[int, int]:
    """``(home_goals, away_goals)`` from a finished football-data match dict.

    Reads ``score.fullTime`` (the real API shape); falls back to 0-0 when the
    score block is absent (e.g. a winner-only test fixture), which never
    corrupts scoring because the OUTCOME is derived from ``score.winner``.
    """
    ft = (match.get("score") or {}).get("fullTime") or {}
    try:
        hg = int(ft.get("home") or 0)
        ag = int(ft.get("away") or 0)
    except (TypeError, ValueError):
        return 0, 0
    return hg, ag


def _update_ratings_for_result(
    competition_code: str, home_team: str, away_team: str, outcome: str, period: str
) -> None:
    """Nudge the two teams' Glicko ratings from one finished result.

    Both updates use the opponents' PRE-match ratings (correct Glicko-2
    semantics). Best-effort: any failure is swallowed by the caller so a rating
    error never aborts settlement.
    """
    home_rating = get_rating(home_team)
    away_rating = get_rating(away_team)
    sh = 1.0 if outcome == "HOME" else (0.5 if outcome == "DRAW" else 0.0)
    sa = 1.0 if outcome == "AWAY" else (0.5 if outcome == "DRAW" else 0.0)
    new_home = update_rating(
        home_rating, [(away_rating.rating, away_rating.rd, sh)], period=period
    )
    new_away = update_rating(
        away_rating, [(home_rating.rating, home_rating.rd, sa)], period=period
    )
    upsert_rating(home_team, new_home)
    upsert_rating(away_team, new_away)


def compute_pnl(
    market_price: float | None,
    bet_outcome: str,
    settled_outcome: str,
    stake_usd: float,
) -> float:
    """Realized P&L for one settled bet (pure; no DB).

    ``market_price is None`` (favourite-only bet) -> 0.0. An out-of-range
    price (<=0 or >=1) on a winning bet -> 0.0 rather than a nonsense payout.
    """
    if market_price is None:
        return 0.0
    if bet_outcome != settled_outcome:
        return -float(stake_usd)
    if not (0.0 < market_price < 1.0):
        return 0.0
    return float(stake_usd) * (1.0 / market_price - 1.0)


@dataclass(frozen=True)
class SettlementSummary:
    settled: int
    skipped_in_play: int
    skipped_no_result: int
    kill_switch_tripped: bool
    window_pnl_usd: float
    window_staked_usd: float


class SettlementWatcher:
    def __init__(self, client: FootballDataClient, settings) -> None:
        self._client = client
        self._settings = settings

    async def settle_due(self, now: datetime | None = None) -> SettlementSummary:
        now = now or datetime.now(timezone.utc)
        s = self._settings
        due = list_unsettled_bets_due(now, s.settle_grace_minutes)

        settled = in_play = no_result = 0
        for bet in due:
            try:
                match = await self._client.get_match(bet.fixture_id)
            except Exception as e:  # noqa: BLE001 — one bad fetch shouldn't stop the run
                log.warning("settle_fetch_failed", fixture_id=bet.fixture_id, error=str(e))
                no_result += 1
                continue
            if match is None:
                no_result += 1
                continue
            if match.get("status") not in SETTLED_STATUSES:
                in_play += 1
                continue
            winner = (match.get("score") or {}).get("winner")
            outcome = _WINNER_TO_OUTCOME.get(winner)
            if outcome is None:
                # Final status but winner not populated yet — retry next run.
                no_result += 1
                continue
            pnl = compute_pnl(bet.market_price, bet.outcome, outcome.value, bet.stake_usd)
            record_settlement(bet.id, outcome.value, pnl, now)
            settled += 1
            log.info(
                "bet_settled",
                fixture_id=bet.fixture_id,
                bet_outcome=bet.outcome,
                result=outcome.value,
                pnl_usd=round(pnl, 2),
            )

        # --- Outcome ledger + per-match rating learning (ALL predictions) -----
        # Runs independently of bets: every finished fixture with a stored
        # prediction is scored vs reality (idempotent on fixture_id) and, for
        # rated competitions, nudges the two teams' Glicko ratings ONCE.
        await self._score_outcomes(now)

        tripped, pnl_w, staked_w = self._evaluate_kill_switch()
        log.info(
            "settlement_done",
            settled=settled,
            in_play=in_play,
            no_result=no_result,
            window_pnl_usd=round(pnl_w, 2),
            window_staked_usd=round(staked_w, 2),
            kill_switch_tripped=tripped,
        )
        return SettlementSummary(settled, in_play, no_result, tripped, pnl_w, staked_w)

    async def _score_outcomes(self, now: datetime) -> int:
        """Score every finished, un-scored prediction vs its result.

        For each fixture: fetch the match, if final derive the outcome from
        ``score.winner``, INSERT-OR-IGNORE the scored outcome (idempotent), and
        — only on a NEWLY-inserted row for a rated competition — nudge the two
        teams' Glicko ratings once. Returns the count newly scored. Best-effort
        throughout: neither a fetch nor a rating error aborts settlement.
        """
        due = list_unsettled_predictions_due(now, self._settings.settle_grace_minutes)
        scored = 0
        for pred in due:
            try:
                match = await self._client.get_match(pred.fixture_id)
            except Exception as e:  # noqa: BLE001 — one bad fetch mustn't stop the run
                log.warning("outcome_fetch_failed", fixture_id=pred.fixture_id, error=str(e))
                continue
            if match is None or match.get("status") not in SETTLED_STATUSES:
                continue
            outcome = _WINNER_TO_OUTCOME.get((match.get("score") or {}).get("winner"))
            if outcome is None:
                continue  # final status but winner not populated yet — retry
            hg, ag = _final_goals(match)
            newly = record_prediction_outcome(
                fixture_id=pred.fixture_id,
                competition_code=pred.competition_code,
                p_home=pred.p_home,
                p_draw=pred.p_draw,
                p_away=pred.p_away,
                actual_outcome=outcome.value,
                home_goals=hg,
                away_goals=ag,
                settled_at=now,
            )
            if not newly:
                continue  # already scored — idempotent (no double rating update)
            scored += 1
            log.info(
                "prediction_scored",
                fixture_id=pred.fixture_id,
                pick=("HOME", "DRAW", "AWAY")[
                    max(range(3), key=lambda i: (pred.p_home, pred.p_draw, pred.p_away)[i])
                ],
                result=outcome.value,
            )
            if (pred.competition_code or "").upper() in _RATED_COMPETITIONS:
                try:
                    period = now.date().isoformat()
                    _update_ratings_for_result(
                        pred.competition_code, pred.home_team, pred.away_team,
                        outcome.value, period,
                    )
                    log.info(
                        "rating_updated",
                        fixture_id=pred.fixture_id,
                        home=pred.home_team, away=pred.away_team,
                        result=outcome.value,
                    )
                except Exception as e:  # noqa: BLE001 — never abort settlement
                    log.warning(
                        "rating_update_failed",
                        fixture_id=pred.fixture_id, error=str(e),
                    )
        return scored

    def _evaluate_kill_switch(self) -> tuple[bool, float, float]:
        s = self._settings
        pnl, staked = settled_pnl_window(s.drawdown_window_days)
        if (
            not is_kill_switch_tripped()
            and staked >= s.drawdown_min_staked_usd
            and pnl < -s.drawdown_kill_pct * staked
        ):
            reason = (
                f"drawdown ${pnl:.2f} over {s.drawdown_window_days}d on "
                f"${staked:.0f} staked (limit -{s.drawdown_kill_pct:.0%})"
            )
            trip_kill_switch(reason, pnl, staked)
            log.error("kill_switch_tripped", reason=reason)
        return is_kill_switch_tripped(), pnl, staked
