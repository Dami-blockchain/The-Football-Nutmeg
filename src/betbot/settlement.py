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
from betbot.config import INTERNATIONAL_COMPETITIONS
from betbot.storage.repos import (
    apply_rating_period,
    is_kill_switch_tripped,
    list_unsettled_bets_due,
    record_settlement,
    score_model_prediction,
    settled_pnl_window,
    trip_kill_switch,
)
from betbot.strategy.engine import Outcome

log = get_logger(__name__)

# football-data statuses that mean the result is final.
SETTLED_STATUSES = frozenset({"FINISHED", "AWARDED"})

# score.winner -> our Outcome.
_WINNER_TO_OUTCOME = {
    "HOME_TEAM": Outcome.HOME,
    "AWAY_TEAM": Outcome.AWAY,
    "DRAW": Outcome.DRAW,
}


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
        wc_results: dict[int, tuple[str, str, str]] = {}  # fixture_id -> (home, away, outcome)
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

            # Collect international results to update Glicko ratings (once/fixture).
            comp = (match.get("competition") or {}).get("code")
            if comp in INTERNATIONAL_COMPETITIONS and bet.fixture_id not in wc_results:
                home = (match.get("homeTeam") or {}).get("name")
                away = (match.get("awayTeam") or {}).get("name")
                if home and away:
                    wc_results[bet.fixture_id] = (home, away, outcome.value)
            log.info(
                "bet_settled",
                fixture_id=bet.fixture_id,
                bet_outcome=bet.outcome,
                result=outcome.value,
                pnl_usd=round(pnl, 2),
            )

        # Update Glicko ratings from this matchday's international results.
        if wc_results:
            try:
                n = apply_rating_period(
                    list(wc_results.values()),
                    period=now.date().isoformat(),
                    tau=self._settings.glicko_tau,
                    default_rating=self._settings.glicko_default_rating,
                    default_rd=self._settings.glicko_default_rd,
                    default_vol=self._settings.glicko_default_vol,
                )
                log.info("glicko_period_applied", fixtures=len(wc_results), teams=n)
            except Exception as e:  # noqa: BLE001
                log.error("glicko_period_failed", error=str(e))
            # Score the dual-logged model race (Hedge) for these fixtures.
            for fid, (_h, _a, result) in wc_results.items():
                try:
                    if score_model_prediction(fid, result):
                        log.info("model_race_scored", fixture_id=fid, result=result)
                except Exception as e:  # noqa: BLE001
                    log.error("model_race_score_failed", fixture_id=fid, error=str(e))

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
