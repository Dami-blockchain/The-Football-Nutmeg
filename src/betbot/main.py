"""The Football Smart Manager — CLI entrypoint.

Commands (run as ``tfsm <command>`` or ``betbot <command>``):
    tfsm run-once      Score the next 48h of fixtures and log paper bets.
    tfsm run-daemon    Schedule run-once daily at 08:00 UTC.
    tfsm bets list     Print recent paper bets to stdout.
    tfsm init-db       Create the SQLite schema (called automatically).
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient, FootballDataError
from betbot.data.form import FormService, _parse_kickoff, _parse_team
from betbot.exchanges.limitless import LimitlessAdapter
from betbot.exchanges.limitless_client import LimitlessClient
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket import PolymarketAdapter
from betbot.exchanges.polymarket_gamma import GammaClient
from betbot.exchanges.router import ExchangeRouter
from betbot.backtest import backtest_mock, backtest_stored
from betbot.gate import evaluate_gate
from betbot.logging import configure_logging, get_logger
from betbot.settlement import SettlementWatcher
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    daily_paper_exposure_usd,
    get_kill_switch,
    insert_paper_bet,
    insert_paper_bet_no_market,
    is_kill_switch_tripped,
    list_recent_paper_bets,
    reset_kill_switch,
    upsert_prediction,
)
from betbot.strategy.engine import StrategyEngine

# Repo root (…/tfsm), used to locate config/team_aliases.yaml regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
bets_app = typer.Typer(help="Inspect logged paper bets.")
app.add_typer(bets_app, name="bets")
ks_app = typer.Typer(help="Inspect / reset the drawdown kill switch.")
app.add_typer(ks_app, name="kill-switch")


# ----------------------------------------------------------------------
# Exchange routing
# ----------------------------------------------------------------------
def _build_router(settings) -> tuple[ExchangeRouter, list]:
    """Construct the exchange router for a run.

    Returns the router and the list of HTTP clients the caller must close.
    ``enable_orders`` stays False through Phase 3 (paper only) — live ordering
    is wired in Phase 5; each adapter's double-gate keeps place_order inert
    regardless. Limitless football coverage is sparse and US IPs are geo-blocked
    (surfaced as LimitlessGeoBlockedError, which the router isolates per-adapter),
    so most runs route on Polymarket alone — expected, not an error.
    """
    resolver = TeamAliasResolver.from_yaml(_REPO_ROOT / "config" / "team_aliases.yaml")

    gamma = GammaClient()
    polymarket = PolymarketAdapter(gamma, resolver, enable_orders=False, mode=settings.mode)

    limitless_client = LimitlessClient()
    limitless = LimitlessAdapter(
        limitless_client, resolver, enable_orders=False, mode=settings.mode
    )

    router = ExchangeRouter([polymarket, limitless])
    return router, [gamma, limitless_client]


# ----------------------------------------------------------------------
# Scoring run
# ----------------------------------------------------------------------
async def _score_once() -> int:
    """Pull fixtures in the next 48h, score each, log paper bets."""
    settings = get_settings()
    log = get_logger(__name__)
    init_engine(settings.db_path)

    kill_tripped = is_kill_switch_tripped()
    if kill_tripped:
        log.warning(
            "kill_switch_active",
            note="predictions still recorded, but NO new bets will be logged",
        )

    log.info(
        "starting_scoring_run",
        mode=settings.mode,
        leagues=list(settings.leagues),
        kill_switch_tripped=kill_tripped,
    )

    today = date.today()
    date_from = today.isoformat()
    # +2: football-data dateTo is exclusive, so +2 = include tomorrow.
    date_to = (today + timedelta(days=2)).isoformat()

    paper_bets_logged = 0
    router, _http_clients = _build_router(settings)

    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        form_service = FormService(client, settings)
        engine = StrategyEngine(settings)

        try:
            for league in settings.leagues:
                try:
                    matches = await client.list_scheduled_matches(
                        league, date_from, date_to
                    )
                except FootballDataError as e:
                    log.warning("league_fetch_failed", league=league, error=str(e))
                    continue

                log.info(
                    "league_fetched",
                    league=league,
                    upcoming=len(matches),
                    window=f"{date_from}..{date_to}",
                )
                for m in matches:
                    try:
                        bets = await _score_and_log_one(
                            m, league, form_service, engine, router,
                            settings, kill_tripped,
                        )
                        paper_bets_logged += bets
                    except FootballDataError as e:
                        log.warning(
                            "fixture_score_failed",
                            league=league,
                            match_id=m.get("id"),
                            error=str(e),
                        )
                    except Exception as e:  # noqa: BLE001 — defensive
                        log.error(
                            "fixture_score_unexpected",
                            league=league,
                            match_id=m.get("id"),
                            error=str(e),
                        )
        finally:
            for client in _http_clients:
                await client.close()

    log.info(
        "scoring_run_done",
        paper_bets=paper_bets_logged,
        daily_exposure_usd=round(daily_paper_exposure_usd(), 2),
    )
    return paper_bets_logged


async def _score_and_log_one(
    match: dict,
    league: str,
    form_service: FormService,
    engine: StrategyEngine,
    router: ExchangeRouter,
    settings,
    kill_tripped: bool = False,
) -> int:
    """Score one fixture and log a paper bet.

    Tri-state routing on the model's favourite outcome:

    * a market quote with sufficient edge  → log a market-priced paper bet;
    * a market quote with no edge           → the market vetoes; log NOTHING
      (no favourite fallback);
    * no market quote at all                → fall through to a Phase-1
      favourite-only paper bet.
    """
    log = get_logger(__name__)
    fixture_id = int(match["id"])
    kickoff = _parse_kickoff(match["utcDate"])
    home = _parse_team(match["homeTeam"])
    away = _parse_team(match["awayTeam"])

    fixture_form = await form_service.fixture_form(
        fixture_id=fixture_id,
        competition_code=league,
        kickoff=kickoff,
        home_team=home,
        away_team=away,
    )

    prediction = engine.predict(fixture_form)
    log.info(
        "prediction",
        fixture_id=fixture_id,
        league=league,
        home=prediction.home_team,
        away=prediction.away_team,
        kickoff=kickoff.isoformat(),
        p_home=round(prediction.p_home, 3),
        p_draw=round(prediction.p_draw, 3),
        p_away=round(prediction.p_away, 3),
        home_form=fixture_form.home_form.weighted_points,
        away_form=fixture_form.away_form.weighted_points,
    )

    pred_id = upsert_prediction(prediction, kickoff=kickoff)

    # Kill switch: predictions are still recorded above (useful data), but we
    # refuse to log any new bet while the drawdown kill switch is tripped.
    if kill_tripped:
        log.debug("bet_suppressed_kill_switch", fixture_id=fixture_id)
        return 0

    # Risk gate: stop if today's exposure has already hit the cap.
    if (
        daily_paper_exposure_usd() + settings.fixed_stake_usd
        > settings.daily_exposure_cap_usd
    ):
        log.warning(
            "exposure_cap_reached",
            cap_usd=settings.daily_exposure_cap_usd,
        )
        return 0

    favourite = prediction.best_outcome

    # ---- Market route (Phase 2) --------------------------------------
    quote = await router.find_best_quote(
        prediction.home_team, prediction.away_team, kickoff, favourite
    )
    if quote is not None:
        decision = engine.decide_with_market(prediction, favourite, quote.yes_price)
        if decision is None:
            # no_edge: the market price vetoes this bet. Do NOT fall back to
            # favourite-only logging — the market is the better predictor here.
            log.info(
                "route_no_edge",
                fixture_id=fixture_id,
                outcome=favourite.value,
                market_price=round(quote.yes_price, 3),
                exchange=quote.exchange.value,
            )
            return 0
        inserted = insert_paper_bet(decision, pred_id)
        if inserted:
            log.info(
                "paper_bet_logged_market",
                fixture_id=fixture_id,
                outcome=decision.outcome.value,
                market_price=round(decision.market_price, 3),
                edge=round(decision.edge, 3),
                exchange=quote.exchange.value,
                stake_usd=decision.stake_usd,
            )
            return 1
        log.debug("paper_bet_already_logged", fixture_id=fixture_id)
        return 0

    # ---- No market: Phase-1 favourite-only paper bet -----------------
    p = max(prediction.p_home, prediction.p_draw, prediction.p_away)
    rationale = (
        f"Favourite paper bet (no market found): "
        f"P({favourite.value})={p:.3f}; "
        f"home_form_w={fixture_form.home_form.weighted_points:.2f}, "
        f"away_form_w={fixture_form.away_form.weighted_points:.2f}"
    )
    inserted = insert_paper_bet_no_market(
        prediction=prediction,
        prediction_id=pred_id,
        outcome=favourite,
        stake_usd=settings.fixed_stake_usd,
        rationale=rationale,
    )
    if inserted:
        log.info(
            "paper_bet_logged",
            fixture_id=fixture_id,
            outcome=favourite.value,
            stake_usd=settings.fixed_stake_usd,
        )
        return 1
    log.debug("paper_bet_already_logged", fixture_id=fixture_id)
    return 0


# ----------------------------------------------------------------------
# Settlement run
# ----------------------------------------------------------------------
async def _settle_once():
    """Settle finished bets, compute P&L, evaluate the kill switch."""
    settings = get_settings()
    init_engine(settings.db_path)
    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        watcher = SettlementWatcher(client, settings)
        return await watcher.settle_due()


# ----------------------------------------------------------------------
# CLI commands
# ----------------------------------------------------------------------
@app.command("run-once")
def run_once() -> None:
    """Score the next 48h of fixtures and log paper bets."""
    settings = get_settings()
    configure_logging(settings.log_level)
    n = asyncio.run(_score_once())
    typer.echo(f"Logged {n} paper bet(s).")


@app.command("settle")
def settle_cmd() -> None:
    """Settle finished bets, compute P&L, and update the kill switch."""
    settings = get_settings()
    configure_logging(settings.log_level)
    summary = asyncio.run(_settle_once())
    typer.echo(
        f"Settled {summary.settled} bet(s) "
        f"({summary.skipped_in_play} in-play, {summary.skipped_no_result} no-result). "
        f"Trailing-{settings.drawdown_window_days}d P&L "
        f"${summary.window_pnl_usd:.2f} on ${summary.window_staked_usd:.0f} staked. "
        f"Kill switch: {'TRIPPED' if summary.kill_switch_tripped else 'clear'}."
    )


@app.command("backtest")
def backtest_cmd(
    mode: Annotated[str, typer.Option(help="stored | mock")] = "stored",
    window: Annotated[
        int | None, typer.Option(help="trailing days (stored mode only)")
    ] = None,
) -> None:
    """Backtest the strategy: replay settled bets, or a synthetic diagnostic."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    if mode == "mock":
        r = backtest_mock(edge_threshold=settings.edge_threshold)
    else:
        r = backtest_stored(window)
    typer.echo(
        f"[{mode}] n={r.n}  hit={r.hit_rate:.1%}  ROI={r.roi:+.1%}  "
        f"Brier={r.brier:.3f}  P&L=${r.pnl_usd:+.2f} on ${r.staked_usd:.0f}"
    )
    for outcome, st in sorted(r.per_outcome.items()):
        typer.echo(
            f"   {outcome:<4} n={st.n:>3}  hit={st.hit_rate:.1%}  ROI={st.roi:+.1%}"
        )


@app.command("gate")
def gate_cmd() -> None:
    """Check whether the paper record clears the live-trading gate."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    g = evaluate_gate(settings)
    if g.passed:
        typer.echo("GATE: PASS — paper record clears the live-trading thresholds.")
    else:
        typer.echo("GATE: FAIL")
        for reason in g.reasons:
            typer.echo(f"   - {reason}")
    typer.echo(
        f"  (n={g.result.n}, hit={g.result.hit_rate:.1%}, ROI={g.result.roi:+.1%}, "
        f"window={g.window_days_observed:.1f}d, "
        f"kill_switch={'TRIPPED' if g.kill_switch_tripped else 'clear'})"
    )


@ks_app.command("status")
def kill_switch_status() -> None:
    """Show the drawdown kill-switch state."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    ks = get_kill_switch()
    if ks.tripped_at is None:
        typer.echo("Kill switch: CLEAR")
        return
    typer.echo(f"Kill switch: TRIPPED at {ks.tripped_at:%Y-%m-%d %H:%M} UTC")
    typer.echo(f"  reason: {ks.reason}")
    typer.echo(
        f"  realized P&L ${ks.realized_pnl_usd:.2f} on ${ks.staked_usd:.0f} staked"
    )
    typer.echo("  run `tfsm kill-switch reset` to resume betting.")


@ks_app.command("reset")
def kill_switch_reset() -> None:
    """Clear a tripped kill switch so the bot resumes logging bets."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    reset_kill_switch()
    typer.echo("Kill switch reset to CLEAR.")


@app.command("run-daemon")
def run_daemon(
    cron: Annotated[
        str | None,
        typer.Option(help="Cron expression (UTC). Defaults to settings."),
    ] = None,
) -> None:
    """Run the scoring job on a schedule (default 08:00 UTC daily)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)

    cron_expr = cron or settings.daemon_cron
    trigger = CronTrigger.from_crontab(cron_expr, timezone=timezone.utc)

    async def _tick() -> None:
        # Settle finished bets first (updates the kill switch), then score —
        # so a freshly-tripped kill switch suppresses this tick's new bets.
        await _settle_once()
        await _score_once()

    async def _main() -> None:
        scheduler = AsyncIOScheduler(timezone=timezone.utc)
        scheduler.add_job(_tick, trigger=trigger, id="score_and_settle")
        scheduler.start()
        log.info("daemon_started", cron=cron_expr)
        await _tick()  # immediate first run
        try:
            await asyncio.Event().wait()
        finally:
            scheduler.shutdown(wait=False)

    asyncio.run(_main())


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the SQLite schema."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    typer.echo(f"DB initialized at {settings.db_path.resolve()}")


@bets_app.command("list")
def bets_list(
    since: Annotated[
        str,
        typer.Option(help="How far back to look, e.g. '7d', '24h'."),
    ] = "7d",
) -> None:
    """List recent paper bets."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)

    days = _parse_since(since)
    rows = list_recent_paper_bets(days=days)
    if not rows:
        typer.echo(f"No paper bets in the last {since}.")
        return
    typer.echo(
        f"{'created_at':<17}  {'fixture':>7}  {'out':<4}  {'p':>4}  "
        f"{'mkt':>5}  {'stake':>6}  {'res':<4}  {'pnl':>8}"
    )
    for b in rows:
        ts = b.created_at.strftime("%Y-%m-%d %H:%M") if b.created_at else "?"
        mkt = f"{b.market_price:.2f}" if b.market_price is not None else "-"
        res = b.settled_outcome or "-"
        pnl = f"${b.pnl_usd:+.2f}" if b.pnl_usd is not None else "-"
        typer.echo(
            f"{ts:<17}  {b.fixture_id:>7}  {b.outcome:<4}  "
            f"{b.our_probability:>4.2f}  {mkt:>5}  ${b.stake_usd:>5.0f}  "
            f"{res:<4}  {pnl:>8}"
        )


def _parse_since(s: str) -> int:
    """Convert '7d' or '24h' to a number of days (rounded up)."""
    s = s.strip().lower()
    if s.endswith("d"):
        return max(1, int(s[:-1]))
    if s.endswith("h"):
        return max(1, (int(s[:-1]) + 23) // 24)
    return max(1, int(s))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
