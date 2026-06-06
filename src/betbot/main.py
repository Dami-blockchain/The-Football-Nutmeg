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
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket import PolymarketAdapter
from betbot.exchanges.polymarket_gamma import GammaClient
from betbot.exchanges.router import ExchangeRouter
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    daily_paper_exposure_usd,
    insert_paper_bet,
    insert_paper_bet_no_market,
    list_recent_paper_bets,
    upsert_prediction,
)
from betbot.strategy.engine import StrategyEngine

# Repo root (…/tfsm), used to locate config/team_aliases.yaml regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
bets_app = typer.Typer(help="Inspect logged paper bets.")
app.add_typer(bets_app, name="bets")


# ----------------------------------------------------------------------
# Exchange routing
# ----------------------------------------------------------------------
def _build_router(settings) -> tuple[ExchangeRouter, GammaClient]:
    """Construct the exchange router for a run.

    Returns the router and the underlying GammaClient so the caller can close
    it. ``enable_orders`` stays False in Phase 2 (paper only) — live ordering
    is wired in Phase 5; the adapter's double-gate keeps place_order inert
    regardless.
    """
    resolver = TeamAliasResolver.from_yaml(_REPO_ROOT / "config" / "team_aliases.yaml")
    gamma = GammaClient()
    polymarket = PolymarketAdapter(
        gamma,
        resolver,
        enable_orders=False,
        mode=settings.mode,
    )
    return ExchangeRouter([polymarket]), gamma


# ----------------------------------------------------------------------
# Scoring run
# ----------------------------------------------------------------------
async def _score_once() -> int:
    """Pull fixtures in the next 48h, score each, log paper bets."""
    settings = get_settings()
    log = get_logger(__name__)
    init_engine(settings.db_path)

    log.info(
        "starting_scoring_run",
        mode=settings.mode,
        leagues=list(settings.leagues),
    )

    today = date.today()
    date_from = today.isoformat()
    # +2: football-data dateTo is exclusive, so +2 = include tomorrow.
    date_to = (today + timedelta(days=2)).isoformat()

    paper_bets_logged = 0
    router, gamma = _build_router(settings)

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
                            m, league, form_service, engine, router, settings
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
            await gamma.close()

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
# CLI commands
# ----------------------------------------------------------------------
@app.command("run-once")
def run_once() -> None:
    """Score the next 48h of fixtures and log paper bets."""
    settings = get_settings()
    configure_logging(settings.log_level)
    n = asyncio.run(_score_once())
    typer.echo(f"Logged {n} paper bet(s).")


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

    async def _main() -> None:
        scheduler = AsyncIOScheduler(timezone=timezone.utc)
        scheduler.add_job(_score_once, trigger=trigger, id="daily_score")
        scheduler.start()
        log.info("daemon_started", cron=cron_expr)
        await _score_once()  # immediate first run
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
        f"{'created_at':<20}  {'fixture':>8}  {'outcome':<5}  "
        f"{'p':>5}  {'stake':>6}  rationale"
    )
    for b in rows:
        ts = b.created_at.strftime("%Y-%m-%d %H:%M") if b.created_at else "?"
        typer.echo(
            f"{ts:<20}  {b.fixture_id:>8}  {b.outcome:<5}  "
            f"{b.our_probability:>5.2f}  ${b.stake_usd:>5.0f}  "
            f"{b.rationale[:80]}"
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
