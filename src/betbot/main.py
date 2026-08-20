"""The Football Nutmeg Agent — CLI entrypoint.

Commands (run as ``nutmeg <command>``, ``tfsm <command>`` or ``betbot <command>``):
    tfsm run-once      Score the next 48h of fixtures and log paper bets.
    tfsm run-daemon    Schedule run-once daily at 08:00 UTC.
    tfsm bets list     Print recent paper bets to stdout.
    tfsm init-db       Create the SQLite schema (called automatically).
    tfsm announce      Flag the operator on Telegram BEFORE a change.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient, FootballDataError
from betbot.data.form import FormService, _parse_kickoff, _parse_team
from betbot.data.odds import shared_odds_service
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket import PolymarketAdapter
from betbot.exchanges.polymarket_gamma import GammaClient
from betbot.exchanges.router import ExchangeRouter
from betbot.backtest import backtest_mock, backtest_stored
from betbot.daily_jobs import (
    nairobi_day_bounds,
    register_daily_jobs,
    run_matchday_notice,
    run_result_alerts,
    send_prediction_alert,
)
from betbot.gate import evaluate_gate
from betbot.logging import configure_logging, get_logger
from betbot.notify import announce_change, notify_operator
from betbot.reschedule import (
    alert_job_ids,
    alert_still_valid,
    parse_utc,
    resync_kickoffs,
)
from betbot.scheduling import add_async_job, unawaitable_jobs
from betbot.settlement import SettlementWatcher
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    daily_paper_exposure_usd,
    get_kill_switch,
    insert_paper_bet,
    insert_paper_bet_no_market,
    list_recent_paper_bets,
    predictions_for_kickoff_range,
    reset_kill_switch,
    upsert_prediction,
)
from betbot.strategy.engine import StrategyEngine
from betbot.strategy.club_engine import ClubStrategyEngine
from betbot.strategy.cl_engine import EuropeanStrategyEngine
from betbot.strategy.odds_anchor import anchor_prediction

# Repo root (…/tfsm), used to locate config/team_aliases.yaml regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
bets_app = typer.Typer(help="Inspect logged paper bets.")
app.add_typer(bets_app, name="bets")
ks_app = typer.Typer(help="Inspect / reset the drawdown kill switch.")
app.add_typer(ks_app, name="kill-switch")
glicko_app = typer.Typer(help="Glicko-2 club ratings.")
app.add_typer(glicko_app, name="glicko")


# ----------------------------------------------------------------------
# Exchange routing (READ-ONLY — predictions-only build)
# ----------------------------------------------------------------------
def _build_router(settings) -> tuple[ExchangeRouter, list]:
    """Construct a READ-ONLY exchange router for a run.

    Returns ``(router, http_clients_to_close)``.

    This build fetches Polymarket market PRICES only — it anchors predictions
    and computes the edge-based bet/no-bet recommendation. It NEVER places an
    order or moves funds: the Polymarket adapter is read-only and carries NO
    order-placement path or signing key at all (only ``find_market`` /
    ``get_orderbook`` exist), so there is nothing here to arm. Limitless has been
    removed entirely; only Polymarket read pricing remains.
    """
    resolver = TeamAliasResolver.from_yaml(_REPO_ROOT / "config" / "team_aliases.yaml")

    gamma = GammaClient()
    pm = PolymarketAdapter(gamma, resolver)  # read-only: no key, no order path
    router = ExchangeRouter(
        [pm],
        min_plausible_price=settings.min_plausible_price,
        max_plausible_price=settings.max_plausible_price,
    )
    return router, [gamma]


# ----------------------------------------------------------------------
# Scoring run
# ----------------------------------------------------------------------
async def _score_once() -> int:
    """Pull fixtures in the next 48h, score each, log a paper reco per fixture.

    Predictions-only: reads Polymarket PRICES to anchor the edge-based
    bet/no-bet recommendation and logs a "paper reco" row. No order is ever
    placed and no funds move.
    """
    settings = get_settings()
    log = get_logger(__name__)
    init_engine(settings.db_path)

    log.info(
        "starting_scoring_run",
        leagues=list(settings.leagues),
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
        club_engine = (
            ClubStrategyEngine(settings)
            if settings.club_ensemble_enabled
            else engine
        )
        cl_engine = (
            EuropeanStrategyEngine(settings)
            if settings.cl_elo_enabled
            else engine
        )
        # ONE odds service for the whole run: a single shared TTL cache means a
        # 20-fixture Saturday costs ONE HTTP GET, not twenty (the Highlightly
        # lesson). None when BETBOT_ODDS_ANCHOR is off — the default.
        odds_service = shared_odds_service(settings)

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
                # Engine routing: domestic top-5 leagues -> club Glicko/DC
                # ensemble; Champions League -> cross-league ClubElo engine
                # (R2), which falls back to naive internally on unresolved
                # teams / a stale snapshot, and is `engine` outright when
                # BETBOT_CL_ELO=false.
                eng = cl_engine if league == "CL" else club_engine
                for m in matches:
                    try:
                        bets = await _score_and_log_one(
                            m, league, form_service, eng, router, settings,
                            odds_service=odds_service,
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
    *,
    odds_service=None,
) -> int:
    """Score one fixture and log a paper reco (recommendation record).

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
    # Free pre-match odds anchoring (flag-gated, default OFF). Every scored
    # fixture gets anchored to a de-vigged bookmaker line, not just the ones
    # Polymarket happens to list. Returns the prediction UNCHANGED on any
    # failure — unresolvable name, missing row, dead feed.
    prediction = await anchor_prediction(
        prediction,
        league=league,
        kickoff=kickoff,
        settings=settings,
        odds_service=odds_service,
        engine=engine,
    )
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
        home_xg=prediction.home_xg,
        away_xg=prediction.away_xg,
        home_form=fixture_form.home_form.weighted_points,
        away_form=fixture_form.away_form.weighted_points,
    )

    pred_id = upsert_prediction(prediction, kickoff=kickoff)

    # Predictions-only: recos are NEVER suppressed (there is no trading and no
    # drawdown kill switch gating them). They always flow.

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

    # NOTE (single-anchor invariant): with BETBOT_ODDS_ANCHOR on, this picks
    # the leg from the ODDS-ANCHORED triple, which flipped the favourite on
    # 4.1% of held-out fixtures (n=1752, scripts/double_anchor_report.py). That
    # is deliberate, not a second anchor — the anchored triple is the better
    # calibrated one (the gate: RPS 0.2033 -> 0.2005), and whichever leg is
    # chosen, decide_with_market prices it from the RAW model against the
    # exchange, so the bookmaker never reaches the edge number.
    favourite = prediction.best_outcome

    # ---- Market route (Phase 2 + Phase 5 live placement) -------------
    route = await router.find_best_route(
        prediction.home_team, prediction.away_team, kickoff, favourite
    )
    if route is not None:
        quote = route.quote
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
        else:
            log.debug("paper_bet_already_logged", fixture_id=fixture_id)
        return 1 if inserted else 0

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
# Two-alert pre-match scheduling plan (pure — unit-testable)
# ----------------------------------------------------------------------
def plan_kickoff_alert_jobs(settings, preds, now):
    """Return the per-fixture DateTrigger job plan for the two-alert model.

    For each prediction in ``preds`` this yields UP TO two entries:
      ``predict_early_<fid>`` at ``KO - early_alert_lead_minutes(competition)``
      ``predict_late_<fid>``  at ``KO - lineup_confirm_lead_minutes()``
    A fire time already at/before ``now`` is DROPPED (past-time skip). Pure and
    offline: no scheduler, no network — just ``[(job_id, run_at_utc), ...]`` in
    schedule order, so ``_schedule_kickoff_alerts`` and its test share one source
    of truth for the offsets.
    """
    plan: list[tuple[str, datetime]] = []
    for p in preds:
        early_lead = timedelta(
            minutes=settings.early_alert_lead_minutes(p.competition_code)
        )
        late_lead = timedelta(minutes=settings.lineup_confirm_lead_minutes())
        ko = p.kickoff
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        fid = p.fixture_id
        for tag, run_at in (
            ("early", ko - early_lead),
            ("late", ko - late_lead),
        ):
            if run_at <= now:
                continue  # firing time already past — skip
            plan.append((f"predict_{tag}_{fid}", run_at))
    return plan


def drop_alert_jobs(scheduler, fixture_ids) -> list[str]:
    """Remove both pre-match jobs for each fixture. Returns the ids removed.

    A rescheduled fixture's OLD jobs are still sitting on the scheduler pinned to
    the dead kickoff, and re-planning does not touch them: the planner registers
    `predict_early_<fid>` at the NEW time, which replaces the old job only when
    the fixture still falls inside the pass's window. A match moved out of the
    window keeps its old job and fires a phantom alert — the one that charges a
    reveal for a match nobody plays. So the old jobs come off explicitly.
    """
    removed: list[str] = []
    for fixture_id in fixture_ids:
        for job_id in alert_job_ids(fixture_id):
            try:
                scheduler.remove_job(job_id)
            except Exception:  # noqa: BLE001 — not registered / already fired
                continue
            removed.append(job_id)
    return removed


# ----------------------------------------------------------------------
# Alert-coverage self-check (the "silent no-op" watchdog)
# ----------------------------------------------------------------------
# How long ahead the watchdog expects alert jobs to already be registered.
# The daily re-scan covers ONE Nairobi day, so a fixture more than ~12h out
# legitimately has no jobs yet; inside 12h the daily pass has always run and a
# missing job is a real fault, not a timing artefact.
ALERT_WATCHDOG_HORIZON_HOURS = 12


def audit_alert_coverage(scheduler, plan) -> list[str]:
    """Job ids from ``plan`` that are NOT registered on ``scheduler``.

    Pure and offline. ``plan`` is :func:`plan_kickoff_alert_jobs` output, which
    has already dropped fire times in the past — and APScheduler drops one-off
    DateTrigger jobs once they fire — so a job id present in the plan but
    absent from the scheduler means the alert will simply never be sent.
    """
    registered = {job.id for job in scheduler.get_jobs()}
    return [job_id for job_id, _run_at in plan if job_id not in registered]


async def report_alert_coverage(scheduler, plan, *, settings, send_fn=None) -> list[str]:
    """Audit alert coverage and, on a gap, log ERROR + Telegram the operator.

    A missing alert job used to be a SILENT no-op — the daily re-scan was
    registered as a sync lambda around an ``async def`` for weeks and nothing
    said so. Coverage gaps are now loud: ERROR in the log and a push to
    ``settings.telegram_allowed_user_id``. Returns the missing job ids so
    callers (and tests) can assert on them.
    """
    missing = audit_alert_coverage(scheduler, plan)
    if not missing:
        return []
    log = get_logger(__name__)
    log.error(
        "alert_coverage_gap",
        missing_count=len(missing),
        missing_job_ids=missing,
    )
    body = (
        "*\U0001f6a8 Alert scheduling fault*\n\n"
        f"{len(missing)} pre-match/lineup alert job(s) are NOT registered on "
        "the scheduler, so those alerts will NOT be sent:\n"
        + "\n".join(f"- `{jid}`" for jid in missing[:20])
        + "\n\nThe daemon needs a restart / investigation."
    )
    # Through notify_operator, not a raw send: this runs from the HOURLY
    # watchdog, and a gap that stays unfixed all day would otherwise push 24
    # identical messages — which trains the operator to ignore the one signal
    # this whole watchdog exists to send. The cooldown for
    # "alert_coverage_gap" caps it, and a delivery failure is logged at
    # ERROR rather than swallowed.
    await notify_operator(
        settings, body, kind="alert_coverage_gap", send_fn=send_fn
    )
    return missing


# ----------------------------------------------------------------------
# Per-fixture re-scoring (pre-match lineup-adjusted alert, R4b)
# ----------------------------------------------------------------------
def _build_engines(settings, form_service):
    """Return ``(club_engine, cl_engine)`` mirroring _score_once's routing."""
    base = StrategyEngine(settings)
    club_engine = (
        ClubStrategyEngine(settings) if settings.club_ensemble_enabled else base
    )
    cl_engine = (
        EuropeanStrategyEngine(settings) if settings.cl_elo_enabled else base
    )
    return club_engine, cl_engine


async def score_fixture_adjusted(
    settings,
    fixture_id: int,
    *,
    home_rating_adj: float = 0.0,
    away_rating_adj: float = 0.0,
):
    """Re-score ONE fixture lineup-adjusted, returning ``(Prediction, kickoff)``.

    Rebuilds the same FixtureForm the daily scoring run uses — the fixture's
    teams/kickoff come from football-data (:meth:`get_match`), and the form
    snapshots from :class:`FormService` — then calls the SAME engine the daily
    run routes to (``cl_engine`` for CL, else ``club_engine``) with R4a's
    ``home_rating_adj`` / ``away_rating_adj``. With both adjustments 0.0 the
    result is byte-identical to the baseline. Returns ``(None, None)`` if the
    fixture can't be fetched (network gap / unknown id) so the caller falls
    back to the stored baseline prediction.
    """
    log = get_logger(__name__)
    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        match = await client.get_match(fixture_id)
        if not match:
            log.info("rescore_no_match", fixture_id=fixture_id)
            return None, None
        league = str((match.get("competition") or {}).get("code") or "")
        kickoff = _parse_kickoff(match["utcDate"])
        home = _parse_team(match["homeTeam"])
        away = _parse_team(match["awayTeam"])

        form_service = FormService(client, settings)
        club_engine, cl_engine = _build_engines(settings, form_service)
        engine = cl_engine if league == "CL" else club_engine

        fixture_form = await form_service.fixture_form(
            fixture_id=fixture_id,
            competition_code=league,
            kickoff=kickoff,
            home_team=home,
            away_team=away,
        )
        prediction = engine.predict(
            fixture_form,
            home_rating_adj=home_rating_adj,
            away_rating_adj=away_rating_adj,
        )
        # Same anchor as the daily run, so the pre-match alert and the stored
        # baseline are priced the same way. Flag-gated, default OFF.
        prediction = await anchor_prediction(
            prediction,
            league=league,
            kickoff=kickoff,
            settings=settings,
            odds_service=shared_odds_service(settings),
            engine=engine,
        )
        return prediction, kickoff


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


@glicko_app.command("ratings")
def glicko_ratings() -> None:
    """List current Glicko-2 ratings, strongest first."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    from betbot.storage.repos import all_ratings

    rows = all_ratings()
    if not rows:
        typer.echo("No Glicko ratings yet. Seed with: python scripts/seed_glicko_club.py")
        return
    typer.echo(f"{'team':<28} {'rating':>8} {'RD':>6} {'vol':>7}")
    for name, r in rows:
        typer.echo(f"{name:<28} {r.rating:>8.1f} {r.rd:>6.1f} {r.volatility:>7.4f}")


@glicko_app.command("seed")
def glicko_seed() -> None:
    """Bootstrap club Glicko-2 ratings from fetched club results."""
    import runpy

    runpy.run_path(
        str(_REPO_ROOT / "scripts" / "seed_glicko_club.py"), run_name="__main__"
    )


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
        # Refresh the cross-league Elo snapshot the CL engine reads, so a
        # fixture is priced off today's ratings. Non-fatal + off the event
        # loop; on failure the engine keeps the last snapshot.
        _s = get_settings()
        if _s.cl_elo_enabled:
            try:
                from betbot.data.clubelo import refresh_latest
                await asyncio.to_thread(
                    refresh_latest, Path(_s.clubelo_latest_path)
                )
            except Exception as e:  # noqa: BLE001 — never crash the tick
                get_logger(__name__).warning("clubelo_refresh_tick_failed", error=str(e))
        # Score recos for the next 48h, then settle finished recos (accuracy
        # tracking). No orders are placed on either step.
        await _settle_once()
        await _score_once()

    async def _club_refresh_tick() -> None:
        # Weekly: refresh results + re-seed club Glicko + refit club DC so
        # ratings track the season instead of freezing at seed time.
        # Subprocess (not import): the scripts are argparse mains; isolation
        # means a bad refresh can never corrupt the daemon.
        import subprocess

        def _run() -> None:
            for script in ("fetch_club_results.py", "seed_glicko_club.py",
                           "fit_dixon_coles_club.py"):
                subprocess.run(
                    [".venv/bin/python", f"scripts/{script}"],
                    cwd=str(_REPO_ROOT), timeout=1800, check=True,
                    capture_output=True,
                )
        try:
            await asyncio.to_thread(_run)
            get_logger(__name__).info("club_data_refreshed")
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("club_refresh_failed", error=str(e))

    async def _season_title_refresh_tick() -> None:
        # Weekly: refresh the season-title Monte-Carlo cache for each domestic
        # league so /title stays current. Subprocess per league (isolation +
        # never crash the daemon); non-fatal if one fails. Budget-safe: one
        # football-data.org call per league per week.
        import subprocess

        _s = get_settings()
        leagues = [c for c in _s.leagues if c in ("PL", "PD", "BL1", "SA", "FL1")]

        def _run() -> None:
            for code in leagues:
                try:
                    subprocess.run(
                        [".venv/bin/python", "scripts/simulate_season.py",
                         "--league", code],
                        cwd=str(_REPO_ROOT), timeout=900, check=True,
                        capture_output=True,
                    )
                except Exception as e:  # noqa: BLE001 — one league must not sink the rest
                    get_logger(__name__).warning(
                        "season_title_league_failed", league=code, error=str(e))
            # Champions League winner projection (tournament sim, ClubElo-seeded).
            try:
                subprocess.run(
                    [".venv/bin/python", "scripts/simulate_cl.py"],
                    cwd=str(_REPO_ROOT), timeout=900, check=True,
                    capture_output=True,
                )
            except Exception as e:  # noqa: BLE001 — CL must not sink the leagues
                get_logger(__name__).warning("cl_winner_refresh_failed", error=str(e))
        try:
            await asyncio.to_thread(_run)
            get_logger(__name__).info("season_title_refreshed", leagues=len(leagues))
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("season_title_refresh_failed", error=str(e))

    async def _matchday_notice_tick() -> None:
        # Morning heads-up (Africa/Nairobi): FREE fixture list, no prediction.
        try:
            await run_matchday_notice(get_settings())
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("matchday_notice_failed", error=str(e))

    async def _settle_and_results_tick() -> None:
        # Periodic (every ~2h): settle finished fixtures (score outcomes, update
        # ratings) then broadcast end-of-match RESULT ALERTS. This is what lands
        # results within ~2h of full time instead of waiting for the 08:00 tick.
        # Both steps are independently try/excepted so neither can crash the
        # daemon. Result alerts are FREE (no reveal-ledger write, no charge).
        try:
            await _settle_once()
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("periodic_settle_failed", error=str(e))
        try:
            await run_result_alerts(get_settings())
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("result_alerts_failed", error=str(e))

    async def _alert_matches_upstream(settings, fixture_id: int) -> bool:
        """Re-read the fixture upstream and say whether this alert should fire.

        Returns True on any fetch failure: a network blip must not silence a
        legitimate alert, and the stale-alert case is already covered by the
        hourly re-sync. Only a CONFIRMED mismatch (match moved, postponed,
        cancelled) suppresses the send.
        """
        try:
            async with FootballDataClient(
                api_key=settings.football_data_api_key,
                base_url=settings.football_data_base_url,
                rate_limit_per_min=settings.football_data_rate_limit_per_min,
            ) as client:
                match = await client.get_match(fixture_id)
        except Exception as e:  # noqa: BLE001 — never block on a fetch failure
            get_logger(__name__).warning(
                "prematch_guard_fetch_failed", fixture_id=fixture_id, error=str(e)
            )
            return True
        if match is None:
            return True
        kickoff = parse_utc(match.get("utcDate"))
        status = match.get("status")
        league = str((match.get("competition") or {}).get("code") or "")
        if alert_still_valid(
            datetime.now(timezone.utc),
            kickoff,
            status,
            early_lead_minutes=settings.early_alert_lead_minutes(league),
        ):
            return True
        get_logger(__name__).info(
            "prematch_alert_skipped_stale",
            fixture_id=fixture_id,
            upstream_kickoff=kickoff.isoformat() if kickoff else None,
            status=status,
            note="fixture moved or is not being played — not charging a reveal",
        )
        return False

    async def _fire_prediction_alert(fixture_id: int) -> None:
        # Pre-match lineup-adjusted, gated. Wire the re-scoring helper so the
        # alert re-scores off the confirmed XI; lineup_fn defaults to the
        # production LineupService inside send_prediction_alert.
        try:
            settings = get_settings()
            # send_prediction_alert calls rescore_fn(fixture_id, home_adj,
            # away_adj), but score_fixture_adjusted's real signature is
            # (settings, fixture_id, *, home_rating_adj, away_rating_adj).
            # Wire a closure that adapts the positional shape (and forwards the
            # (Prediction, kickoff) return unchanged) so the alert re-scores
            # instead of raising "takes 2 positional arguments but 3 were given".
            async def _rescore(fid, home_adj, away_adj):
                return await score_fixture_adjusted(
                    settings, fid,
                    home_rating_adj=home_adj, away_rating_adj=away_adj,
                )

            # Last line of defence on the money path. A fixture can move in the
            # gap between the last re-sync and this fire time, and the early
            # alert CHARGES a reveal — so confirm against upstream that the
            # match really is about to kick off before spending the user's
            # credit. One call, only ever on the alert path.
            if not await _alert_matches_upstream(settings, fixture_id):
                return

            await send_prediction_alert(
                settings, fixture_id, rescore_fn=_rescore,
            )
        except Exception as e:  # noqa: BLE001 — never crash
            get_logger(__name__).warning(
                "prematch_alert_failed", fixture_id=fixture_id, error=str(e),
            )

    async def _schedule_kickoff_alerts(scheduler) -> None:
        # TWO-alert pre-match model. For each of today's scored fixtures schedule
        # TWO one-off DateTrigger jobs, BOTH calling send_prediction_alert:
        #   predict_early_<fid> at KO - early_alert_lead(competition) (PL 70,
        #     else 55) — the XI isn't posted yet -> fresh MODEL prediction with
        #     the "lineups not yet confirmed" note;
        #   predict_late_<fid>  at KO - lineup_confirm_lead_minutes() (default
        #     10, all leagues) — the XI is now posted -> confirmed XI + lineup-
        #     adjusted prediction.
        # Runs at daemon start AND daily (05:00 UTC) so a long-running daemon
        # keeps picking up newly-scored fixtures. Jobs are idempotent
        # (replace_existing on per-fixture ids); a fire time already in the past
        # is skipped. Every tick is try/excepted so a failure can't crash the
        # daemon.
        #
        # MONEY INVARIANT: both alerts hit send_prediction_alert, which gates
        # via the reveal ledger, so the fixture is charged EXACTLY ONCE — the
        # early alert reveals+charges; the late alert finds it already-revealed
        # and re-shows it FREE with the updated lineup-adjusted content.
        try:
            _s = get_settings()
            now = datetime.now(timezone.utc)
            # Re-read upstream kickoffs FIRST, so the plan below is built off
            # where the fixtures actually are rather than where they were when
            # they were scored. Any fixture that moved has its old jobs pulled:
            # a match shifted out of today's window would otherwise keep a job
            # pinned to the dead time and fire a phantom, charging alert.
            try:
                async with FootballDataClient(
                    api_key=_s.football_data_api_key,
                    base_url=_s.football_data_base_url,
                    rate_limit_per_min=_s.football_data_rate_limit_per_min,
                ) as _client:
                    changes = await resync_kickoffs(_client, _s, now=now)
                if changes:
                    removed = drop_alert_jobs(
                        scheduler, [c.fixture_id for c in changes]
                    )
                    get_logger(__name__).info(
                        "kickoff_change_jobs_dropped",
                        fixtures=[c.fixture_id for c in changes],
                        removed=removed,
                    )
            except Exception as e:  # noqa: BLE001 — re-sync is best-effort
                get_logger(__name__).warning(
                    "kickoff_resync_failed", error=str(e)
                )
            start, end, _day = nairobi_day_bounds(now)
            preds = predictions_for_kickoff_range(start, end)
            plan = plan_kickoff_alert_jobs(_s, preds, now)
            scheduled = 0
            for job_id, run_at in plan:
                # job_id is predict_early_<fid> / predict_late_<fid>; recover the
                # fixture id (last underscore-delimited token) for the closure.
                fid = int(job_id.rsplit("_", 1)[1])

                async def _fire(fixture_id=fid) -> None:
                    await _fire_prediction_alert(fixture_id)

                add_async_job(
                    scheduler,
                    _fire,
                    trigger=DateTrigger(run_date=run_at, timezone=timezone.utc),
                    id=job_id,
                    replace_existing=True,
                )
                scheduled += 1
            get_logger(__name__).info(
                "prematch_alerts_scheduled", scheduled=scheduled,
            )
            # Self-check: everything we just planned must actually be on the
            # scheduler. A pass that quietly schedules nothing is the failure
            # mode that hid this bug for days, so it is now loud.
            await report_alert_coverage(scheduler, plan, settings=_s)
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("schedule_kickoff_alerts_failed", error=str(e))

    async def _alert_coverage_watchdog(scheduler) -> None:
        # Hourly independent check that the SCHEDULING ITSELF is happening.
        # report_alert_coverage inside _schedule_kickoff_alerts only catches a
        # pass that ran and lost jobs; this catches a pass that never ran at
        # all (the actual production failure). Looks ahead
        # ALERT_WATCHDOG_HORIZON_HOURS, by which point the daily re-scan has
        # always covered the fixture. The audit itself is DB-only; the
        # scheduling pass it now runs first costs one football-data call per
        # league per hour, which is what buys same-day reschedule pickup.
        try:
            # Re-run the full scheduling pass first. It re-syncs kickoffs and is
            # idempotent (per-fixture job ids, replace_existing, past fire times
            # skipped), so a fixture pulled INTO today by a reschedule picks up
            # its alerts within the hour instead of waiting for the 05:00 pass
            # that has already been and gone. Auditing without this could only
            # ever report the gap; now it closes it.
            await _schedule_kickoff_alerts(scheduler)
            _s = get_settings()
            now = datetime.now(timezone.utc)
            # Clamp to the end of the CURRENT Nairobi day. Scheduling passes
            # only ever cover today-on-the-Nairobi-clock, but scoring writes
            # predictions 48h ahead — so a fixture on the NEXT Nairobi day that
            # is already inside the horizon has legitimately not been scheduled
            # yet, and flagging it would fire an hourly false alarm until the
            # 05:00 UTC pass. Alarming on a non-problem trains the operator to
            # ignore the exact signal this watchdog exists to send.
            _, day_end, _day = nairobi_day_bounds(now)
            horizon = min(
                now + timedelta(hours=ALERT_WATCHDOG_HORIZON_HOURS), day_end
            )
            preds = predictions_for_kickoff_range(now, horizon)
            plan = plan_kickoff_alert_jobs(_s, preds, now)
            await report_alert_coverage(scheduler, plan, settings=_s)
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning(
                "alert_coverage_watchdog_failed", error=str(e)
            )

    async def _player_minutes_refresh_tick() -> None:
        # Weekly: refresh the api-football player-minutes cache (R4a importance
        # data) so the lineup adjustment stays warm. Subprocess (not import),
        # mirroring _club_refresh_tick — isolation + never crash the daemon.
        # Budget-safe: once/week.
        import subprocess

        _s = get_settings()

        def _run() -> None:
            args = [".venv/bin/python", "scripts/fetch_player_minutes.py",
                    "--season", str(_s.api_football_season)]
            subprocess.run(
                args, cwd=str(_REPO_ROOT), timeout=1800, check=True,
                capture_output=True,
            )
        try:
            await asyncio.to_thread(_run)
            get_logger(__name__).info("player_minutes_refreshed")
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("player_minutes_refresh_failed", error=str(e))

    async def _main() -> None:
        s = get_settings()
        init_engine(s.db_path)  # cron jobs may fire before the first scoring tick
        scheduler = AsyncIOScheduler(timezone=timezone.utc)
        add_async_job(scheduler, _tick, trigger=trigger, id="score_and_settle")
        add_async_job(
            scheduler,
            _club_refresh_tick,
            trigger=CronTrigger.from_crontab("0 6 * * 1", timezone=timezone.utc),
            id="club_data_refresh",
        )
        register_daily_jobs(scheduler, s, matchday_notice=_matchday_notice_tick)
        # Re-scan for today's fixtures daily so the daemon keeps scheduling
        # pre-match alerts as new fixtures get scored.
        #
        # The callable MUST be the coroutine function itself with ``scheduler``
        # bound through args=. It was a sync ``lambda: _schedule_kickoff_alerts
        # (scheduler)`` until 2026-08-19, which APScheduler CALLED and then
        # discarded the coroutine of — so this re-scan never once executed and
        # every fixture scored after daemon start got NO pre-match and NO
        # lineup alert. add_async_job now rejects that shape outright.
        add_async_job(
            scheduler,
            _schedule_kickoff_alerts,
            args=(scheduler,),
            trigger=CronTrigger.from_crontab("0 5 * * *", timezone=timezone.utc),
            id="reschedule_kickoff_alerts",
        )
        # Hourly: verify the alert jobs the DB says should exist actually do.
        add_async_job(
            scheduler,
            _alert_coverage_watchdog,
            args=(scheduler,),
            trigger=IntervalTrigger(hours=1, timezone=timezone.utc),
            id="alert_coverage_watchdog",
        )
        # Weekly (Mon 05:30 UTC): refresh the player-minutes importance cache so
        # the lineup adjustment stays warm. Budget-safe (once/week).
        add_async_job(
            scheduler,
            _player_minutes_refresh_tick,
            trigger=CronTrigger.from_crontab("30 5 * * 1", timezone=timezone.utc),
            id="player_minutes_refresh",
        )
        # Weekly (Mon 06:30 UTC, after the club re-seed at 06:00): refresh the
        # season-title Monte-Carlo cache so /title tracks the season. One
        # football-data.org call per domestic league per week.
        add_async_job(
            scheduler,
            _season_title_refresh_tick,
            trigger=CronTrigger.from_crontab("30 6 * * 1", timezone=timezone.utc),
            id="season_title_refresh",
        )
        # Periodic (every 2h): settle finished fixtures + fire RESULT ALERTS, so
        # results land within ~2h of full time rather than at the next 08:00 tick.
        add_async_job(
            scheduler,
            _settle_and_results_tick,
            trigger=IntervalTrigger(hours=2, timezone=timezone.utc),
            id="settle_and_results",
        )
        # Belt-and-braces: catch anything registered through a raw add_job
        # that bypassed add_async_job. Loud, but never fatal to the daemon.
        _broken = unawaitable_jobs(scheduler)
        if _broken:
            log.error("scheduler_jobs_not_awaitable", job_ids=_broken)
            # Registration-time, so it fires once per daemon start — but a
            # dropped job is a SILENT job, which is exactly the class of fault
            # that ran unnoticed for days. It goes to the human too.
            await notify_operator(
                s,
                "*\U0001f6a8 Scheduler fault*\n\n"
                f"{len(_broken)} registered job(s) will NEVER RUN — "
                "APScheduler would call them and throw away the coroutine:\n"
                + "\n".join(f"- `{jid}`" for jid in _broken[:20])
                + "\n\nThese were registered bypassing add_async_job. "
                "The affected features are silently dead until this is fixed.",
                kind="scheduler_jobs_not_awaitable",
            )
        scheduler.start()
        log.info(
            "daemon_started",
            cron=cron_expr,
            matchday_alert_hour_nairobi=s.matchday_alert_hour,
            early_pl_lead_min=s.pl_lineup_alert_lead_minutes,
            early_lead_min_default=s.lineup_alert_lead_minutes_default,
            late_confirm_lead_min=s.lineup_confirm_lead_minutes(),
        )
        await _tick()  # immediate first run
        await run_result_alerts(s)  # fire any pending result alerts on startup
        await _schedule_kickoff_alerts(scheduler)  # schedule today's reminders now
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


@app.command("announce")
def announce_cmd(
    what: Annotated[
        str, typer.Argument(help="What is about to change, in one line.")
    ],
    rollback: Annotated[
        str,
        typer.Option(
            "--rollback", "-r",
            help="How to undo it if it goes wrong. State one.",
        ),
    ] = "",
    who: Annotated[
        str, typer.Option("--who", help="Who is making the change.")
    ] = "",
) -> None:
    """Flag the operator on Telegram BEFORE changing anything.

    Standing rule: the operator is told what is about to happen *before* a
    change is committed or a flag is flipped — not after, and not only in a
    log file. This is the entry point for that, callable from a shell script,
    a Makefile or an agent without importing the package:

        tfsm announce "merge feat/operator-notify to main" \\
             --rollback "git revert <sha> && systemctl restart tfsm" \\
             --who ronaldo

    Exits NON-ZERO if the message did not reach Telegram, so the announcement
    can gate the change itself:

        tfsm announce "..." --rollback "..." && git merge feat/x

    Announcements are never rate-limited — each one is a deliberate statement
    about a distinct change.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    if not rollback.strip():
        typer.echo(
            "warning: no --rollback given; the announcement will say so.",
            err=True,
        )
    ok = announce_change(settings, what, rollback=rollback, who=who)
    if not ok:
        typer.echo(
            "FAILED to announce — the operator has NOT been told. "
            "Do not make the change yet.",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo("Announced to the operator on Telegram. Change not yet applied.")


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
