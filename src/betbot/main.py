"""The Football Nutmeg Agent — CLI entrypoint.

Commands (run as ``nutmeg <command>``, ``tfsm <command>`` or ``betbot <command>``):
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
from apscheduler.triggers.interval import IntervalTrigger

from betbot.config import INTERNATIONAL_COMPETITIONS, get_settings
from betbot.data.api_football import ApiFootballClient
from betbot.data.football_data import FootballDataClient, FootballDataError
from betbot.data.form import FormService, _parse_kickoff, _parse_team
from betbot.exchanges.base import ExchangeName
from betbot.exchanges.limitless import LimitlessAdapter
from betbot.exchanges.limitless_client import LimitlessClient
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket import PolymarketAdapter
from betbot.exchanges.polymarket_gamma import GammaClient
from betbot.exchanges.router import ExchangeRouter
from betbot.backtest import backtest_mock, backtest_stored
from betbot.daily_jobs import (
    persist_arb_opportunities,
    register_daily_jobs,
    run_arb_digest,
    run_daily_report,
)
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
    list_users,
    reset_kill_switch,
    upsert_prediction,
)
from betbot.strategy.availability import availability_penalty
from betbot.strategy.engine import StrategyEngine
from betbot.strategy.international_engine import InternationalStrategyEngine
from betbot.strategy.club_engine import ClubStrategyEngine
# NOTE: betbot.wallet pulls in web3 (the `api` extra) at import time, so it's
# imported lazily inside the functions that need it — keeping `betbot.main`
# importable (and testable) on the base install.

# Repo root (…/tfsm), used to locate config/team_aliases.yaml regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
bets_app = typer.Typer(help="Inspect logged paper bets.")
app.add_typer(bets_app, name="bets")
ks_app = typer.Typer(help="Inspect / reset the drawdown kill switch.")
app.add_typer(ks_app, name="kill-switch")
glicko_app = typer.Typer(help="Glicko-2 ratings (international / World Cup).")
app.add_typer(glicko_app, name="glicko")
arb_app = typer.Typer(help="Cross-venue arbitrage.")
app.add_typer(arb_app, name="arb")


# ----------------------------------------------------------------------
# Exchange routing
# ----------------------------------------------------------------------
class _ClientGroup:
    """A closeable bag of HTTP clients created lazily during a run.

    Lets ``_build_router`` hand the caller ONE entry in its close list that, on
    ``close()``, closes every per-user client built after the router was made.
    Closing is best-effort: one client's failure doesn't block the rest.
    """

    def __init__(self) -> None:
        self._clients: list = []

    def add(self, client) -> None:
        self._clients.append(client)

    async def close(self) -> None:
        for c in self._clients:
            try:
                await c.close()
            except Exception as e:  # noqa: BLE001 — cleanup must not raise
                get_logger(__name__).warning("client_close_failed", error=str(e))


def _build_router(settings) -> tuple[ExchangeRouter, list, object]:
    """Construct the exchange router for a run.

    Returns ``(router, http_clients_to_close, make_signer_adapters)``.

    ``enable_orders`` stays False through Phase 3 (paper only) — live ordering
    is wired in Phase 5; each adapter's double-gate keeps place_order inert
    regardless. Limitless football coverage is sparse and US IPs are geo-blocked
    (surfaced as LimitlessGeoBlockedError, which the router isolates per-adapter),
    so most runs route on Polymarket alone — expected, not an error.

    ``make_signer_adapters(signing_key, funder)`` rebuilds the venue adapters
    bound to a *different* signing key — this is how multi-tenant placement
    trades each user's own wallet. Market discovery/scoring is wallet-independent
    and done once via ``router``; only the final ``place_order`` is per-user.
    """
    from betbot.wallet import get_private_key

    resolver = TeamAliasResolver.from_yaml(_REPO_ROOT / "config" / "team_aliases.yaml")
    enable = settings.mode == "live"  # double-gate: place_order also checks mode

    # Fall back to the agent wallet key (the one you deposit into) for signing
    # if explicit per-venue keys aren't set.
    agent_key = get_private_key(settings.wallet_keyfile)

    gamma = GammaClient()
    limitless_client = LimitlessClient(
        api_key=settings.limitless_api_key,
        api_secret=settings.limitless_api_secret,
    )

    # Per-user LimitlessClients (one per linked key) are created lazily as
    # funded users are processed, and live for the run. They're gathered in a
    # _ClientGroup so the caller's uniform ``await client.close()`` cleanup loop
    # closes all of them without knowing they exist.
    user_clients = _ClientGroup()

    def make_signer_adapters(
        *,
        signing_key: str | None,
        funder: str | None = None,
        limitless_creds: dict | None = None,
        agent_fallback: bool = False,
    ) -> dict:
        """Venue adapters bound to one signing key.

        Polymarket is always present. Limitless is per-account:

        * ``limitless_creds`` given → build a dedicated ``LimitlessClient`` with
          THAT user's api_key/secret and a ``LimitlessAdapter`` signing with
          THAT user's wallet key, so the order authenticates AND signs as the
          key owner (Limitless keys are scoped to their own wallet/profile).
        * ``agent_fallback=True`` → reuse the shared agent client built from
          ``settings.limitless_api_key/secret`` (the no-user / global path,
          unchanged from before).
        * otherwise (a real user with no linked key) → OMIT Limitless entirely,
          so the user's Polymarket leg still trades but no Limitless order is
          ever attempted under the wrong (shared/agent) credentials.

        Shares the Polymarket HTTP client (``gamma``) across all adapters.
        """
        pm = PolymarketAdapter(
            gamma, resolver, enable_orders=enable, mode=settings.mode,
            private_key=signing_key, funder=funder,
            signature_type=settings.polymarket_signature_type,
        )
        adapters: dict = {ExchangeName.POLYMARKET: pm}

        lm_client: LimitlessClient | None = None
        if limitless_creds is not None:
            lm_client = LimitlessClient(
                api_key=limitless_creds.get("api_key", ""),
                api_secret=limitless_creds.get("api_secret", ""),
            )
            user_clients.add(lm_client)
        elif agent_fallback:
            lm_client = limitless_client

        if lm_client is not None:
            adapters[ExchangeName.LIMITLESS] = LimitlessAdapter(
                lm_client, resolver, enable_orders=enable, mode=settings.mode,
                private_key=signing_key, fee_rate_bps=settings.limitless_fee_rate_bps,
                max_order_usd=settings.max_bet_usd,
            )
        return adapters

    base = make_signer_adapters(
        signing_key=settings.polymarket_private_key or agent_key,
        funder=settings.polymarket_funder or None,
        agent_fallback=True,
    )
    router = ExchangeRouter(
        [base[ExchangeName.POLYMARKET], base[ExchangeName.LIMITLESS]],
        min_plausible_price=settings.min_plausible_price,
        max_plausible_price=settings.max_plausible_price,
    )
    # ``user_clients`` is a single closeable that fans out to every per-user
    # Limitless client built mid-run, so the caller's uniform close loop reaches
    # them all.
    return router, [gamma, limitless_client, user_clients], make_signer_adapters


# ----------------------------------------------------------------------
# Player-availability (injury) adjustment — WC only, OFF by default
# ----------------------------------------------------------------------
async def _availability_adjustments(
    af_client: ApiFootballClient | None,
    settings,
    home_name: str,
    away_name: str,
) -> tuple[float, float]:
    """Return ``(home_rating_adj, away_rating_adj)`` Glicko-point shifts.

    A COARSE absence-count signal: count each team's unavailable players and
    map that to a small, capped NEGATIVE Glicko adjustment. v1 does NOT weight
    by player importance and is OFF by default until validated on the
    calibration report.

    GRACEFUL by contract: any error, a missing key, or the feature being
    disabled returns ``(0.0, 0.0)`` — this must NEVER break or skip a
    prediction. Logged once per failure, not raised.
    """
    if af_client is None or not af_client.has_key:
        return 0.0, 0.0
    log = get_logger(__name__)
    try:
        home_id = await af_client.resolve_team_id(home_name)
        away_id = await af_client.resolve_team_id(away_name)
        season = settings.api_football_season

        async def _adj(team_id: int | None) -> float:
            if team_id is None:
                return 0.0
            injuries = await af_client.get_injuries(team_id, season)
            return availability_penalty(
                len(injuries),
                per_player=settings.injury_penalty_per_player,
                cap=settings.injury_penalty_cap,
            )

        home_adj = await _adj(home_id)
        away_adj = await _adj(away_id)
        if home_adj or away_adj:
            log.info(
                "availability_adjustment",
                home=home_name, away=away_name,
                home_adj=round(home_adj, 1), away_adj=round(away_adj, 1),
            )
        return home_adj, away_adj
    except Exception as e:  # noqa: BLE001 — availability must never break scoring
        log.warning(
            "availability_fetch_failed",
            home=home_name, away=away_name, error=str(e),
            note="falling back to no adjustment (0/0)",
        )
        return 0.0, 0.0


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

    # Live pre-flight: only place real orders if the gate is clear. If it fails
    # we still score + log paper bets (data keeps flowing), just no live orders.
    live_orders = settings.mode == "live"
    if live_orders and settings.require_gate:
        gate = evaluate_gate(settings)
        if not gate.passed:
            log.error(
                "live_gate_failed",
                reasons=gate.reasons,
                note="paper bets still logged; NO live orders this run",
            )
            live_orders = False
    elif live_orders and not settings.require_gate:
        log.warning(
            "live_gate_skipped",
            note="BETBOT_REQUIRE_GATE=false — trading live with no paper-history gate",
        )

    log.info(
        "starting_scoring_run",
        mode=settings.mode,
        leagues=list(settings.leagues),
        kill_switch_tripped=kill_tripped,
        live_orders=live_orders,
    )

    today = date.today()
    date_from = today.isoformat()
    # +2: football-data dateTo is exclusive, so +2 = include tomorrow.
    date_to = (today + timedelta(days=2)).isoformat()

    paper_bets_logged = 0
    router, _http_clients, make_signer_adapters = _build_router(settings)
    # Per-run cache of per-user signer adapters (keyed by wallet keyfile), so we
    # build one ClobClient per user, not one per bet.
    user_adapter_cache: dict[str, dict] = {}

    # Optional player-availability (injury) client — built only when the WC
    # adjustment is enabled AND a key is configured. Closed in the finally
    # below. When absent, _availability_adjustments returns (0, 0) — a NO-OP.
    af_client: ApiFootballClient | None = None
    if settings.lineup_adjust_enabled and settings.api_football_key:
        af_client = ApiFootballClient(
            api_key=settings.api_football_key,
            base_url=settings.api_football_base_url,
            rate_limit_per_min=settings.api_football_rate_limit_per_min,
        )
        log.info("availability_adjustment_enabled", season=settings.api_football_season)

    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        form_service = FormService(client, settings)
        engine = StrategyEngine(settings)
        intl_engine = InternationalStrategyEngine(settings)
        club_engine = (
            ClubStrategyEngine(settings)
            if settings.club_ensemble_enabled
            else engine
        )

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
                # Engine routing: World Cup -> Glicko/DC intl ensemble;
                # domestic top-5 leagues -> club Glicko/DC ensemble;
                # Champions League -> naive form engine (cross-league club
                # ratings aren't calibrated against each other yet).
                is_intl = league in INTERNATIONAL_COMPETITIONS
                if is_intl:
                    eng = intl_engine
                elif league != "CL":
                    eng = club_engine
                else:
                    eng = engine
                for m in matches:
                    try:
                        bets = await _score_and_log_one(
                            m, league, form_service, eng, router,
                            settings, kill_tripped, live_orders,
                            make_signer_adapters, user_adapter_cache,
                            # Injury adjustment is WC-only: pass the client only
                            # for international competitions (None elsewhere).
                            af_client=af_client if is_intl else None,
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
            if af_client is not None:
                await af_client.close()

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
    live_orders: bool = False,
    make_signer_adapters=None,
    user_adapter_cache: dict | None = None,
    af_client: ApiFootballClient | None = None,
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

    # WC-only player-availability (injury) adjustment. Graceful: returns (0, 0)
    # when disabled, keyless, or on any fetch error — never breaks the
    # prediction. Only the InternationalStrategyEngine accepts the rating-shift
    # kwargs, so we pass them ONLY when there's a non-zero shift to apply; the
    # zero case calls predict(fixture_form) exactly as before, preserving
    # compatibility with the club engine (and any mock engine in tests).
    home_adj = away_adj = 0.0
    if league in INTERNATIONAL_COMPETITIONS:
        home_adj, away_adj = await _availability_adjustments(
            af_client, settings, home.name, away.name
        )
    if home_adj or away_adj:
        prediction = engine.predict(
            fixture_form, home_rating_adj=home_adj, away_rating_adj=away_adj
        )
    else:
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

    # ---- Market route (Phase 2 + Phase 5 live placement) -------------
    route = await router.find_best_route(
        prediction.home_team, prediction.away_team, kickoff, favourite
    )
    if route is not None:
        quote = route.quote
        # "Bet every match" mode (WC): bypass the edge filter for international
        # competitions so we take a position on every match that has a market.
        bet_every = (
            settings.international_bet_every_match
            and prediction.competition_code in INTERNATIONAL_COMPETITIONS
        )
        decision = engine.decide_with_market(
            prediction, favourite, quote.yes_price, require_edge=not bet_every
        )
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
        # Phase 5: place the real order on every active user's own wallet
        # (gated; skipped for WC and when not live).
        await _place_live_for_users(
            route, prediction, decision, settings, live_orders,
            make_signer_adapters, user_adapter_cache if user_adapter_cache is not None else {},
        )
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


async def _place_one(
    adapter, route, prediction, decision, stake_usd, max_price, *, who
) -> None:
    """Place a single real order on one venue adapter (one wallet)."""
    log = get_logger(__name__)
    try:
        result = await adapter.place_order(
            route.market, decision.outcome, stake_usd, max_price
        )
        log.info(
            "live_order_placed",
            who=who,
            fixture_id=prediction.fixture_id,
            exchange=route.quote.exchange.value,
            outcome=decision.outcome.value,
            stake_usd=round(stake_usd, 2),
            order_id=result.order_id,
            status=result.status,
        )
    except Exception as e:  # noqa: BLE001 — paper row persists even if order fails
        log.error(
            "live_order_failed",
            who=who,
            fixture_id=prediction.fixture_id,
            exchange=route.quote.exchange.value,
            error=str(e),
        )


async def _place_live_for_users(
    route, prediction, decision, settings, live_orders,
    make_signer_adapters, user_adapter_cache: dict,
) -> None:
    """Place the decided bet on EVERY active user's own wallet — heavily gated.

    Non-custodial multi-tenant placement: market discovery + the bet decision
    are wallet-independent and already done by the caller; here we re-sign and
    submit the same order from each user's isolated wallet, sized to that user's
    own balance. One user's order never touches another's funds.

    No-ops unless ``live_orders`` (live mode + gate clear). NEVER places for
    INTERNATIONAL_COMPETITIONS (World Cup) unless explicitly allowed — that
    guard is the hard rule. The paper bet has already been logged and persists
    regardless of whether any live order succeeds. When no users are registered,
    falls back to the single global agent wallet (backward-compatible).
    """
    log = get_logger(__name__)
    if not live_orders:
        return
    if (
        prediction.competition_code in INTERNATIONAL_COMPETITIONS
        and not settings.allow_international_live
    ):
        log.info(
            "live_order_skipped_international",
            fixture_id=prediction.fixture_id,
            competition=prediction.competition_code,
            note="set BETBOT_ALLOW_INTERNATIONAL_LIVE=true to enable",
        )
        return

    max_price = min(0.99, route.quote.yes_price + settings.order_slippage)
    venue = route.quote.exchange

    users = list_users()
    if not users:
        # Backward-compatible: trade the single global agent wallet. (No wallet
        # import on this path — keeps the web3 dependency off the single-wallet
        # flow and its tests.)
        await _place_one(
            route.adapter, route, prediction, decision,
            decision.stake_usd, max_price, who="agent",
        )
        return

    if make_signer_adapters is None:  # defensive — caller always supplies it live
        log.error("multi_user_placement_unconfigured", fixture_id=prediction.fixture_id)
        return

    # Per-user path: size each order against that user's balance on the venue's
    # chain. betbot.wallet pulls in web3, so import it only here.
    from betbot.wallet import get_private_key, load_limitless_creds, usdc_balance

    chain = "polygon" if venue == ExchangeName.POLYMARKET else "base"
    rpc = settings.polygon_rpc_url if chain == "polygon" else settings.base_rpc_url

    for u in users:
        who = f"user:{u.telegram_user_id}"
        # Per-user isolation: one user's bad key / RPC blip / adapter build error
        # must not abort placement for everyone else (or the next fixture). The
        # decided paper bet is already persisted regardless.
        try:
            key = get_private_key(Path(u.wallet_keyfile))
            if not key:
                log.warning("user_key_missing", who=who, keyfile=u.wallet_keyfile)
                continue
            bal = usdc_balance(u.wallet_address, chain, rpc)
            stake = min(decision.stake_usd, bal.usdc) if bal.ok else 0.0
            if stake < settings.min_user_stake_usd:
                log.info(
                    "user_bet_skipped_low_balance",
                    who=who, chain=chain,
                    balance=round(bal.usdc, 2) if bal.ok else None,
                    need=settings.min_user_stake_usd,
                )
                continue
            adapters = user_adapter_cache.get(u.wallet_keyfile)
            if adapters is None:
                # Each user trades their OWN Limitless account: pass THEIR linked
                # api_key/secret (or None → no Limitless leg for them). Polymarket
                # is always built and unaffected.
                creds = load_limitless_creds(
                    settings.secrets_dir, u.telegram_user_id
                )
                adapters = make_signer_adapters(
                    signing_key=key,
                    funder=u.wallet_address,
                    limitless_creds=creds,
                )
                user_adapter_cache[u.wallet_keyfile] = adapters
            adapter = adapters.get(venue)
            if adapter is None:
                # Venue is Limitless but this user hasn't linked a Limitless key.
                # Skip their Limitless leg only — their Polymarket leg (a separate
                # fixture/route) still trades. Logged once per (user, fixture).
                log.info(
                    "user_limitless_unlinked_skipped",
                    who=who,
                    fixture_id=prediction.fixture_id,
                    venue=str(getattr(venue, "value", venue)),
                    note="no linked Limitless key — skipping this user's Limitless leg",
                )
                continue
            await _place_one(
                adapter, route, prediction, decision, stake, max_price, who=who,
            )
        except Exception as e:  # noqa: BLE001 — isolate per-user placement failures
            log.error(
                "user_placement_failed",
                who=who,
                fixture_id=prediction.fixture_id,
                error=str(e),
            )


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


@app.command("calibration")
def calibration_cmd() -> None:
    """Show WC model calibration — RPS/Brier + reliability over settled matches."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    from betbot.calibration import build_report
    from betbot.reports import format_calibration_report
    from betbot.storage.repos import scored_model_predictions

    report = build_report(scored_model_predictions())
    # Strip Telegram Markdown for the terminal.
    text = format_calibration_report(report).replace("```", "").replace("*", "")
    typer.echo(text)


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
        typer.echo("No Glicko ratings yet. Seed with: python scripts/seed_glicko.py")
        return
    typer.echo(f"{'team':<28} {'rating':>8} {'RD':>6} {'vol':>7}")
    for name, r in rows:
        typer.echo(f"{name:<28} {r.rating:>8.1f} {r.rd:>6.1f} {r.volatility:>7.4f}")


@glicko_app.command("seed")
def glicko_seed() -> None:
    """Bootstrap ratings (Path 1 results-CSV, or Path 2 World-Cup teams)."""
    import runpy

    runpy.run_path(str(_REPO_ROOT / "scripts" / "seed_glicko.py"), run_name="__main__")


async def _run_arb_scan(settings, limit: int, min_margin: float) -> list:
    """Scan Polymarket + Limitless + SX Bet; return complete opportunities at or
    above ``min_margin``, best first. Builds + closes its own clients."""
    import re
    from datetime import datetime, timezone

    from betbot.exchanges.arbitrage import ArbScanner
    from betbot.exchanges.sxbet import SXBetAdapter, SXBetClient

    win = re.compile(r"will\s+(.+?)\s+win", re.IGNORECASE)
    router, clients, _make_adapters = _build_router(settings)
    gamma = clients[0]
    # SX Bet is scan-only — added to the arb scan, NOT the betting router.
    sx_client = SXBetClient()
    sx = SXBetAdapter(
        sx_client, TeamAliasResolver.from_yaml(_REPO_ROOT / "config" / "team_aliases.yaml")
    )
    clients = clients + [sx_client]
    scanner = ArbScanner(router.adapters + [sx])
    found = []
    try:
        events = await gamma.list_soccer_events(limit=limit)
        for e in events:
            teams = []
            for m in e.get("markets") or []:
                if m.get("closed"):
                    continue
                mt = win.search(m.get("question", "") or "")
                if mt:
                    teams.append(mt.group(1).strip())
            if len(teams) < 2:
                continue
            opp = await scanner.scan(teams[0], teams[1], datetime.now(timezone.utc))
            if opp and opp.complete and opp.margin >= min_margin:
                found.append(opp)
    finally:
        for c in clients:
            await c.close()
    found.sort(key=lambda o: o.margin, reverse=True)
    return found


async def _arb_scan(settings, limit: int, min_margin: float) -> None:
    found = await _run_arb_scan(settings, limit, min_margin)
    if not found:
        typer.echo("No cross-venue matches found across venues "
                   "(per-match overlap is rare — Limitless/SX list few matches).")
        return
    typer.echo(f"{'match':<32}{'sum':>6}{'margin':>9}   legs (outcome:venue@price)")
    for o in found[:25]:
        legs = "  ".join(f"{ov}:{lg.exchange.value[:4]}@{lg.price:.2f}"
                         for ov, lg in o.legs.items())
        tag = " ARB" if o.margin > 0 else ""
        typer.echo(f"{(o.home_team + ' v ' + o.away_team)[:31]:<32}"
                   f"{o.price_sum:>6.2f}{o.margin:>+8.1%}{tag}   {legs}")


async def _arb_scan_and_notify(settings) -> int:
    """Scan and push a Telegram alert for each real (margin>0) opportunity,
    then hand the opportunities to the (self-gating) executor."""
    from betbot.notify import send_telegram

    log = get_logger(__name__)
    found = await _run_arb_scan(
        settings, settings.arb_scan_limit, settings.arb_notify_min_margin
    )
    # Persist every hit so the 21:00 daily report can count today's total.
    persist_arb_opportunities(found)
    sent = 0
    for o in found:
        legs = "\n".join(f"  {ov}: {lg.exchange.value} @ {lg.price:.3f}"
                         for ov, lg in o.legs.items())
        msg = (f"🔔 *Arb opportunity*\n*{o.home_team} v {o.away_team}*\n"
               f"locked margin *{o.margin:+.1%}* (cost {o.price_sum:.3f})\n{legs}")
        if await send_telegram(settings, msg):
            sent += 1
    if found:
        log.info("arb_notified", opportunities=len(found), sent=sent)
        await _maybe_execute_arbs(settings, found)
    return sent


async def _maybe_execute_arbs(settings, opportunities: list) -> None:
    """Hand scanned opportunities to the gated arb executor.

    The executor self-gates (BETBOT_ARB_EXECUTE + live mode + kill switch +
    margin/sizing/balance/daily-cap checks) and notifies Telegram on every
    execution or abort. The flag check here just avoids building live adapters
    on every tick while execution is off (the default).
    """
    if not opportunities or not settings.arb_execute:
        return
    from betbot.arb_executor import ArbExecutor

    log = get_logger(__name__)
    init_engine(settings.db_path)
    router, clients, _make_adapters = _build_router(settings)
    adapters = {a.name: a for a in router.adapters}
    try:
        executor = ArbExecutor(settings, adapters)
        results = await executor.execute_all(opportunities)
        log.info(
            "arb_execute_tick_done",
            considered=len(opportunities),
            statuses=[r.status for r in results],
        )
    finally:
        for c in clients:
            await c.close()


@arb_app.command("scan")
def arb_scan(
    limit: Annotated[int, typer.Option(help="max soccer events to scan")] = 100,
    min_margin: Annotated[float, typer.Option(help="report at/above this margin")] = -0.05,
) -> None:
    """Scan for cross-venue arbitrage (read-only). Positive margin = locked profit."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_arb_scan(settings, limit, min_margin))


@arb_app.command("watch")
def arb_watch() -> None:
    """One-shot scan + Telegram alert per opportunity (the daemon also does this
    automatically every BETBOT_ARB_SCAN_INTERVAL_MIN minutes)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)  # scan results are persisted for the daily report
    n = asyncio.run(_arb_scan_and_notify(settings))
    typer.echo(f"Notified {n} opportunity(ies).")


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

    async def _arb_tick() -> None:
        try:
            await _arb_scan_and_notify(get_settings())
        except Exception as e:  # noqa: BLE001 — never let the watcher crash the daemon
            get_logger(__name__).warning("arb_tick_failed", error=str(e))

    async def _arb_digest_tick() -> None:
        # 09:00 Africa/Nairobi: one scan + digest to operator AND all users.
        try:
            s = get_settings()
            await run_arb_digest(
                s,
                lambda: _run_arb_scan(s, s.arb_scan_limit, s.arb_notify_min_margin),
            )
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("arb_digest_failed", error=str(e))

    async def _daily_report_tick() -> None:
        # 21:00 Africa/Nairobi: trades/settlements/P&L/arb-count/balances.
        try:
            await run_daily_report(get_settings())
        except Exception as e:  # noqa: BLE001 — never crash the daemon
            get_logger(__name__).warning("daily_report_failed", error=str(e))

    async def _deposit_tick() -> None:
        # Deposit pipeline: detect new user USDC, CCTP-bridge it to the
        # trading chains, run venue approvals. Lazy import — betbot.bridge
        # pulls in web3 (the `api` extra), which the base install lacks.
        try:
            from betbot.bridge import run_deposit_scan

            await run_deposit_scan(get_settings())
        except Exception as e:  # noqa: BLE001 — never let the scanner crash the daemon
            get_logger(__name__).warning("deposit_tick_failed", error=str(e))

    async def _main() -> None:
        s = get_settings()
        init_engine(s.db_path)  # cron jobs may fire before the first scoring tick
        scheduler = AsyncIOScheduler(timezone=timezone.utc)
        scheduler.add_job(_tick, trigger=trigger, id="score_and_settle")
        scheduler.add_job(
            _arb_tick,
            trigger=IntervalTrigger(minutes=s.arb_scan_interval_min),
            id="arb_watch",
        )
        register_daily_jobs(
            scheduler, s,
            arb_digest=_arb_digest_tick,
            daily_report=_daily_report_tick,
        )
        scheduler.add_job(
            _deposit_tick,
            trigger=IntervalTrigger(minutes=s.deposit_scan_minutes),
            id="deposit_scan",
        )
        scheduler.start()
        log.info(
            "daemon_started",
            cron=cron_expr,
            arb_scan_min=s.arb_scan_interval_min,
            arb_digest_hour_nairobi=s.arb_digest_hour if s.arb_digest_enabled else None,
            daily_report_hour_nairobi=(
                s.daily_report_hour if s.daily_report_enabled else None
            ),
            deposit_scan_min=s.deposit_scan_minutes,
        )
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
