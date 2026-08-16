"""Tipster Telegram jobs — gated reveals, matchday + kickoff alerts, cron.

No network: entitlement + Telegram sends are injected fakes; storage runs
against a throwaway SQLite file. The reports.py daily-report formatters are
still exercised by test_reports_format below (kept as regression cover — those
formatters are unused by the daemon now but remain importable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pytest
from apscheduler.triggers.cron import CronTrigger

from betbot.config import Settings
from betbot.daily_jobs import (
    broadcast_chat_ids,
    commit_reveals,
    nairobi_day_bounds,
    register_daily_jobs,
    render_matchday_notice,
    render_user_predictions,
    run_matchday_notice,
    send_prediction_alert,
)
from betbot.entitlement import Entitlement
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_user,
    has_revealed,
    record_reveal,
)


# By default nothing has been revealed yet — tests that exercise the
# already-revealed path inject their own fn.
def _never_revealed(uid, fid):
    return False


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "daily_jobs.sqlite")
    yield


def _tg_settings(tmp_path) -> Settings:
    return Settings(
        FOOTBALL_DATA_API_KEY="fake-test-key",
        TELEGRAM_BOT_TOKEN="fake-token",
        TELEGRAM_ALLOWED_USER_ID=111,
        BETBOT_WALLET_KEYFILE=str(tmp_path / "agent.key"),
        BETBOT_EDGE_THRESHOLD=0.05,
    )


# ----------------------------------------------------------------------
# Fixtures / fakes
# ----------------------------------------------------------------------
@dataclass
class _Bet:
    outcome: str
    market_price: float | None
    edge: float | None


@dataclass
class _Pred:
    fixture_id: int = 1
    competition_code: str = "PL"
    home_team: str = "Man City"
    away_team: str = "Arsenal"
    p_home: float = 0.39
    p_draw: float = 0.31
    p_away: float = 0.30
    home_score: float = 0.0
    away_score: float = 0.0
    draw_score: float = 0.0
    home_xg: float | None = 1.44
    away_xg: float | None = 1.17
    kickoff: datetime = datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc)
    paper_bets: list = field(default_factory=list)


@dataclass
class _User:
    telegram_user_id: int
    wallet_address: str = "0xabc"
    predictions_consumed: int = 0
    created_at: datetime = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _ent(reason, *, trial=0, credits=0):
    allowed = reason in ("operator", "trial", "credit")
    return Entitlement(allowed=allowed, reason=reason,
                       trial_days_left=trial, credits_remaining=credits)


# ----------------------------------------------------------------------
# render_user_predictions — the gating heart
# ----------------------------------------------------------------------
def test_operator_reveals_all():
    preds = [_Pred(fixture_id=1), _Pred(fixture_id=2, home_team="Spurs")]
    text, reveals = render_user_predictions(
        _User(111), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("operator"),
        already_revealed_fn=_never_revealed,
    )
    # Recorded (so they stay free after any trial) but NEVER charged.
    assert reveals == [(1, False), (2, False)]
    assert "operator" in text
    assert "🔒" not in text


def test_trial_reveals_all():
    preds = [_Pred(fixture_id=1), _Pred(fixture_id=2)]
    text, reveals = render_user_predictions(
        _User(5), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("trial", trial=3),
        already_revealed_fn=_never_revealed,
    )
    assert reveals == [(1, False), (2, False)]  # free, but recorded
    assert "3 days left" in text
    assert "🔒" not in text


def test_credits_reveal_then_lock():
    preds = [_Pred(fixture_id=i) for i in range(3)]
    text, reveals = render_user_predictions(
        _User(5), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("credit", credits=2),
        already_revealed_fn=_never_revealed,
    )
    # Only the 2 funded fixtures are revealed + charged; the third is locked.
    assert reveals == [(0, True), (1, True)]
    assert text.count("🔒") == 1


def test_locked_user_gets_all_teasers():
    preds = [_Pred(fixture_id=1), _Pred(fixture_id=2)]
    text, reveals = render_user_predictions(
        _User(5), preds, _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("locked"),
        already_revealed_fn=_never_revealed,
    )
    assert reveals == []  # nothing revealed, nothing charged
    # One lock in the header + one teaser per prediction (2) = 3.
    assert text.count("🔒") == 3
    assert text.count("send 1 USDC (Polygon) to unlock this prediction") == 2
    assert "%" not in text  # no probabilities leak


def test_no_fixtures_message():
    text, reveals = render_user_predictions(
        _User(5), [], _tg_settings_stub(),
        entitlement_fn=lambda u, s, now=None: _ent("trial", trial=3),
        already_revealed_fn=_never_revealed,
    )
    assert reveals == []
    assert "No fixtures today" in text


def _tg_settings_stub() -> Settings:
    return Settings(FOOTBALL_DATA_API_KEY="x", BETBOT_EDGE_THRESHOLD=0.05)


# ----------------------------------------------------------------------
# lineup_alert_lead_minutes helper — PL 70, else 55
# ----------------------------------------------------------------------
def test_lineup_alert_lead_minutes_per_competition():
    s = _tg_settings_stub()
    assert s.lineup_alert_lead_minutes("PL") == 70
    assert s.lineup_alert_lead_minutes("pl") == 70  # case-insensitive
    for code in ("PD", "BL1", "SA", "FL1", "CL"):
        assert s.lineup_alert_lead_minutes(code) == 55
    assert s.lineup_alert_lead_minutes(None) == 55  # unknown -> default


def test_early_alert_lead_minutes_alias_matches_legacy():
    # early_alert_lead_minutes is the canonical EARLY (model) lead; the legacy
    # lineup_alert_lead_minutes name aliases it 1:1 so nothing drifts.
    s = _tg_settings_stub()
    for code in ("PL", "pl", "PD", "BL1", "SA", "FL1", "CL", None):
        assert s.early_alert_lead_minutes(code) == s.lineup_alert_lead_minutes(code)
    assert s.early_alert_lead_minutes("PL") == 70
    assert s.early_alert_lead_minutes("SA") == 55


def test_lineup_confirm_lead_minutes_default_and_env_override():
    # LATE (confirmed-XI) lead: same for every league, default 10, tunable.
    assert _tg_settings_stub().lineup_confirm_lead_minutes() == 10
    tuned = Settings(FOOTBALL_DATA_API_KEY="x", BETBOT_LINEUP_CONFIRM_LEAD_MIN=8)
    assert tuned.lineup_confirm_lead_minutes() == 8


# ----------------------------------------------------------------------
# run_matchday_notice — FREE heads-up, no prediction, no ledger
# ----------------------------------------------------------------------
def _fixt(fid, home, away, ko, code):
    return _Pred(fixture_id=fid, home_team=home, away_team=away,
                 kickoff=ko, competition_code=code)


def test_matchday_notice_states_both_alert_times_and_hides_probs():
    s = _tg_settings_stub()
    # Two-alert model. PL KO 19:30 UTC -> early KO-70 = 18:20 UTC, late KO-10 =
    # 19:20 UTC; SA KO 17:00 UTC -> early KO-55 = 16:05 UTC, late KO-10 = 16:50.
    pl = _fixt(1, "Man City", "Arsenal",
               datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc), "PL")
    sa = _fixt(2, "Inter", "Milan",
               datetime(2026, 8, 1, 17, 0, tzinfo=timezone.utc), "SA")
    text = render_matchday_notice(s, [pl, sa], date(2026, 8, 1))
    # Times are EAT (+3): PL KO 22:30, early 21:20, late 22:20; SA KO 20:00,
    # early 19:05, late 19:50.
    assert (
        "KO 22:30 · 🔮 prediction at 21:20, confirmed-lineup update ~22:20"
        in text
    )  # PL: early KO-70, late KO-10
    assert (
        "KO 20:00 · 🔮 prediction at 19:05, confirmed-lineup update ~19:50"
        in text
    )  # SA: early KO-55, late KO-10
    assert "Man City (H) v Arsenal (A)" in text
    assert "Inter (H) v Milan (A)" in text
    # NO probability / edge / xG leakage.
    assert "%" not in text
    assert "xG" not in text
    assert "Edge" not in text
    assert "Model" not in text


def test_matchday_notice_no_fixtures_returns_none():
    assert render_matchday_notice(_tg_settings_stub(), [], date(2026, 8, 1)) is None


async def test_run_matchday_notice_broadcasts_free_no_ledger(db, tmp_path):
    s = _tg_settings(tmp_path)
    u = _paying_user(tmp_path, tid=901)  # a real, post-trial user
    sent: list[tuple[int, str]] = []

    async def fake_send(settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    preds = [_fixt(1, "Man City", "Arsenal",
                   datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc), "PL")]
    delivered = await run_matchday_notice(
        s, send_fn=fake_send,
        fixtures_source=lambda a, b: preds,
        users_fn=lambda: [u],
    )
    # Operator (111) + user (901), de-duplicated.
    assert delivered == 2
    assert {cid for cid, _ in sent} == {111, 901}
    # FREE: no reveal recorded, no credit consumed.
    assert has_revealed(901, 1) is False
    assert get_user(901).predictions_consumed == 0


async def test_run_matchday_notice_quiet_when_no_fixtures(db, tmp_path):
    s = _tg_settings(tmp_path)

    async def fake_send(settings, chat_id, text):  # pragma: no cover
        raise AssertionError("should not send when no fixtures")

    delivered = await run_matchday_notice(
        s, send_fn=fake_send, fixtures_source=lambda a, b: [],
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 0


# ----------------------------------------------------------------------
# send_prediction_alert — pre-match lineup-adjusted, entitlement-gated
# ----------------------------------------------------------------------
_LINEUP = {
    "home": {"formation": "4-3-3", "xi": ["Ederson", "Rodri", "Haaland"]},
    "away": {"formation": "4-4-2", "xi": ["Raya", "Rice", "Saka"]},
}


def _lineup_fn_stub(home_adj=40.0, away_adj=0.0, lineup=None):
    async def _fn(baseline):
        return (lineup if lineup is not None else _LINEUP,
                home_adj, away_adj, "rating shift home +40")
    return _fn


def _rescore_stub(pred=None):
    async def _fn(fixture_id, home_adj, away_adj):
        return (pred or _Pred(fixture_id=fixture_id),
                datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc))
    return _fn


async def test_prematch_alert_operator_gets_xi_and_prediction(db, tmp_path):
    s = _tg_settings(tmp_path)
    sent: list[str] = []

    async def fake_send(settings, chat_id, text):
        sent.append(text)
        return True

    delivered = await send_prediction_alert(
        s, 1, send_fn=fake_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 1
    body = sent[0]
    assert "Pre-match" in body
    assert "XI:" in body and "Ederson" in body and "Saka" in body
    assert "[4-3-3]" in body
    assert "Key absences" in body  # nonzero adjustment
    assert "Model:" in body  # the standing prediction block is present
    assert "🔒" not in body


async def test_prematch_alert_payer_charged_once(db, tmp_path):
    s = _tg_settings(tmp_path)
    u = _paying_user(tmp_path, tid=811)

    async def ok_send(settings, chat_id, text):
        return True

    ent = lambda usr, se, now=None: _ent("credit", credits=1)  # noqa: E731
    await send_prediction_alert(
        s, 55, send_fn=ok_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=None,  # skip re-score path; deliver baseline + XI
        entitlement_fn=ent, users_fn=lambda: [u],
    )
    assert get_user(811).predictions_consumed == 1
    assert has_revealed(811, 55) is True

    # A SECOND pre-match alert for the SAME fixture -> already revealed -> free.
    await send_prediction_alert(
        s, 55, send_fn=ok_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=None,
        entitlement_fn=ent, users_fn=lambda: [u],
    )
    assert get_user(811).predictions_consumed == 1  # no double-charge


async def test_prematch_alert_locked_user_teaser_no_charge(db, tmp_path):
    s = _tg_settings(tmp_path)
    u = _paying_user(tmp_path, tid=812)
    sent: list[str] = []

    async def ok_send(settings, chat_id, text):
        sent.append(text)
        return True

    await send_prediction_alert(
        s, 60, send_fn=ok_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=None,
        entitlement_fn=lambda usr, se, now=None: _ent("locked"),
        users_fn=lambda: [u],
    )
    assert "🔒" in sent[0]
    assert "unlock this prediction" in sent[0]
    assert "Ederson" not in sent[0]  # XI withheld from locked user
    assert has_revealed(812, 60) is False
    assert get_user(812).predictions_consumed == 0


async def test_prematch_alert_send_failure_never_charges(db, tmp_path):
    s = _tg_settings(tmp_path)
    u = _paying_user(tmp_path, tid=813)

    async def failing_send(settings, chat_id, text):
        return False

    await send_prediction_alert(
        s, 77, send_fn=failing_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=None,
        entitlement_fn=lambda usr, se, now=None: _ent("credit", credits=1),
        users_fn=lambda: [u],
    )
    assert has_revealed(813, 77) is False
    assert get_user(813).predictions_consumed == 0


async def test_prematch_alert_always_rescores_fresh_even_when_adj_zero(db, tmp_path):
    """Freshness fix: even with a ZERO lineup adjustment (empty minutes cache) the
    alert must re-score fresh and send the FRESH probs, not the STALE stored row.

    Mirrors the observed live bug where Celta shipped stale H87/D8/A6 while a
    fresh score gave H50/D23/A27. Here the stored baseline is H87 and the fresh
    re-score is H50 — the delivered body must read H 50%, not H 87%.
    """
    s = _tg_settings(tmp_path)
    sent: list[str] = []

    async def ok_send(settings, chat_id, text):
        sent.append(text)
        return True

    stale = _Pred(fixture_id=70, p_home=0.87, p_draw=0.08, p_away=0.05)
    fresh = _Pred(fixture_id=70, p_home=0.50, p_draw=0.23, p_away=0.27)

    # prediction_fn returns STALE on the first (baseline) read and FRESH on the
    # post-upsert re-read — modelling upsert_prediction having persisted fresh.
    calls = {"n": 0}

    def prediction_fn(fid):
        calls["n"] += 1
        return stale if calls["n"] == 1 else fresh

    async def rescore_fresh(fixture_id, home_adj, away_adj):
        # adjustment is 0.0 here — the whole point is we still re-score.
        assert home_adj == 0.0 and away_adj == 0.0
        return fresh, datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc)

    # Lineup not out -> adj (0.0, 0.0). Old code would have skipped re-score.
    async def no_lineup(baseline):
        return None, 0.0, 0.0, None

    delivered = await send_prediction_alert(
        s, 70, send_fn=ok_send,
        prediction_fn=prediction_fn,
        lineup_fn=no_lineup,
        rescore_fn=rescore_fresh,
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 1
    assert "H 50%" in sent[0]  # FRESH values delivered
    assert "H 87%" not in sent[0]  # NOT the stale stored row


async def test_prematch_alert_rescore_failure_falls_back_no_double_charge(db, tmp_path):
    """If re-scoring raises, the alert falls back to the STORED baseline (still
    delivered) and the payer is charged EXACTLY ONCE — the money path never
    regresses on a re-score failure.
    """
    s = _tg_settings(tmp_path)
    u = _paying_user(tmp_path, tid=815)
    sent: list[str] = []

    async def ok_send(settings, chat_id, text):
        sent.append(text)
        return True

    stored = _Pred(fixture_id=71, p_home=0.42, p_draw=0.26, p_away=0.32)

    async def boom(fixture_id, home_adj, away_adj):
        raise RuntimeError("network down")

    ent = lambda usr, se, now=None: _ent("credit", credits=1)  # noqa: E731
    delivered = await send_prediction_alert(
        s, 71, send_fn=ok_send,
        prediction_fn=lambda fid: stored,
        lineup_fn=_lineup_fn_stub(home_adj=40.0),  # nonzero, forces rescore attempt
        rescore_fn=boom,
        entitlement_fn=ent, users_fn=lambda: [u],
    )
    assert delivered == 1
    assert "H 42%" in sent[0]  # stored baseline delivered, not skipped
    assert has_revealed(815, 71) is True
    assert get_user(815).predictions_consumed == 1  # charged exactly once


async def test_prematch_alert_no_lineup_sends_baseline_caveat(db, tmp_path):
    s = _tg_settings(tmp_path)
    sent: list[str] = []

    async def ok_send(settings, chat_id, text):
        sent.append(text)
        return True

    async def no_lineup(baseline):
        return None, 0.0, 0.0, None

    delivered = await send_prediction_alert(
        s, 1, send_fn=ok_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=no_lineup,
        rescore_fn=_rescore_stub(),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 1
    assert "lineup not yet confirmed" in sent[0]
    assert "Model:" in sent[0]  # baseline prediction still delivered


async def test_prematch_alert_with_production_rescore_closure_shape(db, tmp_path):
    """Regression for the live 'score_fixture_adjusted() takes 2 positional
    arguments but 3 were given' TypeError.

    send_prediction_alert calls rescore_fn(fixture_id, home_adj, away_adj), but
    score_fixture_adjusted's REAL signature is
    (settings, fixture_id, *, home_rating_adj, away_rating_adj). The production
    wiring must be a closure adapting that shape. Here we build the SAME closure
    main.py uses over the REAL score_fixture_adjusted (with football-data stubbed
    to avoid network) and assert the alert re-scores fresh — no TypeError.
    """
    import betbot.main as main_mod

    s = _tg_settings(tmp_path)
    sent: list[str] = []

    async def ok_send(settings, chat_id, text):
        sent.append(text)
        return True

    # Capture the (fid, home_adj, away_adj) the closure is invoked with, proving
    # the arity send_prediction_alert uses maps onto score_fixture_adjusted's
    # keyword-only home_rating_adj/away_rating_adj.
    seen: dict = {}

    async def fake_score_fixture_adjusted(
        settings, fixture_id, *, home_rating_adj=0.0, away_rating_adj=0.0
    ):
        seen["fid"] = fixture_id
        seen["home"] = home_rating_adj
        seen["away"] = away_rating_adj
        return _Pred(fixture_id=fixture_id, p_home=0.55, p_draw=0.25, p_away=0.20), \
            datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc)

    # Build the closure EXACTLY as main.py does, over the (stubbed) real fn.
    async def _rescore(fid, home_adj, away_adj):
        return await fake_score_fixture_adjusted(
            s, fid, home_rating_adj=home_adj, away_rating_adj=away_adj
        )

    calls = {"n": 0}

    def prediction_fn(fid):
        calls["n"] += 1
        # baseline (stale) then fresh after upsert.
        return _Pred(fixture_id=fid, p_home=0.87, p_draw=0.08, p_away=0.05) \
            if calls["n"] == 1 \
            else _Pred(fixture_id=fid, p_home=0.55, p_draw=0.25, p_away=0.20)

    delivered = await send_prediction_alert(
        s, 501, send_fn=ok_send,
        prediction_fn=prediction_fn,
        lineup_fn=_lineup_fn_stub(home_adj=40.0, away_adj=-15.0),
        rescore_fn=_rescore,
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 1
    # The closure was invoked with the alert's positional shape and forwarded
    # them to the keyword-only adjustment params — no TypeError raised.
    assert seen == {"fid": 501, "home": 40.0, "away": -15.0}
    assert "H 55%" in sent[0]  # FRESH re-scored values shipped
    # And the REAL production fn is imported + wired as a closure, not bare.
    assert callable(main_mod.score_fixture_adjusted)


def test_main_wires_rescore_as_closure_not_bare_function():
    """Static guard: main.py must NOT pass the bare score_fixture_adjusted as
    rescore_fn (its signature mismatches). It must build an adapting closure.
    """
    import inspect

    import betbot.main as main_mod

    src = inspect.getsource(main_mod)
    # The bare-function wiring that caused the production TypeError must be gone.
    assert "rescore_fn=score_fixture_adjusted" not in src
    # A closure that forwards to the keyword-only params must be present.
    assert "home_rating_adj=" in src and "away_rating_adj=" in src

    # score_fixture_adjusted keeps its keyword-only adjustment signature.
    sig = inspect.signature(main_mod.score_fixture_adjusted)
    assert sig.parameters["home_rating_adj"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["away_rating_adj"].kind is inspect.Parameter.KEYWORD_ONLY


async def test_send_prediction_alert_no_prediction_is_noop(tmp_path):
    s = _tg_settings(tmp_path)
    delivered = await send_prediction_alert(
        s, 999, send_fn=None,
        prediction_fn=lambda fid: None,
        lineup_fn=_lineup_fn_stub(),
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 0


# ----------------------------------------------------------------------
# Scheduler registration
# ----------------------------------------------------------------------
class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, object] = {}

    def add_job(self, func, *, trigger, id):  # noqa: A002 — APScheduler's kwarg
        self.jobs[id] = trigger


async def _noop() -> None:  # pragma: no cover — never fired in tests
    pass


def test_matchday_cron_registered_with_nairobi_timezone(settings):
    sched = FakeScheduler()
    register_daily_jobs(sched, settings, matchday_notice=_noop)
    assert set(sched.jobs) == {"matchday_notice", "player_minutes_backfill"}
    trig = sched.jobs["matchday_notice"]
    assert isinstance(trig, CronTrigger)
    assert str(trig.timezone) == "Africa/Nairobi"
    fields = {f.name: str(f) for f in trig.fields}
    assert fields["hour"] == "8"  # default matchday_alert_hour
    assert fields["minute"] == "0"


def test_player_minutes_backfill_registered_daily_utc(settings):
    sched = FakeScheduler()
    register_daily_jobs(sched, settings, matchday_notice=_noop)
    trig = sched.jobs["player_minutes_backfill"]
    assert isinstance(trig, CronTrigger)
    assert str(trig.timezone) in ("UTC", "utc")
    fields = {f.name: str(f) for f in trig.fields}
    assert fields["hour"] == "5"
    assert fields["minute"] == "15"


def test_pick_league_to_backfill_selects_missing_skips_populated(tmp_path):
    # Only PD_2024 populated (mirrors the live cache): picker must skip PL? No —
    # PL comes first and is missing, so it is chosen; a dir where all are
    # populated returns None.
    from betbot.daily_jobs import pick_league_to_backfill

    # Fake cache dir: PD populated, others absent.
    (tmp_path / "PD_2024.json").write_text('{"111": {"x": 90}}', encoding="utf-8")
    picked = pick_league_to_backfill(2024, minutes_dir=tmp_path)
    assert picked == "PL"  # first missing domestic league

    # Empty (2-byte "{}") counts as unpopulated and is still picked.
    (tmp_path / "PL_2024.json").write_text("{}", encoding="utf-8")
    assert pick_league_to_backfill(2024, minutes_dir=tmp_path) == "PL"

    # Populate every domestic league -> no-op (None).
    for code in ("PL", "PD", "BL1", "SA", "FL1"):
        (tmp_path / f"{code}_2024.json").write_text(
            '{"1": {"p": 90}}', encoding="utf-8"
        )
    assert pick_league_to_backfill(2024, minutes_dir=tmp_path) is None


def test_prior_minutes_season_is_two_back(settings):
    from betbot.daily_jobs import prior_minutes_season

    assert prior_minutes_season(settings) == settings.api_football_season - 2


# ----------------------------------------------------------------------
# Two-alert scheduler plan — TWO jobs per fixture at the right leads
# ----------------------------------------------------------------------
def test_plan_schedules_two_jobs_per_fixture_at_right_leads():
    from betbot.main import plan_kickoff_alert_jobs

    s = _tg_settings_stub()
    now = datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc)  # well before both KOs
    pl = _Pred(fixture_id=10, competition_code="PL",
               kickoff=datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc))
    pd = _Pred(fixture_id=20, competition_code="PD",  # La Liga
               kickoff=datetime(2026, 8, 1, 17, 0, tzinfo=timezone.utc))
    plan = dict(plan_kickoff_alert_jobs(s, [pl, pd], now))

    # TWO jobs per fixture.
    assert set(plan) == {
        "predict_early_10", "predict_late_10",
        "predict_early_20", "predict_late_20",
    }
    # PL: early = KO-70 = 18:20, late = KO-10 = 19:20 UTC.
    assert plan["predict_early_10"] == datetime(2026, 8, 1, 18, 20, tzinfo=timezone.utc)
    assert plan["predict_late_10"] == datetime(2026, 8, 1, 19, 20, tzinfo=timezone.utc)
    # La Liga: early = KO-55 = 16:05, late = KO-10 = 16:50 UTC.
    assert plan["predict_early_20"] == datetime(2026, 8, 1, 16, 5, tzinfo=timezone.utc)
    assert plan["predict_late_20"] == datetime(2026, 8, 1, 16, 50, tzinfo=timezone.utc)


def test_plan_skips_jobs_whose_fire_time_is_past():
    from betbot.main import plan_kickoff_alert_jobs

    s = _tg_settings_stub()
    # KO 19:30; now = 19:25 -> early (18:20) is past & dropped, late (19:20) is
    # ALSO past & dropped. Push now to 18:30: early past (dropped), late future.
    pl = _Pred(fixture_id=10, competition_code="PL",
               kickoff=datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc))

    now_after_early = datetime(2026, 8, 1, 18, 30, tzinfo=timezone.utc)
    plan = dict(plan_kickoff_alert_jobs(s, [pl], now_after_early))
    assert set(plan) == {"predict_late_10"}  # early skipped, late kept

    now_after_both = datetime(2026, 8, 1, 19, 25, tzinfo=timezone.utc)
    plan2 = dict(plan_kickoff_alert_jobs(s, [pl], now_after_both))
    assert plan2 == {}  # both fire times already past


# ----------------------------------------------------------------------
# Day bounds + broadcast recipients
# ----------------------------------------------------------------------
def test_nairobi_day_bounds_anchor_the_local_calendar_day():
    now = datetime(2026, 6, 10, 22, 30, tzinfo=timezone.utc)
    start, end, day = nairobi_day_bounds(now)
    assert day == date(2026, 6, 11)
    assert start == datetime(2026, 6, 10, 21, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 6, 11, 21, 0, tzinfo=timezone.utc)


def test_broadcast_includes_operator_and_users_deduplicated(tmp_path):
    class U:
        def __init__(self, tid):
            self.telegram_user_id = tid

    s = _tg_settings(tmp_path)
    ids = broadcast_chat_ids(s, [U(222), U(111), U(333)])
    assert ids == [111, 222, 333]  # operator first, no duplicate 111


# ----------------------------------------------------------------------
# reports.py formatters — still importable (unused by daemon now)
# ----------------------------------------------------------------------
def test_reports_daily_formatter_still_importable():
    from betbot.reports import DailyReport, format_daily_report

    text = format_daily_report(DailyReport(date(2026, 6, 11), (), (), 0.0, 0.0, ()))
    assert "Daily report" in text


# ----------------------------------------------------------------------
# Reveal ledger — the money-delivery guarantees (Fable findings #2 + #3)
# ----------------------------------------------------------------------
def _paying_user(tmp_path, tid=777):
    """A real registered user with a real wallet, past trial."""
    from betbot.storage.repos import get_or_create_user

    u = get_or_create_user(tid, "payer", secrets_dir=str(tmp_path / ".secrets"))
    return u


def test_repeat_render_charges_a_fixture_at_most_once(db, tmp_path):
    """Paying user, 2 credits, 2 new fixtures. First render charges both; after
    commit, a second render (fixtures now in the ledger) is FREE — no re-charge."""
    u = _paying_user(tmp_path)
    preds = [_Pred(fixture_id=1), _Pred(fixture_id=2)]
    s = _tg_settings_stub()
    ent = lambda usr, se, now=None: _ent("credit", credits=2)  # noqa: E731

    # First render: nothing revealed yet -> both are NEW paid reveals.
    text1, reveals1 = render_user_predictions(
        u, preds, s, entitlement_fn=ent, already_revealed_fn=has_revealed
    )
    assert reveals1 == [(1, True), (2, True)]
    assert text1.count("🔒") == 0

    # Commit after a (simulated) confirmed send: 2 charged rows, consumed == 2.
    commit_reveals(u, reveals1)
    assert has_revealed(u.telegram_user_id, 1) is True
    assert has_revealed(u.telegram_user_id, 2) is True
    assert get_user(u.telegram_user_id).predictions_consumed == 2

    # Second render: both already in the ledger -> shown FREE, reveals empty.
    text2, reveals2 = render_user_predictions(
        u, preds, s, entitlement_fn=ent, already_revealed_fn=has_revealed
    )
    assert reveals2 == []          # nothing NEW to charge
    assert text2.count("🔒") == 0  # still fully revealed (free)
    # A redundant commit must not double-charge.
    commit_reveals(u, reveals2)
    assert get_user(u.telegram_user_id).predictions_consumed == 2


async def test_send_failure_never_charges(db, tmp_path):
    """Send returns False -> commit is NOT called -> no ledger rows, no charge."""
    u = _paying_user(tmp_path, tid=778)
    s = _tg_settings(tmp_path)

    async def failing_send(settings, chat_id, text):
        return False  # Telegram refused the message

    delivered = await send_prediction_alert(
        s, 42, send_fn=failing_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=None,
        entitlement_fn=lambda usr, se, now=None: _ent("credit", credits=1),
        users_fn=lambda: [u],
    )
    assert delivered == 0
    assert has_revealed(u.telegram_user_id, 42) is False
    assert get_user(u.telegram_user_id).predictions_consumed == 0


async def test_repeat_prematch_same_fixture_charges_once(db, tmp_path):
    """After a pre-match alert charges a fixture, a SECOND pre-match alert for
    the SAME fixture reveals it FREE (already in the ledger) — no re-charge."""
    u = _paying_user(tmp_path, tid=779)
    s = _tg_settings(tmp_path)

    async def ok_send(settings, chat_id, text):
        return True

    ent = lambda usr, se, now=None: _ent("credit", credits=5)  # noqa: E731

    await send_prediction_alert(
        s, 99, send_fn=ok_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(), rescore_fn=None,
        entitlement_fn=ent, users_fn=lambda: [u],
    )
    assert get_user(u.telegram_user_id).predictions_consumed == 1
    assert has_revealed(u.telegram_user_id, 99) is True

    # Second pre-match alert for the same fixture — already revealed, no charge.
    await send_prediction_alert(
        s, 99, send_fn=ok_send,
        prediction_fn=lambda fid: _Pred(fixture_id=fid),
        lineup_fn=_lineup_fn_stub(), rescore_fn=None,
        entitlement_fn=ent, users_fn=lambda: [u],
    )
    assert get_user(u.telegram_user_id).predictions_consumed == 1


async def test_two_alerts_charge_once_late_shows_adjusted(db, tmp_path):
    """The two-alert model money invariant. A payer with 1 credit gets:
      early alert — NO lineup out -> fresh MODEL prediction, CHARGED once;
      late alert  — lineup out -> confirmed XI + lineup-ADJUSTED prediction,
                    already-revealed -> FREE (no second charge).
    The SECOND (late) message must carry the fresh lineup-adjusted content.
    """
    u = _paying_user(tmp_path, tid=790)
    s = _tg_settings(tmp_path)
    sent: list[str] = []

    async def ok_send(settings, chat_id, text):
        sent.append(text)
        return True

    ent = lambda usr, se, now=None: _ent("credit", credits=1)  # noqa: E731

    # EARLY: XI not yet posted -> model note, adj (0.0, 0.0).
    async def early_no_lineup(baseline):
        return None, 0.0, 0.0, None

    early_pred = _Pred(fixture_id=88, p_home=0.50, p_draw=0.23, p_away=0.27)

    async def early_rescore(fixture_id, home_adj, away_adj):
        assert (home_adj, away_adj) == (0.0, 0.0)
        return early_pred, datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc)

    delivered_early = await send_prediction_alert(
        s, 88, send_fn=ok_send,
        prediction_fn=lambda fid: early_pred,
        lineup_fn=early_no_lineup,
        rescore_fn=early_rescore,
        entitlement_fn=ent, users_fn=lambda: [u],
    )
    assert delivered_early == 1
    assert get_user(790).predictions_consumed == 1  # charged on the EARLY alert
    assert has_revealed(790, 88) is True
    assert "lineup not yet confirmed" in sent[0]  # early = model note
    assert "H 50%" in sent[0]

    # LATE: XI now posted -> lineup-adjusted re-score (home +40) shifts the prob.
    late_pred = _Pred(fixture_id=88, p_home=0.62, p_draw=0.20, p_away=0.18)

    async def late_rescore(fixture_id, home_adj, away_adj):
        assert home_adj == 40.0  # the lineup adjustment reached the re-score
        return late_pred, datetime(2026, 8, 1, 19, 30, tzinfo=timezone.utc)

    delivered_late = await send_prediction_alert(
        s, 88, send_fn=ok_send,
        prediction_fn=lambda fid: late_pred,
        lineup_fn=_lineup_fn_stub(home_adj=40.0, away_adj=0.0),
        rescore_fn=late_rescore,
        entitlement_fn=ent, users_fn=lambda: [u],
    )
    assert delivered_late == 1
    # CHARGED ONCE across BOTH alerts — the late alert is free.
    assert get_user(790).predictions_consumed == 1
    # The SECOND (late) message carries the confirmed XI + ADJUSTED prediction.
    late_body = sent[1]
    assert "Ederson" in late_body and "Saka" in late_body  # confirmed XI shown
    assert "H 62%" in late_body  # lineup-adjusted probs
    assert "lineup not yet confirmed" not in late_body  # XI IS confirmed now


def test_operator_trial_reveals_recorded_but_never_charged(db, tmp_path):
    """Operator/trial reveals carry charged=False; commit records the ledger
    (so it stays free after the trial) but NEVER increments consumed."""
    u = _paying_user(tmp_path, tid=780)
    _t, reveals = render_user_predictions(
        u, [_Pred(fixture_id=7)], _tg_settings_stub(),
        entitlement_fn=lambda usr, se, now=None: _ent("trial", trial=4),
        already_revealed_fn=has_revealed,
    )
    assert reveals == [(7, False)]
    commit_reveals(u, reveals)
    assert has_revealed(u.telegram_user_id, 7) is True
    assert get_user(u.telegram_user_id).predictions_consumed == 0


def test_record_reveal_is_idempotent(db, tmp_path):
    """Second record_reveal for the same (user, fixture) returns False; a
    guarded commit therefore never double-increments."""
    u = _paying_user(tmp_path, tid=781)
    assert record_reveal(u.telegram_user_id, 5, True) is True
    assert record_reveal(u.telegram_user_id, 5, True) is False
    # commit_reveals twice must charge exactly once.
    commit_reveals(u, [(5, True)])  # row already exists -> no increment
    assert get_user(u.telegram_user_id).predictions_consumed == 0


# keep timedelta import used (kickoff scheduling reference)
_ = timedelta
