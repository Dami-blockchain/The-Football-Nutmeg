"""High-conviction alert filter (BETBOT_HIGH_CONF_ALERTS_ONLY).

Covers the four surfaces the gate touches:

  1. the pure gate (:func:`betbot.main.high_conf_alert_passes`) — above/below
     threshold, DRAW top-pick exclusion, and flag-off = unchanged behaviour
     (down to NOT reading the probability fields);
  2. the planner (:func:`betbot.main.plan_kickoff_alert_jobs`) — a suppressed
     fixture yields NO jobs, and the coverage watchdog therefore does NOT
     mistake it for a missing alert (THE trap: the watchdog that once caught a
     real outage must stay quiet when the gate deliberately drops a fixture);
  3. the fire-time re-check inside ``_fire_prediction_alert`` — a fixture that
     no longer clears the bar is suppressed at fire time, never sent;
  4. the message format (:mod:`betbot.notify`) — band constants, the live-tally
     wording, and the honest "no market price" path — plus the repo tally
     (:func:`betbot.storage.repos.high_conf_band_tally`).

HONEST framing (stated once, not re-litigated): the gate raises the alert HIT
RATE, not profit — measured ROI on gated subsets was −1.3% [−4.3, +1.9].
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import betbot.main as main
from betbot import notify
from betbot.storage.db import init_engine
from betbot.storage.repos import high_conf_band_tally, record_prediction_outcome

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
KO = NOW + timedelta(hours=6)


def _pred(fid, ph, pd, pa, *, code="PL", ko=KO):
    return SimpleNamespace(
        fixture_id=fid,
        competition_code=code,
        kickoff=ko,
        p_home=ph,
        p_draw=pd,
        p_away=pa,
        home_team="Arsenal",
        away_team="Everton",
    )


def _hc(settings, *, on=True, min_p=0.65):
    return settings.model_copy(
        update={"high_conf_alerts_only": on, "high_conf_alert_min_p": min_p}
    )


# ----------------------------------------------------------------------
# 1. The pure gate
# ----------------------------------------------------------------------
def test_gate_off_passes_everything_without_reading_probabilities(settings):
    # Flag OFF must be byte-identical to before, right down to NOT touching the
    # probability fields — the planner still accepts minimal fixture doubles.
    bare = SimpleNamespace(fixture_id=1, competition_code="PL", kickoff=KO)
    passes, pick, p = main.high_conf_alert_passes(settings, bare)
    assert passes is True
    assert (pick, p) == ("", 0.0)


def test_gate_on_home_favourite_above_threshold_passes(settings):
    passes, pick, p = main.high_conf_alert_passes(_hc(settings), _pred(1, 0.71, 0.18, 0.11))
    assert passes is True
    assert pick == "HOME"
    assert p == pytest.approx(0.71)


def test_gate_on_away_favourite_above_threshold_passes(settings):
    passes, pick, _ = main.high_conf_alert_passes(_hc(settings), _pred(1, 0.11, 0.18, 0.71))
    assert passes is True
    assert pick == "AWAY"


def test_gate_on_below_threshold_is_suppressed(settings):
    passes, pick, p = main.high_conf_alert_passes(_hc(settings), _pred(1, 0.55, 0.25, 0.20))
    assert passes is False
    assert pick == "HOME"
    assert p == pytest.approx(0.55)


def test_gate_on_draw_top_pick_is_suppressed_even_above_threshold(settings):
    # Draw is the argmax AND above 0.65, but the draw is never alerted.
    passes, pick, _ = main.high_conf_alert_passes(_hc(settings), _pred(1, 0.15, 0.70, 0.15))
    assert passes is False
    assert pick == "DRAW"


def test_gate_on_exactly_at_threshold_passes(settings):
    passes, _, _ = main.high_conf_alert_passes(_hc(settings, min_p=0.65), _pred(1, 0.65, 0.20, 0.15))
    assert passes is True  # >= is inclusive


# ----------------------------------------------------------------------
# 2. The planner + the coverage-watchdog trap
# ----------------------------------------------------------------------
def test_planner_flag_off_schedules_every_fixture(settings):
    preds = [_pred(1, 0.55, 0.25, 0.20), _pred(2, 0.40, 0.35, 0.25)]
    ids = {jid for jid, _ in main.plan_kickoff_alert_jobs(settings, preds, NOW)}
    assert ids == {
        "predict_early_1", "predict_late_1",
        "predict_early_2", "predict_late_2",
    }


def test_planner_flag_on_drops_suppressed_fixtures(settings):
    preds = [_pred(1, 0.72, 0.18, 0.10), _pred(2, 0.50, 0.30, 0.20)]
    ids = {jid for jid, _ in main.plan_kickoff_alert_jobs(_hc(settings), preds, NOW)}
    # fid 1 clears the bar; fid 2 is suppressed -> no jobs at all.
    assert ids == {"predict_early_1", "predict_late_1"}


def test_planner_logs_each_suppression_when_asked(settings, monkeypatch):
    events: list[tuple[str, dict]] = []

    class _Rec:
        def info(self, event, **kw):
            events.append((event, kw))

        def __getattr__(self, _n):  # warning/error unused here
            return lambda *a, **k: None

    monkeypatch.setattr(main, "get_logger", lambda _n: _Rec())
    preds = [_pred(1, 0.72, 0.18, 0.10), _pred(2, 0.50, 0.30, 0.20)]
    main.plan_kickoff_alert_jobs(_hc(settings), preds, NOW, log_suppressed=True)
    supp = [kw for ev, kw in events if ev == "prematch_alert_suppressed_low_conf"]
    assert len(supp) == 1
    assert supp[0]["fixture_id"] == 2
    assert supp[0]["p"] == pytest.approx(0.50)


def test_planner_does_not_log_suppressions_on_audit_only_calls(settings, monkeypatch):
    # The default (log_suppressed=False) keeps report_alert_coverage / the
    # watchdog's pre-heal audit quiet so only the scheduling pass records it.
    events: list[str] = []
    monkeypatch.setattr(
        main, "get_logger",
        lambda _n: SimpleNamespace(info=lambda ev, **k: events.append(ev)),
    )
    preds = [_pred(2, 0.50, 0.30, 0.20)]
    main.plan_kickoff_alert_jobs(_hc(settings), preds, NOW)  # log_suppressed default
    assert "prematch_alert_suppressed_low_conf" not in events


class _FakeJob:
    def __init__(self, job_id):
        self.id = job_id


class _FakeScheduler:
    def __init__(self):
        self.jobs: list[_FakeJob] = []

    def add(self, job_id):
        self.jobs.append(_FakeJob(job_id))

    def get_jobs(self):
        return list(self.jobs)


def test_suppressed_fixture_is_not_reported_as_missing_coverage(settings):
    """THE trap: a deliberately-suppressed fixture must NOT read as a gap.

    Because the gate lives in the planner (the single source of truth for both
    scheduling and the audit), a suppressed fixture is simply absent from the
    plan, so ``audit_alert_coverage`` never sees a job id for it to miss.
    """
    preds = [_pred(1, 0.72, 0.18, 0.10), _pred(2, 0.50, 0.30, 0.20)]
    hc = _hc(settings)

    gated_plan = main.plan_kickoff_alert_jobs(hc, preds, NOW)
    sched = _FakeScheduler()
    for jid, _ in gated_plan:  # register exactly what the gate planned
        sched.add(jid)

    # No gap: everything planned is registered, and the suppressed fixture is
    # not even in the plan to be missed.
    assert main.audit_alert_coverage(sched, gated_plan) == []

    # Contrast — the trap the design avoids: had the plan been built WITHOUT the
    # gate (as it would be if suppression happened at scheduling time only), the
    # suppressed fixture's jobs would be in the plan, absent from the scheduler,
    # and flagged as a phantom gap.
    ungated_plan = main.plan_kickoff_alert_jobs(settings, preds, NOW)
    assert set(main.audit_alert_coverage(sched, ungated_plan)) == {
        "predict_early_2", "predict_late_2",
    }


def test_scheduling_pass_stays_silent_when_the_only_omission_is_suppressed(
    monkeypatch, settings
):
    """End-to-end: a scheduling pass with the gate ON does not false-alarm.

    Mirrors test_scheduler_jobs' gap test, but here the 'missing' fixture is
    deliberately suppressed, so the watchdog must stay quiet — the operator must
    keep trusting the one check that caught the sync-lambda outage.
    """
    from tests.test_scheduler_jobs import (
        _RecordingScheduler, _daemon_jobs, _operator_settings,
    )

    s = _hc(_operator_settings(settings))
    jobs = _daemon_jobs(monkeypatch, s)
    rescan = next(j for j in jobs if j.id == "reschedule_kickoff_alerts")

    # The scheduling pass reads the WALL clock (datetime.now), NOT the frozen
    # module NOW above, so fixtures must kick off in the real future or every
    # job is dropped as past-time and nothing schedules — scheduled=0 for the
    # WRONG reason. Anchor to now, exactly as the healthy sibling in
    # test_scheduler_jobs does (kickoff = now + 6h).
    _ko = datetime.now(timezone.utc) + timedelta(hours=6)
    preds = [_pred(1, 0.72, 0.18, 0.10, ko=_ko), _pred(2, 0.50, 0.30, 0.20, ko=_ko)]
    monkeypatch.setattr(
        main, "predictions_for_kickoff_range", lambda _st, _en: preds
    )

    sent: list[tuple[int, str]] = []

    async def _send(_settings, chat_id, text):
        sent.append((chat_id, text))
        return True

    monkeypatch.setattr(notify, "send_telegram_to", _send)

    sched = _RecordingScheduler()
    asyncio.run(rescan.func(sched, *rescan.args[1:]))

    ids = {j.id for j in sched.jobs}
    assert "predict_early_1" in ids and "predict_late_1" in ids
    assert "predict_early_2" not in ids and "predict_late_2" not in ids
    assert sent == [], "suppressed fixture triggered a phantom coverage alarm"


# ----------------------------------------------------------------------
# 3. Fire-time re-check
# ----------------------------------------------------------------------
def test_fire_time_recheck_suppresses_when_flag_flips_on(monkeypatch, settings):
    """A job scheduled while the gate was off must not fire an off-band alert
    once the gate is turned on before it fires."""
    from tests.test_scheduler_jobs import _RecordingScheduler, _daemon_jobs

    # Schedule with the gate OFF so the fixture gets a job at all.
    jobs = _daemon_jobs(monkeypatch, settings)
    rescan = next(j for j in jobs if j.id == "reschedule_kickoff_alerts")
    # Wall-clock anchor (the rescan drops past-time jobs; a frozen-NOW kickoff
    # would schedule nothing to later re-check). See the note in the scheduling
    # pass test above.
    _ko = datetime.now(timezone.utc) + timedelta(hours=6)
    monkeypatch.setattr(
        main, "predictions_for_kickoff_range",
        lambda _st, _en: [_pred(999, 0.72, 0.18, 0.10, ko=_ko)],
    )
    sched = _RecordingScheduler()
    asyncio.run(rescan.func(sched, *rescan.args[1:]))
    fire = next(j for j in sched.jobs if j.id == "predict_early_999")

    # Now flip the gate ON, and let the STORED baseline be sub-threshold.
    monkeypatch.setattr(main, "get_settings", lambda: _hc(settings))
    monkeypatch.setattr(
        main, "prediction_for_fixture",
        lambda _fid: _pred(999, 0.50, 0.30, 0.20),
    )

    sent: list = []

    async def _spa(*a, **k):
        sent.append((a, k))
        return 1

    monkeypatch.setattr(main, "send_prediction_alert", _spa)

    asyncio.run(fire.func())
    assert sent == [], "fire-time re-check let a suppressed fixture through"


# ----------------------------------------------------------------------
# 4a. Band constants + live-tally wording
# ----------------------------------------------------------------------
def test_band_fixture_counts_are_derived_from_the_keep_fractions():
    assert notify.band_fixture_count(0.55) == 2422
    assert notify.band_fixture_count(0.60) == 1657
    assert notify.band_fixture_count(0.65) == 1041  # round(0.147 * 7082)
    assert notify.band_fixture_count(0.70) == 538


def test_band_line_default_threshold_small_live_sample():
    line = notify.format_band_line(0.65, (3, 4))
    assert "p>=0.65 hits 72.8% [70.1–75.4] on 1,041 walk-forward fixtures (2022–26)." in line
    assert "This season live: 3/4 — too few to mean anything yet." in line


def test_band_line_meaningful_live_sample_shows_percentage():
    line = notify.format_band_line(0.65, (24, 32))
    assert "This season live: 24/32 (75%)." in line
    assert "too few" not in line


def test_band_line_no_settled_and_unavailable():
    assert "This season live: none settled yet." in notify.format_band_line(0.65, (0, 0))
    assert "This season live: not available." in notify.format_band_line(0.65, None)


def test_band_line_off_table_threshold_uses_the_weaker_band_at_or_below():
    # 0.62 is not a table key -> report the 0.60 band (conservative), not 0.65.
    line = notify.format_band_line(0.62, (1, 2))
    assert "p>=0.6 hits 70.2%" in line
    assert "1,657 walk-forward fixtures" in line


# ----------------------------------------------------------------------
# 4b. The full high-confidence message
# ----------------------------------------------------------------------
def test_high_conf_message_home_favourite_no_market(settings):
    pred = _pred(1, 0.71, 0.18, 0.11, code="PL", ko=datetime(2026, 8, 24, 14, 0, tzinfo=timezone.utc))
    body = notify.format_high_conf_alert(pred, _hc(settings), live_tally=(3, 4))
    assert "HIGH-CONFIDENCE ALERT" in body
    assert "Arsenal (HOME) v Everton (AWAY)" in body
    assert ", PL," in body
    assert "KO 17:00 EAT" in body  # 14:00 UTC -> 17:00 EAT (UTC+3)
    assert "Model: HOME 71% / draw 18% / away 11%" in body
    # Matching is dead: the Market field must say so, not fabricate a price.
    assert "Market: unavailable (no live quote)" in body
    # Honest: with no quote the reason must NOT claim an edge-vs-price check.
    assert "*NO BET* (default; no live price to assess edge)" in body
    assert "Band record: p>=0.65 hits 72.8%" in body


def test_high_conf_message_labels_away_favourite():
    from betbot.config import Settings

    s = Settings(_env_file=None, FOOTBALL_DATA_API_KEY="x", BETBOT_HIGH_CONF_ALERTS_ONLY=True)
    pred = _pred(1, 0.11, 0.18, 0.71)
    body = notify.format_high_conf_alert(pred, s, live_tally=None)
    assert "Model: home 11% / draw 18% / AWAY 71%" in body


def test_high_conf_message_renders_a_market_quote_when_supplied(settings):
    pred = _pred(1, 0.71, 0.18, 0.11)
    body = notify.format_high_conf_alert(
        pred, _hc(settings), market=("HOME", 0.68, 1.47), live_tally=(3, 4)
    )
    assert "Market: HOME 68% (1.47)" in body
    # With a real quote the edge-vs-price framing is legitimate.
    assert "*NO BET* (default; edge vs price below threshold)" in body


# ----------------------------------------------------------------------
# 4c. The live-season tally repo helper (club-only, current season)
# ----------------------------------------------------------------------
@pytest.fixture
def _season(monkeypatch):
    from betbot.config import get_settings

    monkeypatch.setenv("BETBOT_SEASON_START", "2026-08-01")
    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_high_conf_band_tally_counts_only_in_band_club_fixtures(tmp_path, _season):
    init_engine(tmp_path / "tally.sqlite")
    this_season = datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc)
    settled = datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)
    world_cup = datetime(2026, 7, 18, 21, 0, tzinfo=timezone.utc)

    def _out(fid, ph, pd, pa, actual, *, code="PL", ko=this_season):
        record_prediction_outcome(
            fixture_id=fid, competition_code=code,
            p_home=ph, p_draw=pd, p_away=pa,
            actual_outcome=actual, home_goals=1, away_goals=0,
            settled_at=settled, kickoff=ko,
        )

    _out(1, 0.72, 0.18, 0.10, "HOME")            # in band, correct
    _out(2, 0.70, 0.20, 0.10, "AWAY")            # in band, wrong
    _out(3, 0.55, 0.25, 0.20, "HOME")            # below 0.65 -> excluded
    _out(4, 0.15, 0.70, 0.15, "DRAW")            # draw top-pick -> excluded
    _out(5, 0.80, 0.12, 0.08, "HOME", code="WC", ko=world_cup)  # World Cup -> excluded

    hits, n = high_conf_band_tally(0.65)
    assert (hits, n) == (1, 2)


# ----------------------------------------------------------------------
# Reviewer fix 1: scheduled=0 must be distinguishable from a dead pass
# ----------------------------------------------------------------------
def test_scheduling_headline_reports_the_suppressed_count(monkeypatch, settings):
    """An all-sub-threshold matchday logs scheduled=0 WITH suppressed>0, so the
    operator's 'grep prematch_alerts_scheduled for a non-zero count' health
    check can tell a gated no-op apart from the sync-lambda outage."""
    from tests.test_scheduler_jobs import _RecordingScheduler, _daemon_jobs

    events: list[tuple[str, dict]] = []

    class _Rec:
        def info(self, event, **kw):
            events.append((event, kw))

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    s = _hc(settings)
    jobs = _daemon_jobs(monkeypatch, s)
    rescan = next(j for j in jobs if j.id == "reschedule_kickoff_alerts")
    monkeypatch.setattr(
        main, "predictions_for_kickoff_range",
        lambda _st, _en: [_pred(1, 0.50, 0.30, 0.20), _pred(2, 0.40, 0.35, 0.25)],
    )
    monkeypatch.setattr(main, "get_logger", lambda _n: _Rec())

    asyncio.run(rescan.func(_RecordingScheduler(), *rescan.args[1:]))

    headline = [kw for ev, kw in events if ev == "prematch_alerts_scheduled"]
    assert headline, "the scheduling pass logged no headline"
    assert headline[0] == {"scheduled": 0, "suppressed": 2}


def test_scheduling_headline_suppressed_is_zero_when_flag_off(monkeypatch, settings):
    from tests.test_scheduler_jobs import _RecordingScheduler, _daemon_jobs

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        main, "get_logger",
        lambda _n: SimpleNamespace(
            info=lambda ev, **kw: events.append((ev, kw)),
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
    )
    jobs = _daemon_jobs(monkeypatch, settings)  # gate OFF
    rescan = next(j for j in jobs if j.id == "reschedule_kickoff_alerts")
    # Wall-clock anchor so the rescan actually schedules (early + late); a
    # frozen-NOW kickoff is past-time now and would drop both jobs. See above.
    _ko = datetime.now(timezone.utc) + timedelta(hours=6)
    monkeypatch.setattr(
        main, "predictions_for_kickoff_range",
        lambda _st, _en: [_pred(1, 0.40, 0.35, 0.25, ko=_ko)],
    )
    asyncio.run(rescan.func(_RecordingScheduler(), *rescan.args[1:]))
    headline = [kw for ev, kw in events if ev == "prematch_alerts_scheduled"]
    assert headline and headline[0]["suppressed"] == 0
    assert headline[0]["scheduled"] == 2  # early + late, nothing dropped


# ----------------------------------------------------------------------
# Reviewer fix 2: display/gate drift after the always-fresh rescore
# ----------------------------------------------------------------------
def _stateful_pred_fn(first, rest):
    state = {"n": 0}

    def _fn(_fid):
        state["n"] += 1
        return first if state["n"] == 1 else rest

    return _fn


async def test_rescore_drift_falls_back_to_standard_body(tmp_path):
    """Stored row clears the bar, rescored row does not -> the alert must NOT
    wear a HIGH-CONFIDENCE banner over a sub-band Model line."""
    from tests.test_daily_jobs import (
        _Pred, _User, _ent, _lineup_fn_stub, _rescore_stub, _tg_settings,
    )
    from betbot.daily_jobs import send_prediction_alert

    init_engine(tmp_path / "drift.sqlite")
    s = _hc(_tg_settings(tmp_path))
    sent: list[str] = []

    async def fake_send(_se, _cid, txt):
        sent.append(txt)
        return True

    pf = _stateful_pred_fn(
        _Pred(fixture_id=1, p_home=0.72, p_draw=0.18, p_away=0.10),  # stored: clears
        _Pred(fixture_id=1, p_home=0.55, p_draw=0.25, p_away=0.20),  # rescored: does not
    )
    delivered = await send_prediction_alert(
        s, 1, send_fn=fake_send,
        prediction_fn=pf,
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )
    assert delivered == 1
    body = sent[0]
    assert "HIGH-CONFIDENCE" not in body  # banner dropped on drift
    assert "Model: H 55% / D 25% / A 20%" in body  # standard body, honest number


async def test_rescored_row_still_clearing_keeps_the_high_conf_body(tmp_path):
    from tests.test_daily_jobs import (
        _Pred, _User, _ent, _lineup_fn_stub, _rescore_stub, _tg_settings,
    )
    from betbot.daily_jobs import send_prediction_alert

    init_engine(tmp_path / "clear.sqlite")
    s = _hc(_tg_settings(tmp_path))
    sent: list[str] = []

    async def fake_send(_se, _cid, txt):
        sent.append(txt)
        return True

    pf = _stateful_pred_fn(
        _Pred(fixture_id=1, p_home=0.72, p_draw=0.18, p_away=0.10),
        _Pred(fixture_id=1, p_home=0.71, p_draw=0.18, p_away=0.11),  # still clears
    )
    await send_prediction_alert(
        s, 1, send_fn=fake_send,
        prediction_fn=pf,
        lineup_fn=_lineup_fn_stub(),
        rescore_fn=_rescore_stub(),
        entitlement_fn=lambda u, se, now=None: _ent("operator"),
        users_fn=lambda: [_User(111)],
    )
    body = sent[0]
    assert "HIGH-CONFIDENCE ALERT" in body
    assert "Model: HOME 71% / draw 18% / away 11%" in body
    assert "Band record: p>=0.65 hits 72.8%" in body
