"""R10a: outcome ledger + per-match rating learning + result alert + budget.

All DB-backed tests use a tmp SQLite; network is mocked. Pins:
  * settlement scores EVERY prediction (not just bets) into prediction_outcomes,
  * a HOME win nudges the home Glicko UP / away DOWN, idempotent per fixture,
  * run_result_alerts is free + entitled-only (operator always),
  * result_notified prevents re-sending,
  * the shared LineupService issues ONE /matches per league/day across fixtures.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot.storage.db import init_engine
from betbot.storage.models import PredictionOutcome
from betbot.storage.repos import (
    get_rating,
    prediction_outcomes_since,
    rating_exists,
    record_reveal,
    score_prediction,
    track_record,
    upsert_prediction,
    upsert_rating,
)
from betbot.settlement import SettlementWatcher
from betbot.strategy.engine import Prediction
from betbot.strategy.glicko import Glicko2Rating

NOW = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "outcome.sqlite")
    yield


class FakeFD:
    def __init__(self, results):
        self._results = results

    async def get_match(self, fixture_id):
        return self._results.get(fixture_id)


def _finished(winner, hg, ag):
    return {
        "status": "FINISHED",
        "score": {"winner": winner, "fullTime": {"home": hg, "away": ag}},
    }


def _seed_pred(fixture_id, code, home, away, ph, pd, pa, kickoff):
    pred = Prediction(
        fixture_id=fixture_id, competition_code=code, home_team=home, away_team=away,
        p_home=ph, p_draw=pd, p_away=pa, home_score=1.0, away_score=0.0, draw_score=2.4,
    )
    return upsert_prediction(pred, kickoff=kickoff)


# ----------------------------------------------------------------------
# Pure scoring
# ----------------------------------------------------------------------
def test_score_prediction_correct_home():
    pick, correct, brier, rps, ll = score_prediction(0.6, 0.25, 0.15, "HOME")
    assert pick == "HOME"
    assert correct is True
    # Brier = (0.6-1)^2 + 0.25^2 + 0.15^2
    assert brier == pytest.approx(0.16 + 0.0625 + 0.0225)
    assert 0.0 <= rps <= 1.0
    assert ll == pytest.approx(-__import__("math").log(0.6))


def test_score_prediction_wrong_pick():
    pick, correct, *_ = score_prediction(0.2, 0.3, 0.5, "HOME")
    assert pick == "AWAY"
    assert correct is False


# ----------------------------------------------------------------------
# Outcome scoring at settlement (ALL predictions, not just bets)
# ----------------------------------------------------------------------
async def test_settlement_scores_prediction_outcome(db, settings):
    past = NOW - timedelta(minutes=200)
    _seed_pred(101, "PL", "Arsenal", "Chelsea", 0.6, 0.25, 0.15, past)
    w = SettlementWatcher(FakeFD({101: _finished("HOME_TEAM", 2, 0)}), settings)
    await w.settle_due(now=NOW)

    rows = prediction_outcomes_since(365)
    assert len(rows) == 1
    r = rows[0]
    assert r.fixture_id == 101
    assert r.actual_outcome == "HOME"
    assert r.predicted_pick == "HOME"
    assert r.correct is True
    assert r.home_goals == 2 and r.away_goals == 0
    assert 0.0 <= r.brier <= 2.0
    assert 0.0 <= r.rps <= 1.0

    tr = track_record(365)
    assert tr["n"] == 1 and tr["hits"] == 1 and tr["hit_rate"] == 1.0


async def test_settlement_scores_without_a_bet(db, settings):
    # No paper bet at all — outcome scoring must still run.
    past = NOW - timedelta(minutes=200)
    _seed_pred(202, "PD", "Real", "Barca", 0.2, 0.3, 0.5, past)
    w = SettlementWatcher(FakeFD({202: _finished("AWAY_TEAM", 0, 3)}), settings)
    await w.settle_due(now=NOW)
    rows = prediction_outcomes_since(365)
    assert len(rows) == 1
    assert rows[0].correct is True  # picked AWAY (0.5), result AWAY


# ----------------------------------------------------------------------
# Per-match rating learning (idempotent)
# ----------------------------------------------------------------------
async def test_home_win_nudges_ratings_and_is_idempotent(db, settings):
    past = NOW - timedelta(minutes=200)
    _seed_pred(303, "PL", "HomeFC", "AwayFC", 0.5, 0.3, 0.2, past)
    # Both teams must have REAL rating rows (the weekly re-seed creates them);
    # settlement only NUDGES existing ratings, never fabricates new ones.
    upsert_rating("HomeFC", Glicko2Rating(1520.0, 80.0, 0.06, "2026-06-01"))
    upsert_rating("AwayFC", Glicko2Rating(1480.0, 80.0, 0.06, "2026-06-01"))
    before_home = get_rating("HomeFC").rating
    before_away = get_rating("AwayFC").rating

    fd = FakeFD({303: _finished("HOME_TEAM", 1, 0)})
    w = SettlementWatcher(fd, settings)
    await w.settle_due(now=NOW)

    after_home = get_rating("HomeFC").rating
    after_away = get_rating("AwayFC").rating
    assert after_home > before_home  # home won -> rating up
    assert after_away < before_away  # away lost -> rating down

    # Re-settle the SAME fixture: outcome ledger is unique on fixture_id, so the
    # rating must NOT move again (no double update).
    await w.settle_due(now=NOW + timedelta(minutes=1))
    assert get_rating("HomeFC").rating == pytest.approx(after_home)
    assert get_rating("AwayFC").rating == pytest.approx(after_away)


async def test_unknown_team_scores_but_never_fabricates_a_rating(db, settings):
    # A just-promoted club with NO seeded rating: the outcome must be scored,
    # but NO rating row may be created — a fabricated near-default row would
    # defeat the club engine's unknown-team fallback to the form engine.
    past = NOW - timedelta(minutes=200)
    _seed_pred(313, "PL", "PromotedFC", "AwayFC", 0.5, 0.3, 0.2, past)
    upsert_rating("AwayFC", Glicko2Rating(1480.0, 80.0, 0.06, "2026-06-01"))
    away_before = get_rating("AwayFC").rating

    w = SettlementWatcher(FakeFD({313: _finished("HOME_TEAM", 1, 0)}), settings)
    await w.settle_due(now=NOW)

    assert len(prediction_outcomes_since(365)) == 1  # still scored
    assert not rating_exists("PromotedFC")  # no junk row
    assert get_rating("AwayFC").rating == pytest.approx(away_before)  # untouched


async def test_stale_backfill_is_silent_and_never_nudges_ratings(db, settings):
    # A fixture that finished LONG ago (e.g. rows predating the outcome loop):
    # scored into the ledger pre-notified (no result-alert blast on deploy) and
    # ratings are NOT nudged (already inside the weekly re-seed's history).
    from betbot import daily_jobs

    old = NOW - timedelta(days=10)
    _seed_pred(323, "PL", "HomeFC", "AwayFC", 0.6, 0.25, 0.15, old)
    upsert_rating("HomeFC", Glicko2Rating(1520.0, 80.0, 0.06, "2026-06-01"))
    upsert_rating("AwayFC", Glicko2Rating(1480.0, 80.0, 0.06, "2026-06-01"))
    before = (get_rating("HomeFC").rating, get_rating("AwayFC").rating)

    w = SettlementWatcher(FakeFD({323: _finished("HOME_TEAM", 2, 0)}), settings)
    await w.settle_due(now=NOW)

    rows = prediction_outcomes_since(365)
    assert len(rows) == 1
    assert rows[0].result_notified is True  # pre-notified: no alert ever
    assert (get_rating("HomeFC").rating, get_rating("AwayFC").rating) == before

    # And the result-alert job must find nothing to send.
    object.__setattr__(settings, "telegram_allowed_user_id", 999)

    async def _boom_send(s, chat_id, text):
        raise AssertionError("stale backfill must not broadcast result alerts")

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=_boom_send, users_fn=lambda: []
    )
    assert n == 0


async def test_ancient_predictions_outside_lookback_are_ignored(db, settings):
    # Kickoff beyond the 30-day lookback: never fetched, never scored — a first
    # deploy must not replay months of history through the API.
    ancient = NOW - timedelta(days=40)
    _seed_pred(333, "WC", "OldA", "OldB", 0.4, 0.3, 0.3, ancient)

    class ExplodingFD:
        async def get_match(self, fixture_id):
            raise AssertionError("ancient fixture must not be fetched")

    w = SettlementWatcher(ExplodingFD(), settings)
    await w.settle_due(now=NOW)
    assert prediction_outcomes_since(365) == []


# ----------------------------------------------------------------------
# Result alerts — free, entitled-only, operator always, no re-send
# ----------------------------------------------------------------------
class _User:
    def __init__(self, uid):
        self.telegram_user_id = uid


def _seed_outcome(fixture_id, code="PL", correct=True, notified=False):
    from betbot.storage.db import session_scope

    with session_scope() as s:
        s.add(PredictionOutcome(
            fixture_id=fixture_id, competition_code=code,
            predicted_home=0.6, predicted_draw=0.25, predicted_away=0.15,
            predicted_pick="HOME", actual_outcome="HOME" if correct else "AWAY",
            correct=correct, brier=0.1, rps=0.1, log_loss=0.5,
            home_goals=2, away_goals=0, result_notified=notified,
            settled_at=datetime.now(timezone.utc),
        ))


async def test_result_alert_entitled_only_and_free(db, settings, monkeypatch):
    from betbot import daily_jobs

    _seed_pred(404, "PL", "Arsenal", "Spurs", 0.6, 0.25, 0.15, NOW)
    _seed_outcome(404)

    # user 111 saw the prediction; user 222 did NOT.
    record_reveal(111, 404, charged=True)

    settings_obj = settings
    object.__setattr__(settings_obj, "telegram_allowed_user_id", 999)  # operator

    sent_to: list[tuple[int, str]] = []

    async def fake_send(s, chat_id, text):
        sent_to.append((chat_id, text))
        return True

    # Assert the money path is NEVER touched by result alerts.
    def _boom(*a, **k):  # noqa: ANN001
        raise AssertionError("result alerts must not touch the money path")

    monkeypatch.setattr(daily_jobs, "record_reveal", _boom, raising=False)
    monkeypatch.setattr(daily_jobs, "increment_predictions_consumed", _boom, raising=False)

    n = await daily_jobs.run_result_alerts(
        settings_obj,
        send_fn=fake_send,
        users_fn=lambda: [_User(111), _User(222)],
    )

    recipients = {cid for cid, _ in sent_to}
    assert 999 in recipients  # operator always
    assert 111 in recipients  # saw the prediction
    assert 222 not in recipients  # did not see it -> no result alert
    assert n == len(sent_to)
    # Content: score + correct/wrong + model probs.
    body = sent_to[0][1]
    assert "Full time: Arsenal 2-0 Spurs" in body
    assert "correct" in body
    assert "H 60%" in body


async def test_result_alert_not_resent(db, settings, monkeypatch):
    from betbot import daily_jobs

    _seed_pred(505, "PL", "A", "B", 0.6, 0.25, 0.15, NOW)
    _seed_outcome(505)
    record_reveal(111, 505, charged=False)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)

    calls: list[int] = []

    async def fake_send(s, chat_id, text):
        calls.append(chat_id)
        return True

    await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send, users_fn=lambda: [_User(111)]
    )
    first = len(calls)
    assert first >= 1
    # Second run: the fixture is now flagged result_notified -> nothing sent.
    await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send, users_fn=lambda: [_User(111)]
    )
    assert len(calls) == first


# ----------------------------------------------------------------------
# Budget fix — shared LineupService issues ONE /matches per league/day
# ----------------------------------------------------------------------
class _CountingHighlightly:
    """Records every /matches + /lineup call; returns both fixtures' matches."""

    def __init__(self, matches, lineup):
        self._matches = matches
        self._lineup = lineup
        self.match_calls: list[tuple[str, str]] = []
        self.lineup_calls: list[object] = []

    async def list_matches(self, league_name, date):
        self.match_calls.append((league_name, date))
        return self._matches

    async def get_lineup(self, match_id):
        self.lineup_calls.append(match_id)
        return self._lineup

    async def close(self):
        pass


class _FakeAf:
    async def close(self):
        pass


class _Baseline:
    def __init__(self, code, home, away, kickoff):
        self.competition_code = code
        self.home_team = home
        self.away_team = away
        self.kickoff = kickoff


async def test_shared_lineup_service_one_matches_per_league_day(monkeypatch):
    from betbot import daily_jobs
    from betbot.config import Settings
    from betbot.data.lineup_service import LineupService

    s = Settings(
        _env_file=None, FOOTBALL_DATA_API_KEY="x",
        HIGHLIGHTLY_API_KEY="hl", API_FOOTBALL_KEY="af", BETBOT_AF_SEASON=2026,
    )
    ko = datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc)
    matches = [
        {"match_id": 1, "home_name": "Real Madrid", "away_name": "Barcelona"},
        {"match_id": 2, "home_name": "Sevilla", "away_name": "Valencia"},
    ]
    hl = _CountingHighlightly(matches, lineup={"home": {"xi": []}, "away": {"xi": []}})

    # Force the module singleton to use OUR counting Highlightly client.
    daily_jobs._LINEUP_SERVICE = None
    daily_jobs._LINEUP_SERVICE_SETTINGS = None

    def _fake_service(settings):
        if daily_jobs._LINEUP_SERVICE is None:
            daily_jobs._LINEUP_SERVICE = LineupService(
                settings, client=_FakeAf(), highlightly=hl
            )
            daily_jobs._LINEUP_SERVICE_SETTINGS = settings
        return daily_jobs._LINEUP_SERVICE

    monkeypatch.setattr(daily_jobs, "_lineup_service", _fake_service)

    fn = daily_jobs._default_lineup_fn(s)
    # Two DIFFERENT fixtures, SAME league (La Liga) + SAME day.
    await fn(_Baseline("PD", "Real Madrid CF", "FC Barcelona", ko))
    await fn(_Baseline("PD", "Sevilla FC", "Valencia CF", ko))

    # The shared per-(league,date) cache means only ONE /matches fetch total…
    assert len(hl.match_calls) == 1
    assert hl.match_calls[0] == ("La Liga", "2026-08-16")
    # …but one /lineups per fixture.
    assert len(hl.lineup_calls) == 2

    # Clean up the module singleton so it can't leak into other tests.
    daily_jobs._LINEUP_SERVICE = None
    daily_jobs._LINEUP_SERVICE_SETTINGS = None


async def test_lineup_service_does_not_cache_empty_day_list():
    # list_matches returns [] on transient API errors; with the process-wide
    # singleton an empty result must NOT be cached, or one failed fetch would
    # poison every later alert that league-day (incl. the confirmed-XI update).
    from betbot.config import Settings
    from betbot.data.lineup_service import LineupService

    s = Settings(
        _env_file=None, FOOTBALL_DATA_API_KEY="x",
        HIGHLIGHTLY_API_KEY="hl", API_FOOTBALL_KEY="af", BETBOT_AF_SEASON=2026,
    )

    class FlakyHighlightly:
        def __init__(self):
            self.calls = 0

        async def list_matches(self, league_name, date):
            self.calls += 1
            if self.calls == 1:
                return []  # transient failure shape
            return [{"match_id": 7, "home_name": "A", "away_name": "B"}]

        async def close(self):
            pass

    hl = FlakyHighlightly()
    svc = LineupService(s, client=_FakeAf(), highlightly=hl)

    assert await svc._day_matches("La Liga", "2026-08-16") == []
    # Second call must RETRY (not serve the poisoned empty cache)…
    assert len(await svc._day_matches("La Liga", "2026-08-16")) == 1
    assert hl.calls == 2
    # …and the good result IS cached.
    assert len(await svc._day_matches("La Liga", "2026-08-16")) == 1
    assert hl.calls == 2


# ----------------------------------------------------------------------
# Result-alert HIGH-CONVICTION gate — the result path must agree with the
# pre-match path by construction (reuses main.high_conf_alert_passes).
# ----------------------------------------------------------------------
class _GatePred:
    """Minimal stored-prediction stand-in carrying the H/D/A triple + names."""

    def __init__(self, home, away, ph, pd, pa):
        self.home_team = home
        self.away_team = away
        self.p_home = ph
        self.p_draw = pd
        self.p_away = pa


def _notified_flag(fixture_id):
    """Read result_notified straight from the DB for the given fixture."""
    from betbot.storage.db import session_scope

    with session_scope() as s:
        row = (
            s.query(PredictionOutcome)
            .filter(PredictionOutcome.fixture_id == fixture_id)
            .one()
        )
        return row.result_notified


async def test_result_gate_off_alerts_everything(db, settings, monkeypatch):
    # Flag OFF (the fixture default): a LOW-confidence, DRAW-topped fixture that
    # would fail the gate must STILL alert — byte-identical legacy behaviour.
    from betbot import daily_jobs

    assert settings.high_conf_alerts_only is False  # guard the premise
    _seed_outcome(701)
    record_reveal(111, 701, charged=False)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)

    preds = {701: _GatePred("A", "B", 0.30, 0.45, 0.25)}  # top pick DRAW, p<0.65
    sent: list[int] = []

    async def fake_send(s, chat_id, text):
        sent.append(chat_id)
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert set(sent) == {999, 111}  # operator + revealed user both alerted
    assert n == 2
    assert _notified_flag(701) is True  # consumed as usual


async def test_result_gate_on_only_high_conf_non_draw_alerts(db, settings, monkeypatch):
    # Flag ON: three settled fixtures — one clears the gate, two do not
    # (one low-p, one high-p but DRAW-topped). Only the clearing one alerts;
    # the two suppressed ones are marked notified so they never re-queue.
    from betbot import daily_jobs

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)

    _seed_outcome(801)  # PASS: HOME 0.72
    _seed_outcome(802)  # FAIL: top p 0.50 (< 0.65)
    _seed_outcome(803)  # FAIL: DRAW-topped even though 0.70
    # Only the PASSING fixture (801) was revealed at alert time. 802/803 were
    # never revealed, so with the drift fix they stay suppressible — the
    # reveal-honouring OR-clause rescues only fixtures a user actually saw.
    record_reveal(111, 801, charged=False)

    preds = {
        801: _GatePred("Arsenal", "Spurs", 0.72, 0.18, 0.10),
        802: _GatePred("C", "D", 0.50, 0.30, 0.20),
        803: _GatePred("E", "F", 0.20, 0.70, 0.10),
    }
    sent: list[tuple[int, str]] = []

    async def fake_send(s, chat_id, text):
        sent.append((chat_id, text))
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    # Only fixture 801 alerted (operator + revealed user = 2 messages).
    assert n == 2
    assert {cid for cid, _ in sent} == {999, 111}
    assert all("Arsenal" in body for _, body in sent)  # only the passing fixture
    # All three are flagged notified: the passing one after send, the two
    # suppressed ones as a deliberate consume (no backlog).
    assert _notified_flag(801) is True
    assert _notified_flag(802) is True
    assert _notified_flag(803) is True


async def test_result_gate_on_missing_prediction_suppressed_and_consumed(
    db, settings
):
    # Flag ON + NO stored prediction: cannot have cleared the pre-match gate,
    # so it is suppressed WITHOUT raising, and still marked notified.
    from betbot import daily_jobs

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    _seed_outcome(901)
    # NOT revealed to anyone: a missing prediction that was never alerted
    # cannot have cleared the pre-match gate, so it is suppressed.

    # RECORD sends rather than raising: run_result_alerts wraps each send in
    # a per-send try/except, so a raised AssertionError inside the sender is
    # swallowed (logged result_alert_send_failed) and the old _boom_send made
    # this test inert — it passed even on ungated base code. Asserting the
    # sender was NEVER reached proves the gate suppresses UPSTREAM of the
    # send loop.
    calls: list[tuple[int, str]] = []

    async def _recording_send(s, chat_id, text):
        calls.append((chat_id, text))
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=_recording_send,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: None,  # no stored prediction
    )
    assert n == 0
    assert calls == []  # send was NEVER reached (gate suppressed upstream)
    assert _notified_flag(901) is True  # consumed, never re-queues


async def test_result_gate_off_missing_prediction_still_alerts(db, settings):
    # Flag OFF + None prediction must NOT raise and must still alert with the
    # "Home"/"Away" placeholders (legacy behaviour preserved end to end).
    from betbot import daily_jobs

    assert settings.high_conf_alerts_only is False
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    _seed_outcome(1001)

    sent: list[tuple[int, str]] = []

    async def fake_send(s, chat_id, text):
        sent.append((chat_id, text))
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [],
        prediction_fn=lambda fid: None,
    )
    assert n == 1  # operator only
    assert "Home" in sent[0][1] and "Away" in sent[0][1]
    assert _notified_flag(1001) is True


# ----------------------------------------------------------------------
# High-conf DRIFT — the alert-time promise on the RESULT path.
# upsert_prediction overwrites p_* in place on rescore, so the live stored
# row can read BELOW the bar by settlement. If we ALERTED it (any user has a
# reveal row) we still owe the result; if we never alerted it, suppress.
# Real 2026-09-04 cases: Stuttgart 0.654->0.617, Ipswich v Liverpool AWAY
# 0.691->0.639 (both alerted, both correct, both silently dropped).
# ----------------------------------------------------------------------
def _make_pred(fixture_id, home, away, ph, pd, pa):
    return Prediction(
        fixture_id=fixture_id, competition_code="PL", home_team=home, away_team=away,
        p_home=ph, p_draw=pd, p_away=pa, home_score=1.0, away_score=0.0, draw_score=2.4,
    )


async def test_result_drift_below_bar_with_reveal_still_alerts(db, settings):
    from betbot import daily_jobs

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)

    _seed_outcome(1101)
    record_reveal(111, 1101, charged=False)  # revealed at alert time

    # Live stored row has DRIFTED below 0.65 since the alert fired.
    preds = {1101: _GatePred("Stuttgart", "Augsburg", 0.617, 0.23, 0.153)}
    sent: list[int] = []

    async def fake_send(s, chat_id, text):
        sent.append(chat_id)
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert set(sent) == {999, 111}  # honoured: operator + the user who saw it
    assert n == 2
    assert _notified_flag(1101) is True


async def test_result_drift_below_bar_without_reveal_suppressed(db, settings):
    from betbot import daily_jobs

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)

    _seed_outcome(1102)  # never revealed to anyone

    preds = {1102: _GatePred("C", "D", 0.60, 0.25, 0.15)}
    sent: list[int] = []

    async def fake_send(s, chat_id, text):
        sent.append(chat_id)
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert n == 0 and sent == []          # below bar + never revealed -> suppressed
    assert _notified_flag(1102) is True   # consumed, never re-queues


async def test_result_gate_off_below_bar_unchanged(db, settings):
    # Flag OFF: the reveal-honouring OR-clause must not change legacy behaviour —
    # a below-bar fixture NEVER revealed still alerts (operator).
    from betbot import daily_jobs

    assert settings.high_conf_alerts_only is False
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    _seed_outcome(1103)

    preds = {1103: _GatePred("E", "F", 0.55, 0.25, 0.20)}
    sent: list[int] = []

    async def fake_send(s, chat_id, text):
        sent.append(chat_id)
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [],
        prediction_fn=lambda fid: preds.get(fid),
    )
    assert sent == [999] and n == 1
    assert _notified_flag(1103) is True


async def test_result_gate_on_missing_prediction_but_revealed_is_honoured(db, settings):
    # Flag ON + NO stored prediction row, but the fixture WAS revealed at alert
    # time (item 2: honour it, log it, DO NOT raise on the None pred). The
    # result goes out with the Home/Away placeholders.
    from betbot import daily_jobs

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    _seed_outcome(1105)
    record_reveal(111, 1105, charged=False)

    sent: list[tuple[int, str]] = []

    async def fake_send(s, chat_id, text):
        sent.append((chat_id, text))
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send,
        users_fn=lambda: [_User(111)],
        prediction_fn=lambda fid: None,
    )
    assert {cid for cid, _ in sent} == {999, 111}
    assert n == 2
    assert "Home" in sent[0][1] and "Away" in sent[0][1]  # placeholder names
    assert _notified_flag(1105) is True


async def test_result_total_send_failure_leaves_unnotified_for_retry(db, settings):
    # Item 4: every recipient errors -> result_notified stays False so the
    # 2-hourly pass retries inside the 3-day window (the column means "handled",
    # never "sent to nobody").
    from betbot import daily_jobs

    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    _seed_outcome(1104)

    async def boom_send(s, chat_id, text):
        raise RuntimeError("telegram down")

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=boom_send, users_fn=lambda: [],
        prediction_fn=lambda fid: None,
    )
    assert n == 0
    assert _notified_flag(1104) is False  # NOT consumed -> will retry


# ----------------------------------------------------------------------
# Pre-match DRIFT downgrade note (item 3): present when the rescored row
# has slipped below the bar, absent when it still clears it.
# ----------------------------------------------------------------------
def _free_operator_entitlement(user, settings, *, now=None):
    from betbot.entitlement import Entitlement

    return Entitlement(
        allowed=True, reason="operator", trial_days_left=-1, credits_remaining=-1
    )


async def _capture_prematch(settings, pred):
    from betbot import daily_jobs

    object.__setattr__(settings, "telegram_allowed_user_id", 999)
    sent: list[str] = []

    async def fake_send(s, chat_id, text):
        sent.append(text)
        return True

    async def lineup_fn(_baseline):
        # Non-empty lineup so the no-XI gap-report path (which reads a
        # PredictionRow.kickoff the injected Prediction lacks) is skipped.
        return ({"home": {"xi": []}, "away": {"xi": []}}, 0.0, 0.0, None)

    await daily_jobs.send_prediction_alert(
        settings, pred.fixture_id,
        send_fn=fake_send,
        prediction_fn=lambda fid: pred,
        lineup_fn=lineup_fn,
        rescore_fn=None,
        entitlement_fn=_free_operator_entitlement,
        users_fn=lambda: [_User(999)],
    )
    return sent


async def test_prematch_drift_note_present_when_below_bar(db, settings):
    from betbot.daily_jobs import HIGH_CONF_DOWNGRADE_NOTE

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)

    pred = _make_pred(1201, "Stuttgart", "Augsburg", 0.617, 0.23, 0.153)
    sent = await _capture_prematch(settings, pred)
    assert len(sent) == 1
    assert HIGH_CONF_DOWNGRADE_NOTE in sent[0]


async def test_prematch_no_drift_note_when_above_bar(db, settings):
    from betbot.daily_jobs import HIGH_CONF_DOWNGRADE_NOTE

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)

    pred = _make_pred(1202, "Arsenal", "Spurs", 0.72, 0.18, 0.10)
    sent = await _capture_prematch(settings, pred)
    assert len(sent) == 1
    assert HIGH_CONF_DOWNGRADE_NOTE not in sent[0]

# ----------------------------------------------------------------------
# F2: empty audience is consumed, never retried on an impossible send.
# ----------------------------------------------------------------------
async def test_result_no_audience_is_consumed_not_retried(db, settings):
    from betbot import daily_jobs

    assert settings.high_conf_alerts_only is False
    object.__setattr__(settings, "telegram_allowed_user_id", None)  # no operator
    _seed_outcome(1106)  # revealed to nobody

    sent: list[int] = []

    async def fake_send(s, chat_id, text):
        sent.append(chat_id)
        return True

    n = await daily_jobs.run_result_alerts(
        settings, send_fn=fake_send, users_fn=lambda: [],
        prediction_fn=lambda fid: None,
    )
    assert n == 0 and sent == []          # nobody to send to
    assert _notified_flag(1106) is True   # consumed, not retried every 2h


# ----------------------------------------------------------------------
# F4: with NO lineup, the "lineup not yet confirmed" caveat and the drift
# downgrade note both ride on adj_note (the f"{adj_note}\n{NOTE}" branch).
# Assert the caveat precedes the note and both precede the team line.
# ----------------------------------------------------------------------
class _KickoffPred:
    def __init__(self, fixture_id, home, away, ph, pd, pa, kickoff):
        self.fixture_id = fixture_id
        self.competition_code = "PL"
        self.home_team = home
        self.away_team = away
        self.p_home = ph
        self.p_draw = pd
        self.p_away = pa
        self.home_xg = None
        self.away_xg = None
        self.kickoff = kickoff


async def test_prematch_drift_note_stacks_after_lineup_caveat(db, settings):
    from betbot import daily_jobs
    from betbot.daily_jobs import HIGH_CONF_DOWNGRADE_NOTE

    object.__setattr__(settings, "high_conf_alerts_only", True)
    object.__setattr__(settings, "high_conf_alert_min_p", 0.65)
    object.__setattr__(settings, "telegram_allowed_user_id", 999)

    ko = datetime(2026, 9, 6, 18, 30, tzinfo=timezone.utc)
    pred = _KickoffPred(1203, "Stuttgart", "Augsburg", 0.617, 0.23, 0.153, ko)

    sent: list[str] = []

    async def fake_send(s, chat_id, text):
        sent.append(text)
        return True

    async def noop_operator(*a, **k):
        return True

    async def lineup_fn(_baseline):
        return (None, 0.0, 0.0, None)  # NO lineup -> caveat + downgrade stack

    await daily_jobs.send_prediction_alert(
        settings, pred.fixture_id,
        send_fn=fake_send,
        prediction_fn=lambda fid: pred,
        lineup_fn=lineup_fn,
        rescore_fn=None,
        entitlement_fn=_free_operator_entitlement,
        users_fn=lambda: [_User(999)],
        operator_send_fn=noop_operator,
    )
    assert len(sent) == 1
    body = sent[0]
    caveat_i = body.index("lineup not yet confirmed")
    note_i = body.index(HIGH_CONF_DOWNGRADE_NOTE)
    team_i = body.index("Stuttgart (H)")
    assert caveat_i < note_i < team_i
