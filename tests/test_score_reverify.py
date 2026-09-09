"""DEFECT 1 — re-verify recently-settled scorelines against the source, mark
rows verified, page on a winner conflict, and send forward-only corrections.

Covers the build plus the two review findings:
  * goals-only drift (winner unchanged) is corrected + returned + row stamped;
  * FINDING A: a null or winner-inconsistent source score is NEVER applied
    (P1, P1b) — the goals are the payload here, not a harmless fallback;
  * a winner FLIP is never auto-applied, PAGES the operator, is counted;
  * provider lastUpdated stored; verify counter caps re-checks;
  * FINDING B: run_score_reverification corrects ONLY where a result was
    actually published — a pending (unsent) row is deferred (P5) and a stale
    backfill row is suppressed (P4) — else same audience + group, group failure
    isolated from DMs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import betbot.settlement as st
from betbot import daily_jobs
from betbot.settlement import CorrectedScore, ScoreReverifySummary, SettlementWatcher
from betbot.storage.models import PredictionOutcome
from betbot.storage.repos import record_reveal

from tests.test_outcome_loop import FakeFD, _GatePred, _User, _seed_outcome

BROADCAST_ID = -1003880403502
NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    from betbot.storage.db import init_engine

    init_engine(tmp_path / "score_reverify.sqlite")
    yield


def _match(winner, hg, ag, last_updated="2026-09-09T00:20:32Z"):
    return {
        "status": "FINISHED",
        "lastUpdated": last_updated,
        "score": {"winner": winner, "fullTime": {"home": hg, "away": ag}},
    }


def _row(fixture_id):
    from betbot.storage.db import session_scope

    with session_scope() as s:
        r = (
            s.query(PredictionOutcome)
            .filter(PredictionOutcome.fixture_id == fixture_id)
            .one()
        )
        return (
            r.home_goals, r.away_goals, r.actual_outcome, r.correct,
            r.score_verify_count, r.score_verified_at, r.source_last_updated,
        )


def _seed_old_outcome(fixture_id, hours_ago, verify_count=0):
    from betbot.storage.db import session_scope

    with session_scope() as s:
        s.add(PredictionOutcome(
            fixture_id=fixture_id, competition_code="CL",
            predicted_home=0.71, predicted_draw=0.12, predicted_away=0.17,
            predicted_pick="HOME", actual_outcome="HOME",
            correct=True, brier=0.1, rps=0.1, log_loss=0.3,
            home_goals=2, away_goals=0, result_notified=True,
            score_verify_count=verify_count,
            settled_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        ))


def _published_correction(fixture_id, hg, ag):
    """A CorrectedScore for a fixture whose result WAS published: notified, and
    kicked off recently (not a stale backfill)."""
    return CorrectedScore(
        fixture_id, hg, ag,
        result_notified=True, kickoff=NOW - timedelta(hours=5), settled_at=NOW,
    )


# ----------------------------------------------------------------------
# SettlementWatcher.reverify_recent_scores
# ----------------------------------------------------------------------
async def test_corrects_provisional_scoreline_and_stamps_row(db, settings):
    _seed_outcome(575324, code="CL")  # HOME 2-0 (the bug)
    fd = FakeFD({575324: _match("HOME_TEAM", 1, 0)})
    summary = await SettlementWatcher(fd, settings).reverify_recent_scores()

    assert isinstance(summary, ScoreReverifySummary)
    assert [cs.fixture_id for cs in summary.corrected] == [575324]
    cs = summary.corrected[0]
    assert (cs.home_goals, cs.away_goals) == (1, 0)
    hg, ag, outcome, correct, vcount, vat, lu = _row(575324)
    assert (hg, ag) == (1, 0)          # scoreline corrected
    assert outcome == "HOME" and correct is True   # winner/pick untouched
    assert vcount == 1 and vat is not None          # stamped verified
    assert lu == "2026-09-09T00:20:32Z"             # provider lastUpdated stored


async def test_p1_null_fulltime_never_applied(db, settings):
    # FINDING A: FINISHED + winner set but fullTime nulls must NOT become 0-0.
    _seed_outcome(10, code="CL")  # HOME 2-0
    fd = FakeFD({10: _match("HOME_TEAM", None, None)})
    summary = await SettlementWatcher(fd, settings).reverify_recent_scores()

    assert summary.corrected == []
    hg, ag, *_rest, vcount, _vat, _lu = _row(10)
    assert (hg, ag) == (2, 0)              # real score untouched
    assert vcount in (0, None)             # NOT stamped -> retries next day


async def test_p1b_goals_contradict_winner_never_applied(db, settings):
    # FINDING A: winner HOME but goals 0-3 (inconsistent provisional payload).
    _seed_outcome(11, code="CL")  # HOME 2-0
    fd = FakeFD({11: _match("HOME_TEAM", 0, 3)})
    summary = await SettlementWatcher(fd, settings).reverify_recent_scores()

    assert summary.corrected == []
    hg, ag, *_rest, vcount, _vat, _lu = _row(11)
    assert (hg, ag) == (2, 0)
    assert vcount in (0, None)


async def test_match_stamps_verified_but_corrects_nothing(db, settings):
    _seed_outcome(1, code="CL")  # HOME 2-0
    fd = FakeFD({1: _match("HOME_TEAM", 2, 0)})
    summary = await SettlementWatcher(fd, settings).reverify_recent_scores()

    assert summary.corrected == []
    assert summary.checked == 1
    _, _, _, _, vcount, vat, _ = _row(1)
    assert vcount == 1 and vat is not None


async def test_winner_flip_pages_operator_never_auto_applies(db, settings, monkeypatch):
    paged: list[dict] = []

    async def fake_notify(settings, text, **kw):
        paged.append({"text": text, **kw})
        return True

    monkeypatch.setattr(st, "notify_operator", fake_notify)

    _seed_outcome(2, code="CL")  # stored HOME 2-0
    fd = FakeFD({2: _match("AWAY_TEAM", 0, 2)})  # source flipped the WINNER
    summary = await SettlementWatcher(fd, settings).reverify_recent_scores()

    assert summary.corrected == []
    assert summary.winner_conflicts == 1
    assert _row(2)[:4] == (2, 0, "HOME", True)          # nothing auto-applied
    assert len(paged) == 1
    assert paged[0]["dedupe_key"] == "outcome_winner_conflict:2"


async def test_verify_count_cap_stops_rechecking(db, settings):
    _seed_old_outcome(3, hours_ago=1, verify_count=3)
    fd = FakeFD({3: _match("HOME_TEAM", 1, 0)})
    summary = await SettlementWatcher(fd, settings).reverify_recent_scores(max_checks=3)

    assert summary.checked == 0
    assert _row(3)[:2] == (2, 0)


async def test_outside_window_is_skipped(db, settings):
    _seed_old_outcome(3, hours_ago=100)
    fd = FakeFD({3: _match("HOME_TEAM", 1, 0)})
    summary = await SettlementWatcher(fd, settings).reverify_recent_scores(window_hours=72)

    assert summary.corrected == [] and summary.checked == 0
    assert _row(3)[:2] == (2, 0)


# ----------------------------------------------------------------------
# daily_jobs.run_score_reverification — forward-only correction alerts
# ----------------------------------------------------------------------
class _FakeWatcher:
    def __init__(self, summary):
        self._summary = summary

    async def reverify_recent_scores(self, now=None, window_hours=72):
        return self._summary


def _set(settings, **kw):
    for k, v in kw.items():
        object.__setattr__(settings, k, v)


async def test_correction_sent_where_published_plus_group(db, settings):
    _set(settings, telegram_allowed_user_id=999, broadcast_chat_id=BROADCAST_ID)
    _seed_outcome(575324, code="CL")
    record_reveal(111, 575324, charged=False)

    summary = ScoreReverifySummary(1, [_published_correction(575324, 1, 0)], 0)
    sent: list[tuple[int, str]] = []

    async def fake_send(s, cid, txt):
        sent.append((cid, txt))
        return True

    n = await daily_jobs.run_score_reverification(
        settings, watcher=_FakeWatcher(summary), send_fn=fake_send,
        prediction_fn=lambda fid: _GatePred("PAE AEK", "LASK Linz", 0.71, 0.12, 0.17),
        users_fn=lambda: [_User(111)],
    )
    assert n == 1
    assert {cid for cid, _ in sent} == {999, 111, BROADCAST_ID}
    body = sent[0][1]
    assert "Result correction" in body and "1-0" in body
    assert "call is unaffected" in body


async def test_p4_stale_backfill_gets_no_correction(db, settings):
    # FINDING B: pre-notified, never-sent stale backfill -> silent DB fix only.
    _set(settings, telegram_allowed_user_id=999, broadcast_chat_id=BROADCAST_ID,
         high_conf_alerts_only=True, high_conf_alert_min_p=0.65)
    cs = CorrectedScore(
        40, 1, 0, result_notified=True,
        kickoff=NOW - timedelta(days=6), settled_at=NOW,  # stale
    )
    summary = ScoreReverifySummary(1, [cs], 0)
    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    n = await daily_jobs.run_score_reverification(
        settings, watcher=_FakeWatcher(summary), send_fn=fake_send,
        prediction_fn=lambda fid: _GatePred("A", "B", 0.7, 0.2, 0.1),
        users_fn=lambda: [_User(111)],
    )
    assert n == 0 and sent == []


async def test_p5_pending_unsent_row_defers(db, settings):
    # FINDING B: result_notified False -> the result alert hasn't gone out yet
    # and will carry the corrected goals itself; a correction now is premature.
    _set(settings, telegram_allowed_user_id=999, broadcast_chat_id=BROADCAST_ID)
    record_reveal(111, 50, charged=False)
    cs = CorrectedScore(
        50, 1, 0, result_notified=False,
        kickoff=NOW - timedelta(hours=5), settled_at=NOW,
    )
    summary = ScoreReverifySummary(1, [cs], 0)
    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    n = await daily_jobs.run_score_reverification(
        settings, watcher=_FakeWatcher(summary), send_fn=fake_send,
        prediction_fn=lambda fid: _GatePred("A", "B", 0.7, 0.2, 0.1),
        users_fn=lambda: [_User(111)],
    )
    assert n == 0 and sent == []


async def test_no_correction_where_never_published(db, settings):
    # Gate ON, below-bar, never revealed -> published predicate fails -> silent.
    _set(settings, telegram_allowed_user_id=999, broadcast_chat_id=BROADCAST_ID,
         high_conf_alerts_only=True, high_conf_alert_min_p=0.65)
    summary = ScoreReverifySummary(1, [_published_correction(701, 1, 0)], 0)
    sent: list[int] = []

    async def fake_send(s, cid, txt):
        sent.append(cid)
        return True

    n = await daily_jobs.run_score_reverification(
        settings, watcher=_FakeWatcher(summary), send_fn=fake_send,
        prediction_fn=lambda fid: _GatePred("A", "B", 0.30, 0.45, 0.25),  # DRAW-top
        users_fn=lambda: [_User(111)],
    )
    assert n == 0 and sent == []


async def test_correction_group_failure_isolated_from_dms(db, settings):
    _set(settings, telegram_allowed_user_id=999, broadcast_chat_id=BROADCAST_ID)
    record_reveal(111, 575324, charged=False)
    summary = ScoreReverifySummary(1, [_published_correction(575324, 1, 0)], 0)
    dm: list[int] = []

    async def flaky_send(s, cid, txt):
        if cid == BROADCAST_ID:
            raise RuntimeError("group boom")
        dm.append(cid)
        return True

    n = await daily_jobs.run_score_reverification(
        settings, watcher=_FakeWatcher(summary), send_fn=flaky_send,
        prediction_fn=lambda fid: _GatePred("PAE AEK", "LASK Linz", 0.71, 0.12, 0.17),
        users_fn=lambda: [_User(111)],
    )
    assert n == 1 and set(dm) == {999, 111}
