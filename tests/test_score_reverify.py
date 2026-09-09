"""DEFECT 1 — re-verify recently-settled scorelines against the source.

football-data.org's free tier occasionally flips a match to FINISHED with a
PROVISIONAL fullTime score and corrects it hours later. record_prediction_outcome
is one-shot (idempotent on fixture_id) and never re-reads, so the wrong scoreline
stayed published (2026-09-08 fixture 575324: stored 2-0, source later 1-0).
SettlementWatcher.reverify_recent_scores re-fetches recent outcomes and corrects
a goals-only drift when the winner is unchanged; a winner FLIP is never
auto-applied (logged loud for the review gate instead).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot.settlement import SettlementWatcher
from betbot.storage.models import PredictionOutcome

from tests.test_outcome_loop import FakeFD, _finished, _seed_outcome


@pytest.fixture
def db(tmp_path):
    from betbot.storage.db import init_engine

    init_engine(tmp_path / "score_reverify.sqlite")
    yield


def _goals(fixture_id):
    from betbot.storage.db import session_scope

    with session_scope() as s:
        row = (
            s.query(PredictionOutcome)
            .filter(PredictionOutcome.fixture_id == fixture_id)
            .one()
        )
        return row.home_goals, row.away_goals, row.actual_outcome, row.correct


def _seed_old_outcome(fixture_id, hours_ago):
    """Seed a HOME 2-0 outcome settled ``hours_ago`` hours in the past."""
    from betbot.storage.db import session_scope

    with session_scope() as s:
        s.add(PredictionOutcome(
            fixture_id=fixture_id, competition_code="CL",
            predicted_home=0.71, predicted_draw=0.12, predicted_away=0.17,
            predicted_pick="HOME", actual_outcome="HOME",
            correct=True, brier=0.1, rps=0.1, log_loss=0.3,
            home_goals=2, away_goals=0, result_notified=True,
            settled_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        ))


async def test_corrects_provisional_scoreline_winner_unchanged(db, settings):
    # Stored 2-0 (the 575324 bug); source now says 1-0, winner still HOME.
    _seed_outcome(575324, code="CL")  # HOME 2-0
    fd = FakeFD({575324: _finished("HOME_TEAM", 1, 0)})
    watcher = SettlementWatcher(fd, settings)

    n = await watcher.reverify_recent_scores(window_hours=72)

    assert n == 1
    hg, ag, outcome, correct = _goals(575324)
    assert (hg, ag) == (1, 0)          # scoreline corrected
    assert outcome == "HOME"            # winner (and pick correctness) untouched
    assert correct is True


async def test_no_change_when_source_matches(db, settings):
    _seed_outcome(1, code="CL")  # HOME 2-0
    fd = FakeFD({1: _finished("HOME_TEAM", 2, 0)})
    watcher = SettlementWatcher(fd, settings)

    n = await watcher.reverify_recent_scores(window_hours=72)

    assert n == 0
    assert _goals(1)[:2] == (2, 0)


async def test_winner_flip_is_not_auto_applied(db, settings, monkeypatch):
    # Stored HOME 2-0; source now says AWAY 0-2 — a winner FLIP. Must NOT be
    # silently re-scored; logged as outcome_winner_conflict instead.
    import betbot.settlement as st

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(st.log, "warning", lambda evt, **kw: events.append((evt, kw)))

    _seed_outcome(2, code="CL")  # HOME 2-0
    fd = FakeFD({2: _finished("AWAY_TEAM", 0, 2)})
    watcher = SettlementWatcher(fd, settings)

    n = await watcher.reverify_recent_scores(window_hours=72)

    assert n == 0
    assert _goals(2) == (2, 0, "HOME", True)   # nothing changed
    assert any(evt == "outcome_winner_conflict" for evt, _ in events)


async def test_outside_window_is_skipped(db, settings):
    # Settled 100h ago; a 72h window must not touch it even if the source drifted.
    _seed_old_outcome(3, hours_ago=100)
    fd = FakeFD({3: _finished("HOME_TEAM", 1, 0)})
    watcher = SettlementWatcher(fd, settings)

    n = await watcher.reverify_recent_scores(window_hours=72)

    assert n == 0
    assert _goals(3)[:2] == (2, 0)
