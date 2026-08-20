"""Rescheduled fixtures keep their pre-match alerts.

The live fault this pins: Celta-Osasuna was scored with kickoff 2026-08-16
19:30, football-data later moved it to 2026-08-27 18:30, and nothing ever
re-read that column. The alert jobs stayed pinned to the 16th — firing for a
match nobody played, burning the user's one paid reveal through the ledger, so
the alert on the real day came back already-revealed and free.

Pins, in the order the fault propagates:
  * upsert_prediction REFRESHES kickoff (it silently dropped it before);
  * update_prediction_kickoff moves EVERY row for a fixture, not just the newest;
  * plan_kickoff_changes spots moved/postponed fixtures and stays silent on the
    merely-absent (absence is ambiguous, not a cancellation);
  * resync_kickoffs resolves a fixture that left the league window with ONE
    per-fixture fetch, and persists what it finds;
  * drop_alert_jobs pulls the old jobs for a moved fixture;
  * alert_still_valid refuses to fire — and so refuses to charge — when the
    fixture upstream is no longer about to kick off.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from betbot.main import drop_alert_jobs, plan_kickoff_alert_jobs
from betbot.reschedule import (
    KickoffChange,
    alert_job_ids,
    alert_still_valid,
    parse_utc,
    plan_kickoff_changes,
    resync_kickoffs,
)
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    update_prediction_kickoff,
    upcoming_prediction_fixtures,
    upsert_prediction,
)
from betbot.strategy.engine import Prediction

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
OLD_KO = datetime(2026, 8, 20, 19, 30, tzinfo=timezone.utc)
NEW_KO = datetime(2026, 8, 27, 18, 30, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "resync.sqlite")
    yield


def _seed(fixture_id, kickoff, code="PD"):
    pred = Prediction(
        fixture_id=fixture_id, competition_code=code,
        home_team="Celta", away_team="Osasuna",
        p_home=0.4, p_draw=0.3, p_away=0.3,
        home_score=1.0, away_score=0.0, draw_score=2.4,
    )
    return upsert_prediction(pred, kickoff=kickoff)


class FakeClient:
    """Stands in for FootballDataClient over the two endpoints the re-sync uses."""

    def __init__(self, league_matches=None, matches=None):
        self._league_matches = league_matches or {}
        self._matches = matches or {}
        self.league_calls: list[tuple[str, str, str]] = []
        self.match_calls: list[int] = []

    async def list_matches(self, code, date_from, date_to, *, status=None):
        self.league_calls.append((code, date_from, date_to))
        return self._league_matches.get(code, [])

    async def get_match(self, fixture_id):
        self.match_calls.append(fixture_id)
        return self._matches.get(fixture_id)


def _match(fixture_id, utc_date, status="TIMED"):
    return {"id": fixture_id, "utcDate": utc_date, "status": status}


# ----------------------------------------------------------------------
# Storage: the stale column itself
# ----------------------------------------------------------------------
def test_upsert_prediction_refreshes_kickoff(db):
    """A same-day re-score with a corrected kickoff must move the stored row.

    send_prediction_alert already re-scores through score_fixture_adjusted,
    which reads the CURRENT kickoff from football-data and passes it here — the
    update branch just threw it away.
    """
    row_id = _seed(1, OLD_KO)
    again_id = _seed(1, NEW_KO)  # same (fixture_id, run_date) -> update branch
    assert again_id == row_id, "expected the update branch, not a second row"
    stored = upcoming_prediction_fixtures(NOW, NOW + timedelta(days=30))
    assert stored[1][0] == NEW_KO


def test_update_prediction_kickoff_moves_every_row(db):
    """Stale earlier-run_date rows drive settlement, so they move too."""
    from betbot.storage.db import session_scope
    from betbot.storage.models import PredictionRow
    from sqlalchemy import select

    _seed(2, OLD_KO)
    # A second row for the same fixture from an earlier daily run.
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=2, competition_code="PD", kickoff=OLD_KO,
                run_date="2026-08-19", home_team="Celta", away_team="Osasuna",
                p_home=0.4, p_draw=0.3, p_away=0.3,
                home_score=1.0, away_score=0.0, draw_score=2.4,
            )
        )

    changed = update_prediction_kickoff(2, NEW_KO)
    assert changed == 2

    with session_scope() as s:
        kos = [
            r.kickoff.replace(tzinfo=timezone.utc)
            for r in s.execute(
                select(PredictionRow).where(PredictionRow.fixture_id == 2)
            ).scalars()
        ]
    assert kos == [NEW_KO, NEW_KO]


def test_update_prediction_kickoff_reports_only_real_changes(db):
    _seed(3, OLD_KO)
    assert update_prediction_kickoff(3, OLD_KO) == 0


def test_upcoming_prediction_fixtures_takes_the_freshest_row(db):
    from betbot.storage.db import session_scope
    from betbot.storage.models import PredictionRow

    _seed(4, OLD_KO)  # today's run_date
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=4, competition_code="PD",
                kickoff=OLD_KO - timedelta(hours=2), run_date="2026-01-01",
                home_team="Celta", away_team="Osasuna",
                p_home=0.4, p_draw=0.3, p_away=0.3,
                home_score=1.0, away_score=0.0, draw_score=2.4,
            )
        )
    stored = upcoming_prediction_fixtures(NOW, NOW + timedelta(days=2))
    assert stored == {4: (OLD_KO, "PD")}


# ----------------------------------------------------------------------
# Pure planning
# ----------------------------------------------------------------------
def test_plan_spots_a_fixture_moved_later():
    changes = plan_kickoff_changes({7: OLD_KO}, {7: (NEW_KO, "TIMED")})
    assert changes == [KickoffChange(7, OLD_KO, NEW_KO, "TIMED")]
    assert changes[0].is_dead is False


def test_plan_spots_a_fixture_moved_earlier():
    earlier = OLD_KO - timedelta(days=1)
    changes = plan_kickoff_changes({7: OLD_KO}, {7: (earlier, "SCHEDULED")})
    assert changes[0].new_kickoff == earlier


def test_plan_is_silent_when_nothing_moved():
    assert plan_kickoff_changes({7: OLD_KO}, {7: (OLD_KO, "TIMED")}) == []


def test_plan_treats_naive_stored_kickoffs_as_utc():
    """SQLite hands back naive datetimes; a tz mismatch must not read as a move."""
    assert plan_kickoff_changes(
        {7: OLD_KO.replace(tzinfo=None)}, {7: (OLD_KO, "TIMED")}
    ) == []


def test_plan_flags_a_postponed_fixture_even_at_the_same_time():
    """POSTPONED keeps the original utcDate — the kickoff alone never changes."""
    changes = plan_kickoff_changes({7: OLD_KO}, {7: (OLD_KO, "POSTPONED")})
    assert len(changes) == 1
    assert changes[0].is_dead is True
    assert changes[0].new_kickoff is None


def test_plan_ignores_fixtures_absent_upstream():
    """Absence is ambiguous — it must not be read as a cancellation."""
    assert plan_kickoff_changes({7: OLD_KO}, {}) == []


# ----------------------------------------------------------------------
# The re-sync, end to end
# ----------------------------------------------------------------------
async def test_resync_persists_a_move_seen_in_the_league_window(db, settings):
    _seed(11, OLD_KO)
    client = FakeClient(
        league_matches={"PD": [_match(11, "2026-08-21T16:00:00Z")]}
    )
    changes = await resync_kickoffs(client, settings, now=NOW)

    assert [c.fixture_id for c in changes] == [11]
    stored = upcoming_prediction_fixtures(NOW, NOW + timedelta(days=30))
    assert stored[11][0] == datetime(2026, 8, 21, 16, 0, tzinfo=timezone.utc)
    assert client.match_calls == [], "a league hit needs no per-fixture call"


async def test_resync_resolves_a_fixture_that_left_the_window(db, settings):
    """The live case: moved 11 days out, so the league fetch no longer has it."""
    _seed(12, OLD_KO)
    client = FakeClient(
        league_matches={"PD": []},
        matches={12: _match(12, "2026-08-27T18:30:00Z")},
    )
    changes = await resync_kickoffs(client, settings, now=NOW)

    assert client.match_calls == [12]
    assert changes[0].new_kickoff == NEW_KO
    stored = upcoming_prediction_fixtures(NOW, NOW + timedelta(days=30))
    assert stored[12][0] == NEW_KO


async def test_resync_reports_a_postponement_without_rewriting_the_kickoff(
    db, settings
):
    _seed(13, OLD_KO)
    client = FakeClient(
        league_matches={"PD": [_match(13, "2026-08-20T19:30:00Z", "POSTPONED")]}
    )
    changes = await resync_kickoffs(client, settings, now=NOW)

    assert changes[0].is_dead is True
    stored = upcoming_prediction_fixtures(NOW, NOW + timedelta(days=30))
    assert stored[13][0] == OLD_KO, "no new time is known — leave the row alone"


async def test_resync_costs_one_call_per_league(db, settings):
    _seed(14, OLD_KO)
    _seed(15, OLD_KO + timedelta(hours=1))
    client = FakeClient(
        league_matches={
            "PD": [
                _match(14, "2026-08-20T19:30:00Z"),
                _match(15, "2026-08-20T20:30:00Z"),
            ]
        }
    )
    await resync_kickoffs(client, settings, now=NOW)
    codes = [c[0] for c in client.league_calls]
    assert codes == list(settings.leagues)
    assert len(codes) == len(set(codes)), "no league fetched twice"


async def test_resync_survives_a_failing_league(db, settings):
    """One dead league must not cost the others their re-sync.

    The PD fixture is left alone this pass — its league fetch failed, so its
    absence says nothing about whether it moved — while the PL fixture, whose
    league answered, is corrected as normal.
    """

    class Boom(FakeClient):
        async def list_matches(self, code, date_from, date_to, *, status=None):
            if code == "PD":
                raise RuntimeError("upstream 500")
            return await super().list_matches(
                code, date_from, date_to, status=status
            )

    _seed(16, OLD_KO, code="PD")
    _seed(17, OLD_KO, code="PL")
    client = Boom(
        league_matches={"PL": [_match(17, "2026-08-21T16:00:00Z")]},
        matches={16: _match(16, "2026-08-27T18:30:00Z")},
    )
    changes = await resync_kickoffs(client, settings, now=NOW)

    assert [c.fixture_id for c in changes] == [17]
    assert client.match_calls == [], "the dead league must not fan out"


async def test_resync_corrects_a_fixture_that_looks_overdue(db, settings):
    """The live shape of the bug, end to end.

    Celta-Osasuna: stored 2026-08-16 19:30, moved upstream to 2026-08-27 18:30.
    On our clock the stored kickoff is four days PAST, so a forward-only window
    would never look at it again — settlement would keep asking for a result
    that cannot exist and the alerts would stay pinned to the dead day.
    """
    stale = NOW - timedelta(days=4)
    _seed(21, stale)
    client = FakeClient(
        league_matches={"PD": []},
        matches={21: _match(21, "2026-08-27T18:30:00Z")},
    )
    changes = await resync_kickoffs(client, settings, now=NOW)

    assert [c.fixture_id for c in changes] == [21]
    assert changes[0].new_kickoff == NEW_KO
    stored = upcoming_prediction_fixtures(NOW - timedelta(days=7), NOW + timedelta(days=30))
    assert stored[21][0] == NEW_KO


async def test_resync_leaves_alone_a_fixture_older_than_the_backfill(db, settings):
    _seed(22, NOW - timedelta(days=30))
    client = FakeClient()
    assert await resync_kickoffs(client, settings, now=NOW, backfill_days=7) == []
    assert client.league_calls == []


async def test_resync_skips_fixtures_already_settled(db, settings):
    """A settled fixture's kickoff no longer matters — and it must not cost a call."""
    from betbot.storage.repos import record_prediction_outcome

    _seed(23, NOW - timedelta(days=2))
    record_prediction_outcome(
        fixture_id=23, competition_code="PD",
        p_home=0.4, p_draw=0.3, p_away=0.3, actual_outcome="HOME",
        home_goals=1, away_goals=0, settled_at=NOW, result_notified=True,
    )
    client = FakeClient(matches={23: _match(23, "2026-08-27T18:30:00Z")})
    assert await resync_kickoffs(client, settings, now=NOW) == []
    assert client.match_calls == []


async def test_resync_does_nothing_without_stored_fixtures(db, settings):
    client = FakeClient()
    assert await resync_kickoffs(client, settings, now=NOW) == []
    assert client.league_calls == []


# ----------------------------------------------------------------------
# Job hygiene: the old jobs come off
# ----------------------------------------------------------------------
class FakeScheduler:
    def __init__(self, job_ids):
        self.jobs = set(job_ids)

    def remove_job(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        self.jobs.discard(job_id)


def test_drop_alert_jobs_removes_both_ids_for_a_fixture():
    sched = FakeScheduler({"predict_early_9", "predict_late_9", "predict_early_8"})
    removed = drop_alert_jobs(sched, [9])
    assert sorted(removed) == ["predict_early_9", "predict_late_9"]
    assert sched.jobs == {"predict_early_8"}


def test_drop_alert_jobs_tolerates_a_job_that_already_fired():
    sched = FakeScheduler({"predict_late_9"})
    assert drop_alert_jobs(sched, [9]) == ["predict_late_9"]


def test_drop_alert_job_ids_match_what_the_planner_registers(settings):
    """A drift between the two would leave orphans firing off dead kickoffs."""
    row = SimpleNamespace(
        fixture_id=42, competition_code="PL", kickoff=NOW + timedelta(hours=5)
    )
    planned = {job_id for job_id, _ in plan_kickoff_alert_jobs(settings, [row], NOW)}
    assert planned == set(alert_job_ids(42))


# ----------------------------------------------------------------------
# The money guard at fire time
# ----------------------------------------------------------------------
def test_alert_fires_at_its_planned_time():
    ko = NOW + timedelta(minutes=70)
    assert alert_still_valid(NOW, ko, "TIMED", early_lead_minutes=70) is True


def test_alert_fires_for_the_late_lineup_slot():
    ko = NOW + timedelta(minutes=10)
    assert alert_still_valid(NOW, ko, "TIMED", early_lead_minutes=70) is True


def test_alert_is_suppressed_when_the_fixture_moved_later():
    """The phantom-charge case: job pinned to the old day, match days away."""
    ko = NOW + timedelta(days=7)
    assert alert_still_valid(NOW, ko, "TIMED", early_lead_minutes=70) is False


def test_alert_is_suppressed_once_the_match_has_kicked_off():
    ko = NOW - timedelta(minutes=45)
    assert alert_still_valid(NOW, ko, "IN_PLAY", early_lead_minutes=70) is False


def test_alert_tolerates_a_slightly_late_scheduler():
    ko = NOW - timedelta(minutes=5)
    assert alert_still_valid(NOW, ko, "TIMED", early_lead_minutes=70) is True


def test_alert_is_suppressed_for_a_postponed_fixture():
    ko = NOW + timedelta(minutes=70)
    assert alert_still_valid(NOW, ko, "POSTPONED", early_lead_minutes=70) is False


def test_alert_is_suppressed_when_upstream_has_no_kickoff():
    assert alert_still_valid(NOW, None, "TIMED", early_lead_minutes=70) is False


def test_alert_guard_handles_a_naive_kickoff():
    ko = (NOW + timedelta(minutes=70)).replace(tzinfo=None)
    assert alert_still_valid(NOW, ko, "TIMED", early_lead_minutes=70) is True


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-08-27T18:30:00Z", NEW_KO),
        ("2026-08-27T18:30:00+00:00", NEW_KO),
        ("2026-08-27T20:30:00+02:00", NEW_KO),
        ("not-a-date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_utc(raw, expected):
    assert parse_utc(raw) == expected


# ----------------------------------------------------------------------
# Review regressions
# ----------------------------------------------------------------------
async def test_a_postponed_fixture_is_not_re_planned_from_the_stale_row(
    db, settings, monkeypatch
):
    """The drop is worthless unless the dead fixture also leaves the PLAN.

    A POSTPONED fixture has no new time to write, so its stored kickoff still
    reads as today. Dropping its jobs and then re-planning from that untouched
    row re-registers both at exactly the dead times — a guaranteed no-op that
    leaves the phantom, charging alert in place.
    """
    from betbot.reschedule import KickoffChange

    _seed(31, OLD_KO)
    changes = [KickoffChange(31, OLD_KO, None, "POSTPONED")]
    dead = {c.fixture_id for c in changes if c.is_dead}
    assert dead == {31}

    # The plan the scheduler builds must not contain the dead fixture.
    from betbot.storage.repos import predictions_for_kickoff_range

    preds = [
        p
        for p in predictions_for_kickoff_range(
            OLD_KO - timedelta(hours=12), OLD_KO + timedelta(hours=12)
        )
        if p.fixture_id not in dead
    ]
    assert preds == []
    assert plan_kickoff_alert_jobs(settings, preds, NOW) == []


async def test_a_failed_league_does_not_fan_out_per_fixture_calls(db, settings):
    """A rate-limited league must not be answered with a burst of single calls."""

    class Boom(FakeClient):
        async def list_matches(self, code, date_from, date_to, *, status=None):
            raise RuntimeError("429 rate limited")

    for fid in (41, 42, 43):
        _seed(fid, OLD_KO)
    client = Boom(matches={fid: _match(fid, "2026-08-27T18:30:00Z") for fid in (41, 42, 43)})
    changes = await resync_kickoffs(client, settings, now=NOW)

    assert client.match_calls == [], "a failed league fetch must not fan out"
    assert changes == []


def test_an_unreadable_date_on_a_healthy_fixture_is_left_alone():
    """A malformed field must not read as a postponement and kill the alerts."""
    assert plan_kickoff_changes({7: OLD_KO}, {7: (None, "TIMED")}) == []


def test_the_late_alert_guard_uses_the_late_lead():
    """Sized off the EARLY lead, the late alert would pass 70 minutes out.

    It would then ship "confirmed XI" content against lineups that do not exist
    yet on a fixture that had quietly moved an hour later.
    """
    ko = NOW + timedelta(minutes=70)
    assert alert_still_valid(NOW, ko, "TIMED", early_lead_minutes=10) is False
    assert alert_still_valid(NOW, ko, "TIMED", early_lead_minutes=70) is True
