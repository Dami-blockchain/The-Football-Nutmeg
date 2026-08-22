"""The record covers this season's CLUB football, and nothing else.

The live fault: /record said "4 hits in 8 since 17 August", and two of those
eight were France-England and Spain-Argentina at the World Cup, played 18-19
JULY. They passed every filter the ledger had — settled 17 August (so inside
the epoch), non-degenerate probabilities — because the only date the ledger
stored was the SETTLEMENT time, not the kickoff.

Pins:
  * the outcome row now records the kickoff, and settlement supplies it;
  * international fixtures are excluded by an allowlist, so the next EURO or
    friendly is excluded too, not just the code "WC";
  * a fixture played before the season start is excluded even when it was
    settled inside the window;
  * a row that cannot be dated is excluded rather than guessed at;
  * the startup backfill dates historical rows from `predictions`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionOutcome, PredictionRow
from betbot.storage.repos import (
    prediction_outcomes_since,
    record_prediction_outcome,
    track_record,
)

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
THIS_SEASON = datetime(2026, 8, 17, 19, 0, tzinfo=timezone.utc)
WORLD_CUP = datetime(2026, 7, 18, 21, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "season.sqlite")
    yield


@pytest.fixture(autouse=True)
def _season(monkeypatch):
    """Switch the season scope back ON for this file.

    conftest's ``_no_season_scope`` disables it suite-wide (most tests pin a
    June "now"); these tests exist to exercise it, so they opt back in — the
    same shape as test_confidence_ledger.py re-enabling the epoch.
    """
    from betbot.config import get_settings

    monkeypatch.setenv("BETBOT_SEASON_START", "2026-08-01")
    monkeypatch.setenv("BETBOT_ACCURACY_LEDGER_EPOCH", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _record(fixture_id, code, kickoff, *, outcome="HOME", settled=NOW):
    return record_prediction_outcome(
        fixture_id=fixture_id, competition_code=code,
        p_home=0.5, p_draw=0.25, p_away=0.25,
        actual_outcome=outcome, home_goals=1, away_goals=0,
        settled_at=settled, kickoff=kickoff,
    )


# ----------------------------------------------------------------------
# The live fault
# ----------------------------------------------------------------------
def test_a_world_cup_tie_settled_this_season_is_still_excluded(db):
    """Settled 17 Aug, played 18 Jul — the exact shape that inflated /record."""
    _record(1, "PD", THIS_SEASON)
    _record(2, "WC", WORLD_CUP, settled=datetime(2026, 8, 17, 18, tzinfo=timezone.utc))

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [1]


def test_every_international_code_is_excluded_not_just_wc(db):
    """An allowlist, so the next EURO or friendly does not walk back in."""
    _record(1, "PD", THIS_SEASON)
    for i, code in enumerate(("WC", "EC", "FRIENDLY", "NL"), start=2):
        _record(i, code, THIS_SEASON)

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [1]


@pytest.mark.parametrize("code", ["PL", "PD", "BL1", "SA", "FL1", "CL"])
def test_every_club_competition_is_included(db, code):
    """CL included: it is club football, played by the same club engine."""
    _record(1, code, THIS_SEASON)
    assert len(prediction_outcomes_since(365)) == 1


def test_a_club_match_from_last_season_is_excluded(db):
    _record(1, "PD", datetime(2026, 5, 20, 19, 0, tzinfo=timezone.utc))
    _record(2, "PD", THIS_SEASON)

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [2]


def test_a_match_on_the_season_boundary_is_included(db):
    """The boundary is inclusive — 1 August IS the new season."""
    _record(1, "PD", datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))
    assert len(prediction_outcomes_since(365)) == 1


def test_an_undated_row_is_excluded_rather_than_guessed(db):
    """A record that guesses at a season is worse than one that abstains."""
    _record(1, "PD", None)
    _record(2, "PD", THIS_SEASON)

    rows = prediction_outcomes_since(365)
    assert [r.fixture_id for r in rows] == [2]


def test_naive_kickoffs_are_treated_as_utc(db):
    """SQLite hands back naive datetimes; a tz slip must not drop a real row."""
    _record(1, "PD", THIS_SEASON)
    with session_scope() as s:
        row = s.query(PredictionOutcome).filter_by(fixture_id=1).one()
        row.kickoff = THIS_SEASON.replace(tzinfo=None)
    assert len(prediction_outcomes_since(365)) == 1


# ----------------------------------------------------------------------
# The figure the bot quotes
# ----------------------------------------------------------------------
def test_track_record_counts_only_club_season_matches(db):
    _record(1, "PD", THIS_SEASON, outcome="HOME")   # HIT (pick is HOME)
    _record(2, "PD", THIS_SEASON, outcome="AWAY")   # miss
    _record(3, "WC", WORLD_CUP, outcome="HOME")     # excluded
    _record(4, "PD", datetime(2026, 5, 1, tzinfo=timezone.utc), outcome="HOME")

    tr = track_record(365)
    assert tr["n"] == 2
    assert tr["hits"] == 1
    assert tr["hit_rate"] == 0.5


def test_track_record_is_zero_when_the_season_has_no_club_results(db):
    _record(1, "WC", WORLD_CUP)
    tr = track_record(365)
    assert tr["n"] == 0
    assert tr["hit_rate"] == 0.0


# ----------------------------------------------------------------------
# Storage + migration
# ----------------------------------------------------------------------
async def test_settlement_stores_the_kickoff(db, settings):
    """Without this the row cannot be placed in a season at all."""
    from betbot.settlement import SettlementWatcher

    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=99, competition_code="PD", kickoff=THIS_SEASON,
                run_date="2026-08-17", home_team="A", away_team="B",
                p_home=0.5, p_draw=0.25, p_away=0.25,
                home_score=1.0, away_score=0.0, draw_score=2.4,
            )
        )

    class FakeFD:
        async def get_match(self, fixture_id):
            return {
                "status": "FINISHED",
                "score": {"winner": "HOME_TEAM", "fullTime": {"home": 2, "away": 0}},
            }

    w = SettlementWatcher(FakeFD(), settings)
    await w.settle_due(now=THIS_SEASON + timedelta(hours=3))

    with session_scope() as s:
        row = s.query(PredictionOutcome).filter_by(fixture_id=99).one()
        stored = row.kickoff
    assert stored is not None
    if stored.tzinfo is None:
        stored = stored.replace(tzinfo=timezone.utc)
    assert stored == THIS_SEASON


def test_startup_backfills_kickoffs_for_historical_rows(tmp_path):
    """Rows predating the column must be dated, or the record loses its history."""
    from sqlalchemy import text

    import betbot.storage.db as dbmod
    from betbot.storage.db import _backfill_outcome_kickoffs

    init_engine(tmp_path / "backfill.sqlite")
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=77, competition_code="PD", kickoff=THIS_SEASON,
                run_date="2026-08-17", home_team="A", away_team="B",
                p_home=0.5, p_draw=0.25, p_away=0.25,
                home_score=1.0, away_score=0.0, draw_score=2.4,
            )
        )
    _record(77, "PD", None)  # column exists but this row is undated

    _backfill_outcome_kickoffs(dbmod._engine)

    with session_scope() as s:
        ko = s.execute(
            text("SELECT kickoff FROM prediction_outcomes WHERE fixture_id = 77")
        ).scalar_one()
    assert ko is not None
    assert len(prediction_outcomes_since(365)) == 1


# ----------------------------------------------------------------------
# Review regressions
# ----------------------------------------------------------------------
def test_a_configured_extra_league_counts_in_the_record(db, monkeypatch):
    """The record must cover exactly what the bot predicts.

    ``Settings.leagues`` has no env alias — it is overridden in code — but every
    other stage of the pipeline reads it, so a competition added there would
    otherwise be fetched, predicted, bet on and settled, then silently dropped
    from the record, quietly under-counting.
    """
    import betbot.config as cfg

    base = cfg.get_settings()
    widened = base.model_copy(update={"leagues": ("PL", "PD", "DED")})
    monkeypatch.setattr(cfg, "get_settings", lambda: widened)

    _record(1, "DED", THIS_SEASON)
    assert len(prediction_outcomes_since(365)) == 1


def test_competition_codes_are_matched_case_insensitively(db):
    """form.py stores whatever the API returns; siblings all normalise first."""
    _record(1, "pd", THIS_SEASON)
    assert len(prediction_outcomes_since(365)) == 1


def test_an_empty_competition_code_is_not_club_football():
    from betbot.storage.repos import _is_club_competition

    assert _is_club_competition("") is False


def test_the_season_boundary_rolls_forward_with_the_football_year():
    """A hardcoded date silently expires; the record would then span two seasons."""
    from betbot.storage.repos import _current_season_start

    # July 2027 is still 2026/27 — the season has not turned over yet.
    assert _current_season_start(
        datetime(2027, 7, 15, tzinfo=timezone.utc)
    ) == datetime(2026, 8, 1, tzinfo=timezone.utc)
    # August 2027 is 2027/28.
    assert _current_season_start(
        datetime(2027, 8, 15, tzinfo=timezone.utc)
    ) == datetime(2027, 8, 1, tzinfo=timezone.utc)


def test_an_unparseable_season_start_falls_back_to_auto_not_to_off(monkeypatch):
    """A typo must not silently disable scoping and readmit every old fixture."""
    from betbot.config import get_settings
    from betbot.storage.repos import _current_season_start, _season_start

    monkeypatch.setenv("BETBOT_SEASON_START", "not-a-date")
    get_settings.cache_clear()
    try:
        assert _season_start() == _current_season_start()
    finally:
        get_settings.cache_clear()


def test_out_of_scope_rows_are_not_billed_as_degenerate(db, caplog):
    """The degenerate log traces ledger poison; counting scope drops corrupts it."""
    import logging

    _record(1, "PD", THIS_SEASON)
    _record(2, "WC", WORLD_CUP)
    with caplog.at_level(logging.INFO):
        prediction_outcomes_since(365)
    assert "accuracy_ledger_degenerate_rows_excluded" not in caplog.text


def test_the_backfill_is_write_free_once_every_row_is_dated(tmp_path):
    """api, bot and daemon all boot at once; a needless write lock can kill one."""
    import betbot.storage.db as dbmod
    from betbot.storage.db import _backfill_outcome_kickoffs

    init_engine(tmp_path / "nowrite.sqlite")
    _record(1, "PD", THIS_SEASON)  # already dated

    calls = []
    real_begin = dbmod._engine.begin

    def _spy_begin(*a, **kw):
        calls.append(1)
        return real_begin(*a, **kw)

    dbmod._engine.begin = _spy_begin
    try:
        _backfill_outcome_kickoffs(dbmod._engine)
    finally:
        dbmod._engine.begin = real_begin
    assert calls == [], "took a write lock with nothing to backfill"


def test_the_backfill_survives_a_locked_database(tmp_path, monkeypatch):
    """Best-effort and idempotent — the next startup retries. Never die at boot."""
    from sqlalchemy.exc import OperationalError

    import betbot.storage.db as dbmod
    from betbot.storage.db import _backfill_outcome_kickoffs

    init_engine(tmp_path / "locked.sqlite")
    _record(1, "PD", None)  # undated -> the backfill will want to write

    def _boom(*a, **kw):
        raise OperationalError("UPDATE", {}, Exception("database is locked"))

    monkeypatch.setattr(dbmod._engine, "begin", _boom)
    _backfill_outcome_kickoffs(dbmod._engine)  # must not raise


def test_the_kickoff_index_exists_after_an_additive_migration(tmp_path):
    """create_all does not index an existing table; without this, schema drift."""
    from sqlalchemy import text

    import betbot.storage.db as dbmod

    init_engine(tmp_path / "idx.sqlite")
    with dbmod._engine.connect() as conn:
        names = {
            r[0]
            for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='index'")
            )
        }
    assert "ix_prediction_outcomes_kickoff" in names


# ----------------------------------------------------------------------
# Review follow-ups: orphan backfill row + concurrent boot migration race
# ----------------------------------------------------------------------
def test_the_backfill_is_write_free_when_the_only_undated_row_is_an_orphan(tmp_path):
    """An outcome with no matching prediction can never be dated.

    Its kickoff subquery yields NULL, so it stays undated — and the OLD trigger
    (``kickoff IS NULL``) would re-select it on every boot forever, taking a
    write lock the "write-free after the first run" contract forbids. api, bot
    and daemon all boot at once, so a needless recurring write lock is exactly
    the collision that killed the Telegram bot on the first deploy. The
    fillable-row guard must exclude the orphan so no write is ever attempted.
    """
    import betbot.storage.db as dbmod
    from betbot.storage.db import _backfill_outcome_kickoffs

    init_engine(tmp_path / "orphan.sqlite")
    _record(999, "PD", None)  # undated AND no matching predictions row -> orphan

    calls = []
    real_begin = dbmod._engine.begin

    def _spy_begin(*a, **kw):
        calls.append(1)
        return real_begin(*a, **kw)

    dbmod._engine.begin = _spy_begin
    try:
        _backfill_outcome_kickoffs(dbmod._engine)
        # Second boot: still no write, and the row is still (correctly) undated.
        _backfill_outcome_kickoffs(dbmod._engine)
    finally:
        dbmod._engine.begin = real_begin
    assert calls == [], "took a write lock over an unfillable orphan row"


def test_a_datable_row_still_backfills_even_next_to_an_orphan(tmp_path):
    """The orphan guard must not stop real rows from being dated."""
    from sqlalchemy import text

    import betbot.storage.db as dbmod
    from betbot.storage.db import _backfill_outcome_kickoffs

    init_engine(tmp_path / "mixed.sqlite")
    with session_scope() as s:
        s.add(
            PredictionRow(
                fixture_id=55, competition_code="PD", kickoff=THIS_SEASON,
                run_date="2026-08-17", home_team="A", away_team="B",
                p_home=0.5, p_draw=0.25, p_away=0.25,
                home_score=1.0, away_score=0.0, draw_score=2.4,
            )
        )
    _record(55, "PD", None)   # datable (has a prediction)
    _record(999, "PD", None)  # orphan (no prediction)

    _backfill_outcome_kickoffs(dbmod._engine)

    with session_scope() as s:
        dated = s.execute(
            text("SELECT kickoff FROM prediction_outcomes WHERE fixture_id = 55")
        ).scalar_one()
        orphan = s.execute(
            text("SELECT kickoff FROM prediction_outcomes WHERE fixture_id = 999")
        ).scalar_one()
    assert dated is not None
    assert orphan is None  # left undated, not guessed


def test_concurrent_boots_racing_the_same_column_migration_do_not_crash(tmp_path):
    """The real three-way boot race, reproduced with threads.

    deploy/start-services.sh launches api, bot and daemon at once; each opens
    its own engine and runs _apply_additive_migrations. Before the fix the ALTER
    loser raised "duplicate column name" and died at boot (the rollout had to
    STAGGER the starts to avoid it). Here N engines race the identical ALTER on
    a shared file; not one may raise, and the column must exist exactly once.
    """
    import threading

    from sqlalchemy import create_engine, inspect, text

    from betbot.storage.db import _apply_additive_migrations

    db_file = tmp_path / "race.sqlite"
    url = f"sqlite:///{db_file.absolute()}"

    # Seed a table that is MISSING an additive column, the way a pre-migration
    # deploy's on-disk schema is. `users` predates arb_interest/predictions_
    # consumed in _ADDITIVE_COLUMNS.
    seed = create_engine(url, future=True, connect_args={"check_same_thread": False})
    with seed.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
    seed.dispose()

    n = 8
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def _boot():
        eng = create_engine(
            url, future=True, connect_args={"check_same_thread": False}
        )
        try:
            barrier.wait()  # all threads hit the ALTER together
            _apply_additive_migrations(eng)
        except BaseException as e:  # noqa: BLE001 — the whole point is to catch it
            errors.append(e)
        finally:
            eng.dispose()

    threads = [threading.Thread(target=_boot) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"a racing boot crashed: {errors!r}"

    check = create_engine(url, future=True)
    cols = [c["name"] for c in inspect(check).get_columns("users")]
    check.dispose()
    assert cols.count("arb_interest") == 1
    assert cols.count("predictions_consumed") == 1
