"""TASK D — the sold-triple snapshot on the reveal ledger.

Covers the additive schema migration (on a COPY of an OLD-schema DB), the
persistence of the ACTUALLY-RENDERED triple with the reveal row, the
first-reveal-wins idempotency (an existing row is never overwritten), the
sold-basis high-conf tally over NULL/legacy + real rows, and ``format_result``
with and without a stored triple.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionReveal
from betbot.storage.repos import (
    high_conf_band_tally,
    high_conf_band_tally_sold,
    record_prediction_outcome,
    record_reveal,
    sold_triple,
)

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
THIS_SEASON = datetime(2026, 9, 4, 19, 0, tzinfo=timezone.utc)
WORLD_CUP = datetime(2026, 7, 18, 21, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "reveal.sqlite")
    yield


# ----------------------------------------------------------------------
# Migration on a COPY of an OLD-schema DB (throwaway file, never live)
# ----------------------------------------------------------------------
def test_additive_migration_on_old_schema_db(tmp_path):
    """A DB whose prediction_reveals predates the triple columns gains them on
    init_engine, keeps its legacy row (triple reads NULL), and accepts new
    triple-carrying reveals."""
    path = tmp_path / "old_schema.sqlite"
    # Build the OLD prediction_reveals table by hand — no p_home/p_draw/p_away.
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE prediction_reveals (
            id INTEGER PRIMARY KEY,
            telegram_user_id INTEGER,
            fixture_id INTEGER,
            charged BOOLEAN DEFAULT 0,
            revealed_at DATETIME,
            CONSTRAINT uq_reveal_user_fixture UNIQUE (telegram_user_id, fixture_id)
        )
        """
    )
    con.execute(
        "INSERT INTO prediction_reveals "
        "(telegram_user_id, fixture_id, charged, revealed_at) "
        "VALUES (7, 100, 1, '2026-09-01 10:00:00')"
    )
    con.commit()
    cols_before = {r[1] for r in con.execute("PRAGMA table_info(prediction_reveals)")}
    con.close()
    assert {"p_home", "p_draw", "p_away"} & cols_before == set()

    # Migrate by opening the engine on the SAME file.
    init_engine(path)

    con = sqlite3.connect(path)
    cols_after = {r[1] for r in con.execute("PRAGMA table_info(prediction_reveals)")}
    assert {"p_home", "p_draw", "p_away"} <= cols_after
    # Legacy row preserved, and its new triple columns are NULL.
    legacy = con.execute(
        "SELECT charged, p_home, p_draw, p_away FROM prediction_reveals "
        "WHERE fixture_id = 100"
    ).fetchone()
    con.close()
    assert legacy[0] == 1
    assert legacy[1:] == (None, None, None)

    # The unique constraint still bites (idempotency intact).
    assert record_reveal(7, 100, True) is False
    # A brand-new triple-carrying reveal persists.
    assert record_reveal(9, 200, True, 0.71, 0.19, 0.10) is True
    assert sold_triple(200) == (0.71, 0.19, 0.10)
    # The legacy fixture has no stored triple.
    assert sold_triple(100) is None


# ----------------------------------------------------------------------
# Persistence of the rendered triple
# ----------------------------------------------------------------------
def test_reveal_persists_rendered_triple(db):
    assert record_reveal(5, 300, False, 0.68, 0.20, 0.12) is True
    with session_scope() as s:
        row = s.query(PredictionReveal).filter_by(fixture_id=300).one()
        assert (row.p_home, row.p_draw, row.p_away) == (0.68, 0.20, 0.12)
        assert row.charged is False
    assert sold_triple(300) == (0.68, 0.20, 0.12)


def test_reveal_without_triple_stores_null(db):
    assert record_reveal(5, 301, False) is True
    assert sold_triple(301) is None


# ----------------------------------------------------------------------
# First reveal wins — an existing row is NEVER overwritten
# ----------------------------------------------------------------------
def test_re_reveal_does_not_overwrite_sold_triple(db):
    assert record_reveal(5, 400, True, 0.72, 0.18, 0.10) is True
    # A later reveal of the same (user, fixture) with a DIFFERENT (drifted)
    # triple is a no-op and must not change the stored sold triple.
    assert record_reveal(5, 400, True, 0.50, 0.25, 0.25) is False
    assert sold_triple(400) == (0.72, 0.18, 0.10)


def test_sold_triple_takes_earliest_reveal_across_users(db):
    """Two users bought the same fixture at different triples — the EARLIEST
    reveal is the canonical sold call."""
    with session_scope() as s:
        s.add(PredictionReveal(
            telegram_user_id=1, fixture_id=500, charged=True,
            p_home=0.80, p_draw=0.12, p_away=0.08,
            revealed_at=datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc),
        ))
        s.add(PredictionReveal(
            telegram_user_id=2, fixture_id=500, charged=True,
            p_home=0.66, p_draw=0.20, p_away=0.14,
            revealed_at=datetime(2026, 9, 4, 11, 0, tzinfo=timezone.utc),
        ))
    assert sold_triple(500) == (0.80, 0.12, 0.08)


# ----------------------------------------------------------------------
# Sold-basis high-conf tally: recovers drifted calls, ignores NULL/legacy rows
# ----------------------------------------------------------------------
def _outcome(fid, code, ph, pd, pa, *, outcome, kickoff=THIS_SEASON):
    record_prediction_outcome(
        fixture_id=fid, competition_code=code,
        p_home=ph, p_draw=pd, p_away=pa,
        actual_outcome=outcome, home_goals=2, away_goals=0,
        settled_at=NOW, kickoff=kickoff,
    )


def test_sold_tally_recovers_drifted_call_that_standard_tally_drops(db):
    """A correct call sold at HOME 0.72 but rescored down to HOME 0.50 by
    settlement: the standard tally (post-rescore triple) drops it from BOTH
    numerator and denominator; the sold tally (stored triple) keeps it."""
    # Outcome carries the POST-rescore triple, below the 0.65 bar; correct.
    _outcome(565791, "PL", 0.50, 0.25, 0.25, outcome="HOME")
    # The sold triple was above the bar.
    record_reveal(1, 565791, True, 0.72, 0.18, 0.10)

    assert high_conf_band_tally(0.65) == (0, 0)      # dropped post-rescore
    assert high_conf_band_tally_sold(0.65) == (1, 1)  # recovered from sold triple


def test_sold_tally_ignores_legacy_null_triple_rows(db):
    """A fixture revealed only on a legacy (NULL-triple) row has no sold basis
    and is excluded from the sold tally, even though it is in scope and its
    post-rescore triple clears the bar (so the standard tally counts it)."""
    _outcome(560000, "PL", 0.70, 0.18, 0.12, outcome="HOME")
    record_reveal(1, 560000, True)  # NULL triple (legacy shape)

    assert high_conf_band_tally(0.65) == (1, 1)       # standard counts it
    assert high_conf_band_tally_sold(0.65) == (0, 0)  # no sold triple -> excluded


def test_sold_tally_respects_scope_and_band_and_draw(db, monkeypatch):
    """Sold tally honours club-only + season scope and the band/draw rules."""
    from betbot.config import get_settings

    monkeypatch.setenv("BETBOT_SEASON_START", "2026-08-01")
    get_settings.cache_clear()
    try:
        # In scope, sold above bar, correct -> counts as a hit.
        _outcome(1, "PL", 0.55, 0.25, 0.20, outcome="HOME")
        record_reveal(1, 1, True, 0.71, 0.19, 0.10)
        # Out of season (World Cup date) -> excluded by season scope.
        _outcome(2, "PL", 0.55, 0.25, 0.20, outcome="HOME", kickoff=WORLD_CUP)
        record_reveal(1, 2, True, 0.90, 0.05, 0.05)
        # Non-club competition -> excluded.
        _outcome(3, "WC", 0.55, 0.25, 0.20, outcome="HOME")
        record_reveal(1, 3, True, 0.90, 0.05, 0.05)
        # Sold triple's top pick is a DRAW -> never a high-conf call.
        _outcome(4, "PL", 0.30, 0.40, 0.30, outcome="DRAW")
        record_reveal(1, 4, True, 0.20, 0.70, 0.10)
        # Sold below the bar -> excluded from the band.
        _outcome(5, "PL", 0.55, 0.25, 0.20, outcome="HOME")
        record_reveal(1, 5, True, 0.60, 0.22, 0.18)

        assert high_conf_band_tally_sold(0.65) == (1, 1)
    finally:
        get_settings.cache_clear()


# ----------------------------------------------------------------------
# format_result with / without a stored sold triple
# ----------------------------------------------------------------------
class _Outcome:
    fixture_id = 1
    predicted_pick = "HOME"
    correct = True
    home_goals = 2
    away_goals = 0
    predicted_home = 0.50
    predicted_draw = 0.25
    predicted_away = 0.25


def test_format_result_without_sold_triple_is_unchanged():
    from betbot.tips import format_result

    body = format_result(_Outcome(), "Arsenal", "Chelsea")
    assert "As shown to you" not in body
    assert "Model had H 50% / D 25% / A 25%" in body


def test_format_result_quotes_sold_triple_when_present():
    from betbot.tips import format_result

    body = format_result(
        _Outcome(), "Arsenal", "Chelsea", sold_triple=(0.72, 0.18, 0.10)
    )
    assert "As shown to you: H 72% / D 18% / A 10%" in body
    # The post-rescore model triple is still shown alongside it.
    assert "Model had H 50% / D 25% / A 25%" in body
