"""SQLAlchemy engine + session helpers.

Single SQLite file. The engine is process-global; ``init_engine`` is
idempotent. Tests reset the globals via ``monkeypatch``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from betbot.logging import get_logger
from betbot.storage.models import Base

log = get_logger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


# Idempotent, additive column migrations for tables that predate a new column.
# ``create_all`` only creates MISSING tables — it never alters an existing one —
# so a column added to an already-deployed table needs an explicit ALTER. Each
# entry is (table, column, "DDL type + default"). Applied only when the column
# is absent, so this is safe to run on every startup.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("users", "arb_interest", "BOOLEAN NOT NULL DEFAULT 0"),
    ("users", "predictions_consumed", "INTEGER NOT NULL DEFAULT 0"),
    ("model_predictions", "c_home", "FLOAT"),
    ("model_predictions", "c_draw", "FLOAT"),
    ("model_predictions", "c_away", "FLOAT"),
    ("model_predictions", "rps_challenger", "FLOAT"),
    ("model_predictions", "m_home", "FLOAT"),
    ("model_predictions", "m_draw", "FLOAT"),
    ("model_predictions", "m_away", "FLOAT"),
    ("model_predictions", "rps_mov", "FLOAT"),
    # Expected-goals readout (display-only) — DC lambdas for the two sides.
    ("predictions", "home_xg", "REAL"),
    ("predictions", "away_xg", "REAL"),
    # When the match was played, so the record can be scoped to a season.
    ("prediction_outcomes", "kickoff", "DATETIME"),
)

# Indexes for columns added by _ADDITIVE_COLUMNS. ``create_all`` only indexes
# tables it CREATES, so a column added by ALTER on a deployed database would
# never get the index its model declares — the schema would then differ between
# the test suite (fresh db, indexed) and production (full scan), which is the
# kind of drift that only shows up under load.
_ADDITIVE_INDEXES: tuple[tuple[str, str, str], ...] = (
    ("prediction_outcomes", "ix_prediction_outcomes_kickoff", "kickoff"),
)


def _apply_additive_migrations(engine: Engine) -> None:
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl in _ADDITIVE_COLUMNS:
            if table not in existing_tables:
                continue  # create_all just made it with the column already
            cols = {c["name"] for c in insp.get_columns(table)}
            if column in cols:
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        for table, index, column in _ADDITIVE_INDEXES:
            if table not in existing_tables:
                continue
            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {index} ON {table} ({column})")
            )


def _backfill_outcome_kickoffs(engine: Engine) -> None:
    """Fill ``prediction_outcomes.kickoff`` from ``predictions`` where NULL.

    Rows written before the column existed have no kickoff, and the record
    excludes what it cannot date — so without this every historical row would
    silently vanish from the accuracy figure. The kickoff is taken from the
    freshest prediction row for the fixture, which is the same one settlement
    scored. Idempotent (only touches NULLs) and safe to run at every startup.
    """
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    if not {"prediction_outcomes", "predictions"} <= tables:
        return
    if "kickoff" not in {c["name"] for c in insp.get_columns("prediction_outcomes")}:
        return
    # READ first, and take a write lock ONLY when there is something to write.
    # api, bot and daemon all call init_engine simultaneously at deploy, and an
    # unconditional UPDATE here would contend for SQLite's single writer at
    # every boot — the collision that "took down the Telegram bot on the first
    # deploy" (see init_engine). After the first run this SELECT finds nothing
    # and the startup path stays write-free, exactly as the column migrations
    # above are.
    try:
        with engine.connect() as conn:
            pending = conn.execute(
                text(
                    "SELECT 1 FROM prediction_outcomes WHERE kickoff IS NULL LIMIT 1"
                )
            ).first()
        if pending is None:
            return
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE prediction_outcomes SET kickoff = (
                        SELECT p.kickoff FROM predictions p
                        WHERE p.fixture_id = prediction_outcomes.fixture_id
                        ORDER BY p.run_date DESC, p.id DESC LIMIT 1
                    )
                    WHERE kickoff IS NULL
                    """
                )
            )
    except OperationalError as e:
        # "database is locked" — another process is mid-write (a settlement
        # pass). Backfilling is best-effort and idempotent: the next startup
        # retries it. Never take a service down at boot for it.
        log.warning("outcome_kickoff_backfill_skipped", error=str(e))


def init_engine(db_path: Path) -> Engine:
    """Create the engine + schema. Idempotent."""
    global _engine, _SessionLocal
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path.absolute()}"
    engine = create_engine(
        url,
        future=True,
        connect_args={"check_same_thread": False},
    )
    # ``create_all`` (checkfirst=True) is not atomic ACROSS PROCESSES: the api,
    # bot and daemon are launched simultaneously by deploy/start-services.sh and
    # each calls init_engine, so two can both see a table missing and race to
    # CREATE it — the loser raised "table already exists" and crashed (it took
    # down the Telegram bot on the first deploy). Swallow that specific race:
    # the table exists either way, which is all we need.
    try:
        Base.metadata.create_all(engine)
    except OperationalError as e:
        if "already exists" not in str(e).lower():
            raise
    _apply_additive_migrations(engine)
    _backfill_outcome_kickoffs(engine)
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session. Commits on success, rolls back on error."""
    if _SessionLocal is None:
        raise RuntimeError("init_engine() must be called before session_scope().")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
