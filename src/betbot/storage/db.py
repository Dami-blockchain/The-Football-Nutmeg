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
from sqlalchemy.orm import Session, sessionmaker

from betbot.storage.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


# Idempotent, additive column migrations for tables that predate a new column.
# ``create_all`` only creates MISSING tables — it never alters an existing one —
# so a column added to an already-deployed table needs an explicit ALTER. Each
# entry is (table, column, "DDL type + default"). Applied only when the column
# is absent, so this is safe to run on every startup.
_ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("users", "arb_interest", "BOOLEAN NOT NULL DEFAULT 0"),
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
    Base.metadata.create_all(engine)
    _apply_additive_migrations(engine)
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
