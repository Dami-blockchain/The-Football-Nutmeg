"""SQLAlchemy engine + session helpers.

Single SQLite file. The engine is process-global; ``init_engine`` is
idempotent. Tests reset the globals via ``monkeypatch``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from betbot.storage.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


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
