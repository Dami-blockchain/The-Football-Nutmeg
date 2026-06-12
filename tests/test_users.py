"""Multi-tenant user accounts — each user gets an isolated wallet."""

from __future__ import annotations

from betbot.storage.db import init_engine
from betbot.storage.repos import (
    get_or_create_user,
    get_user,
    list_users,
    set_arb_interest,
)

import pytest


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "users.sqlite")
    return tmp_path


def test_create_user_generates_isolated_wallet(db):
    u = get_or_create_user(123, "alice", secrets_dir=str(db / "secrets"))
    assert u.telegram_user_id == 123
    assert u.wallet_address.startswith("0x")
    assert (db / "secrets" / "users" / "123.key").exists()


def test_get_or_create_is_idempotent(db):
    u1 = get_or_create_user(123, "alice", secrets_dir=str(db / "secrets"))
    u2 = get_or_create_user(123, "renamed", secrets_dir=str(db / "secrets"))
    assert u1.wallet_address == u2.wallet_address  # wallet not regenerated
    assert get_user(123).wallet_address == u1.wallet_address


def test_distinct_users_get_distinct_wallets(db):
    a = get_or_create_user(1, "a", secrets_dir=str(db / "secrets"))
    b = get_or_create_user(2, "b", secrets_dir=str(db / "secrets"))
    assert a.wallet_address != b.wallet_address  # isolated, not pooled
    assert len(list_users()) == 2


def test_new_user_defaults_arb_interest_false(db):
    u = get_or_create_user(9, "a", secrets_dir=str(db / "secrets"))
    assert u.arb_interest is False
    set_arb_interest(9, True)
    assert get_user(9).arb_interest is True


def test_additive_migration_adds_arb_interest_to_legacy_db(tmp_path):
    """A users table created WITHOUT arb_interest (legacy prod) gets the column
    added idempotently on the next init_engine — no manual migration needed."""
    import sqlite3

    from sqlalchemy import inspect

    from betbot.storage import db as db_mod

    dbfile = tmp_path / "legacy.sqlite"
    # Hand-build a legacy users table missing the new column.
    conn = sqlite3.connect(dbfile)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, telegram_user_id INTEGER, "
        "name TEXT, wallet_address TEXT, wallet_keyfile TEXT, active BOOLEAN, "
        "created_at TIMESTAMP, updated_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO users (telegram_user_id, name, wallet_address, "
        "wallet_keyfile, active) VALUES (77, 'legacy', '0xabc', '/k', 1)"
    )
    conn.commit()
    conn.close()

    cols_before = {
        r[1]
        for r in sqlite3.connect(dbfile).execute("PRAGMA table_info(users)")
    }
    assert "arb_interest" not in cols_before

    engine = db_mod.init_engine(dbfile)
    cols_after = {c["name"] for c in inspect(engine).get_columns("users")}
    assert "arb_interest" in cols_after
    # Existing row survives and reads back with the default.
    assert get_user(77).arb_interest is False
