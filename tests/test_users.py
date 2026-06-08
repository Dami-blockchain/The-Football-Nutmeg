"""Multi-tenant user accounts — each user gets an isolated wallet."""

from __future__ import annotations

from betbot.storage.db import init_engine
from betbot.storage.repos import get_or_create_user, get_user, list_users

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
