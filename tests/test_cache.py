"""Tests for the TTL cache."""

from __future__ import annotations

import time

from betbot.utils.cache import TTLCache


def test_set_and_get() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    assert c.get("a") == 1


def test_expiry() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=0.05)
    c.set("a", 1)
    time.sleep(0.1)
    assert c.get("a") is None


def test_eviction_at_max_size() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=60, max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # should evict "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_clear() -> None:
    c: TTLCache[int] = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.clear()
    assert c.get("a") is None
    assert len(c) == 0
