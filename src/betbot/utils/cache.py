"""Tiny TTL cache. Used by the football-data client.

Not threadsafe; we run a single asyncio loop, which is enough.
"""

from __future__ import annotations

import time
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: float, max_size: int = 1024) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[object, tuple[float, T]] = {}

    def get(self, key: object) -> T | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: object, value: T) -> None:
        if len(self._store) >= self._max_size:
            # Evict the oldest entry. O(n) but n is bounded.
            oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            del self._store[oldest_key]
        self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
