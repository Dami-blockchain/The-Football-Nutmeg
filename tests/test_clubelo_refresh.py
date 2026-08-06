"""Unit tests for the ClubElo snapshot refresh (no network — urlopen mocked)."""

from __future__ import annotations

import io

import betbot.data.clubelo as clubelo


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_refresh_writes_snapshot(tmp_path, monkeypatch):
    payload = b"Rank,Club,Country,Level,Elo,From,To\n1,Man City,ENG,1,2000,2026-01-01,2026-01-07\n"
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", lambda *a, **k: _Resp(payload))
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest) is True
    assert dest.exists()
    assert "Man City" in dest.read_text()


def test_refresh_rejects_bad_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(clubelo.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(b"<html>nope</html>"))
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest) is False
    assert not dest.exists()


def test_refresh_survives_network_error(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _boom)
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest) is False
    assert not dest.exists()
