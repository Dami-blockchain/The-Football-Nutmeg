"""Unit tests for the ClubElo snapshot refresh (no network — urlopen mocked).

Covers the three failure modes that actually bit us:

* the upstream hanging (timeout) and then recovering -> the retry must save it;
* the upstream hanging for good -> bounded retries, existing snapshot intact;
* the snapshot silently going stale -> detectable and loud.
"""

from __future__ import annotations

import io
import socket
from datetime import date, timedelta
from pathlib import Path

import betbot.data.clubelo as clubelo

HEADER = "Rank,Club,Country,Level,Elo,From,To"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _payload(*, frm: date | None = None, rows: int = 60) -> bytes:
    """A structurally real snapshot: enough rows and in-band Elo values."""
    frm = frm or date.today()
    to = frm + timedelta(days=10)  # ClubElo's To is a FUTURE validity horizon
    lines = [HEADER]
    for i in range(rows):
        lines.append(f"{i + 1},Club {i},ENG,1,{2000 - i * 5}.5,{frm.isoformat()},{to.isoformat()}")
    return ("\n".join(lines) + "\n").encode()


def _urlopen_returning(payload: bytes):
    def _f(*a, **k):
        return _Resp(payload)

    return _f


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_refresh_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _urlopen_returning(_payload()))
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest, sleep=lambda _s: None) is True
    assert dest.exists()
    assert "Club 0" in dest.read_text()


def test_refresh_rejects_bad_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(clubelo.urllib.request, "urlopen",
                        _urlopen_returning(b"<html>nope</html>"))
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest, sleep=lambda _s: None) is False
    assert not dest.exists()


def test_refresh_survives_network_error(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _boom)
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest, sleep=lambda _s: None) is False
    assert not dest.exists()


# --------------------------------------------------------------------------
# retry behaviour — the actual outage shape
# --------------------------------------------------------------------------


def test_timeout_then_successful_retry(tmp_path, monkeypatch):
    """Two hangs then a good response: the retry must recover the snapshot."""
    calls = {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise socket.timeout("timed out")
        return _Resp(_payload())

    slept: list[float] = []
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _flaky)
    dest = tmp_path / "clubelo_latest.csv"

    assert clubelo.refresh_latest(dest, retries=3, sleep=slept.append) is True
    assert calls["n"] == 3
    assert dest.exists() and "Club 0" in dest.read_text()
    # backed off between attempts, and backed off further each time
    assert len(slept) == 2
    assert slept[1] > slept[0]


def test_retries_are_bounded_and_do_not_hammer(tmp_path, monkeypatch):
    """Exhausted retries: exactly `retries` attempts, no more."""
    calls = {"n": 0}

    def _always_timeout(*a, **k):
        calls["n"] += 1
        raise socket.timeout("timed out")

    slept: list[float] = []
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _always_timeout)
    dest = tmp_path / "clubelo_latest.csv"

    assert clubelo.refresh_latest(dest, retries=3, sleep=slept.append) is False
    assert calls["n"] == 3
    assert len(slept) == 2  # no sleep after the final failure
    assert not dest.exists()


def test_exhausted_retries_leave_the_previous_snapshot_intact(tmp_path, monkeypatch):
    """A failed refresh must never destroy the snapshot we still have."""
    dest = tmp_path / "clubelo_latest.csv"
    good = _payload().decode()
    dest.write_text(good)

    def _always_timeout(*a, **k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _always_timeout)
    assert clubelo.refresh_latest(dest, retries=2, sleep=lambda _s: None) is False
    assert dest.read_text() == good


def test_bad_payload_is_not_retried_and_does_not_clobber(tmp_path, monkeypatch):
    """A wrong body is the server's answer, not a blip — one attempt, keep the old file."""
    dest = tmp_path / "clubelo_latest.csv"
    good = _payload().decode()
    dest.write_text(good)

    calls = {"n": 0}

    def _garbage(*a, **k):
        calls["n"] += 1
        return _Resp(b"Rank,Club,Country,Level,Elo,From,To\n1,X,ENG,1,99999,2026-01-01,2026-01-07\n")

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _garbage)
    assert clubelo.refresh_latest(dest, retries=3, sleep=lambda _s: None) is False
    assert calls["n"] == 1
    assert dest.read_text() == good


def test_write_is_atomic_no_temp_files_left(tmp_path, monkeypatch):
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _urlopen_returning(_payload()))
    dest = tmp_path / "sub" / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest, sleep=lambda _s: None) is True
    assert [p.name for p in dest.parent.iterdir()] == ["clubelo_latest.csv"]


# --------------------------------------------------------------------------
# staleness detection
# --------------------------------------------------------------------------


def test_status_reports_fresh_snapshot_as_fresh(tmp_path):
    dest = tmp_path / "c.csv"
    dest.write_bytes(_payload(frm=date.today()))
    st = clubelo.snapshot_status(dest)
    assert st.exists and not st.stale
    assert st.age_days == 0.0
    assert st.snapshot_date == date.today()
    assert st.clubs == 60


def test_status_detects_a_stale_snapshot(tmp_path):
    dest = tmp_path / "c.csv"
    dest.write_bytes(_payload(frm=date.today() - timedelta(days=9)))
    st = clubelo.snapshot_status(dest, stale_after_days=3)
    assert st.stale
    assert st.age_days == 9.0
    assert st.reason == "stale_9.0d"


def test_age_is_measured_from_From_not_the_future_To_column(tmp_path):
    """The original bug: To is a future horizon, so age came out NEGATIVE."""
    frm = date.today() - timedelta(days=20)
    dest = tmp_path / "c.csv"
    dest.write_bytes(_payload(frm=frm))  # To = frm + 10d
    st = clubelo.snapshot_status(dest, stale_after_days=3)
    assert st.snapshot_date == frm
    assert st.age_days == 20.0 and st.age_days > 0
    assert st.stale


def test_status_flags_missing_and_unparseable_files(tmp_path):
    missing = clubelo.snapshot_status(tmp_path / "nope.csv")
    assert missing.stale and not missing.exists and missing.reason == "missing"

    junk = tmp_path / "junk.csv"
    junk.write_text("not a clubelo file at all\n")
    st = clubelo.snapshot_status(junk)
    assert st.stale and st.reason == "unparseable"


def test_failed_refresh_reports_the_stale_file_it_fell_back_on(tmp_path, monkeypatch):
    """The whole point: the outage is survivable, the silence is not."""
    dest = tmp_path / "c.csv"
    dest.write_bytes(_payload(frm=date.today() - timedelta(days=11)))

    def _always_timeout(*a, **k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _always_timeout)

    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(clubelo.log, "error", lambda ev, **kw: seen.append((ev, kw)))
    monkeypatch.setattr(clubelo.log, "warning", lambda ev, **kw: None)

    assert clubelo.refresh_latest(dest, retries=2, sleep=lambda _s: None) is False

    events = {ev for ev, _ in seen}
    assert "clubelo_refresh_failed" in events
    assert "clubelo_snapshot_stale" in events
    stale_kw = next(kw for ev, kw in seen if ev == "clubelo_snapshot_stale")
    assert stale_kw["age_days"] == 11.0  # the age is IN the alert


def test_check_snapshot_freshness_returns_the_status_seam(tmp_path):
    dest = tmp_path / "c.csv"
    dest.write_bytes(_payload(frm=date.today() - timedelta(days=30)))
    st = clubelo.check_snapshot_freshness(dest, stale_after_days=3)
    assert isinstance(st, clubelo.SnapshotStatus)
    assert st.stale and st.age_days == 30.0
    assert st.as_log_fields()["path"] == str(Path(dest))


# --------------------------------------------------------------------------
# historical snapshots (the backtest cache shares this one fetch path)
# --------------------------------------------------------------------------


def test_snapshot_date_pins_the_requested_day(tmp_path, monkeypatch):
    seen: list[str] = []

    def _capture(url, *a, **k):
        seen.append(url)
        return _Resp(_payload())

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _capture)
    dest = tmp_path / "2024-03-01.csv"
    assert clubelo.refresh_latest(
        dest, snapshot_date=date(2024, 3, 1), sleep=lambda _s: None
    ) is True
    assert seen == ["http://api.clubelo.com/2024-03-01"]


def test_historical_fetch_does_not_raise_a_staleness_alarm(tmp_path, monkeypatch):
    """A 2024 backtest snapshot is old by design — that is not a live outage."""
    dest = tmp_path / "2024-03-01.csv"
    dest.write_bytes(_payload(frm=date(2024, 3, 1)))

    def _always_timeout(*a, **k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _always_timeout)
    seen: list[str] = []
    monkeypatch.setattr(clubelo.log, "error", lambda ev, **kw: seen.append(ev))
    monkeypatch.setattr(clubelo.log, "warning", lambda ev, **kw: None)

    assert clubelo.refresh_latest(
        dest, retries=1, snapshot_date=date(2024, 3, 1), sleep=lambda _s: None
    ) is False
    assert "clubelo_refresh_failed" in seen
    assert "clubelo_snapshot_stale" not in seen
