"""``_club_refresh_tick`` fault-tolerance + operator paging.

The live regression: the tick ran the three club-learning scripts in one loop
with ``check=True``, so the FIRST failure (football-data.co.uk 503 in the fetch
step) cancelled the Glicko re-seed AND the Dixon-Coles refit, and only logged a
warning nobody read. These tests pin the fix:

* every step runs independently — a fetch failure no longer cancels the
  re-seed/refit;
* a failed step PAGES the operator (not just a log line).
"""

from __future__ import annotations

import asyncio

import pytest

import betbot.main as main


class _StopBeforeStart(Exception):
    pass


class _FakeJob:
    def __init__(self, func, job_id, args, kwargs, trigger):
        self.func, self.id = func, job_id
        self.args, self.kwargs, self.trigger = tuple(args or ()), dict(kwargs or {}), trigger


class _RecordingScheduler:
    def __init__(self, *_a, **_kw):
        self.jobs = []

    def add_job(self, func, trigger=None, *, id=None, args=None, kwargs=None, **_rest):
        self.jobs = [j for j in self.jobs if j.id != id]
        self.jobs.append(_FakeJob(func, id, args, kwargs, trigger))
        return self.jobs[-1]

    def get_jobs(self):
        return list(self.jobs)

    def start(self):
        raise _StopBeforeStart

    def shutdown(self, wait=False):
        pass


def _club_refresh_job(monkeypatch, settings):
    captured = {}

    def _factory(*_a, **_kw):
        captured["s"] = _RecordingScheduler()
        return captured["s"]

    monkeypatch.setattr(main, "AsyncIOScheduler", _factory)
    monkeypatch.setattr(main, "init_engine", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    with pytest.raises(_StopBeforeStart):
        main.run_daemon()
    job = next(j for j in captured["s"].jobs if j.id == "club_data_refresh")
    return job.func


class _Proc:
    def __init__(self, rc, stderr=""):
        self.returncode, self.stderr, self.stdout = rc, stderr, ""


def _install_notify_recorder(monkeypatch):
    pages = []

    async def _fake_notify(settings, text, **kw):
        pages.append({"kind": kw.get("kind"), "text": text})
        return True

    monkeypatch.setattr(main, "notify_operator", _fake_notify)
    return pages


def _fake_run_writing(report_json, rp):
    """subprocess.run stand-in where the fetch step WRITES the report — as the
    real fetch script does. The tick unlinks any stale report before running,
    so a pre-seeded file would be gone; the fresh write is what it must read.
    """
    def _run(argv, **kw):
        if argv[-1].endswith("fetch_club_results.py"):
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(report_json)
        return _Proc(0)
    return _run


def test_fetch_failure_does_not_cancel_reseed_or_refit(monkeypatch, settings, tmp_path):
    # No report file at the patched repo root -> report{} path exercised too.
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    tick = _club_refresh_job(monkeypatch, settings)
    pages = _install_notify_recorder(monkeypatch)

    calls = []

    def _fake_run(argv, **kw):
        script = argv[-1]
        calls.append(script)
        # Only the FETCH step fails (503); the other two succeed.
        rc = 1 if script.endswith("fetch_club_results.py") else 0
        return _Proc(rc, stderr="HTTP Error 503" if rc else "")

    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)

    asyncio.run(tick())

    # The re-seed and the refit BOTH ran despite the fetch failing.
    assert any("seed_glicko_club.py" in c for c in calls), "re-seed was skipped"
    assert any("fit_dixon_coles_club.py" in c for c in calls), "DC refit was skipped"
    # The operator was paged about the failed step.
    assert any(p["kind"] == "club_refresh_step_failed" for p in pages), (
        "a failed refresh step did not page the operator"
    )


def test_all_steps_ok_does_not_page(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    tick = _club_refresh_job(monkeypatch, settings)
    pages = _install_notify_recorder(monkeypatch)

    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Proc(0))

    asyncio.run(tick())

    assert pages == [], f"a clean refresh should not page, got {pages}"


def test_fallback_report_unmapped_pages(monkeypatch, settings, tmp_path):
    # A coverage report with an unmapped club must reach the operator.
    rp = tmp_path / "data" / "club_fallback_report.json"
    content = ('{"couk_has_current": true, "fallback_used": true, '
               '"unmapped": ["Weird FC (PL)"], "coverage": 0.9, '
               '"fallback_rows": 5, "rejected_partitions": []}')
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    tick = _club_refresh_job(monkeypatch, settings)
    pages = _install_notify_recorder(monkeypatch)

    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run_writing(content, rp))

    asyncio.run(tick())

    assert any(p["kind"] == "club_refresh_unmapped" for p in pages), (
        "an unmapped fallback club did not page the operator"
    )


def test_stale_current_season_pages(monkeypatch, settings, tmp_path):
    # Total outage + a fallback that added nothing: current season is STALE.
    rp = tmp_path / "data" / "club_fallback_report.json"
    content = ('{"couk_has_current": false, "fallback_used": false, '
               '"fallback_rows": 0, "unmapped": [], "rejected_partitions": []}')
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    tick = _club_refresh_job(monkeypatch, settings)
    pages = _install_notify_recorder(monkeypatch)

    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run_writing(content, rp))

    asyncio.run(tick())

    assert any(p["kind"] == "club_refresh_stale" for p in pages), (
        "a silently-stale current season did not page the operator"
    )


def test_rejected_partition_pages(monkeypatch, settings, tmp_path):
    # A shield-200 that was refused must reach the operator.
    rp = tmp_path / "data" / "club_fallback_report.json"
    content = ('{"couk_has_current": true, "fallback_used": false, '
               '"unmapped": [], "rejected_partitions": [{"league": "PL", '
               '"season": "2526", "reason": "empty", "fresh": 0, '
               '"existing": 380}]}')
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    tick = _club_refresh_job(monkeypatch, settings)
    pages = _install_notify_recorder(monkeypatch)

    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run_writing(content, rp))

    asyncio.run(tick())

    assert any(p["kind"] == "club_refresh_rejected" for p in pages), (
        "a rejected .co.uk partition did not page the operator"
    )


def test_stale_report_from_prior_week_is_not_read(monkeypatch, settings, tmp_path):
    # A report left by a PREVIOUS run must be deleted before this run, so a
    # fetch that crashes before writing one cannot be read as this run.
    (tmp_path / "data").mkdir()
    rp = tmp_path / "data" / "club_fallback_report.json"
    rp.write_text('{"couk_has_current": false, "fallback_used": false, '
                  '"fallback_rows": 0, "unmapped": ["Ghost FC (PL)"]}')
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    tick = _club_refresh_job(monkeypatch, settings)
    pages = _install_notify_recorder(monkeypatch)

    # Fetch "runs" but does NOT rewrite the report (simulates a crash after the
    # unlink); the stale report must be gone, so no unmapped/stale page fires.
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Proc(0))

    asyncio.run(tick())

    assert not rp.exists(), "stale report was not cleared before the run"
    assert not any(p["kind"] in ("club_refresh_unmapped", "club_refresh_stale")
                   for p in pages), "a prior week's report leaked into this run"
