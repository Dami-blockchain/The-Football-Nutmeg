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
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "club_fallback_report.json").write_text(
        '{"fallback_used": true, "unmapped": ["Weird FC (PL)"], '
        '"coverage": 0.9, "fallback_rows": 5}'
    )
    monkeypatch.setattr(main, "_REPO_ROOT", tmp_path)
    tick = _club_refresh_job(monkeypatch, settings)
    pages = _install_notify_recorder(monkeypatch)

    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Proc(0))

    asyncio.run(tick())

    assert any(p["kind"] == "club_refresh_unmapped" for p in pages), (
        "an unmapped fallback club did not page the operator"
    )
