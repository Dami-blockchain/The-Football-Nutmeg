"""Read-only rescore-drift report for high-conf pre-match alerts.

MEASUREMENT ONLY. Reads :class:`betbot.storage.models.RescoreDriftLog` — the
per-stage snapshots ``send_prediction_alert`` / ``run_result_alerts`` write —
and prints how far the ALERT-TIME call drifts by each later stage
(``kickoff_60``, ``result``). For each fixture the earliest observation is the
baseline; a later stage's delta is that baseline pick's probability at the later
stage minus at baseline (positive = firmed up, negative = eased).

Nothing here changes alerting, gating, or thresholds. The context for the
request was "4 of 6 rescored alerts drifted downward; n=6 is noise" — so the
report STATES n EXPLICITLY and is meant to be read once n>=30.

Usage (on the droplet, against the live DB — read-only):
    python scripts/rescore_drift_report.py
"""
from __future__ import annotations

from betbot.config import get_settings
from betbot.storage.db import init_engine
from betbot.storage.repos import rescore_drift_stats


def main() -> None:
    settings = get_settings()
    init_engine(settings.db_path)
    stats = rescore_drift_stats()

    n_fix = stats["n_fixtures"]
    print("Rescore-drift report (MEASUREMENT ONLY)")
    print(f"Fixtures with an alert-time baseline: n = {n_fix}")
    if n_fix < 30:
        print("  (n < 30 — too few to mean anything yet; noise, not a trend.)")
    stages = stats["stages"]
    if not stages:
        print("No later-stage observations recorded yet.")
        return
    print()
    print(f"{'stage':<12} {'n':>4} {'up':>4} {'down':>5} {'flat':>5} {'mean_delta':>11}")
    print("-" * 44)
    # Show known stages first in lifecycle order, then any extras.
    order = ["kickoff_60", "result"]
    ordered = [s for s in order if s in stages] + [
        s for s in stages if s not in order
    ]
    for stage in ordered:
        st = stages[stage]
        print(
            f"{stage:<12} {st['n']:>4} {st['up']:>4} {st['down']:>5} "
            f"{st['flat']:>5} {st['mean_delta']:>+11.4f}"
        )


if __name__ == "__main__":
    main()
