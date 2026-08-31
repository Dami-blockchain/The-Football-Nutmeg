"""Fit the club ensemble's isotonic calibration artifact from RAW dual-log triples.

Production runs IDENTITY calibration today: ``IsotonicCalibrator.fit`` had no
caller and ``data/ensemble_calibration_club.json`` does not exist, so the club
engine's ``calibrate()`` is a no-op. This CLI is the missing fit path.

Train/serve-skew trap (the reason this must read the dual-log, not the served
predictions): the club engine applies calibration to the RAW log-pool triple at
serve time. A calibrator refit on ALREADY-calibrated serving probabilities would
learn a second map on top of the first, and each refit would compound the skew.
So we train ONLY on the RAW pre-calibration ensemble triples that
``model_predictions`` stores at prediction time (``e_home/e_draw/e_away``, the
log-pool BEFORE calibrate — see ``repos.upsert_model_prediction``), scored
against the realised outcome. Those raw triples are captured continuously and
independently of whether any artifact exists, so the first fit never trains on
its own output.

Per-outcome isotonic (PAV) map from predicted probability -> observed frequency,
then the club engine renormalises the three transformed marginals.

Minimum-sample guard: refuses to fit below ``--min-n`` (default 500). The clean
settled club sample is ~45 today — FAR too small; a calibrator fit on dozens of
matches would overfit noise and could easily hurt live RPS. Do NOT lower the
guard to force an early artifact.

Run (repo root, venv active) once the dual-log has >= 500 settled club rows:
    python scripts/fit_ensemble_calibration_club.py
    python scripts/fit_ensemble_calibration_club.py --out data/ensemble_calibration_club.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from betbot.config import get_settings
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import ModelPrediction
from betbot.strategy.ensemble import IsotonicCalibrator

log = get_logger("fit_ensemble_calibration_club")

MIN_FIT_N = 500  # standing minimum; see module docstring.
_IDX = {"HOME": 0, "DRAW": 1, "AWAY": 2}


def _load_raw_triples() -> list[tuple[tuple[float, float, float], str]]:
    """Every SETTLED dual-log row as (raw_ensemble_triple, outcome)."""
    out: list[tuple[tuple[float, float, float], str]] = []
    with session_scope() as s:
        rows = s.execute(
            select(ModelPrediction).where(ModelPrediction.outcome.is_not(None))
        ).scalars()
        for r in rows:
            if r.outcome not in _IDX:
                continue
            out.append(((r.e_home, r.e_draw, r.e_away), r.outcome))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="artifact path (default: settings.ensemble_calibration_club_path)")
    ap.add_argument("--min-n", type=int, default=MIN_FIT_N,
                    help=f"refuse to fit below this many settled rows (default {MIN_FIT_N})")
    args = ap.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)

    out_path = args.out or Path(settings.ensemble_calibration_club_path)
    samples = _load_raw_triples()
    n = len(samples)

    if n < args.min_n:
        raise SystemExit(
            f"REFUSING to fit: {n} settled club dual-log rows < min-n={args.min_n}.\n"
            f"A calibrator fit on so few matches overfits noise and can hurt live\n"
            f"RPS. Let model_predictions accumulate to >= {args.min_n} settled rows,\n"
            f"then re-run. No artifact written; production keeps identity calibration."
        )

    # Per-outcome: predicted marginal vs 1/0 outcome indicator.
    result: dict[str, dict[str, list[float]]] = {}
    for key, idx in (("home", 0), ("draw", 1), ("away", 2)):
        predicted = [triple[idx] for triple, _ in samples]
        observed = [1.0 if _IDX[o] == idx else 0.0 for _, o in samples]
        cal = IsotonicCalibrator().fit(predicted, observed)
        result[key] = {"xs": cal._xs, "ys": cal._ys}  # noqa: SLF001 — serialising own state

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path} from {n} settled club dual-log rows "
          f"(RAW pre-calibration triples).")


if __name__ == "__main__":
    main()
