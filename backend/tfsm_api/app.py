"""FastAPI backend for The Football Nutmeg Agent.

Async endpoints that wrap the bot's own functions (never shelling out) plus a
wallet/deposit layer for the Telegram bot and frontend. Auth: if
``TFSM_API_TOKEN`` is set, every ``/api`` route requires
``Authorization: Bearer <token>``; if unset, the app assumes a localhost-only
bind and allows requests (with a logged warning). The built React app in
``frontend/dist`` is served at ``/`` when present.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text as sa_text

from betbot.backtest import backtest_mock, backtest_stored
from betbot.config import get_settings
from betbot.gate import evaluate_gate
from betbot.logging import configure_logging, get_logger
from betbot.storage.db import init_engine
from betbot.storage.repos import (
    daily_paper_exposure_usd,
    get_kill_switch,
    is_kill_switch_tripped,
    list_recent_paper_bets,
    list_recent_predictions,
    reset_kill_switch,
    settled_pnl_window,
)
from betbot.wallet import wallet_summary

log = get_logger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"

#: Wall-clock time this process started, used for the health uptime figure.
_BOOTED_AT = time.time()


def _booted_commit() -> str:
    """Short git SHA of the working tree AS THIS PROCESS IMPORTED IT.

    Resolved once, at import, and never refreshed. That is the point: the
    running daemon/API keep executing the code they loaded at boot, so after a
    `git merge` the checkout moves on and the processes do not. On 2026-08-22
    all three services were still running ``0e7b8db`` while ``main`` had been
    at ``fafc83b`` for ~3.5h, and nothing exposed the gap. A monitored probe
    that reports the BOOTED commit turns "is the deploy actually live?" into a
    one-line diff against ``git rev-parse --short HEAD``.

    Never raises: a missing git binary or a non-repo deploy yields "unknown".
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 — health must never fail on its own stamp
        return "unknown"
    return out.stdout.strip() or "unknown" if out.returncode == 0 else "unknown"


#: Resolved at import so it reflects the deployed commit, not the current one.
_BOOTED_COMMIT = _booted_commit()


def _db_ok() -> tuple[bool, str | None]:
    """Round-trip a trivial query so the probe proves the DB is REACHABLE.

    ``init_engine`` succeeding at startup says nothing about the file still
    being readable now (a deleted/locked/permission-changed sqlite file fails
    only at query time), so the check issues a real ``SELECT 1``.
    """
    try:
        from betbot.storage.db import session_scope

        with session_scope() as session:
            session.execute(sa_text("SELECT 1"))
        return True, None
    except Exception as e:  # noqa: BLE001 — report the fault, never raise
        return False, type(e).__name__

# NOTE: BETBOT_MODE was deliberately REMOVED from this set. The predictions-only
# pivot deleted the Settings field, and Settings uses ``extra="ignore"`` — so a
# POST /api/settings writing BETBOT_MODE=live edited .env, returned 200, and
# changed absolutely nothing. With real money in scope, a control that reports
# success while doing nothing is worse than no control: it reads as "live
# trading is on" (or "off") when neither is true. Trading mode is now a build
# fact (config.LIVE_ORDER_PATH_AVAILABLE), so this endpoint correctly 400s it.
EDITABLE_SETTINGS = {
    "BETBOT_EDGE_THRESHOLD", "BETBOT_FIXED_STAKE_USD",
    "BETBOT_MAX_BET_USD", "BETBOT_DAILY_EXPOSURE_CAP_USD",
    "BETBOT_DRAWDOWN_KILL_PCT", "BETBOT_DRAWDOWN_WINDOW_DAYS",
    "BETBOT_DRAWDOWN_MIN_STAKED_USD", "BETBOT_GATE_MIN_BETS",
    "BETBOT_GATE_MIN_WINDOW_DAYS", "BETBOT_GATE_MIN_HIT_RATE", "BETBOT_GATE_MIN_ROI",
}
# Knobs the running daemon only re-reads on restart — which is ALL of them.
#
# This used to be {"BETBOT_MODE"}, implying the other eleven took effect live.
# They do not. ``get_settings`` is an lru_cache singleton and, as config.py puts
# it, "the production daemon only reads it once at startup". This endpoint
# clears the cache in the API PROCESS, so the API immediately reports the new
# value — while the daemon that actually sizes and places the bets keeps using
# the old one until it is restarted. An operator who lowered
# BETBOT_MAX_BET_USD, saw the dashboard agree, and believed the daemon had
# obeyed would be wrong about a risk control, with real money live.
RESTART_REQUIRED = set(EDITABLE_SETTINGS)


# ---- auth ------------------------------------------------------------
async def require_auth(authorization: str | None = Header(default=None)) -> None:
    token = get_settings().api_token
    if not token:
        return  # unset -> localhost-only deployment, no auth
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


# ---- models ----------------------------------------------------------
class SettingsUpdate(BaseModel):
    key: str
    value: str


def _bet_dict(b) -> dict:
    return {
        "fixture_id": b.fixture_id,
        "outcome": b.outcome,
        "our_probability": round(b.our_probability, 3),
        "market_price": b.market_price,
        "edge": b.edge,
        "stake_usd": b.stake_usd,
        "created_at": b.created_at.isoformat() if b.created_at else None,
        "settled_outcome": b.settled_outcome,
        "pnl_usd": b.pnl_usd,
        "rationale": b.rationale,
    }


def _prediction_dict(p) -> dict:
    return {
        "fixture_id": p.fixture_id,
        "competition_code": p.competition_code,
        "home_team": p.home_team,
        "away_team": p.away_team,
        "kickoff": p.kickoff.isoformat() if p.kickoff else None,
        "p_home": round(p.p_home, 3),
        "p_draw": round(p.p_draw, 3),
        "p_away": round(p.p_away, 3),
    }


def _backtest_dict(r) -> dict:
    return {
        "n": r.n, "wins": r.wins, "hit_rate": round(r.hit_rate, 4),
        "roi": round(r.roi, 4), "brier": round(r.brier, 4),
        "pnl_usd": round(r.pnl_usd, 2), "staked_usd": round(r.staked_usd, 2),
        "per_outcome": {
            o: {"n": s.n, "wins": s.wins, "hit_rate": round(s.hit_rate, 4),
                "roi": round(s.roi, 4), "pnl_usd": round(s.pnl_usd, 2)}
            for o, s in r.per_outcome.items()
        },
    }


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings.db_path)
    if not settings.api_token:
        log.warning("api_no_token", note="TFSM_API_TOKEN unset — bind to localhost only")

    app = FastAPI(title="The Football Nutmeg Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_origin_regex=r"https?://.*",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- public ----
    @app.get("/api/health")
    async def health():
        """Liveness + the state a monitor actually needs to page on.

        This is a MONITORED PROBE, so it reports facts it verifies rather than
        a bare literal: the DB is queried, not assumed, and the trading mode is
        the derived :attr:`Settings.mode` (a build fact — see
        ``config.LIVE_ORDER_PATH_AVAILABLE``), not a config string that could
        disagree with what the code can actually do.

        Status codes: 200 while the process can serve, 503 once a dependency
        it cannot work without is down. A TRIPPED KILL SWITCH IS NOT 503 — the
        kill switch firing is the system working as designed, and paging on it
        as an outage would train the operator to ignore the probe.
        """
        db_ok, db_error = _db_ok()
        payload = {
            "status": "ok" if db_ok else "degraded",
            "mode": get_settings().mode,
            "checks": {"db": {"ok": db_ok, "error": db_error}},
            "kill_switch": {"tripped": is_kill_switch_tripped()},
            "build": {"commit": _BOOTED_COMMIT},
            "uptime_seconds": round(time.time() - _BOOTED_AT, 1),
        }
        if not db_ok:
            log.error("health_degraded", db_error=db_error)
            return JSONResponse(status_code=503, content=payload)
        return payload

    # ---- protected ----
    @app.get("/api/status", dependencies=[Depends(require_auth)])
    async def status() -> dict:
        s = get_settings()
        ks = get_kill_switch()
        g = evaluate_gate(s)
        pnl7, staked7 = settled_pnl_window(s.drawdown_window_days)
        return {
            "mode": s.mode,
            "kill_switch": {
                "tripped": ks.tripped_at is not None,
                "reason": ks.reason,
                "tripped_at": ks.tripped_at.isoformat() if ks.tripped_at else None,
            },
            "gate": {"passed": g.passed, "reasons": g.reasons,
                     "window_days": round(g.window_days_observed, 1)},
            "performance": _backtest_dict(g.result),
            "trailing_window": {"days": s.drawdown_window_days,
                                "pnl_usd": round(pnl7, 2), "staked_usd": round(staked7, 2)},
            "daily_exposure_usd": round(daily_paper_exposure_usd(), 2),
            "wallet": wallet_summary(s),
        }

    @app.get("/api/bets", dependencies=[Depends(require_auth)])
    async def bets(days: int = 14) -> dict:
        return {"bets": [_bet_dict(b) for b in list_recent_paper_bets(days=days)]}

    @app.get("/api/predictions", dependencies=[Depends(require_auth)])
    async def predictions(days: int = 7) -> dict:
        return {"predictions": [_prediction_dict(p) for p in list_recent_predictions(days=days)]}

    @app.get("/api/backtest", dependencies=[Depends(require_auth)])
    async def backtest(mode: str = "stored") -> dict:
        if mode == "mock":
            return _backtest_dict(backtest_mock(edge_threshold=get_settings().edge_threshold))
        return _backtest_dict(backtest_stored())

    @app.get("/api/gate", dependencies=[Depends(require_auth)])
    async def gate() -> dict:
        g = evaluate_gate(get_settings())
        return {"passed": g.passed, "reasons": g.reasons,
                "window_days": round(g.window_days_observed, 1),
                "kill_switch_tripped": g.kill_switch_tripped,
                "performance": _backtest_dict(g.result)}

    @app.get("/api/wallet", dependencies=[Depends(require_auth)])
    async def wallet() -> dict:
        return wallet_summary(get_settings())

    @app.get("/api/kill-switch", dependencies=[Depends(require_auth)])
    async def kill_switch() -> dict:
        ks = get_kill_switch()
        return {"tripped": ks.tripped_at is not None, "reason": ks.reason,
                "realized_pnl_usd": ks.realized_pnl_usd, "staked_usd": ks.staked_usd}

    @app.post("/api/kill-switch/reset", dependencies=[Depends(require_auth)])
    async def kill_switch_reset() -> dict:
        reset_kill_switch()
        return {"tripped": is_kill_switch_tripped()}

    @app.post("/api/score", dependencies=[Depends(require_auth)])
    async def score() -> dict:
        from betbot.main import _score_once
        n = await _score_once()
        return {"paper_bets_logged": n}

    @app.post("/api/settle", dependencies=[Depends(require_auth)])
    async def settle() -> dict:
        from betbot.main import _settle_once
        summary = await _settle_once()
        return {"settled": summary.settled, "kill_switch_tripped": summary.kill_switch_tripped,
                "window_pnl_usd": round(summary.window_pnl_usd, 2)}

    @app.get("/api/settings", dependencies=[Depends(require_auth)])
    async def read_settings() -> dict:
        s = get_settings()
        return {
            "mode": s.mode, "edge_threshold": s.edge_threshold,
            "fixed_stake_usd": s.fixed_stake_usd, "max_bet_usd": s.max_bet_usd,
            "daily_exposure_cap_usd": s.daily_exposure_cap_usd,
            "drawdown_kill_pct": s.drawdown_kill_pct,
            "drawdown_window_days": s.drawdown_window_days,
            "drawdown_min_staked_usd": s.drawdown_min_staked_usd,
            "gate_min_bets": s.gate_min_bets, "gate_min_window_days": s.gate_min_window_days,
            "gate_min_hit_rate": s.gate_min_hit_rate, "gate_min_roi": s.gate_min_roi,
        }

    @app.post("/api/settings", dependencies=[Depends(require_auth)])
    async def write_settings(update: SettingsUpdate) -> dict:
        if update.key not in EDITABLE_SETTINGS:
            raise HTTPException(status_code=400, detail=f"{update.key} is not editable")
        _update_env_file(_REPO_ROOT / ".env", update.key, update.value)
        get_settings.cache_clear()
        init_engine(get_settings().db_path)
        return {"key": update.key, "value": update.value,
                "restart_required": update.key in RESTART_REQUIRED}

    # ---- serve frontend ----
    if _FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

    return app


def _update_env_file(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    out, found = [], False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n")


app = create_app()
