#!/usr/bin/env python3
"""Manual override: position the agent treasury's trading funds NOW.

The daemon's deposit-scan tick already runs the agent-treasury rebalancer in
the background (see ``betbot.bridge.TreasuryRebalancer`` /
``BETBOT_TREASURY_AUTO_REBALANCE``), so under normal operation no manual bridge
is ever needed. This script stays as an operator override — run it to drive a
single rebalance pass on demand (e.g. to position funds before the next tick,
or after topping up the agent wallet on one chain).

It is the SAME code path as the daemon: it reads the agent USDC on both trading
chains, and if one is below its target by more than the threshold while the
other holds a surplus, it opens/advances ONE CCTP bridge leg. Crash-safety and
the single-in-flight invariant are identical — re-running it is safe and simply
resumes whatever leg is already active.

Usage:
    PYTHONPATH=src python scripts/bridge_agent_funds.py

Respects the same gates as the daemon (``BETBOT_AUTO_BRIDGE`` and
``BETBOT_TREASURY_AUTO_REBALANCE``); set them in ``.env``.
"""

from __future__ import annotations

from betbot.bridge import TreasuryRebalancer
from betbot.config import get_settings
from betbot.logging import get_logger
from betbot.storage.db import init_engine

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    init_engine(settings.db_path)
    log.info("bridge_agent_funds_manual_run")
    TreasuryRebalancer(settings).rebalance_agent_treasury()


if __name__ == "__main__":
    main()
