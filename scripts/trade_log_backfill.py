"""Read-only BACKFILL + report for the operator's Polymarket trading.

Writes NOWHERE except a scratch SQLite copy (never the live DB) and stdout.
Makes a consistent copy of the live DB (for the predictions needed to match),
runs the trade-log sync against the live wallet (public data-api, read-only),
and prints the backfill summary + the unmatched/other trade list.

Usage (from the worktree root, so ./config aliases resolve):
    PYTHONPATH=src .venv/bin/python scripts/trade_log_backfill.py \
        --wallet 0x52Ed... [--since 2026-08-01] [--live-db ./data/betbot.sqlite]

The wallet is supplied on the CLI / env — never hardcoded here.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _consistent_copy(live_db: Path, dest: Path) -> None:
    src = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)
    try:
        out = sqlite3.connect(str(dest))
        try:
            src.backup(out)  # online backup API: a consistent snapshot
        finally:
            out.close()
    finally:
        src.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", default=os.environ.get("BETBOT_OPERATOR_PM_WALLET", ""))
    ap.add_argument("--since", default="2026-08-01")
    ap.add_argument("--live-db", default="./data/betbot.sqlite")
    args = ap.parse_args()
    if not args.wallet:
        raise SystemExit("--wallet (or BETBOT_OPERATOR_PM_WALLET) is required")

    from betbot.config import Settings
    from betbot.storage.db import init_engine
    from betbot import trade_log as tl

    live = Path(args.live_db).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="tradelog_backfill_"))
    scratch = tmp / "scratch.sqlite"
    _consistent_copy(live, scratch)
    init_engine(scratch)  # all writes land in the scratch copy only

    settings = Settings(_env_file=None, BETBOT_OPERATOR_PM_WALLET=args.wallet)
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    stats = tl.sync(settings, start=since)
    trades = tl.repos.list_operator_trades(since=tl.DEFAULT_BACKFILL_START)

    # live /positions for open marks
    positions_feed = []
    try:
        with tl.PmDataClient() as c:
            positions_feed = c.fetch_positions(args.wallet)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] positions fetch failed: {type(e).__name__}")
    positions = tl.build_positions(trades, positions_feed)

    print("=" * 72)
    print(f"BACKFILL: pulled {stats['pulled']} activity rows, "
          f"stored {stats['inserted']} fills (since {args.since})")
    print(f"Positions aggregated: {len(positions)} "
          f"(from {len(trades)} fills; scratch db: {scratch})")
    print("=" * 72)
    print(tl.render_report(positions).replace("```", "").strip())
    print("=" * 72)

    # Per-position detail
    print("\nPER-POSITION DETAIL (oldest first):")
    hdr = f"{'date':10} {'comp':4} {'class':11} {'conf':4} {'cost':>9} {'pnl':>9} {'roi':>7} {'entry':>5} {'hit':4} {'title'}"
    print(hdr)
    for p in sorted(positions, key=lambda x: x.first_ts or datetime.min.replace(tzinfo=timezone.utc)):
        won = p.won
        hit = "-" if won is None else ("Y" if won else "N")
        conf = "-" if p.high_conf is None else ("hi" if p.high_conf else "lo")
        d = p.first_ts.date().isoformat() if p.first_ts else "?"
        openflag = " [OPEN]" if p.is_open else ""
        print(f"{d:10} {str(p.competition_code or '-'):4} {p.classification:11} "
              f"{conf:4} {p.bought_usdc:9.2f} {p.realized_pnl:9.2f} "
              f"{p.roi*100:6.1f}% {p.avg_entry_price:5.2f} {hit:4} {p.title}{openflag}")

    # Unmatched / contrarian / draw
    other = [p for p in positions if p.classification in (tl.UNMATCHED, tl.CONTRARIAN, tl.DRAW_BET)]
    print(f"\nUNMATCHED / CONTRARIAN / DRAW positions ({len(other)}):")
    for p in other:
        print(f"  [{p.classification}] {p.title}  cost ${p.bought_usdc:.2f} "
              f"pnl {p.realized_pnl:+.2f}")


if __name__ == "__main__":
    main()
