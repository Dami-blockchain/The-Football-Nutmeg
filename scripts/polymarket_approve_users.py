"""Polymarket (Polygon) approvals for EVERY registered user's wallet.

Multi-tenant companion to ``polymarket_approve.py``. Each user trades from their
OWN isolated wallet (``.secrets/users/<id>.key``); before that wallet can place
a Polymarket order it needs the same ERC-20 allowances + CTF operator approval
the agent wallet has. This iterates the active users and approves each one.

SAFETY: dry-run by default — prints what it WOULD do per wallet. Pass --confirm
to send real transactions (each wallet needs a little MATIC for gas). Idempotent:
already-approved wallets print ✓ and cost nothing.

Usage:
    python scripts/polymarket_approve_users.py            # dry-run, all users
    python scripts/polymarket_approve_users.py --confirm  # send txns
    python scripts/polymarket_approve_users.py --user 12345 --confirm  # one user
"""

from __future__ import annotations

import argparse
import os
import sys

from web3 import Web3

from betbot.config import get_settings
from betbot.storage.db import init_engine
from betbot.storage.repos import get_user, list_users
from betbot.wallet import get_private_key

# Import the sibling approval script regardless of launcher: ensure scripts/ is
# on sys.path so `python scripts/polymarket_approve_users.py` resolves it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polymarket_approve import approve_for_account  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="send real transactions")
    ap.add_argument("--user", type=int, default=None,
                    help="approve only this telegram_user_id (default: all active users)")
    args = ap.parse_args()

    s = get_settings()
    init_engine(s.db_path)

    if args.user is not None:
        u = get_user(args.user)
        users = [u] if u is not None else []
        if not users:
            raise SystemExit(f"no registered user with telegram_user_id={args.user}")
    else:
        users = list_users()
    if not users:
        raise SystemExit("no active users registered — nothing to approve.")

    w3 = Web3(Web3.HTTPProvider(s.polygon_rpc_url))
    print(f"approving {len(users)} user wallet(s) on Polygon ({w3.eth.chain_id}); "
          f"confirm={args.confirm}\n")

    failed = 0
    for u in users:
        print(f"=== user {u.telegram_user_id} ({u.name}) — {u.wallet_address} ===")
        key = get_private_key(u.wallet_keyfile)
        if not key:
            print(f"  ! keyfile missing ({u.wallet_keyfile}) — skipping\n")
            failed += 1
            continue
        acct = w3.eth.account.from_key(key)
        if acct.address.lower() != u.wallet_address.lower():
            print(f"  ! keyfile address {acct.address} != stored {u.wallet_address} — skipping\n")
            failed += 1
            continue
        try:
            approve_for_account(w3, acct, confirm=args.confirm)
        except Exception as e:  # noqa: BLE001 — one bad wallet shouldn't stop the rest
            print(f"  ! approval failed: {e}")
            failed += 1
        print()

    print(f"processed {len(users)} user(s); {failed} skipped/failed.")
    print("done." if args.confirm else "dry-run complete (pass --confirm to send).")


if __name__ == "__main__":
    main()
