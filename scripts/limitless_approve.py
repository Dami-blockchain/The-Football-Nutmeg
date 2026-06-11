"""Limitless (Base) token approvals for live trading — Phase 5.

Run ONCE per funding wallet before live trading on Limitless. Approves USDC for
the Limitless CTF Exchange and sets the CTF (ERC-1155) operator approval.

SAFETY: dry-run by default; pass --confirm to send real transactions (costs ETH
on Base for gas).

The Limitless exchange (verifyingContract + approval spender) is PER-MARKET
(``market.venue.exchange``). Pass the exchange address you'll trade against via
--exchange / LIMITLESS_EXCHANGE, or pass --market <slug> and the script reads
``venue.exchange`` from the live market payload (the same address order signing
uses). Confirm the CTF address against the Limitless docs.

Usage:
    python scripts/limitless_approve.py --exchange 0x...            # dry-run
    python scripts/limitless_approve.py --market <slug>             # dry-run
    python scripts/limitless_approve.py --exchange 0x... --confirm  # send
"""

from __future__ import annotations

import argparse
import os

from web3 import Web3

from betbot.config import get_settings
from betbot.wallet import CHAINS

USDC = CHAINS["base"]["usdc"]  # 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913
# Base CTF (conditional tokens) — NO safe default: this differs from Polymarket's
# Polygon CTF. Confirm the Base address against Limitless docs and pass via
# LIMITLESS_CTF. (Left empty on purpose so a wrong address can't be used silently.)
CTF = os.environ.get("LIMITLESS_CTF", "")

MAX_UINT = 2**256 - 1
ERC20_ABI = [
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]
ERC1155_ABI = [
    {"name": "setApprovalForAll", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}],
     "outputs": []},
    {"name": "isApprovedForAll", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "operator", "type": "address"}],
     "outputs": [{"name": "", "type": "bool"}]},
]


def _exchange_for_market(slug: str) -> str:
    """Read the PER-MARKET venue.exchange address from the live API."""
    import httpx

    resp = httpx.get(f"https://api.limitless.exchange/markets/{slug}", timeout=20.0)
    resp.raise_for_status()
    exchange = ((resp.json() or {}).get("venue") or {}).get("exchange") or ""
    if not exchange:
        raise SystemExit(f"market {slug!r} has no venue.exchange in its payload")
    print(f"market {slug}: venue.exchange = {exchange}")
    return exchange


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default=os.environ.get("LIMITLESS_EXCHANGE", ""),
                    help="Limitless CTF Exchange address (market.venue.exchange)")
    ap.add_argument("--market", default="",
                    help="Market slug — resolves venue.exchange from the live API")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    if not args.exchange and args.market:
        args.exchange = _exchange_for_market(args.market)
    if not args.exchange:
        raise SystemExit("--exchange (or LIMITLESS_EXCHANGE, or --market <slug>) is required")

    s = get_settings()
    key = s.limitless_private_key
    if not key:
        raise SystemExit("LIMITLESS_PRIVATE_KEY not set")

    w3 = Web3(Web3.HTTPProvider(s.base_rpc_url))
    acct = w3.eth.account.from_key(key)
    exch = Web3.to_checksum_address(args.exchange)
    print(f"wallet: {acct.address}")
    print(f"chain:  Base ({w3.eth.chain_id})   exchange={exch}   confirm={args.confirm}")

    def _send(fn):
        tx = fn.build_transaction(
            {"from": acct.address, "nonce": w3.eth.get_transaction_count(acct.address)}
        )
        signed = acct.sign_transaction(tx)
        h = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"    tx {h.hex()}")
        w3.eth.wait_for_transaction_receipt(h)

    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC), abi=ERC20_ABI)
    if usdc.functions.allowance(acct.address, exch).call() >= MAX_UINT // 2:
        print("  ✓ USDC already approved for exchange")
    else:
        print("  → approve USDC for exchange")
        if args.confirm:
            _send(usdc.functions.approve(exch, MAX_UINT))

    if not CTF:
        print("  ! LIMITLESS_CTF not set — skipping CTF setApprovalForAll. "
              "Confirm the Base CTF address with Limitless docs and re-run.")
        print("done." if args.confirm else "dry-run complete (pass --confirm to send).")
        return

    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=ERC1155_ABI)
    if ctf.functions.isApprovedForAll(acct.address, exch).call():
        print("  ✓ CTF already approved for exchange")
    else:
        print("  → setApprovalForAll CTF for exchange")
        if args.confirm:
            _send(ctf.functions.setApprovalForAll(exch, True))

    print("done." if args.confirm else "dry-run complete (pass --confirm to send).")


if __name__ == "__main__":
    main()
