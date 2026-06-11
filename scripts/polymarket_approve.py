"""Polymarket (Polygon) token approvals for live trading — Phase 5.

Run ONCE per funding wallet before live trading on Polymarket. Sets the ERC-20
allowances and the CTF (ERC-1155) operator approval the CTF Exchange V2 needs.

SAFETY: dry-run by default — prints what it WOULD do. Pass --confirm to send
real transactions (costs MATIC for gas).

⚠️ ADDRESS VERIFICATION: Polymarket migrated to CTF Exchange V2 (2026-04-28) and
the doc deliberately does NOT hardcode the V2 addresses. Confirm the pUSD token,
V2 Exchange, Neg-Risk Exchange, and Collateral Onramp addresses against
https://docs.polymarket.com/resources/contracts and pass them via flags/env
before sending anything. The defaults below are placeholders to be checked.

The USDC→pUSD wrap step is V2-specific; its exact flow must be confirmed against
the Polymarket docs/SDK — this script approves USDC to the onramp but leaves the
wrap call itself behind an explicit, documented TODO rather than guessing.

Usage:
    python scripts/polymarket_approve.py            # dry-run
    python scripts/polymarket_approve.py --confirm  # send txns
"""

from __future__ import annotations

import argparse
import os

from web3 import Web3

from betbot.config import get_settings

# --- addresses (verified against docs.polymarket.com/resources/contracts,
#     2026-06-08; all Polygon / chainId 137). Override via env if they change. ---
USDC = os.environ.get("POLYMARKET_USDC", "0x3c499c542cEF5E3811e1192ce70d8cc03d5c3359")
PUSD = os.environ.get("POLYMARKET_PUSD", "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
CTF = os.environ.get("POLYMARKET_CTF", "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
EXCHANGE_V2 = os.environ.get("POLYMARKET_EXCHANGE", "0xE111180000d2663C0091e4f400237545B87B996B")
NEG_RISK = os.environ.get("POLYMARKET_NEG_RISK", "0xe2222d279d744050d28e00520010520000310F59")
ONRAMP = os.environ.get("POLYMARKET_ONRAMP", "0x93070a847efEf7F70739046A929D47a521F5B8ee")

MAX_UINT = 2**256 - 1
ERC20_ABI = [
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
]
ERC1155_ABI = [
    {"name": "setApprovalForAll", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}],
     "outputs": []},
    {"name": "isApprovedForAll", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "operator", "type": "address"}],
     "outputs": [{"name": "", "type": "bool"}]},
]


def approve_for_account(w3, acct, *, confirm: bool) -> None:
    """Set every Polymarket allowance/operator approval for one account.

    Extracted so multi-tenant tooling can approve each user's own wallet by
    passing that user's ``acct`` (see scripts/polymarket_approve_users.py).
    Idempotent: already-approved spenders are skipped.
    """
    print(f"wallet: {acct.address}")
    print(f"chain:  Polygon ({w3.eth.chain_id})   confirm={confirm}")
    args = argparse.Namespace(confirm=confirm)

    spenders = [("CTF Exchange V2", EXCHANGE_V2)]
    if NEG_RISK:
        spenders.append(("Neg-Risk Exchange", NEG_RISK))

    def _erc20_approve(token, spender_name, spender):
        c = w3.eth.contract(address=Web3.to_checksum_address(token), abi=ERC20_ABI)
        cur = c.functions.allowance(acct.address, Web3.to_checksum_address(spender)).call()
        if cur >= MAX_UINT // 2:
            print(f"  ✓ {token[:8]}… already approved for {spender_name}")
            return
        print(f"  → approve {token[:8]}… for {spender_name} ({spender[:8]}…)")
        if not args.confirm:
            return
        tx = c.functions.approve(Web3.to_checksum_address(spender), MAX_UINT).build_transaction(
            {"from": acct.address, "nonce": w3.eth.get_transaction_count(acct.address)}
        )
        signed = acct.sign_transaction(tx)
        h = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"    tx {h.hex()}")
        w3.eth.wait_for_transaction_receipt(h)

    print("ERC-20 approvals (pUSD -> exchanges):")
    for name, spender in spenders:
        _erc20_approve(PUSD, name, spender)
    if ONRAMP:
        print("USDC -> Collateral Onramp (for the wrap step):")
        _erc20_approve(USDC, "Collateral Onramp", ONRAMP)
    else:
        print("  (POLYMARKET_ONRAMP unset — skipping USDC->pUSD wrap approval; "
              "confirm the onramp address + wrap flow on docs.polymarket.com)")

    print("CTF (ERC-1155) operator approvals:")
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=ERC1155_ABI)
    for name, spender in spenders:
        if ctf.functions.isApprovedForAll(acct.address, Web3.to_checksum_address(spender)).call():
            print(f"  ✓ CTF already approved for {name}")
            continue
        print(f"  → setApprovalForAll CTF for {name}")
        if not args.confirm:
            continue
        tx = ctf.functions.setApprovalForAll(
            Web3.to_checksum_address(spender), True
        ).build_transaction({"from": acct.address, "nonce": w3.eth.get_transaction_count(acct.address)})
        signed = acct.sign_transaction(tx)
        h = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"    tx {h.hex()}")
        w3.eth.wait_for_transaction_receipt(h)

    print("done." if args.confirm else "dry-run complete (pass --confirm to send).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="send real transactions")
    args = ap.parse_args()

    s = get_settings()
    key = s.polymarket_private_key
    if not key:
        raise SystemExit("POLYMARKET_PRIVATE_KEY not set")

    w3 = Web3(Web3.HTTPProvider(s.polygon_rpc_url))
    acct = w3.eth.account.from_key(key)
    approve_for_account(w3, acct, confirm=args.confirm)


if __name__ == "__main__":
    main()
