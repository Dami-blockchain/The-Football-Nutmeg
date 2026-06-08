"""Agent funding wallet — deposit address + USDC balance reads.

The bot has ONE EVM wallet. Because the address is identical on every EVM
chain, the same address receives USDC on both **Polygon** (Polymarket) and
**Base** (Limitless). The private key lives in a 0600 keyfile outside git; the
operator must back it up — it controls real funds.

This module is read-only on-chain (balance reads). It never moves funds; the
operator deposits by sending USDC to the address from their own wallet, and the
bot just reports the balance. Spending happens only through the gated,
double-checked live-order path (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from web3 import Web3

from betbot.logging import get_logger

log = get_logger(__name__)

# Canonical USDC contracts. (Confirm against issuer docs before trusting for
# anything beyond balance display.)
CHAINS: dict[str, dict] = {
    "polygon": {
        "label": "Polygon",
        "chain_id": 137,
        "usdc": "0x3c499c542cEF5E3811e1192ce70d8cc03d5c3359",
        "default_rpc": "https://polygon-rpc.com",
    },
    "base": {
        "label": "Base",
        "chain_id": 8453,
        "usdc": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "default_rpc": "https://mainnet.base.org",
    },
}

_USDC_DECIMALS = 6
_ERC20_BALANCE_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    }
]


@dataclass(frozen=True)
class ChainBalance:
    chain: str
    label: str
    usdc: float
    ok: bool
    error: str | None = None


def get_or_create_address(keyfile: str | Path) -> str:
    """Return the wallet address, generating + persisting a key on first use.

    The key is written 0600. If it already exists it's loaded, never
    overwritten.
    """
    from eth_account import Account

    p = Path(keyfile)
    if p.exists():
        acct = Account.from_key(p.read_text().strip())
        return acct.address

    acct = Account.create()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(acct.key.hex())
    p.chmod(0o600)
    log.warning(
        "wallet_created",
        address=acct.address,
        keyfile=str(p),
        note="BACK UP THIS KEYFILE — it controls real funds",
    )
    return acct.address


def _rpc_for(chain: str, settings) -> str:
    if chain == "polygon":
        return settings.polygon_rpc_url
    if chain == "base":
        return settings.base_rpc_url
    return CHAINS[chain]["default_rpc"]


def usdc_balance(address: str, chain: str, rpc_url: str) -> ChainBalance:
    cfg = CHAINS[chain]
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 15}))
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(cfg["usdc"]),
            abi=_ERC20_BALANCE_ABI,
        )
        raw = contract.functions.balanceOf(
            Web3.to_checksum_address(address)
        ).call()
        return ChainBalance(chain, cfg["label"], raw / 10**_USDC_DECIMALS, True)
    except Exception as e:  # noqa: BLE001 — RPC flakiness shouldn't crash callers
        log.warning("usdc_balance_failed", chain=chain, error=str(e))
        return ChainBalance(chain, cfg["label"], 0.0, False, str(e))


def all_balances(address: str, settings) -> list[ChainBalance]:
    return [usdc_balance(address, c, _rpc_for(c, settings)) for c in CHAINS]


def wallet_summary(settings) -> dict:
    """Address + per-chain USDC balances, for the API / TG bot / frontend."""
    address = get_or_create_address(settings.wallet_keyfile)
    balances = all_balances(address, settings)
    return {
        "address": address,
        "balances": [
            {"chain": b.chain, "label": b.label, "usdc": round(b.usdc, 2), "ok": b.ok}
            for b in balances
        ],
        "total_usdc": round(sum(b.usdc for b in balances if b.ok), 2),
    }
