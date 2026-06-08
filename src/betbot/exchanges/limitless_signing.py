"""EIP-712 order signing for Limitless (Phase 5).

**Limitless / V1-fork ONLY.** Limitless is a fork of Polymarket's *pre-V2*
ctf-exchange: domain name ``"Limitless CTF Exchange"``, version ``"1"``,
chainId 8453 (Base), ``verifyingContract`` is per-market (``market.venue.exchange``).
The order struct is the V1 layout:

    salt, maker, signer, taker, tokenId, makerAmount, takerAmount,
    expiration, nonce, feeRateBps, side, signatureType

Do **NOT** reuse this for Polymarket — Polymarket V2 dropped
taker/expiration/nonce/feeRateBps and added timestamp/metadata/builder, and
its signing is handled entirely by ``py-clob-client-v2``.

The arithmetic helpers (``to_usdc_units``, ``shares_for``, ``random_salt``) are
pure and import-light — ``eth_account`` is imported lazily inside the signing
functions so the helpers and their tests don't require it.
"""

from __future__ import annotations

import secrets
from typing import Any

USDC_DECIMALS = 6
_USDC_SCALE = 10**USDC_DECIMALS

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
LIMITLESS_DOMAIN_NAME = "Limitless CTF Exchange"
LIMITLESS_DOMAIN_VERSION = "1"
BASE_CHAIN_ID = 8453

# CTF Exchange side enum.
BUY = 0
SELL = 1
# signatureType: 0 = EOA (eth_sign / EIP-712), per ctf-exchange.
SIG_TYPE_EOA = 0

_ORDER_FIELDS = [
    {"name": "salt", "type": "uint256"},
    {"name": "maker", "type": "address"},
    {"name": "signer", "type": "address"},
    {"name": "taker", "type": "address"},
    {"name": "tokenId", "type": "uint256"},
    {"name": "makerAmount", "type": "uint256"},
    {"name": "takerAmount", "type": "uint256"},
    {"name": "expiration", "type": "uint256"},
    {"name": "nonce", "type": "uint256"},
    {"name": "feeRateBps", "type": "uint256"},
    {"name": "side", "type": "uint8"},
    {"name": "signatureType", "type": "uint8"},
]


# ---------------------------------------------------------------------------
# Pure arithmetic (no eth_account)
# ---------------------------------------------------------------------------
def to_usdc_units(amount_usd: float) -> int:
    """Dollars -> 6-decimal integer collateral units."""
    if amount_usd < 0:
        raise ValueError("amount must be non-negative")
    return int(round(amount_usd * _USDC_SCALE))


def shares_for(maker_amount_units: int, price: float) -> int:
    """Outcome shares received for spending ``maker_amount_units`` at ``price``.

    takerAmount = makerAmount / price (both 6-decimal). ``price`` is the YES
    price in (0, 1).
    """
    if not (0.0 < price < 1.0):
        raise ValueError("price must be in (0, 1)")
    return int(round(maker_amount_units / price))


def random_salt() -> int:
    """A 256-bit random salt for order uniqueness."""
    return secrets.randbits(256)


def build_order(
    *,
    token_id: int | str,
    maker: str,
    maker_amount_units: int,
    taker_amount_units: int,
    side: int = BUY,
    signer: str | None = None,
    taker: str = ZERO_ADDRESS,
    expiration: int = 0,
    nonce: int = 0,
    fee_rate_bps: int = 0,
    signature_type: int = SIG_TYPE_EOA,
    salt: int | None = None,
) -> dict[str, Any]:
    """Assemble the V1 order struct (field order matters for the hash)."""
    return {
        "salt": salt if salt is not None else random_salt(),
        "maker": maker,
        "signer": signer or maker,
        "taker": taker,
        "tokenId": int(token_id),
        "makerAmount": int(maker_amount_units),
        "takerAmount": int(taker_amount_units),
        "expiration": int(expiration),
        "nonce": int(nonce),
        "feeRateBps": int(fee_rate_bps),
        "side": int(side),
        "signatureType": int(signature_type),
    }


def order_typed_data(
    order: dict[str, Any],
    *,
    verifying_contract: str,
    chain_id: int = BASE_CHAIN_ID,
) -> dict[str, Any]:
    """Full EIP-712 typed-data message for an order."""
    return {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "Order": _ORDER_FIELDS,
        },
        "domain": {
            "name": LIMITLESS_DOMAIN_NAME,
            "version": LIMITLESS_DOMAIN_VERSION,
            "chainId": chain_id,
            "verifyingContract": verifying_contract,
        },
        "primaryType": "Order",
        "message": order,
    }


# ---------------------------------------------------------------------------
# Signing (eth_account imported lazily)
# ---------------------------------------------------------------------------
def sign_order(
    order: dict[str, Any],
    private_key: str,
    *,
    verifying_contract: str,
    chain_id: int = BASE_CHAIN_ID,
) -> dict[str, Any]:
    """Sign an order; return ``{order, signature, hash}``.

    The hash is ``signed.message_hash`` (NOT ``signable.hash``).
    """
    from eth_account import Account
    from eth_account.messages import encode_typed_data

    td = order_typed_data(order, verifying_contract=verifying_contract, chain_id=chain_id)
    signable = encode_typed_data(full_message=td)
    signed = Account.sign_message(signable, private_key)
    return {
        "order": order,
        "signature": "0x" + signed.signature.hex().removeprefix("0x"),
        "hash": "0x" + signed.message_hash.hex().removeprefix("0x"),
    }


def recover_signer(
    order: dict[str, Any],
    signature: str,
    *,
    verifying_contract: str,
    chain_id: int = BASE_CHAIN_ID,
) -> str:
    """Recover the signer address from a signed order (for verification/tests)."""
    from eth_account import Account
    from eth_account.messages import encode_typed_data

    td = order_typed_data(order, verifying_contract=verifying_contract, chain_id=chain_id)
    signable = encode_typed_data(full_message=td)
    return Account.recover_message(signable, signature=signature)
