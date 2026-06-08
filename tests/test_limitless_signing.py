"""Tests for Limitless EIP-712 signing (Phase 5)."""

from __future__ import annotations

import pytest

from betbot.exchanges import limitless_signing as sg

VERIFYING = "0xe3E00BA3a9888d1DE4834269f62ac008b4BB5C47"


# ---- pure helpers (no eth_account) -----------------------------------
def test_to_usdc_units():
    assert sg.to_usdc_units(10.0) == 10_000_000
    assert sg.to_usdc_units(0.5) == 500_000
    with pytest.raises(ValueError):
        sg.to_usdc_units(-1.0)


def test_shares_for():
    # spend 10 USDC at 0.5 -> 20 shares
    assert sg.shares_for(10_000_000, 0.5) == 20_000_000
    with pytest.raises(ValueError):
        sg.shares_for(10_000_000, 0.0)
    with pytest.raises(ValueError):
        sg.shares_for(10_000_000, 1.0)


def test_random_salt_is_256bit_int():
    s = sg.random_salt()
    assert isinstance(s, int) and 0 <= s < 2**256
    assert sg.random_salt() != sg.random_salt()  # practically never equal


def test_build_order_field_order_and_defaults():
    o = sg.build_order(
        token_id="123", maker="0xabc", maker_amount_units=10_000_000,
        taker_amount_units=20_000_000, salt=7,
    )
    assert list(o.keys()) == [
        "salt", "maker", "signer", "taker", "tokenId", "makerAmount",
        "takerAmount", "expiration", "nonce", "feeRateBps", "side", "signatureType",
    ]
    assert o["signer"] == "0xabc"          # defaults to maker
    assert o["taker"] == sg.ZERO_ADDRESS
    assert o["side"] == sg.BUY
    assert o["tokenId"] == 123


def test_typed_data_domain_is_limitless_v1():
    o = sg.build_order(token_id=1, maker="0xabc", maker_amount_units=1, taker_amount_units=2, salt=1)
    td = sg.order_typed_data(o, verifying_contract=VERIFYING)
    assert td["domain"]["name"] == "Limitless CTF Exchange"
    assert td["domain"]["version"] == "1"
    assert td["domain"]["chainId"] == 8453
    assert td["primaryType"] == "Order"


# ---- sign + recover roundtrip (uses eth_account) ---------------------
def test_sign_recover_roundtrip():
    from eth_account import Account

    acct = Account.create()
    order = sg.build_order(
        token_id=42, maker=acct.address, maker_amount_units=sg.to_usdc_units(10),
        taker_amount_units=sg.shares_for(sg.to_usdc_units(10), 0.5), salt=12345,
    )
    signed = sg.sign_order(order, acct.key, verifying_contract=VERIFYING)
    assert signed["signature"].startswith("0x")
    assert signed["hash"].startswith("0x")
    recovered = sg.recover_signer(order, signed["signature"], verifying_contract=VERIFYING)
    assert recovered.lower() == acct.address.lower()


def test_signature_changes_with_contract():
    from eth_account import Account

    acct = Account.create()
    order = sg.build_order(token_id=1, maker=acct.address, maker_amount_units=1, taker_amount_units=2, salt=1)
    a = sg.sign_order(order, acct.key, verifying_contract=VERIFYING)
    b = sg.sign_order(order, acct.key, verifying_contract=sg.ZERO_ADDRESS)
    assert a["signature"] != b["signature"]  # domain-separated
