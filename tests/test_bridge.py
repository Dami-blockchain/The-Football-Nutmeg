"""Deposit-pipeline tests — threshold, idempotency, allocation, resume, gas.

No network, no RPC: every chain-touching primitive of ``DepositPipeline`` is
overridden by ``FakePipeline``, so these tests exercise pure orchestration +
the ``deposits`` table state machine.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from betbot.bridge import (
    BURNED,
    DONE,
    Attestation,
    DepositPipeline,
)
from betbot.storage import repos
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import Deposit


class FakePipeline(DepositPipeline):
    """DepositPipeline with all on-chain primitives replaced by fakes."""

    def __init__(self, settings, *, balances=None, native=None, attestation=None):
        super().__init__(settings)
        self.balances = dict(balances or {})  # (chain, address) -> USDC float
        self.native = dict(native or {})  # (chain, address) -> wei int
        self.attestation = attestation  # None = "not attested yet"
        self.calls: dict[str, list] = defaultdict(list)

    def _usdc_balance(self, chain, address):
        return self.balances.get((chain, address), 0.0)

    def _native_balance_wei(self, chain, address):
        return self.native.get((chain, address), 0)

    def _send_native(self, chain, to_address, amount_wei):
        self.calls["send_native"].append((chain, to_address, amount_wei))
        return "0xgas"

    def _allowance(self, chain, token, owner, spender):
        return 0

    def _approve_erc20(self, chain, acct, token, spender, amount_units):
        self.calls["approve_erc20"].append((chain, token, spender, amount_units))
        return "0xapprove"

    def _deposit_for_burn(
        self, chain, acct, *, amount_units, dest_domain, mint_recipient
    ):
        self.calls["burn"].append((chain, amount_units, dest_domain))
        self.balances[(chain, mint_recipient)] = 0.0  # USDC burned away
        return "0xburn"

    def _fetch_attestation(self, source_chain, burn_tx):
        self.calls["fetch_attestation"].append((source_chain, burn_tx))
        return self.attestation

    def _receive_message(self, dest_chain, att):
        self.calls["receive"].append((dest_chain, att))
        return "0xmint"

    def _approve_polymarket(self, acct):
        self.calls["approve_polymarket"].append(acct.address)

    def _approve_limitless(self, acct):
        self.calls["approve_limitless"].append(acct.address)

    def _user_account(self, leg):
        return SimpleNamespace(address=leg.wallet_address)

    def _agent_account(self):
        return SimpleNamespace(address="0xAGENT")


def _all_deposits() -> list[Deposit]:
    with session_scope() as s:
        rows = list(s.execute(select(Deposit)).scalars())
        s.expunge_all()
        return rows


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "bridge.sqlite")


@pytest.fixture
def user(db, tmp_path):
    return repos.get_or_create_user(
        111, "tester", secrets_dir=str(tmp_path / ".secrets")
    )


@pytest.fixture
def bridge_settings(settings, tmp_path):
    settings.wallet_keyfile = tmp_path / "agent.key"
    settings.auto_bridge = True
    settings.min_deposit_usdc = 10.0
    settings.bridge_split_base = 0.0
    return settings


def test_below_minimum_ignored(bridge_settings, user):
    p = FakePipeline(
        bridge_settings, balances={("ethereum", user.wallet_address): 5.0}
    )
    assert p.run_once() == 0
    assert _all_deposits() == []
    assert p.calls["burn"] == []


def test_auto_bridge_gate_disables_everything(bridge_settings, user):
    bridge_settings.auto_bridge = False
    p = FakePipeline(
        bridge_settings, balances={("ethereum", user.wallet_address): 50.0}
    )
    assert p.run_once() == 0
    assert _all_deposits() == []
    assert p.calls == {}


def test_deposit_bridged_once_end_to_end(bridge_settings, user):
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 50.0},
        attestation=Attestation("0xmsg", "0xatt"),
    )
    assert p.run_once() == 1
    legs = _all_deposits()
    assert len(legs) == 1
    leg = legs[0]
    assert (leg.source_chain, leg.dest_chain) == ("ethereum", "polygon")
    assert leg.amount_usdc == 50.0  # default split: 100% to Polygon
    assert leg.status == DONE
    assert leg.burn_tx == "0xburn" and leg.mint_tx == "0xmint"
    assert len(p.calls["burn"]) == 1
    assert len(p.calls["receive"]) == 1
    assert p.calls["approve_polymarket"] == [user.wallet_address]

    # The same deposit seen again -> exactly one bridge, no new rows.
    assert p.run_once() == 0
    assert len(p.calls["burn"]) == 1
    assert len(_all_deposits()) == 1


def test_detection_skipped_while_leg_in_flight(bridge_settings, user):
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 25.0},
        attestation=None,  # Circle hasn't attested yet
    )
    assert p.run_once() == 1
    assert _all_deposits()[0].status == BURNED
    # Even if the balance read showed funds again mid-flight, the active-leg
    # guard must prevent a second record / second burn.
    p.balances[("ethereum", user.wallet_address)] = 25.0
    assert p.run_once() == 0
    assert len(_all_deposits()) == 1
    assert len(p.calls["burn"]) == 1
    # Attestation arrives -> the SAME leg resumes and completes.
    p.attestation = Attestation("0xmsg", "0xatt")
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.status == DONE
    assert len(p.calls["burn"]) == 1
    assert len(p.calls["receive"]) == 1


def test_resume_after_partial_failure(bridge_settings, user):
    # Simulate a crash AFTER the burn was recorded but BEFORE the mint.
    deposit_id = repos.create_deposit(
        user_id=user.id,
        wallet_address=user.wallet_address,
        source_chain="arbitrum",
        dest_chain="polygon",
        amount_usdc=20.0,
        balance_snapshot=20.0,
        status=BURNED,
    )
    repos.update_deposit(deposit_id, burn_tx="0xoldburn")

    p = FakePipeline(bridge_settings, attestation=None)
    p.run_once()
    assert _all_deposits()[0].status == BURNED  # still awaiting attestation
    assert p.calls["burn"] == []  # NEVER re-burned on resume

    p.attestation = Attestation("0xmsg", "0xatt")
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.status == DONE
    assert leg.burn_tx == "0xoldburn"
    assert leg.mint_tx == "0xmint"
    assert p.calls["burn"] == []
    assert len(p.calls["receive"]) == 1
    assert p.calls["approve_polymarket"] == [user.wallet_address]


def test_mint_replay_advances_instead_of_sticking(bridge_settings, user):
    """receiveMessage reverting with 'nonce already used' means an earlier
    tick's mint landed but wasn't recorded — the leg must advance, not wedge."""
    repos.create_deposit(
        user_id=user.id,
        wallet_address=user.wallet_address,
        source_chain="base",
        dest_chain="polygon",
        amount_usdc=15.0,
        balance_snapshot=15.0,
        status=BURNED,
    )

    class ReplayPipeline(FakePipeline):
        def _receive_message(self, dest_chain, att):
            raise RuntimeError("execution reverted: Nonce already used")

    p = ReplayPipeline(bridge_settings, attestation=Attestation("0xmsg", "0xatt"))
    p.run_once()
    assert _all_deposits()[0].status == DONE


def test_allocation_split_base(bridge_settings, user):
    bridge_settings.bridge_split_base = 0.2
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 100.0},
        attestation=Attestation("0xmsg", "0xatt"),
    )
    assert p.run_once() == 2
    legs = sorted(_all_deposits(), key=lambda d: d.dest_chain)
    assert [(d.dest_chain, d.amount_usdc) for d in legs] == [
        ("base", 20.0),
        ("polygon", 80.0),
    ]
    assert all(d.status == DONE for d in legs)
    # One burn per leg, toward the right CCTP domains (Base=6, Polygon=7).
    assert sorted(c[2] for c in p.calls["burn"]) == [6, 7]
    # Base share triggers the Limitless approval; Polygon the Polymarket one.
    assert p.calls["approve_limitless"] == [user.wallet_address]
    assert p.calls["approve_polymarket"] == [user.wallet_address]


def test_gas_topup_for_empty_wallets(bridge_settings, user):
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 30.0},
        attestation=Attestation("0xmsg", "0xatt"),
    )
    p.run_once()
    # Source-chain gas for approve+burn, then Polygon gas for the venue
    # approvals. The mint needs NO user gas (agent relays receiveMessage).
    assert [c[0] for c in p.calls["send_native"]] == ["ethereum", "polygon"]
    eth_wei = p.calls["send_native"][0][2]
    pol_wei = p.calls["send_native"][1][2]
    assert eth_wei == int(bridge_settings.gas_topup_eth * 10**18)
    assert pol_wei == int(bridge_settings.gas_topup_pol * 10**18)


def test_gas_topup_skipped_when_funded_and_polygon_is_local(bridge_settings, user):
    p = FakePipeline(
        bridge_settings,
        balances={("polygon", user.wallet_address): 40.0},
        native={("polygon", user.wallet_address): 10**18},  # already has POL
    )
    assert p.run_once() == 1
    assert p.calls["send_native"] == []
    # Polygon is a native CCTP domain AND the trading chain: a Polygon
    # deposit needs no bridge at all — straight to venue approvals.
    assert p.calls["burn"] == []
    assert p.calls["receive"] == []
    assert p.calls["approve_polymarket"] == [user.wallet_address]
    leg = _all_deposits()[0]
    assert leg.status == DONE
    assert (leg.source_chain, leg.dest_chain) == ("polygon", "polygon")
    # Done legs form the baseline: the SAME resting balance is not a new deposit.
    assert p.run_once() == 0
    assert len(_all_deposits()) == 1
    # ...but a top-up ABOVE the baseline is.
    p.balances[("polygon", user.wallet_address)] = 65.0
    assert p.run_once() == 1
    new_leg = sorted(_all_deposits(), key=lambda d: d.id)[-1]
    assert new_leg.amount_usdc == 25.0
