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
    BURN_SUBMITTED,
    BURNED,
    DONE,
    GAS_TOPPED_UP,
    MINTED,
    Attestation,
    DepositPipeline,
    SignedTx,
    _extract_cctp_nonce,
)
from betbot.storage import repos
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import Deposit


class FakePipeline(DepositPipeline):
    """DepositPipeline with all on-chain primitives replaced by fakes.

    The burn is modelled in two phases, mirroring production: signing
    (``_sign_deposit_for_burn``) and broadcasting (``_broadcast_signed``).
    ``broadcast_exc`` simulates the receipt-timeout / crash window between
    the two; ``mined`` / ``reverted`` drive ``_tx_status`` for recovery.
    """

    def __init__(self, settings, *, balances=None, native=None, attestation=None):
        super().__init__(settings)
        self.balances = dict(balances or {})  # (chain, address) -> USDC float
        self.native = dict(native or {})  # (chain, address) -> wei int
        self.attestation = attestation  # None = "not attested yet"
        self.calls: dict[str, list] = defaultdict(list)
        self.broadcast_exc: Exception | None = None
        self.mined: set[str] = set()
        self.reverted: set[str] = set()
        self.nonce_used = False  # MessageTransmitter usedNonces verdict
        self._signed: dict[str, tuple] = {}

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

    def _sign_deposit_for_burn(
        self, chain, acct, *, amount_units, dest_domain, mint_recipient
    ):
        tx_hash = f"0xburn{len(self._signed) + 1}"
        self.calls["sign_burn"].append((chain, amount_units, dest_domain))
        self._signed[tx_hash] = (chain, amount_units, dest_domain, mint_recipient)
        return SignedTx(tx_hash=tx_hash, raw=b"raw")

    def _broadcast_signed(self, chain, signed):
        if self.broadcast_exc is not None:
            raise self.broadcast_exc
        c, units, domain, recipient = self._signed[signed.tx_hash]
        self.calls["burn"].append((c, units, domain))
        bal = self.balances.get((c, recipient), 0.0)
        self.balances[(c, recipient)] = max(0.0, round(bal - units / 10**6, 6))
        self.mined.add(signed.tx_hash)

    def _tx_status(self, chain, tx_hash):
        if tx_hash in self.mined:
            return True
        if tx_hash in self.reverted:
            return False
        return None

    def _message_nonce_consumed(self, dest_chain, att):
        self.calls["nonce_check"].append(dest_chain)
        return self.nonce_used

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
    assert leg.burn_tx == "0xburn1" and leg.mint_tx == "0xmint"
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
    # Even if the balance read showed funds again mid-flight (stale RPC),
    # the active-leg guard must prevent a second record / second burn.
    p.balances[("ethereum", user.wallet_address)] = 25.0
    assert p.run_once() == 0
    assert len(_all_deposits()) == 1
    assert len(p.calls["burn"]) == 1
    # The stale read clears (the burn really did consume the funds), then
    # the attestation arrives -> the SAME leg resumes and completes.
    p.balances[("ethereum", user.wallet_address)] = 0.0
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
    """receiveMessage reverting because the message nonce was ALREADY consumed
    (verified on-chain via usedNonces) means an earlier tick's mint landed but
    wasn't recorded — the leg must advance, not wedge."""
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
    p.nonce_used = True  # the on-chain registry confirms the mint executed
    p.run_once()
    assert _all_deposits()[0].status == DONE
    assert p.calls["nonce_check"] == ["polygon"]


def test_account_nonce_error_never_marks_minted(bridge_settings, user):
    """An ordinary ACCOUNT-nonce error from the agent wallet's own submission
    ('nonce too low' from a lagging RPC) contains 'nonce' but has nothing to
    do with the CCTP message nonce. The leg must stay BURNED and retry —
    marking it MINTED would report delivery that never happened."""
    repos.create_deposit(
        user_id=user.id,
        wallet_address=user.wallet_address,
        source_chain="base",
        dest_chain="polygon",
        amount_usdc=15.0,
        balance_snapshot=15.0,
        status=BURNED,
    )

    class StaleNoncePipeline(FakePipeline):
        def _receive_message(self, dest_chain, att):
            raise ValueError("nonce too low: next nonce 41, tx nonce 40")

    p = StaleNoncePipeline(
        bridge_settings, attestation=Attestation("0xmsg", "0xatt")
    )
    p.nonce_used = False  # on-chain registry: message was NOT relayed
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.status == BURNED  # NOT minted — retried next tick
    assert leg.mint_tx is None
    assert "nonce too low" in leg.error
    assert p.calls["nonce_check"] == ["polygon"]


def test_extract_cctp_nonce_reads_v2_header():
    nonce = bytes(range(32))
    message = "0x" + (b"\x01" * 12 + nonce + b"\xff" * 20).hex()
    assert _extract_cctp_nonce(message) == nonce


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


# ----------------------------------------------------------------------
# Burn idempotency across the broadcast/receipt window (funds-loss bug)
# ----------------------------------------------------------------------
def test_burn_hash_persisted_before_broadcast_no_reburn_after_timeout(
    bridge_settings, user
):
    """Receipt timeout AFTER the burn mined: the persisted hash must let the
    leg resume into attestation — never re-run the burn (which would revert
    forever, or worse, consume a second deposit)."""
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 50.0},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p.broadcast_exc = TimeoutError("receipt wait timed out after 120s")
    assert p.run_once() == 1
    leg = _all_deposits()[0]
    assert leg.status == BURN_SUBMITTED
    assert leg.burn_tx == "0xburn1"  # hash recorded BEFORE the broadcast

    # The tx actually mined on-chain (the timeout was only the receipt wait).
    p.mined.add("0xburn1")
    p.balances[("ethereum", user.wallet_address)] = 0.0
    p.broadcast_exc = None
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.status == DONE
    assert leg.burn_tx == "0xburn1"  # the ORIGINAL burn's attestation relayed
    assert len(p.calls["sign_burn"]) == 1  # never signed a second burn
    assert ("ethereum", "0xburn1") in p.calls["fetch_attestation"]


def test_burn_recovery_resubmits_when_broadcast_never_landed(
    bridge_settings, user
):
    """Daemon died before the broadcast reached the network: no receipt AND
    the funds are still in the wallet — safe to re-sign and retry."""
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 50.0},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p.broadcast_exc = ConnectionError("connection dropped before send")
    p.run_once()
    assert _all_deposits()[0].status == BURN_SUBMITTED

    p.broadcast_exc = None
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.status == DONE
    assert leg.burn_tx == "0xburn2"  # the retry's hash, persisted pre-broadcast
    assert len(p.calls["sign_burn"]) == 2
    assert len(p.calls["burn"]) == 1  # exactly ONE burn ever landed


def test_burn_recovery_waits_when_funds_left_but_receipt_unknown(
    bridge_settings, user
):
    """No receipt yet but the USDC left the wallet: the burn almost certainly
    mined and the RPC is lagging — recovery must WAIT, never re-burn (a
    re-burn here is exactly the double-bridge/lost-funds case)."""
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 50.0},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p.broadcast_exc = TimeoutError("receipt wait timed out")
    p.run_once()
    # Funds gone (burn mined) but the receipt still isn't visible.
    p.balances[("ethereum", user.wallet_address)] = 0.0
    p.broadcast_exc = None
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.status == BURN_SUBMITTED  # parked, retried next tick
    assert len(p.calls["sign_burn"]) == 1  # NO second burn
    assert p.calls["burn"] == []


def test_burn_reverted_on_chain_is_retried(bridge_settings, user):
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 50.0},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p.broadcast_exc = TimeoutError("receipt wait timed out")
    p.run_once()
    p.reverted.add("0xburn1")  # mined but reverted — funds untouched
    p.broadcast_exc = None
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.status == DONE
    assert leg.burn_tx == "0xburn2"
    assert len(p.calls["burn"]) == 1


def test_burn_aborts_when_balance_below_leg_amount(bridge_settings, user):
    """Defense in depth: a wallet that no longer holds the leg's amount means
    an unrecorded burn already consumed it — never burn again."""
    deposit_id = repos.create_deposit(
        user_id=user.id,
        wallet_address=user.wallet_address,
        source_chain="ethereum",
        dest_chain="polygon",
        amount_usdc=50.0,
        balance_snapshot=50.0,
        status=GAS_TOPPED_UP,
    )
    p = FakePipeline(
        bridge_settings, balances={("ethereum", user.wallet_address): 0.0}
    )
    p.run_once()
    leg = _all_deposits()[0]
    assert leg.id == deposit_id
    assert leg.status == GAS_TOPPED_UP  # parked with an error, not burned
    assert "burn aborted" in leg.error
    assert p.calls["sign_burn"] == []


# ----------------------------------------------------------------------
# Resume-before-detect + inbound guard (phantom-deposit race)
# ----------------------------------------------------------------------
def test_unrecorded_mint_is_resumed_not_redetected(bridge_settings, user):
    """Mint landed on-chain but the crash ate the status update: the next
    tick must resume the leg FIRST (replay verification advances it) so
    detection never records the minted funds as a phantom new deposit."""
    repos.create_deposit(
        user_id=user.id,
        wallet_address=user.wallet_address,
        source_chain="ethereum",
        dest_chain="polygon",
        amount_usdc=30.0,
        balance_snapshot=30.0,
        status=BURNED,
    )
    repos.update_deposit(_all_deposits()[0].id, burn_tx="0xoldburn")

    class ReplayPipeline(FakePipeline):
        def _receive_message(self, dest_chain, att):
            raise RuntimeError("execution reverted: Nonce already used")

    p = ReplayPipeline(
        bridge_settings,
        # The unrecorded mint's funds are already visible on Polygon.
        balances={("polygon", user.wallet_address): 30.0},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p.nonce_used = True
    created = p.run_once()
    assert created == 0  # NO phantom polygon->polygon deposit
    legs = _all_deposits()
    assert len(legs) == 1
    assert legs[0].status == DONE
    # Baseline reflects the real delivery exactly once.
    assert repos.delivered_to_chain_usdc(user.wallet_address, "polygon") == 30.0


def test_detection_skipped_while_inbound_leg_active(bridge_settings, user):
    """Even when the resume itself can't advance (Circle API down), funds
    sitting on the destination of an in-flight leg must not be detected."""
    repos.create_deposit(
        user_id=user.id,
        wallet_address=user.wallet_address,
        source_chain="ethereum",
        dest_chain="polygon",
        amount_usdc=30.0,
        balance_snapshot=30.0,
        status=BURNED,
    )
    repos.update_deposit(_all_deposits()[0].id, burn_tx="0xoldburn")
    p = FakePipeline(
        bridge_settings,
        balances={("polygon", user.wallet_address): 30.0},
        attestation=None,  # attestation still pending — leg stays BURNED
    )
    assert p.run_once() == 0
    assert len(_all_deposits()) == 1  # no phantom record while in flight


# ----------------------------------------------------------------------
# Public-bot abuse guards (gas cap, registration never spends, gate)
# ----------------------------------------------------------------------
def test_gas_topup_daily_cap_bounds_agent_spend(bridge_settings, user):
    bridge_settings.gas_topup_daily_cap = 1
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 50.0},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p.run_once()
    # Source-chain top-up spent the day's budget; the Polygon venue top-up
    # must be refused — the leg parks at MINTED with the cap error.
    assert len(p.calls["send_native"]) == 1
    leg = _all_deposits()[0]
    assert leg.status == MINTED
    assert "cap" in leg.error

    # The cap is PERSISTED: a daemon restart (fresh pipeline) must not reset it.
    p2 = FakePipeline(
        bridge_settings,
        balances=p.balances,
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p2.run_once()
    assert p2.calls["send_native"] == []
    assert _all_deposits()[0].status == MINTED

    # Cap lifted (0 = disabled): the leg resumes and completes.
    bridge_settings.gas_topup_daily_cap = 0
    p2.run_once()
    assert _all_deposits()[0].status == DONE


def test_registration_alone_never_touches_the_chain(bridge_settings, user):
    """A registered user with empty wallets (i.e. anyone who just messaged
    the public bot) must cause zero on-chain actions and zero records —
    only a detected deposit >= the minimum may spend anything."""
    p = FakePipeline(bridge_settings)  # all balances zero
    assert p.run_once() == 0
    assert _all_deposits() == []
    assert not p.calls


def test_auto_bridge_off_freezes_active_legs_too(bridge_settings, user):
    """BETBOT_AUTO_BRIDGE=false must disable ALL on-chain action — including
    resuming legs that were already in flight when the gate was flipped."""
    repos.create_deposit(
        user_id=user.id,
        wallet_address=user.wallet_address,
        source_chain="ethereum",
        dest_chain="polygon",
        amount_usdc=50.0,
        balance_snapshot=50.0,
        status=GAS_TOPPED_UP,
    )
    bridge_settings.auto_bridge = False
    p = FakePipeline(
        bridge_settings,
        balances={("ethereum", user.wallet_address): 50.0},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    assert p.run_once() == 0
    assert not p.calls
    assert _all_deposits()[0].status == GAS_TOPPED_UP
