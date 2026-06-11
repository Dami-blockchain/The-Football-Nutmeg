"""Agent-treasury rebalancer tests — targets, threshold, in-flight lock, resume.

No network, no RPC: every chain-touching primitive of ``TreasuryRebalancer``
(inherited from ``DepositPipeline``) is overridden by ``FakeTreasury``, so these
tests exercise pure rebalance decision logic + the ``treasury_bridges`` table
state machine. Mirrors tests/test_bridge.py's fake style.
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
    Attestation,
    SignedTx,
    TreasuryRebalancer,
)
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import TreasuryBridge

AGENT = "0xAGENT"


class FakeTreasury(TreasuryRebalancer):
    """TreasuryRebalancer with every on-chain primitive faked.

    Burn is modelled in two phases (sign then broadcast) exactly like
    test_bridge.FakePipeline, so the crash window between them is testable;
    ``mined`` / ``reverted`` drive ``_tx_status`` for resume.
    """

    def __init__(self, settings, *, balances=None, native=None, attestation=None):
        super().__init__(settings)
        self.balances = dict(balances or {})  # (chain, address) -> USDC float
        self.native = dict(native or {})  # (chain, address) -> wei int
        self.attestation = attestation
        self.calls: dict[str, list] = defaultdict(list)
        self.broadcast_exc: Exception | None = None
        self.mined: set[str] = set()
        self.reverted: set[str] = set()
        self.nonce_used = False
        self._signed: dict[str, tuple] = {}

    def _agent_account(self):
        return SimpleNamespace(address=AGENT)

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
        tx_hash = f"0xtburn{len(self._signed) + 1}"
        self.calls["sign_burn"].append((chain, amount_units, dest_domain, mint_recipient))
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
        return "0xtmint"


def _all_bridges() -> list[TreasuryBridge]:
    with session_scope() as s:
        rows = list(s.execute(select(TreasuryBridge)).scalars())
        s.expunge_all()
        return rows


@pytest.fixture
def db(tmp_path):
    init_engine(tmp_path / "treasury.sqlite")


@pytest.fixture
def t_settings(settings, tmp_path):
    settings.wallet_keyfile = tmp_path / "agent.key"
    settings.auto_bridge = True
    settings.treasury_auto_rebalance = True
    settings.treasury_target_polygon_usd = 50.0
    settings.treasury_target_base_usd = 50.0
    settings.treasury_rebalance_threshold_usd = 10.0
    # Give the agent funded gas everywhere so the gas step is a no-op unless a
    # test wants otherwise.
    return settings


def _funded(native):
    return native


# ----------------------------------------------------------------------
# Decision logic
# ----------------------------------------------------------------------
def test_balanced_does_nothing(db, t_settings):
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 50.0, ("base", AGENT): 50.0},
    )
    p.rebalance_agent_treasury()
    assert _all_bridges() == []
    assert p.calls == {}


def test_within_threshold_does_nothing(db, t_settings):
    # Polygon short by 8 (< threshold 10) — dust, ignored.
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 42.0, ("base", AGENT): 58.0},
    )
    p.rebalance_agent_treasury()
    assert _all_bridges() == []
    assert p.calls["burn"] == []


def test_polygon_below_target_bridges_deficit_from_base_surplus(db, t_settings):
    # Polygon 20 (deficit 30), Base 90 (surplus 40). Bridge the deficit (30),
    # capped at the surplus (40) -> 30 from base -> polygon.
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 20.0, ("base", AGENT): 90.0},
        native={("base", AGENT): 10**18},  # agent already has Base gas
        attestation=Attestation("0xmsg", "0xatt"),
    )
    p.rebalance_agent_treasury()
    bridges = _all_bridges()
    assert len(bridges) == 1
    b = bridges[0]
    assert (b.source_chain, b.dest_chain) == ("base", "polygon")
    assert b.amount_usdc == 30.0
    assert b.status == DONE
    # Burn toward Polygon's CCTP domain (7); agent is mint recipient.
    assert len(p.calls["burn"]) == 1
    assert p.calls["burn"][0][2] == 7
    assert p.calls["sign_burn"][0][3] == AGENT
    assert len(p.calls["receive"]) == 1


def test_deficit_capped_at_surplus(db, t_settings):
    # Polygon 0 (deficit 50) but Base only 65 (surplus 15). Bridge only 15.
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 0.0, ("base", AGENT): 65.0},
        native={("base", AGENT): 10**18},
        attestation=Attestation("0xmsg", "0xatt"),
    )
    p.rebalance_agent_treasury()
    b = _all_bridges()[0]
    assert b.amount_usdc == 15.0
    assert (b.source_chain, b.dest_chain) == ("base", "polygon")


def test_deficit_with_no_surplus_does_nothing(db, t_settings):
    # Polygon short, but Base is also at/below its target -> no surplus to pull.
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 20.0, ("base", AGENT): 50.0},
    )
    p.rebalance_agent_treasury()
    assert _all_bridges() == []
    assert p.calls["burn"] == []


def test_surplus_below_threshold_does_nothing(db, t_settings):
    # Polygon deficit 30, Base surplus only 5 -> fundable amount (5) < threshold.
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 20.0, ("base", AGENT): 55.0},
    )
    p.rebalance_agent_treasury()
    assert _all_bridges() == []
    assert p.calls["burn"] == []


# ----------------------------------------------------------------------
# Single-in-flight invariant
# ----------------------------------------------------------------------
def test_only_one_in_flight(db, t_settings):
    # First tick opens a bridge but the attestation isn't ready -> it parks at
    # BURNED. A second tick must NOT open another bridge; it resumes the first.
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 0.0, ("base", AGENT): 100.0},
        native={("base", AGENT): 10**18},
        attestation=None,  # Circle hasn't attested yet
    )
    p.rebalance_agent_treasury()
    assert len(_all_bridges()) == 1
    assert _all_bridges()[0].status == BURNED
    assert len(p.calls["burn"]) == 1

    # Even though Base still shows a big surplus, no second bridge starts.
    p.rebalance_agent_treasury()
    assert len(_all_bridges()) == 1
    assert len(p.calls["burn"]) == 1

    # Attestation arrives -> the SAME leg resumes to done.
    p.attestation = Attestation("0xmsg", "0xatt")
    p.rebalance_agent_treasury()
    assert _all_bridges()[0].status == DONE
    assert len(p.calls["burn"]) == 1
    assert len(p.calls["receive"]) == 1


# ----------------------------------------------------------------------
# Crash safety
# ----------------------------------------------------------------------
def test_resume_after_crash_no_reburn(db, t_settings):
    # A leg persisted at BURN_SUBMITTED (crash between persisting the hash and
    # confirming the receipt). The burn actually mined -> resume to BURNED then
    # mint, NEVER re-burning.
    from betbot.storage import repos

    bid = repos.create_treasury_bridge(
        source_chain="base", dest_chain="polygon", amount_usdc=30.0,
        status=BURN_SUBMITTED,
    )
    repos.update_treasury_bridge(bid, burn_tx="0xtburn1")

    p = FakeTreasury(
        t_settings, attestation=Attestation("0xmsg", "0xatt")
    )
    p.mined.add("0xtburn1")  # the burn DID mine on-chain
    p.rebalance_agent_treasury()
    b = _all_bridges()[0]
    assert b.status == DONE
    assert b.burn_tx == "0xtburn1"  # original burn, never re-signed
    assert b.mint_tx == "0xtmint"
    assert p.calls["sign_burn"] == []
    assert p.calls["burn"] == []
    assert len(p.calls["receive"]) == 1


def test_burn_hash_persisted_before_broadcast(db, t_settings):
    # Receipt timeout AFTER the burn mined: the persisted hash lets the leg
    # resume into attestation, never re-running the burn.
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 0.0, ("base", AGENT): 100.0},
        native={("base", AGENT): 10**18},
        attestation=Attestation("0x" + "00" * 64, "0xatt"),
    )
    p.broadcast_exc = TimeoutError("receipt wait timed out after 120s")
    p.rebalance_agent_treasury()
    b = _all_bridges()[0]
    assert b.status == BURN_SUBMITTED
    assert b.burn_tx == "0xtburn1"  # hash recorded BEFORE the broadcast

    # The tx really mined (only the receipt wait timed out).
    p.mined.add("0xtburn1")
    p.balances[("base", AGENT)] = 50.0  # funds left the source wallet
    p.broadcast_exc = None
    p.rebalance_agent_treasury()
    b = _all_bridges()[0]
    assert b.status == DONE
    assert b.burn_tx == "0xtburn1"
    assert len(p.calls["sign_burn"]) == 1  # never signed a second burn


# ----------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------
def test_auto_rebalance_false_disables(db, t_settings):
    t_settings.treasury_auto_rebalance = False
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 0.0, ("base", AGENT): 100.0},
    )
    p.rebalance_agent_treasury()
    assert _all_bridges() == []
    assert p.calls == {}


def test_auto_bridge_false_disables(db, t_settings):
    t_settings.auto_bridge = False
    p = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 0.0, ("base", AGENT): 100.0},
    )
    p.rebalance_agent_treasury()
    assert _all_bridges() == []
    assert p.calls == {}


def test_auto_bridge_false_freezes_active_leg(db, t_settings):
    # A leg already in flight must NOT resume while the master kill is off.
    from betbot.storage import repos

    repos.create_treasury_bridge(
        source_chain="base", dest_chain="polygon", amount_usdc=30.0,
        status=GAS_TOPPED_UP,
    )
    t_settings.auto_bridge = False
    p = FakeTreasury(
        t_settings,
        balances={("base", AGENT): 100.0},
        attestation=Attestation("0xmsg", "0xatt"),
    )
    p.rebalance_agent_treasury()
    assert _all_bridges()[0].status == GAS_TOPPED_UP
    assert p.calls == {}


# ----------------------------------------------------------------------
# Nonce-already-consumed (mint replay)
# ----------------------------------------------------------------------
def test_nonce_already_consumed_skips_remint(db, t_settings):
    from betbot.storage import repos

    repos.create_treasury_bridge(
        source_chain="base", dest_chain="polygon", amount_usdc=30.0,
        status=BURNED,
    )
    repos.update_treasury_bridge(_all_bridges()[0].id, burn_tx="0xtburn1")

    class ReplayTreasury(FakeTreasury):
        def _receive_message(self, dest_chain, att):
            raise RuntimeError("execution reverted: Nonce already used")

    p = ReplayTreasury(t_settings, attestation=Attestation("0xmsg", "0xatt"))
    p.nonce_used = True  # on-chain registry confirms the mint executed
    p.rebalance_agent_treasury()
    b = _all_bridges()[0]
    assert b.status == DONE
    assert b.mint_tx is None  # no re-mint
    assert p.calls["nonce_check"] == ["polygon"]


def test_account_nonce_error_never_marks_minted(db, t_settings):
    from betbot.storage import repos

    repos.create_treasury_bridge(
        source_chain="base", dest_chain="polygon", amount_usdc=30.0,
        status=BURNED,
    )
    repos.update_treasury_bridge(_all_bridges()[0].id, burn_tx="0xtburn1")

    class StaleNonceTreasury(FakeTreasury):
        def _receive_message(self, dest_chain, att):
            raise ValueError("nonce too low: next nonce 41, tx nonce 40")

    p = StaleNonceTreasury(t_settings, attestation=Attestation("0xmsg", "0xatt"))
    p.nonce_used = False  # NOT relayed on-chain
    p.rebalance_agent_treasury()
    b = _all_bridges()[0]
    assert b.status == BURNED  # parked, retried next tick
    assert b.mint_tx is None
    assert "nonce too low" in b.error
    assert p.calls["nonce_check"] == ["polygon"]


# ----------------------------------------------------------------------
# Combined daemon tick (user scan + treasury rebalance share one tick)
# ----------------------------------------------------------------------
def test_deposit_scan_sync_runs_both(db, t_settings, monkeypatch):
    # The combined sync path runs the user scan FIRST, then the treasury
    # rebalance: agent below target on polygon with a base surplus opens a
    # bridge, while the user scan reports its own (0) created count.
    import betbot.bridge as bridge

    calls = {"user_scan": 0}

    def fake_run_once(self):
        calls["user_scan"] += 1
        return 0

    monkeypatch.setattr(bridge.DepositPipeline, "run_once", fake_run_once)

    prepared = FakeTreasury(
        t_settings,
        balances={("polygon", AGENT): 0.0, ("base", AGENT): 100.0},
        native={("base", AGENT): 10**18},
        attestation=Attestation("0xmsg", "0xatt"),
    )
    monkeypatch.setattr(bridge, "TreasuryRebalancer", lambda settings: prepared)

    created = bridge._deposit_scan_sync(t_settings)
    assert created == 0
    assert calls["user_scan"] == 1
    assert _all_bridges()[0].status == DONE
