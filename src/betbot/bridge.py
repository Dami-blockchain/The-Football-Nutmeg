"""USDC deposit pipeline — Circle CCTP V2 bridging + automatic venue setup.

WHY this exists: users fund their per-user wallet (see ``betbot.wallet`` /
``storage.repos.get_or_create_user``) by sending USDC on whichever chain is
cheap or convenient for them — Ethereum, Arbitrum, Base or Polygon. Trading,
however, happens on Polygon (Polymarket, the live venue) and — once Limitless
live orders are un-parked — Base. This module closes that gap end to end:

1. **Detect**: poll each user wallet's USDC balance on every supported source
   chain. Deposits below ``BETBOT_MIN_DEPOSIT_USDC`` are ignored (logged) —
   bridging dust costs more gas than it's worth.
2. **Allocate**: split the deposit between trading chains. Default 100% to
   Polygon; ``BETBOT_BRIDGE_SPLIT_BASE`` reserves a future Base/Limitless
   share (default 0.0 while Limitless live orders are parked).
3. **Bridge**: Circle CCTP V2 burn-and-mint (no third-party bridge custody):
   ``depositForBurn`` on the source TokenMessengerV2 → poll Circle's
   attestation API (iris-api.circle.com) → ``receiveMessage`` on the
   destination MessageTransmitterV2. We use the *standard* (fee-free,
   finality-threshold 2000) transfer — attestation takes minutes, and the
   scan loop simply resumes the leg on a later tick.
4. **Gas**: user wallets are created empty, so before any user-signed
   transaction the agent wallet tops the user up with a small, configured
   amount of native gas (``BETBOT_GAS_TOPUP_POL`` / ``_ETH``). The CCTP mint
   itself is relayed BY the agent wallet (``destinationCaller`` left open),
   so no destination gas is needed just to receive funds.
5. **Venue accounts**: once funds land on Polygon, run the Polymarket token
   approvals for that user wallet (reusing
   ``scripts/polymarket_approve.approve_for_account``); a Base share
   additionally gets the Limitless USDC approval.

Safety properties:

- **Gated**: ``BETBOT_AUTO_BRIDGE=false`` disables every on-chain action,
  including the resumption of already-active legs.
- **Idempotent**: every leg is a row in the ``deposits`` table with an
  explicit per-step status; a half-finished pipeline resumes from its last
  completed step on the next scan tick. The burn specifically persists its
  signed tx hash (status ``burn_submitted``) BEFORE broadcasting, so a
  receipt timeout or daemon crash after broadcast can never lose the hash
  and re-burn — recovery checks the persisted tx's receipt (and the
  wallet's USDC balance) before deciding anything. See
  ``storage.models.Deposit`` for the dedupe model.
- **Bounded**: agent-paid gas top-ups are capped per wallet per UTC day
  (``BETBOT_GAS_TOPUP_DAILY_CAP``) — the bot is public, so a stranger's
  deposits must not be able to drain the agent wallet through gas spend.
  Registration alone never triggers any on-chain action; only a detected
  deposit >= ``BETBOT_MIN_DEPOSIT_USDC`` does.
- **Explicit**: every on-chain step logs what it did (tx hashes included).

Operator note: the AGENT wallet pays for gas top-ups and mint relays — keep
it funded with a little POL + ETH on the chains you expect deposits on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from betbot.logging import get_logger
from betbot.storage.repos import (
    DEPOSIT_DONE,
    active_treasury_bridge,
    count_gas_topups_since,
    create_deposit,
    create_treasury_bridge,
    delivered_to_chain_usdc,
    has_active_dest_deposit,
    has_active_source_deposit,
    list_active_deposits,
    list_users,
    record_gas_topup,
    update_deposit,
    update_treasury_bridge,
)

log = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_USDC_DECIMALS = 6
_WEI = 10**18

# ---------------------------------------------------------------------------
# CCTP V2 chain registry.
#
# Contract addresses + domain ids verified 2026-06-11 against Circle's docs:
#   https://developers.circle.com/cctp/evm-smart-contracts
# TokenMessengerV2 and MessageTransmitterV2 are deployed at the SAME address
# on every supported EVM mainnet. Polygon PoS *is* a native CCTP V2 domain
# (id 7) — deposits arriving on Polygon need no bridge hop at all, and
# nothing has to detour via Base.
# USDC token addresses verified the same day against
#   https://developers.circle.com/stablecoins/usdc-contract-addresses
# (Polygon/Base entries also match betbot.wallet.CHAINS.)
# ---------------------------------------------------------------------------
TOKEN_MESSENGER_V2 = "0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d"
MESSAGE_TRANSMITTER_V2 = "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64"

CCTP_CHAINS: dict[str, dict] = {
    "ethereum": {
        "label": "Ethereum",
        "chain_id": 1,
        "domain": 0,
        "usdc": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "token_messenger": TOKEN_MESSENGER_V2,
        "message_transmitter": MESSAGE_TRANSMITTER_V2,
        "gas": "ETH",
    },
    "arbitrum": {
        "label": "Arbitrum",
        "chain_id": 42161,
        "domain": 3,
        "usdc": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        "token_messenger": TOKEN_MESSENGER_V2,
        "message_transmitter": MESSAGE_TRANSMITTER_V2,
        "gas": "ETH",
    },
    "base": {
        "label": "Base",
        "chain_id": 8453,
        "domain": 6,
        "usdc": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "token_messenger": TOKEN_MESSENGER_V2,
        "message_transmitter": MESSAGE_TRANSMITTER_V2,
        "gas": "ETH",
    },
    "polygon": {
        "label": "Polygon",
        "chain_id": 137,
        "domain": 7,
        "usdc": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
        "token_messenger": TOKEN_MESSENGER_V2,
        "message_transmitter": MESSAGE_TRANSMITTER_V2,
        "gas": "POL",
    },
}

# Per-leg pipeline statuses (persisted in deposits.status). Local legs
# (source == dest) are created at MINTED — the funds never left the chain.
# BURN_SUBMITTED is the crash-safety checkpoint: the signed burn's tx hash is
# persisted BEFORE broadcasting, so the hash survives a receipt timeout or a
# daemon death mid-broadcast and recovery never re-burns blindly.
DETECTED = "detected"
GAS_TOPPED_UP = "gas_topped_up"
BURN_SUBMITTED = "burn_submitted"
BURNED = "burned"
MINTED = "minted"
DONE = DEPOSIT_DONE  # "done"

# CCTP V2 standard transfer: zero fee, full source-chain finality. The
# fast-transfer lane (lower threshold, nonzero maxFee) is deliberately NOT
# used — deposits aren't latency-sensitive and the standard lane is free.
_MIN_FINALITY_STANDARD = 2000
_MAX_FEE_STANDARD = 0

_ERC20_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
]

_TOKEN_MESSENGER_V2_ABI = [
    {"name": "depositForBurn", "type": "function",
     "stateMutability": "nonpayable",
     "inputs": [
         {"name": "amount", "type": "uint256"},
         {"name": "destinationDomain", "type": "uint32"},
         {"name": "mintRecipient", "type": "bytes32"},
         {"name": "burnToken", "type": "address"},
         {"name": "destinationCaller", "type": "bytes32"},
         {"name": "maxFee", "type": "uint256"},
         {"name": "minFinalityThreshold", "type": "uint32"},
     ],
     "outputs": []},
]

_MESSAGE_TRANSMITTER_V2_ABI = [
    {"name": "receiveMessage", "type": "function",
     "stateMutability": "nonpayable",
     "inputs": [{"name": "message", "type": "bytes"},
                {"name": "attestation", "type": "bytes"}],
     "outputs": [{"name": "", "type": "bool"}]},
    # Replay protection registry: 1 = this message's nonce was consumed (the
    # mint executed). Used to VERIFY "already relayed" instead of guessing
    # from error text — see _step_mint.
    {"name": "usedNonces", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "nonce", "type": "bytes32"}],
     "outputs": [{"name": "", "type": "uint256"}]},
]


class BridgeError(RuntimeError):
    """A deposit-pipeline step failed; the leg stays put and is retried."""


@dataclass(frozen=True)
class Attestation:
    """A complete Circle attestation, ready for ``receiveMessage``."""

    message: str
    attestation: str


@dataclass(frozen=True)
class SignedTx:
    """A locally signed, not-yet-broadcast transaction.

    The hash is known BEFORE broadcast (it's the keccak of the signed
    payload), which is what lets the burn step persist its tx hash first and
    broadcast second — the crash-safety ordering the pipeline relies on.
    """

    tx_hash: str
    raw: bytes


def _extract_cctp_nonce(message_hex: str) -> bytes:
    """The bytes32 message nonce from a CCTP V2 message.

    V2 header layout: version (4 bytes) | sourceDomain (4) | destDomain (4) |
    nonce (32). The nonce is what MessageTransmitterV2.usedNonces keys on.
    """
    raw = bytes.fromhex(message_hex.removeprefix("0x"))
    if len(raw) < 44:
        raise BridgeError("CCTP message too short to contain a nonce")
    return raw[12:44]


def _rpc_for(chain: str, settings) -> str:
    if chain == "polygon":
        return settings.polygon_rpc_url
    if chain == "base":
        return settings.base_rpc_url
    if chain == "ethereum":
        return settings.ethereum_rpc_url
    if chain == "arbitrum":
        return settings.arbitrum_rpc_url
    raise BridgeError(f"unknown chain {chain!r}")


def _address_to_bytes32(address: str) -> bytes:
    """Left-pad an EVM address to the bytes32 CCTP expects for recipients."""
    return b"\x00" * 12 + bytes.fromhex(address.removeprefix("0x"))


class DepositPipeline:
    """State machine driving each deposit leg through the bridge + approvals.

    Chain-touching primitives are isolated as small methods so tests can
    subclass with fakes — orchestration logic is then testable without web3
    or any RPC. Real implementations are deliberately boring: build tx, sign,
    send, wait for receipt, log the hash.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self._w3_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_once(self) -> int:
        """One scan tick: resume unfinished legs, THEN detect new deposits.

        Order matters: a mint that landed on-chain but whose status update
        didn't persist (crash, DB error) must be resumed — and counted into
        the delivered baseline — BEFORE detection reads destination-chain
        balances, or the freshly minted funds would be recorded as a phantom
        brand-new local deposit and permanently inflate the baseline.

        Returns the number of NEW deposit legs recorded this tick.
        """
        if not self.settings.auto_bridge:
            log.info("deposit_scan_disabled", note="BETBOT_AUTO_BRIDGE=false")
            return 0
        resumed: set[int] = set()
        for leg in list_active_deposits():
            resumed.add(leg.id)
            self.process_deposit(leg)
        created = self.detect_new_deposits()
        if created:
            # Advance only the legs detection just created — resumed legs
            # already had their turn this tick.
            for leg in list_active_deposits():
                if leg.id not in resumed:
                    self.process_deposit(leg)
        return created

    def detect_new_deposits(self) -> int:
        """Record deposit legs for new USDC found in user wallets."""
        created = 0
        for user in list_users():
            for chain in CCTP_CHAINS:
                created += self._detect_one(user, chain)
        return created

    def _detect_one(self, user, chain: str) -> int:
        # Guard 1: never re-detect while a leg from this chain is in flight.
        if has_active_source_deposit(user.wallet_address, chain):
            return 0
        # Guard 1b: never detect on a chain that an unfinished leg is bridging
        # TOWARD. A bridged mint can land on-chain before its status persists
        # (crash, DB error, RPC failure during resume); until that leg reaches
        # a recorded delivery, any new balance here could be its funds — and
        # recording them as a fresh local deposit would double-count the
        # amount into the delivered baseline. Real deposits are just picked
        # up a tick later, once the in-flight leg settles.
        if has_active_dest_deposit(user.wallet_address, chain):
            return 0
        balance = self._usdc_balance(chain, user.wallet_address)
        if balance is None:
            return 0  # RPC hiccup — try again next tick
        # Guard 2: on trading chains delivered funds STAY in the wallet, so
        # only the balance above what past legs already delivered is new.
        baseline = delivered_to_chain_usdc(user.wallet_address, chain)
        amount = round(balance - baseline, _USDC_DECIMALS)
        if amount <= 0:
            return 0
        if amount < self.settings.min_deposit_usdc:
            log.info(
                "deposit_below_minimum",
                user=user.telegram_user_id,
                chain=chain,
                amount_usdc=amount,
                min_usdc=self.settings.min_deposit_usdc,
                note="ignored — raise the deposit or lower BETBOT_MIN_DEPOSIT_USDC",
            )
            return 0

        base_amount = round(amount * self.settings.bridge_split_base, _USDC_DECIMALS)
        polygon_amount = round(amount - base_amount, _USDC_DECIMALS)
        created = 0
        for dest, leg_amount in (("polygon", polygon_amount), ("base", base_amount)):
            if leg_amount < 0.01:  # allocation dust — not worth a transaction
                continue
            status = MINTED if dest == chain else DETECTED
            deposit_id = create_deposit(
                user_id=user.id,
                wallet_address=user.wallet_address,
                source_chain=chain,
                dest_chain=dest,
                amount_usdc=leg_amount,
                balance_snapshot=balance,
                status=status,
            )
            log.info(
                "deposit_detected",
                deposit_id=deposit_id,
                user=user.telegram_user_id,
                source_chain=chain,
                dest_chain=dest,
                amount_usdc=leg_amount,
                local=dest == chain,
            )
            created += 1
        return created

    def process_deposit(self, leg) -> None:
        """Advance one leg as far as it can go this tick.

        Each step persists its status BEFORE the next begins, so a crash
        anywhere resumes exactly where it stopped. A failed step records the
        error and leaves the status untouched — retried next tick.
        """
        try:
            if leg.status == DETECTED:
                self._step_source_gas(leg)
            if leg.status == BURN_SUBMITTED:
                # A previous tick crashed/timed out between persisting the
                # burn hash and confirming the receipt — resolve before
                # anything else may run.
                self._resume_submitted_burn(leg)
            if leg.status == GAS_TOPPED_UP:
                self._step_burn(leg)
            if leg.status == BURNED:
                if not self._step_mint(leg):
                    return  # attestation not ready — resume next tick
            if leg.status == MINTED:
                self._step_venue_setup(leg)
        except Exception as e:  # noqa: BLE001 — one bad leg must not kill the scan
            log.error(
                "deposit_step_failed",
                deposit_id=leg.id,
                status=leg.status,
                source_chain=leg.source_chain,
                dest_chain=leg.dest_chain,
                error=str(e),
            )
            update_deposit(leg.id, error=str(e))

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _step_source_gas(self, leg) -> None:
        """The user wallet signs approve+burn on the source chain — fund it."""
        self._ensure_gas(leg.source_chain, leg.wallet_address)
        update_deposit(leg.id, status=GAS_TOPPED_UP)
        leg.status = GAS_TOPPED_UP

    def _step_burn(self, leg) -> None:
        """Approve USDC to the TokenMessenger and burn it toward dest.

        Crash-safety ordering: the burn tx is SIGNED first, its hash persisted
        (status BURN_SUBMITTED), and only then broadcast. If the receipt wait
        times out or the daemon dies after broadcast, the mined burn's hash is
        already on the leg — recovery (:meth:`_resume_submitted_burn`) checks
        that hash instead of ever re-running the burn blindly, which is what
        prevents both the stranded-deposit and the double-burn failure modes.
        """
        cfg = CCTP_CHAINS[leg.source_chain]
        units = int(round(leg.amount_usdc * 10**_USDC_DECIMALS))
        # Defense in depth: never burn unless the wallet demonstrably still
        # holds the leg's amount. A balance below the leg amount means an
        # earlier (unrecorded) burn already consumed it — re-burning would
        # either revert forever or, worse, consume a SECOND deposit.
        balance = self._usdc_balance(leg.source_chain, leg.wallet_address)
        if balance is None:
            raise BridgeError(
                f"burn aborted for deposit {leg.id}: balance read failed"
            )
        if int(round(balance * 10**_USDC_DECIMALS)) < units:
            raise BridgeError(
                f"burn aborted for deposit {leg.id}: source balance {balance}"
                f" below leg amount {leg.amount_usdc} — possible unrecorded burn"
            )
        acct = self._user_account(leg)
        if self._allowance(
            leg.source_chain, cfg["usdc"], leg.wallet_address, cfg["token_messenger"]
        ) < units:
            tx = self._approve_erc20(
                leg.source_chain, acct, cfg["usdc"], cfg["token_messenger"], units
            )
            log.info(
                "deposit_usdc_approved",
                deposit_id=leg.id,
                chain=leg.source_chain,
                spender="TokenMessengerV2",
                tx=tx,
            )
        signed = self._sign_deposit_for_burn(
            leg.source_chain,
            acct,
            amount_units=units,
            dest_domain=CCTP_CHAINS[leg.dest_chain]["domain"],
            mint_recipient=leg.wallet_address,
        )
        # Persist the hash BEFORE broadcasting — the one ordering that makes
        # the burn step idempotent across any crash/timeout window.
        update_deposit(leg.id, status=BURN_SUBMITTED, burn_tx=signed.tx_hash)
        leg.status, leg.burn_tx = BURN_SUBMITTED, signed.tx_hash
        log.info(
            "deposit_burn_submitted",
            deposit_id=leg.id,
            source_chain=leg.source_chain,
            tx=signed.tx_hash,
        )
        self._broadcast_signed(leg.source_chain, signed)
        update_deposit(leg.id, status=BURNED)
        leg.status = BURNED
        log.info(
            "deposit_burned",
            deposit_id=leg.id,
            source_chain=leg.source_chain,
            dest_chain=leg.dest_chain,
            amount_usdc=leg.amount_usdc,
            tx=signed.tx_hash,
        )

    def _resume_submitted_burn(self, leg) -> None:
        """Resolve a leg stranded at BURN_SUBMITTED (crash/timeout window).

        The persisted hash is the source of truth:

        - receipt mined + success → the burn happened; advance to BURNED so
          the attestation for THIS hash gets fetched (funds never stranded).
        - receipt mined + reverted → funds untouched; retry the burn step.
        - no receipt → disambiguate via the wallet's USDC balance: funds
          still present means the broadcast never landed (safe to re-sign —
          the account nonce is unchanged, so a re-send replaces rather than
          duplicates); funds gone means the burn almost certainly mined and
          the RPC is lagging — keep waiting, NEVER re-burn.
        """
        status = self._tx_status(leg.source_chain, leg.burn_tx)
        if status is True:
            update_deposit(leg.id, status=BURNED)
            leg.status = BURNED
            log.info(
                "deposit_burn_confirmed", deposit_id=leg.id, tx=leg.burn_tx
            )
            return
        if status is False:
            update_deposit(leg.id, status=GAS_TOPPED_UP)
            leg.status = GAS_TOPPED_UP
            log.warning(
                "deposit_burn_reverted_retrying",
                deposit_id=leg.id,
                tx=leg.burn_tx,
            )
            return
        units = int(round(leg.amount_usdc * 10**_USDC_DECIMALS))
        balance = self._usdc_balance(leg.source_chain, leg.wallet_address)
        if (
            balance is not None
            and int(round(balance * 10**_USDC_DECIMALS)) >= units
        ):
            update_deposit(leg.id, status=GAS_TOPPED_UP)
            leg.status = GAS_TOPPED_UP
            log.warning(
                "deposit_burn_not_found_retrying",
                deposit_id=leg.id,
                tx=leg.burn_tx,
                note="funds still in wallet — broadcast never landed",
            )
        else:
            log.info(
                "deposit_burn_receipt_pending",
                deposit_id=leg.id,
                tx=leg.burn_tx,
                note="funds left the wallet — waiting for the receipt, not re-burning",
            )

    def _step_mint(self, leg) -> bool:
        """Relay the attested mint on the destination. False = not ready yet.

        The AGENT wallet relays ``receiveMessage`` (the burn left
        ``destinationCaller`` open), so the user wallet needs no destination
        gas just to receive its funds.
        """
        att = self._fetch_attestation(leg.source_chain, leg.burn_tx)
        if att is None:
            log.info(
                "deposit_attestation_pending",
                deposit_id=leg.id,
                burn_tx=leg.burn_tx,
                note="Circle standard-transfer attestation takes minutes; retrying next tick",
            )
            return False
        try:
            mint_tx = self._receive_message(leg.dest_chain, att)
        except Exception as e:  # noqa: BLE001 — replays revert with "Nonce already used"
            # NEVER trust error text: "nonce too low/high" are ordinary
            # ACCOUNT-nonce errors from the agent wallet's own submission
            # (routine on load-balanced public RPCs) and have nothing to do
            # with the CCTP message nonce. Instead VERIFY on-chain: extract
            # the message nonce and ask the MessageTransmitter's replay
            # registry whether it was consumed. Only a confirmed
            # already-relayed mint may advance the leg.
            if self._message_nonce_consumed(leg.dest_chain, att):
                log.warning(
                    "deposit_mint_already_relayed", deposit_id=leg.id, error=str(e)
                )
                mint_tx = None
            else:
                raise
        update_deposit(leg.id, status=MINTED, mint_tx=mint_tx)
        leg.status, leg.mint_tx = MINTED, mint_tx
        log.info(
            "deposit_minted",
            deposit_id=leg.id,
            dest_chain=leg.dest_chain,
            amount_usdc=leg.amount_usdc,
            tx=mint_tx,
        )
        return True

    def _step_venue_setup(self, leg) -> None:
        """Funds have landed — make the wallet tradeable on the venue."""
        acct = self._user_account(leg)
        if leg.dest_chain == "polygon":
            # Approvals are user-signed transactions on Polygon → needs POL.
            self._ensure_gas("polygon", leg.wallet_address)
            self._approve_polymarket(acct)
            log.info(
                "deposit_polymarket_approved",
                deposit_id=leg.id,
                wallet=leg.wallet_address,
            )
        elif leg.dest_chain == "base" and self.settings.bridge_split_base > 0:
            self._ensure_gas("base", leg.wallet_address)
            self._approve_limitless(acct)
            log.info(
                "deposit_limitless_approved",
                deposit_id=leg.id,
                wallet=leg.wallet_address,
            )
        update_deposit(leg.id, status=DONE)
        leg.status = DONE
        log.info(
            "deposit_done",
            deposit_id=leg.id,
            dest_chain=leg.dest_chain,
            amount_usdc=leg.amount_usdc,
        )

    def _ensure_gas(self, chain: str, address: str) -> None:
        """Top the wallet up with native gas from the agent wallet if low.

        Capped per wallet per UTC day (``BETBOT_GAS_TOPUP_DAILY_CAP``): the
        bot is public, so agent-wallet gas spend driven by third-party
        deposits must be bounded. Hitting the cap raises — the leg stays put
        and resumes once the day rolls over (no funds at risk, only delayed).
        """
        amount = (
            self.settings.gas_topup_pol
            if chain == "polygon"
            else self.settings.gas_topup_eth
        )
        needed_wei = int(amount * _WEI)
        have_wei = self._native_balance_wei(chain, address)
        if have_wei >= needed_wei:
            log.debug("gas_topup_skipped", chain=chain, wallet=address)
            return
        cap = self.settings.gas_topup_daily_cap
        if cap > 0:
            day_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if count_gas_topups_since(address, day_start) >= cap:
                raise BridgeError(
                    f"gas top-up daily cap ({cap}) reached for {address};"
                    " leg resumes after the UTC day rolls over"
                )
        tx = self._send_native(chain, address, needed_wei)
        record_gas_topup(wallet_address=address, chain=chain, amount=amount, tx=tx)
        log.info(
            "gas_topup_sent",
            chain=chain,
            to=address,
            amount=amount,
            symbol=CCTP_CHAINS[chain]["gas"],
            tx=tx,
        )

    # ------------------------------------------------------------------
    # Chain primitives (overridden with fakes in tests)
    # ------------------------------------------------------------------
    def _w3(self, chain: str):
        if chain not in self._w3_cache:
            from web3 import Web3
            from web3.middleware import ExtraDataToPOAMiddleware

            w3 = Web3(
                Web3.HTTPProvider(
                    _rpc_for(chain, self.settings), request_kwargs={"timeout": 30}
                )
            )
            # Polygon PoS puts >32 bytes in each block's extraData, which web3's
            # default block formatter rejects — fatal when waiting for a
            # receiveMessage receipt (the mint leg). This middleware trims it;
            # it's a no-op on chains whose extraData is already 32 bytes, so
            # injecting it on every chain is safe.
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            self._w3_cache[chain] = w3
        return self._w3_cache[chain]

    def _agent_account(self):
        from betbot.wallet import get_private_key

        key = get_private_key(self.settings.wallet_keyfile)
        if not key:
            raise BridgeError(
                f"agent wallet keyfile missing: {self.settings.wallet_keyfile}"
            )
        from eth_account import Account

        return Account.from_key(key)

    def _user_account(self, leg):
        from betbot.storage.db import session_scope
        from betbot.storage.models import User
        from betbot.wallet import get_private_key

        with session_scope() as s:
            user = s.get(User, leg.user_id)
            keyfile = user.wallet_keyfile if user else None
        key = get_private_key(keyfile) if keyfile else None
        if not key:
            raise BridgeError(f"user keyfile missing for deposit {leg.id}")
        from eth_account import Account

        return Account.from_key(key)

    def _sign_tx(self, chain: str, acct, tx: dict) -> SignedTx:
        w3 = self._w3(chain)
        tx.setdefault("nonce", w3.eth.get_transaction_count(acct.address))
        tx.setdefault("chainId", CCTP_CHAINS[chain]["chain_id"])
        signed = acct.sign_transaction(tx)
        # Normalise to the canonical 0x form: the hash is persisted (burn_tx)
        # and fed to Circle's attestation API + receipt lookups.
        tx_hash = signed.hash.hex()
        if not tx_hash.startswith("0x"):
            tx_hash = "0x" + tx_hash
        return SignedTx(tx_hash=tx_hash, raw=bytes(signed.raw_transaction))

    def _broadcast_signed(self, chain: str, signed: SignedTx) -> None:
        """Broadcast a signed tx and wait for a successful receipt."""
        w3 = self._w3(chain)
        w3.eth.send_raw_transaction(signed.raw)
        receipt = w3.eth.wait_for_transaction_receipt(signed.tx_hash)
        if receipt.get("status") != 1:
            raise BridgeError(
                f"transaction reverted on {chain}: {signed.tx_hash}"
            )

    def _sign_and_send(self, chain: str, acct, tx: dict) -> str:
        signed = self._sign_tx(chain, acct, tx)
        self._broadcast_signed(chain, signed)
        return signed.tx_hash

    def _tx_status(self, chain: str, tx_hash: str) -> bool | None:
        """Receipt verdict for a tx: True mined+ok, False reverted, None unknown.

        ``None`` (not found / RPC failure) deliberately tells the caller
        nothing — burn recovery treats it as "cannot prove anything" and
        falls back to the balance check rather than guessing.
        """
        try:
            receipt = self._w3(chain).eth.get_transaction_receipt(tx_hash)
        except Exception as e:  # noqa: BLE001 — includes TransactionNotFound
            log.debug("tx_receipt_unavailable", tx=tx_hash, error=str(e))
            return None
        if receipt is None:
            return None
        return receipt.get("status") == 1

    def _message_nonce_consumed(self, dest_chain: str, att: Attestation) -> bool:
        """True iff the MessageTransmitter consumed this message's nonce.

        On-chain ground truth for "the mint already executed". Any failure
        (bad RPC, malformed message) returns False so the original mint
        error propagates and the leg is retried — never silently advanced.
        """
        from web3 import Web3

        try:
            nonce = _extract_cctp_nonce(att.message)
            c = self._w3(dest_chain).eth.contract(
                address=Web3.to_checksum_address(
                    CCTP_CHAINS[dest_chain]["message_transmitter"]
                ),
                abi=_MESSAGE_TRANSMITTER_V2_ABI,
            )
            return int(c.functions.usedNonces(nonce).call()) == 1
        except Exception as e:  # noqa: BLE001 — verification must fail closed
            log.warning("nonce_verification_failed", error=str(e))
            return False

    def _usdc_balance(self, chain: str, address: str) -> float | None:
        from web3 import Web3

        try:
            w3 = self._w3(chain)
            c = w3.eth.contract(
                address=Web3.to_checksum_address(CCTP_CHAINS[chain]["usdc"]),
                abi=_ERC20_ABI,
            )
            raw = c.functions.balanceOf(Web3.to_checksum_address(address)).call()
            return raw / 10**_USDC_DECIMALS
        except Exception as e:  # noqa: BLE001 — RPC flakiness shouldn't kill the scan
            log.warning("deposit_balance_read_failed", chain=chain, error=str(e))
            return None

    def _native_balance_wei(self, chain: str, address: str) -> int:
        from web3 import Web3

        return self._w3(chain).eth.get_balance(Web3.to_checksum_address(address))

    def _send_native(self, chain: str, to_address: str, amount_wei: int) -> str:
        from web3 import Web3

        agent = self._agent_account()
        w3 = self._w3(chain)
        return self._sign_and_send(
            chain,
            agent,
            {
                "from": agent.address,
                "to": Web3.to_checksum_address(to_address),
                "value": amount_wei,
                "gas": 21000,
                "gasPrice": w3.eth.gas_price,
            },
        )

    def _allowance(self, chain: str, token: str, owner: str, spender: str) -> int:
        from web3 import Web3

        c = self._w3(chain).eth.contract(
            address=Web3.to_checksum_address(token), abi=_ERC20_ABI
        )
        return c.functions.allowance(
            Web3.to_checksum_address(owner), Web3.to_checksum_address(spender)
        ).call()

    def _approve_erc20(
        self, chain: str, acct, token: str, spender: str, amount_units: int
    ) -> str:
        from web3 import Web3

        c = self._w3(chain).eth.contract(
            address=Web3.to_checksum_address(token), abi=_ERC20_ABI
        )
        tx = c.functions.approve(
            Web3.to_checksum_address(spender), amount_units
        ).build_transaction({"from": acct.address})
        return self._sign_and_send(chain, acct, tx)

    def _sign_deposit_for_burn(
        self,
        chain: str,
        acct,
        *,
        amount_units: int,
        dest_domain: int,
        mint_recipient: str,
    ) -> SignedTx:
        """Build + sign (NOT send) the depositForBurn — the caller persists
        the hash first, then broadcasts. See _step_burn for why."""
        from web3 import Web3

        cfg = CCTP_CHAINS[chain]
        c = self._w3(chain).eth.contract(
            address=Web3.to_checksum_address(cfg["token_messenger"]),
            abi=_TOKEN_MESSENGER_V2_ABI,
        )
        tx = c.functions.depositForBurn(
            amount_units,
            dest_domain,
            _address_to_bytes32(mint_recipient),
            Web3.to_checksum_address(cfg["usdc"]),
            b"\x00" * 32,  # destinationCaller open: the agent wallet relays
            _MAX_FEE_STANDARD,
            _MIN_FINALITY_STANDARD,
        ).build_transaction({"from": acct.address})
        return self._sign_tx(chain, acct, tx)

    def _fetch_attestation(self, source_chain: str, burn_tx: str) -> Attestation | None:
        """Poll Circle's attestation API once. None = not attested yet."""
        import httpx

        domain = CCTP_CHAINS[source_chain]["domain"]
        url = (
            f"{self.settings.iris_api_url}/v2/messages/{domain}"
            f"?transactionHash={burn_tx}"
        )
        try:
            r = httpx.get(url, timeout=20)
            if r.status_code == 404:
                return None  # not indexed yet
            r.raise_for_status()
            for m in r.json().get("messages") or []:
                att = m.get("attestation")
                if m.get("status") == "complete" and att and att != "PENDING":
                    return Attestation(message=m["message"], attestation=att)
        except Exception as e:  # noqa: BLE001 — attestation polling is retried
            log.warning("attestation_poll_failed", burn_tx=burn_tx, error=str(e))
        return None

    def _receive_message(self, dest_chain: str, att: Attestation) -> str:
        from web3 import Web3

        agent = self._agent_account()
        c = self._w3(dest_chain).eth.contract(
            address=Web3.to_checksum_address(
                CCTP_CHAINS[dest_chain]["message_transmitter"]
            ),
            abi=_MESSAGE_TRANSMITTER_V2_ABI,
        )
        tx = c.functions.receiveMessage(
            bytes.fromhex(att.message.removeprefix("0x")),
            bytes.fromhex(att.attestation.removeprefix("0x")),
        ).build_transaction({"from": agent.address})
        return self._sign_and_send(dest_chain, agent, tx)

    def _approve_polymarket(self, acct) -> None:
        """Run the full Polymarket approval set for this user wallet.

        Reuses ``scripts/polymarket_approve.approve_for_account`` (the
        single source of truth for which spenders need which approvals) —
        loaded by path because ``scripts/`` is not a package. Idempotent:
        already-approved spenders are skipped inside.
        """
        mod = _load_script("polymarket_approve")
        mod.approve_for_account(self._w3("polygon"), acct, confirm=True)

    def _approve_limitless(self, acct) -> None:
        """USDC approval for the Limitless CTF Exchange on Base.

        Mirrors scripts/limitless_approve.py. Only the USDC leg: the CTF
        (ERC-1155) operator approval needs the LIMITLESS_CTF address, which
        is deliberately not defaulted anywhere — and Limitless live orders
        are parked, so the USDC allowance is all the future share needs now.
        """
        exchange = self.settings.limitless_exchange
        if not exchange:
            log.warning(
                "limitless_approval_skipped",
                note="LIMITLESS_EXCHANGE unset — set it before un-parking Base",
            )
            return
        max_uint = 2**256 - 1
        usdc = CCTP_CHAINS["base"]["usdc"]
        if self._allowance("base", usdc, acct.address, exchange) >= max_uint // 2:
            return
        tx = self._approve_erc20("base", acct, usdc, exchange, max_uint)
        log.info("limitless_usdc_approved", wallet=acct.address, tx=tx)


class TreasuryRebalancer(DepositPipeline):
    """Positions the AGENT treasury's own USDC AHEAD of trades.

    The deposit pipeline covers every registered USER wallet, but the
    operator's own seed float (the agent wallet — same address on Polygon and
    Base, key at ``settings.wallet_keyfile``) was not covered, forcing a manual
    ``scripts/bridge_agent_funds.py`` run to position trading funds. This keeps
    target balances on both trading chains topped up in the background so no
    manual bridge is ever needed.

    Why a background job and not a bridge-at-trade-time: CCTP standard-transfer
    attestation takes ~15 minutes, and a trade can't wait that long. Funds are
    moved AHEAD of demand, toward configured per-chain targets.

    It reuses ``DepositPipeline``'s chain primitives (``_usdc_balance``,
    ``_sign_deposit_for_burn``, ``_broadcast_signed``, ``_tx_status``,
    ``_fetch_attestation``, ``_message_nonce_consumed``, ``_receive_message``,
    ``_ensure_gas``, ``_agent_account``, ``CCTP_CHAINS``) — the agent wallet is
    BOTH the burner and the mint relayer (it pays its own gas). Crash-safety
    mirrors the deposit burn exactly (hash persisted before broadcast, verified
    on resume; mint verified via the on-chain nonce registry).

    Idempotency: at most ONE in-flight treasury bridge at a time, enforced by a
    single non-``done`` row in the ``treasury_bridges`` table. A second tick
    while a bridge is active resumes that leg rather than starting another.
    """

    # Only the two trading chains are rebalanced. (Polygon = Polymarket live;
    # Base = Limitless once un-parked.)
    _TRADING_CHAINS = ("polygon", "base")

    def rebalance_agent_treasury(self) -> None:
        """One rebalance tick: resume an in-flight leg, else open one if needed.

        Master gates: ``treasury_auto_rebalance`` AND ``auto_bridge`` (the
        on-chain master kill). Either being false disables ALL action here,
        including resuming a leg that was already in flight when the gate
        flipped — same freeze semantics as the deposit pipeline.
        """
        if not self.settings.auto_bridge:
            log.info(
                "treasury_rebalance_disabled", note="BETBOT_AUTO_BRIDGE=false"
            )
            return
        if not self.settings.treasury_auto_rebalance:
            log.info(
                "treasury_rebalance_disabled",
                note="BETBOT_TREASURY_AUTO_REBALANCE=false",
            )
            return

        active = active_treasury_bridge()
        if active is not None:
            # One in flight already — resume it, never start a second.
            self._process_treasury_leg(active)
            return

        plan = self._plan_rebalance()
        if plan is None:
            return  # balanced (or unreadable) — _plan_rebalance logged why
        source, dest, amount = plan
        bridge_id = create_treasury_bridge(
            source_chain=source,
            dest_chain=dest,
            amount_usdc=amount,
            status=BURNED if source == dest else DETECTED,
        )
        log.info(
            "treasury_rebalance_started",
            bridge_id=bridge_id,
            source_chain=source,
            dest_chain=dest,
            amount_usdc=amount,
        )
        self._process_treasury_leg(active_treasury_bridge())

    def _plan_rebalance(self) -> tuple[str, str, float] | None:
        """Decide the single bridge to run this tick, or None if balanced.

        Reads agent USDC on both trading chains, compares to targets, and — if
        one chain is short by MORE than the threshold while the other holds a
        surplus above its own target — returns
        ``(source, dest, amount)`` with the amount = the deficit, capped at the
        surplus. Dust (deficit <= threshold) is ignored. Never bridges more
        than the surplus, and never below the threshold.
        """
        agent = self._agent_account().address
        targets = {
            "polygon": self.settings.treasury_target_polygon_usd,
            "base": self.settings.treasury_target_base_usd,
        }
        threshold = self.settings.treasury_rebalance_threshold_usd
        balances: dict[str, float] = {}
        for chain in self._TRADING_CHAINS:
            bal = self._usdc_balance(chain, agent)
            if bal is None:
                log.warning(
                    "treasury_balance_read_failed",
                    chain=chain,
                    note="skipping rebalance this tick",
                )
                return None
            balances[chain] = bal

        deficits = {c: round(targets[c] - balances[c], _USDC_DECIMALS) for c in balances}
        surpluses = {c: round(balances[c] - targets[c], _USDC_DECIMALS) for c in balances}

        # The most-underfunded chain that is short by MORE than the threshold.
        dest = max(deficits, key=lambda c: deficits[c])
        if deficits[dest] <= threshold:
            log.info(
                "treasury_balanced",
                polygon_usd=round(balances["polygon"], _USDC_DECIMALS),
                base_usd=round(balances["base"], _USDC_DECIMALS),
                target_polygon=targets["polygon"],
                target_base=targets["base"],
                threshold=threshold,
                note="nothing to do",
            )
            return None

        # Pull from the chain with the largest surplus above its own target.
        source = max(surpluses, key=lambda c: surpluses[c])
        if source == dest or surpluses[source] <= 0:
            log.info(
                "treasury_deficit_without_surplus",
                deficit_chain=dest,
                deficit_usd=deficits[dest],
                note="no surplus chain to fund it — fund the agent wallet",
            )
            return None

        amount = round(min(deficits[dest], surpluses[source]), _USDC_DECIMALS)
        if amount < threshold:
            # The surplus can only cover dust — not worth a bridge.
            log.info(
                "treasury_surplus_below_threshold",
                source_chain=source,
                dest_chain=dest,
                fundable_usd=amount,
                threshold=threshold,
                note="surplus too small to bridge — nothing to do",
            )
            return None
        log.info(
            "treasury_rebalance_planned",
            source_chain=source,
            dest_chain=dest,
            amount_usdc=amount,
            deficit_usd=deficits[dest],
            surplus_usd=surpluses[source],
        )
        return source, dest, amount

    def _process_treasury_leg(self, leg) -> None:
        """Advance one treasury leg as far as it can this tick.

        Mirrors :meth:`DepositPipeline.process_deposit` but with NO user gas
        top-up (the agent funds its own source-chain gas) and NO venue setup
        (the treasury isn't a tradeable user wallet). A failed step records the
        error and leaves the status put — retried next tick.
        """
        if leg is None:
            return
        try:
            if leg.status == DETECTED:
                self._treasury_step_gas(leg)
            if leg.status == BURN_SUBMITTED:
                self._treasury_resume_submitted_burn(leg)
            if leg.status == GAS_TOPPED_UP:
                self._treasury_step_burn(leg)
            if leg.status == BURNED:
                if not self._treasury_step_mint(leg):
                    return  # attestation not ready — resume next tick
            if leg.status == MINTED:
                update_treasury_bridge(leg.id, status=DONE)
                leg.status = DONE
                log.info(
                    "treasury_rebalance_done",
                    bridge_id=leg.id,
                    source_chain=leg.source_chain,
                    dest_chain=leg.dest_chain,
                    amount_usdc=leg.amount_usdc,
                )
        except Exception as e:  # noqa: BLE001 — one bad leg must not kill the scan
            log.error(
                "treasury_step_failed",
                bridge_id=leg.id,
                status=leg.status,
                source_chain=leg.source_chain,
                dest_chain=leg.dest_chain,
                error=str(e),
            )
            update_treasury_bridge(leg.id, error=str(e))

    def _treasury_step_gas(self, leg) -> None:
        """Fund the agent wallet's own source-chain gas for approve+burn."""
        agent = self._agent_account().address
        self._ensure_gas(leg.source_chain, agent)
        update_treasury_bridge(leg.id, status=GAS_TOPPED_UP)
        leg.status = GAS_TOPPED_UP

    def _treasury_step_burn(self, leg) -> None:
        """Approve + burn the agent's USDC toward the deficit chain.

        Same crash-safety ordering as the deposit burn: sign, persist the hash
        (``burn_submitted``), THEN broadcast — so a receipt timeout or crash in
        the broadcast window never loses the hash and re-burns. Defense in
        depth: never burn unless the agent demonstrably still holds the amount
        (a lower balance means an earlier unrecorded burn already consumed it).
        """
        cfg = CCTP_CHAINS[leg.source_chain]
        agent = self._agent_account()
        units = int(round(leg.amount_usdc * 10**_USDC_DECIMALS))
        balance = self._usdc_balance(leg.source_chain, agent.address)
        if balance is None:
            raise BridgeError(
                f"treasury burn aborted for bridge {leg.id}: balance read failed"
            )
        if int(round(balance * 10**_USDC_DECIMALS)) < units:
            raise BridgeError(
                f"treasury burn aborted for bridge {leg.id}: source balance"
                f" {balance} below leg amount {leg.amount_usdc} —"
                " possible unrecorded burn"
            )
        if self._allowance(
            leg.source_chain, cfg["usdc"], agent.address, cfg["token_messenger"]
        ) < units:
            tx = self._approve_erc20(
                leg.source_chain, agent, cfg["usdc"], cfg["token_messenger"], units
            )
            log.info(
                "treasury_usdc_approved",
                bridge_id=leg.id,
                chain=leg.source_chain,
                tx=tx,
            )
        signed = self._sign_deposit_for_burn(
            leg.source_chain,
            agent,
            amount_units=units,
            dest_domain=CCTP_CHAINS[leg.dest_chain]["domain"],
            mint_recipient=agent.address,  # agent -> agent, cross-chain
        )
        update_treasury_bridge(leg.id, status=BURN_SUBMITTED, burn_tx=signed.tx_hash)
        leg.status, leg.burn_tx = BURN_SUBMITTED, signed.tx_hash
        log.info(
            "treasury_burn_submitted",
            bridge_id=leg.id,
            source_chain=leg.source_chain,
            tx=signed.tx_hash,
        )
        self._broadcast_signed(leg.source_chain, signed)
        update_treasury_bridge(leg.id, status=BURNED)
        leg.status = BURNED
        log.info(
            "treasury_burned",
            bridge_id=leg.id,
            source_chain=leg.source_chain,
            dest_chain=leg.dest_chain,
            amount_usdc=leg.amount_usdc,
            tx=signed.tx_hash,
        )

    def _treasury_resume_submitted_burn(self, leg) -> None:
        """Resolve a treasury leg stranded at ``burn_submitted``.

        Identical reasoning to :meth:`DepositPipeline._resume_submitted_burn`:
        the persisted hash is the source of truth (mined+ok -> BURNED;
        reverted -> retry; no receipt -> disambiguate via the agent's USDC
        balance, never re-burning when the funds already left the wallet).
        """
        agent = self._agent_account().address
        status = self._tx_status(leg.source_chain, leg.burn_tx)
        if status is True:
            update_treasury_bridge(leg.id, status=BURNED)
            leg.status = BURNED
            log.info("treasury_burn_confirmed", bridge_id=leg.id, tx=leg.burn_tx)
            return
        if status is False:
            update_treasury_bridge(leg.id, status=GAS_TOPPED_UP)
            leg.status = GAS_TOPPED_UP
            log.warning(
                "treasury_burn_reverted_retrying", bridge_id=leg.id, tx=leg.burn_tx
            )
            return
        units = int(round(leg.amount_usdc * 10**_USDC_DECIMALS))
        balance = self._usdc_balance(leg.source_chain, agent)
        if balance is not None and int(round(balance * 10**_USDC_DECIMALS)) >= units:
            update_treasury_bridge(leg.id, status=GAS_TOPPED_UP)
            leg.status = GAS_TOPPED_UP
            log.warning(
                "treasury_burn_not_found_retrying",
                bridge_id=leg.id,
                tx=leg.burn_tx,
                note="funds still in wallet — broadcast never landed",
            )
        else:
            log.info(
                "treasury_burn_receipt_pending",
                bridge_id=leg.id,
                tx=leg.burn_tx,
                note="funds left the wallet — waiting for the receipt, not re-burning",
            )

    def _treasury_step_mint(self, leg) -> bool:
        """Relay the attested mint on the destination. False = not ready yet.

        The AGENT wallet relays ``receiveMessage`` (it's both burner and
        relayer). A replay revert is verified against the on-chain nonce
        registry before advancing — never trusted from error text.
        """
        att = self._fetch_attestation(leg.source_chain, leg.burn_tx)
        if att is None:
            log.info(
                "treasury_attestation_pending",
                bridge_id=leg.id,
                burn_tx=leg.burn_tx,
                note="Circle standard-transfer attestation takes minutes; retrying next tick",
            )
            return False
        try:
            mint_tx = self._receive_message(leg.dest_chain, att)
        except Exception as e:  # noqa: BLE001 — replays revert with "Nonce already used"
            if self._message_nonce_consumed(leg.dest_chain, att):
                log.warning(
                    "treasury_mint_already_relayed", bridge_id=leg.id, error=str(e)
                )
                mint_tx = None
            else:
                raise
        update_treasury_bridge(leg.id, status=MINTED, mint_tx=mint_tx)
        leg.status, leg.mint_tx = MINTED, mint_tx
        log.info(
            "treasury_minted",
            bridge_id=leg.id,
            dest_chain=leg.dest_chain,
            amount_usdc=leg.amount_usdc,
            tx=mint_tx,
        )
        return True


def _load_script(name: str):
    """Import a module from scripts/ (not a package) by file path."""
    import importlib.util

    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_betbot_scripts_{name}", path)
    if spec is None or spec.loader is None:
        raise BridgeError(f"cannot load script {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _deposit_scan_sync(settings) -> int:
    """One tick: user deposit scan, THEN agent treasury rebalance.

    Both share the deposit-scan interval and the same worker thread. The
    treasury rebalance runs AFTER the user scan so one daemon job handles
    both; its own gates (``treasury_auto_rebalance`` + ``auto_bridge``) decide
    whether it does anything, and any failure inside it must not affect the
    user-scan result already returned.
    """
    created = DepositPipeline(settings).run_once()
    try:
        TreasuryRebalancer(settings).rebalance_agent_treasury()
    except Exception as e:  # noqa: BLE001 — treasury rebalance must not break the user scan
        log.error("treasury_rebalance_failed", error=str(e))
    return created


async def run_deposit_scan(settings) -> int:
    """Daemon entrypoint: one deposit-scan tick (see main.py wiring).

    The pipeline is synchronous web3 code, so it runs in a worker thread to
    keep the daemon's event loop responsive. The agent-treasury rebalance
    rides the same tick, AFTER the user deposit scan.
    """
    from betbot.storage.db import init_engine

    init_engine(settings.db_path)
    if not settings.auto_bridge:
        log.info("deposit_scan_disabled", note="BETBOT_AUTO_BRIDGE=false")
        return 0
    return await asyncio.to_thread(_deposit_scan_sync, settings)
