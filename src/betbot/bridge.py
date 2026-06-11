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

- **Gated**: ``BETBOT_AUTO_BRIDGE=false`` disables every on-chain action.
- **Idempotent**: every leg is a row in the ``deposits`` table with an
  explicit per-step status; a leg is never burned or approved twice, and a
  half-finished pipeline resumes from its last completed step on the next
  scan tick. See ``storage.models.Deposit`` for the dedupe model.
- **Explicit**: every on-chain step logs what it did (tx hashes included).

Operator note: the AGENT wallet pays for gas top-ups and mint relays — keep
it funded with a little POL + ETH on the chains you expect deposits on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from betbot.logging import get_logger
from betbot.storage.repos import (
    DEPOSIT_DONE,
    create_deposit,
    delivered_to_chain_usdc,
    has_active_source_deposit,
    list_active_deposits,
    list_users,
    update_deposit,
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
DETECTED = "detected"
GAS_TOPPED_UP = "gas_topped_up"
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
]


class BridgeError(RuntimeError):
    """A deposit-pipeline step failed; the leg stays put and is retried."""


@dataclass(frozen=True)
class Attestation:
    """A complete Circle attestation, ready for ``receiveMessage``."""

    message: str
    attestation: str


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
        """One scan tick: resume unfinished legs, then detect new deposits.

        Returns the number of NEW deposit legs recorded this tick.
        """
        if not self.settings.auto_bridge:
            log.info("deposit_scan_disabled", note="BETBOT_AUTO_BRIDGE=false")
            return 0
        created = self.detect_new_deposits()
        # Single pass over EVERY unfinished leg — both the ones just created
        # and the ones resuming from an earlier tick (e.g. awaiting Circle's
        # attestation, or whose last step failed).
        for leg in list_active_deposits():
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
        """Approve USDC to the TokenMessenger and burn it toward dest."""
        cfg = CCTP_CHAINS[leg.source_chain]
        units = int(round(leg.amount_usdc * 10**_USDC_DECIMALS))
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
        burn_tx = self._deposit_for_burn(
            leg.source_chain,
            acct,
            amount_units=units,
            dest_domain=CCTP_CHAINS[leg.dest_chain]["domain"],
            mint_recipient=leg.wallet_address,
        )
        update_deposit(leg.id, status=BURNED, burn_tx=burn_tx)
        leg.status, leg.burn_tx = BURNED, burn_tx
        log.info(
            "deposit_burned",
            deposit_id=leg.id,
            source_chain=leg.source_chain,
            dest_chain=leg.dest_chain,
            amount_usdc=leg.amount_usdc,
            tx=burn_tx,
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
        except Exception as e:  # noqa: BLE001 — replays revert with "nonce already used"
            if "nonce" in str(e).lower():
                # A previous tick's mint landed but we crashed before
                # recording it — the funds are there; advance.
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
        """Top the wallet up with native gas from the agent wallet if low."""
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
        tx = self._send_native(chain, address, needed_wei)
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

            self._w3_cache[chain] = Web3(
                Web3.HTTPProvider(
                    _rpc_for(chain, self.settings), request_kwargs={"timeout": 30}
                )
            )
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

    def _sign_and_send(self, chain: str, acct, tx: dict) -> str:
        w3 = self._w3(chain)
        tx.setdefault("nonce", w3.eth.get_transaction_count(acct.address))
        tx.setdefault("chainId", CCTP_CHAINS[chain]["chain_id"])
        signed = acct.sign_transaction(tx)
        h = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(h)
        if receipt.get("status") != 1:
            raise BridgeError(f"transaction reverted on {chain}: {h.hex()}")
        return h.hex()

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

    def _deposit_for_burn(
        self,
        chain: str,
        acct,
        *,
        amount_units: int,
        dest_domain: int,
        mint_recipient: str,
    ) -> str:
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
        return self._sign_and_send(chain, acct, tx)

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


async def run_deposit_scan(settings) -> int:
    """Daemon entrypoint: one deposit-scan tick (see main.py wiring).

    The pipeline is synchronous web3 code, so it runs in a worker thread to
    keep the daemon's event loop responsive.
    """
    from betbot.storage.db import init_engine

    init_engine(settings.db_path)
    if not settings.auto_bridge:
        log.info("deposit_scan_disabled", note="BETBOT_AUTO_BRIDGE=false")
        return 0
    pipeline = DepositPipeline(settings)
    return await asyncio.to_thread(pipeline.run_once)
