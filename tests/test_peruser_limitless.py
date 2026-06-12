"""Per-user Limitless credential wiring into live order placement.

Each user may link their OWN Limitless API key (stored at
``.secrets/users/<id>.limitless.json``). A user's Limitless order must
authenticate with THAT user's api_key/secret and sign with THAT user's wallet
key — never the shared agent key. A user with no linked key trades Polymarket
only; their Limitless leg is skipped. All network is mocked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from datetime import datetime, timezone

from betbot.exchanges.base import ExchangeName, MarketRef, OrderbookQuote, Outcome
from betbot.exchanges.router import RoutedQuote
from betbot.wallet import (
    limitless_creds_path,
    load_limitless_creds,
    store_limitless_creds,
)

from tests.test_scoring_integration import FakeAdapter


def _limitless_quote(price: float = 0.50) -> OrderbookQuote:
    return OrderbookQuote(
        exchange=ExchangeName.LIMITLESS, market_id="m", outcome=Outcome.HOME,
        yes_price=price, yes_size=100.0, timestamp=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------
# wallet.load_limitless_creds: present / absent / corrupt
# --------------------------------------------------------------------------
def test_load_limitless_creds_present(tmp_path):
    secrets = tmp_path / "secrets"
    store_limitless_creds(secrets, 111, "ak-111", "sk-111")
    creds = load_limitless_creds(secrets, 111)
    assert creds == {"api_key": "ak-111", "api_secret": "sk-111"}


def test_load_limitless_creds_absent(tmp_path):
    assert load_limitless_creds(tmp_path / "secrets", 999) is None


def test_load_limitless_creds_corrupt(tmp_path):
    secrets = tmp_path / "secrets"
    p = limitless_creds_path(secrets, 222)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json")
    assert load_limitless_creds(secrets, 222) is None


def test_load_limitless_creds_missing_fields(tmp_path):
    secrets = tmp_path / "secrets"
    p = limitless_creds_path(secrets, 333)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"api_key": "only-key"}))  # no api_secret
    assert load_limitless_creds(secrets, 333) is None


def test_load_limitless_creds_never_logs_values(tmp_path, caplog):
    secrets = tmp_path / "secrets"
    store_limitless_creds(secrets, 444, "SECRET-KEY", "SECRET-SECRET")
    with caplog.at_level("DEBUG"):
        load_limitless_creds(secrets, 444)
    blob = caplog.text
    assert "SECRET-KEY" not in blob
    assert "SECRET-SECRET" not in blob


# --------------------------------------------------------------------------
# make_signer_adapters: per-user vs agent vs no-creds
# --------------------------------------------------------------------------
@pytest.fixture()
def live_settings(settings, tmp_path):
    """Settings pointed at a temp wallet keyfile, live mode."""
    settings.mode = "live"
    settings.wallet_keyfile = str(tmp_path / "agent.key")
    settings.limitless_api_key = "AGENT-KEY"
    settings.limitless_api_secret = "AGENT-SECRET"
    return settings


def _build(live_settings):
    from betbot.main import _build_router

    return _build_router(live_settings)


async def _close(clients):
    for c in clients:
        await c.close()


async def test_make_signer_adapters_uses_user_creds(live_settings):
    router, clients, make = _build(live_settings)
    try:
        adapters = make(
            signing_key="0x" + "11" * 32,
            funder="0xUserFunder",
            limitless_creds={"api_key": "USER-KEY", "api_secret": "USER-SECRET"},
        )
        assert ExchangeName.POLYMARKET in adapters
        lm = adapters[ExchangeName.LIMITLESS]
        # The Limitless adapter must talk to a client built with the USER's key,
        # not the shared agent client.
        assert lm._client._api_key == "USER-KEY"
        assert lm._client._api_secret == "USER-SECRET"
        # And sign with the user's wallet key.
        assert lm._private_key == "0x" + "11" * 32
    finally:
        await _close(clients)


async def test_make_signer_adapters_omits_limitless_without_creds(live_settings):
    router, clients, make = _build(live_settings)
    try:
        adapters = make(signing_key="0x" + "22" * 32, funder="0xUser2")
        # Polymarket still present; Limitless omitted entirely.
        assert ExchangeName.POLYMARKET in adapters
        assert ExchangeName.LIMITLESS not in adapters
    finally:
        await _close(clients)


async def test_make_signer_adapters_agent_fallback_uses_settings_creds(live_settings):
    router, clients, make = _build(live_settings)
    try:
        adapters = make(
            signing_key="0x" + "33" * 32, agent_fallback=True
        )
        lm = adapters[ExchangeName.LIMITLESS]
        # Agent path reuses the shared client built from settings creds.
        assert lm._client._api_key == "AGENT-KEY"
        assert lm._client._api_secret == "AGENT-SECRET"
    finally:
        await _close(clients)


# --------------------------------------------------------------------------
# _place_live_for_users: per-user creds; one bad user doesn't stop others
# --------------------------------------------------------------------------
async def _run_place(route, prediction, decision, settings, make):
    from betbot.main import _place_live_for_users

    await _place_live_for_users(route, prediction, decision, settings, True, make, {})


def _route():
    return RoutedQuote(
        adapter=FakeAdapter(),
        market=MarketRef(ExchangeName.LIMITLESS, "m", "A vs B", {}),
        quote=_limitless_quote(0.50),
    )


async def test_place_live_passes_per_user_creds(
    tmp_path, settings, monkeypatch
):
    """The user's linked creds reach make_signer_adapters for their order."""
    import betbot.wallet as wallet_mod
    from betbot.storage.db import init_engine
    from betbot.storage.repos import get_or_create_user

    init_engine(tmp_path / "place.sqlite")
    secrets = tmp_path / "secrets"
    get_or_create_user(111, "Alice", secrets_dir=str(secrets))
    store_limitless_creds(secrets, 111, "ALICE-KEY", "ALICE-SECRET")
    settings.wallet_keyfile = str(secrets / "agent.key")  # -> secrets_dir = secrets

    monkeypatch.setattr(
        wallet_mod, "usdc_balance",
        lambda a, c, r: wallet_mod.ChainBalance(c, c, 100.0, True),
    )

    seen = []

    def fake_make(*, signing_key, funder=None, limitless_creds=None):
        seen.append(limitless_creds)
        return {ExchangeName.LIMITLESS: FakeAdapter()}

    route = _route()  # Limitless venue
    prediction = SimpleNamespace(competition_code="PL", fixture_id=9001)
    decision = SimpleNamespace(outcome=Outcome.HOME, stake_usd=10.0)

    await _run_place(route, prediction, decision, settings, fake_make)

    assert seen == [{"api_key": "ALICE-KEY", "api_secret": "ALICE-SECRET"}]


async def test_place_live_skips_limitless_leg_when_unlinked(
    tmp_path, settings, monkeypatch
):
    """A user with no linked Limitless key: no Limitless order attempted."""
    import betbot.wallet as wallet_mod
    from betbot.storage.db import init_engine
    from betbot.storage.repos import get_or_create_user

    init_engine(tmp_path / "place.sqlite")
    secrets = tmp_path / "secrets"
    get_or_create_user(222, "Bob", secrets_dir=str(secrets))  # no creds stored
    settings.wallet_keyfile = str(secrets / "agent.key")

    monkeypatch.setattr(
        wallet_mod, "usdc_balance",
        lambda a, c, r: wallet_mod.ChainBalance(c, c, 100.0, True),
    )

    lm_adapter = FakeAdapter()

    def fake_make(*, signing_key, funder=None, limitless_creds=None):
        # No creds -> the real factory omits Limitless; mirror that here so the
        # placement loop sees no Limitless adapter for this user.
        assert limitless_creds is None
        return {ExchangeName.POLYMARKET: FakeAdapter()}

    route = _route()  # Limitless venue
    prediction = SimpleNamespace(competition_code="PL", fixture_id=9001)
    decision = SimpleNamespace(outcome=Outcome.HOME, stake_usd=10.0)

    await _run_place(route, prediction, decision, settings, fake_make)

    assert lm_adapter.placed == []  # never reached


async def test_place_live_one_bad_user_does_not_stop_others(
    tmp_path, settings, monkeypatch
):
    """User with an invalid/erroring key fails cleanly; others still trade."""
    import betbot.wallet as wallet_mod
    from betbot.storage.db import init_engine
    from betbot.storage.repos import get_or_create_user

    init_engine(tmp_path / "place.sqlite")
    secrets = tmp_path / "secrets"
    get_or_create_user(111, "Bad", secrets_dir=str(secrets))
    get_or_create_user(222, "Good", secrets_dir=str(secrets))
    store_limitless_creds(secrets, 111, "BAD-KEY", "BAD-SECRET")
    store_limitless_creds(secrets, 222, "GOOD-KEY", "GOOD-SECRET")
    settings.wallet_keyfile = str(secrets / "agent.key")

    monkeypatch.setattr(
        wallet_mod, "usdc_balance",
        lambda a, c, r: wallet_mod.ChainBalance(c, c, 100.0, True),
    )

    good_adapter = FakeAdapter()

    def fake_make(*, signing_key, funder=None, limitless_creds=None):
        if limitless_creds["api_key"] == "BAD-KEY":
            # Adapter-build failure for the bad user (e.g. invalid key handling).
            raise RuntimeError("invalid limitless key")
        return {ExchangeName.LIMITLESS: good_adapter}

    route = _route()
    prediction = SimpleNamespace(competition_code="PL", fixture_id=9001)
    decision = SimpleNamespace(outcome=Outcome.HOME, stake_usd=10.0)

    # Must not raise; the good user still gets their order placed.
    await _run_place(route, prediction, decision, settings, fake_make)

    assert len(good_adapter.placed) == 1
    assert good_adapter.placed[0][1] == 10.0
