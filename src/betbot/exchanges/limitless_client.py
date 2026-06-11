"""Limitless Exchange REST client — discovery, orderbook + authed endpoints.

Limitless (https://api.limitless.exchange) runs on **Base mainnet** (chain id
8453) and is a fork of Polymarket's CTF Exchange. Football matches are modelled
as a ``group`` market carrying ``metadata.{homeTeam, awayTeam, sportType}`` plus
child ``single`` binary YES/NO markets (one per outcome: home, away, and —
rarely — draw). Per-match World Cup props also exist as standalone
``tradeType="clob"`` / ``marketType="single"`` markets (spreads, totals, BTTS)
with ``metadata.{fixtureId, homeTeam, awayTeam, externalSlug}``. Each market
has its own slug, ``tokens.{yes,no}``, ``prices``, and an orderbook at
``/markets/<slug>/orderbook`` (``{bids, asks}`` with price/size/side).

**Units (verified 2026-06-11 against the live API):** orderbook level sizes and
``settings.minSize`` are in OUTCOME-TOKEN SHARES with 6 decimals
(``"100000000"`` = 100 shares), NOT USDC. The dollar cost of a minimum order is
``min_shares * price``.

Auth: either an ``X-API-Key`` header, or HMAC headers
(``lmts-api-key``/``lmts-timestamp``/``lmts-signature``). A 401 surfaces as
:class:`LimitlessAuthError` (key invalid/revoked) so callers can degrade with a
clear operator message. US IPs get 403/451 — surfaced as
:class:`LimitlessGeoBlockedError` so the router can degrade to Polymarket-only
gracefully.
"""

from __future__ import annotations

from typing import Any

import httpx

from betbot.logging import get_logger

log = get_logger(__name__)

LIMITLESS_BASE = "https://api.limitless.exchange"
BASE_CHAIN_ID = 8453
# USDC on Base — Limitless collateral.
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


class LimitlessError(RuntimeError):
    """A Limitless API request failed."""


class LimitlessGeoBlockedError(LimitlessError):
    """The request was geo-blocked (HTTP 403/451) — e.g. a US IP."""


class LimitlessAuthError(LimitlessError):
    """The API key was rejected (HTTP 401) — invalid or revoked.

    The operator must regenerate the key in the Limitless app and set
    ``LIMITLESS_API_KEY`` (and ``LIMITLESS_API_SECRET`` for HMAC auth).
    """


class LimitlessClient:
    def __init__(
        self,
        base_url: str = LIMITLESS_BASE,
        *,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._api_key = api_key
        self._api_secret = api_secret

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        """Auth headers for an authed endpoint.

        With key + secret: scoped-API-token HMAC headers
        (lmts-api-key/timestamp/signature). Signing string:
        ``{timestamp}\\n{METHOD}\\n{path}\\n{body}`` HMAC-SHA256 with the
        base64-decoded secret; signature base64-encoded. With a key only: a
        plain ``X-API-Key`` header. Returns {} when no key is configured
        (read-only mode).
        """
        if not self._api_key:
            return {}
        if not self._api_secret:
            return {"X-API-Key": self._api_key}
        import base64
        import hashlib
        import hmac
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).isoformat()
        msg = f"{ts}\n{method.upper()}\n{path}\n{body}"
        try:
            key = base64.b64decode(self._api_secret)
        except Exception:  # noqa: BLE001 — fall back to raw bytes if not base64
            key = self._api_secret.encode()
        sig = base64.b64encode(hmac.new(key, msg.encode(), hashlib.sha256).digest()).decode()
        return {"lmts-api-key": self._api_key, "lmts-timestamp": ts, "lmts-signature": sig}

    async def get_profile(self) -> dict[str, Any]:
        """GET /profiles/me (authed).

        The numeric ``id`` is the order ownerId; ``rank.feeRateBps`` is the fee
        rate (basis points) to embed in signed orders. Raises
        :class:`LimitlessAuthError` on 401 (key invalid/revoked).
        """
        url = f"{self._base_url}/profiles/me"
        try:
            resp = await self._client.get(
                url, headers=self._auth_headers("GET", "/profiles/me")
            )
        except httpx.HTTPError as e:
            raise LimitlessError(f"GET /profiles/me failed: {e}") from e
        self._raise_auth_or_geo("GET /profiles/me", resp)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LimitlessError(f"GET /profiles/me -> {resp.status_code}") from e
        data = resp.json()
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _raise_auth_or_geo(what: str, resp: httpx.Response) -> None:
        if resp.status_code == 401:
            raise LimitlessAuthError(
                f"{what} -> 401: Limitless API key invalid/revoked. Regenerate it "
                "in the Limitless app and set LIMITLESS_API_KEY (+ secret)."
            )
        if resp.status_code in (403, 451):
            raise LimitlessGeoBlockedError(
                f"{what} geo-blocked (HTTP {resp.status_code})"
            )

    async def __aenter__(self) -> "LimitlessClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        try:
            resp = await self._client.get(url, params=params)
        except httpx.HTTPError as e:
            raise LimitlessError(f"GET {path} failed: {e}") from e
        self._raise_auth_or_geo(f"GET {path}", resp)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LimitlessError(f"GET {path} -> {resp.status_code}") from e
        return resp.json()

    @staticmethod
    def _as_list(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "markets"):
                v = data.get(key)
                if isinstance(v, list):
                    return v
        return []

    # ------------------------------------------------------------------
    async def list_active_markets(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._as_list(await self._get("/markets/active", params={"limit": limit}))

    async def search_markets(self, query: str) -> list[dict[str, Any]]:
        return self._as_list(await self._get("/markets/search", params={"query": query}))

    async def get_market(self, slug: str) -> dict[str, Any]:
        data = await self._get(f"/markets/{slug}")
        return data if isinstance(data, dict) else {}

    async def get_orderbook(self, slug: str) -> dict[str, Any]:
        data = await self._get(f"/markets/{slug}/orderbook")
        return data if isinstance(data, dict) else {}

    async def post_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a signed order to ``/orders`` (live trading).

        CreateOrderDto (verified 2026-06-11): required ``order{salt, maker,
        signer, tokenId, makerAmount, takerAmount, feeRateBps, side(0=BUY,
        1=SELL), signature, signatureType}``, ``ownerId``, ``orderType``
        ("GTC"|"FAK"|"FOK"), ``marketSlug``; optional ``clientOrderId``
        (idempotency), ``timestamp``, ``recvWindow``.
        """
        import json as _json

        url = f"{self._base_url}/orders"
        body = _json.dumps(payload, separators=(",", ":"))
        headers = {"content-type": "application/json", **self._auth_headers("POST", "/orders", body)}
        try:
            resp = await self._client.post(url, content=body, headers=headers)
        except httpx.HTTPError as e:
            raise LimitlessError(f"POST /orders failed: {e}") from e
        self._raise_auth_or_geo("POST /orders", resp)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LimitlessError(f"POST /orders -> {resp.status_code}: {resp.text[:200]}") from e
        data = resp.json()
        return data if isinstance(data, dict) else {"raw": data}
