# Going live with real funds — read this fully first

**Honest status:** paper mode is solid and verified end-to-end. **Live order
*placement* is built but NOT verified** — I can't test real orders without real
money, so expect the first live order to possibly fail and need debugging.
Start with $1–2, not your whole balance.

## What's verified vs not

| Piece | Status |
|---|---|
| Discovery, pricing, scoring, settlement, P&L, kill switch | ✅ verified |
| Order signing (Limitless EIP-712) | ✅ unit-tested (sign→recover) |
| Limitless `POST /orders` payload shape | ⚠️ **unverified** — may need tweaks |
| Polymarket V2 order placement | ⚠️ **unverified** — see the gap below |
| Cross-venue arb math | ✅ tested; opportunities ⚠️ rare (sparse Limitless) |

**Polymarket V2 gap:** real placement needs API-credential derivation
(`create_or_derive_api_creds`) and the USDC→pUSD wrap + funder/proxy flow, which
the adapter does **not** fully do yet. Limitless (direct USDC on Base + EIP-712 +
`POST /orders`) is the simpler path to try first.

## Steps (small amounts!)

1. **Deposit** a small USDC test amount to the agent wallet — message the bot
   `/deposit` (address `0x608F1144C409E7de0d8164F5e942A390d3a53c0a`). Base is
   simpler than Polygon to start. `/balance` confirms arrival.
2. **Fund gas:** a little **ETH on Base** (Limitless orders + approvals) and/or
   **MATIC on Polygon** (Polymarket approvals). A couple dollars each.
3. **Approvals:**
   - Limitless: confirm the Base CTF address, then
     `LIMITLESS_CTF=0x... python scripts/limitless_approve.py --exchange <market.venue.exchange> --confirm`
   - Polymarket: `python scripts/polymarket_approve.py --confirm` (addresses pre-verified).
4. **`.env`** (the adapters auto-use the agent wallet key):
   ```
   BETBOT_MODE=live
   BETBOT_FIXED_STAKE_USD=1            # tiny while testing
   BETBOT_DAILY_EXPOSURE_CAP_USD=20    # tight
   BETBOT_ALLOW_INTERNATIONAL_LIVE=true   # only if you want WC live
   BETBOT_REQUIRE_GATE=false           # skip the paper-history gate (you accept the risk)
   BETBOT_INTERNATIONAL_BET_EVERY_MATCH=false   # true = bet every WC match (-EV, your call)
   ```
5. **Restart** services (`bash deploy/start-services.sh` or systemd) and run one
   cycle: `tfsm run-once`. Watch `/tmp/daemon.log` for `live_order_placed` or
   `live_order_failed` and debug from there.

## Safety nets that stay on
- **Kill switch** trips at 20% drawdown (lower `BETBOT_DRAWDOWN_KILL_PCT` to be stricter).
- **Daily exposure cap** limits per-day stake.
- Paper bets are still logged even if a live order fails.

## Reality check
WC every-match betting is **−EV** (the Qatar backtest lost 25–35%). Arbs are
**rare** (Limitless rarely lists the same match). Live trading here is an
experiment to run with money you can lose — not an investment.
