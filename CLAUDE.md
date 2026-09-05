# CLAUDE.md — The Football Smart Manager: Build Handover

> **For Claude Code.** This document is the single source of truth for continuing
> the build of *The Football Smart Manager* (`tfsm`), a multi-exchange football
> prediction-market betting bot. Read it fully before touching anything.
>
> The human ("the operator") has been building this over SSH into a VPS using
> `nano`, which is painful and error-prone. You are taking over because you can
> read/write files and run the build-test loop directly. Your first job is to
> get the repo into a clean, verified state; your second is to finish the
> remaining phases.

---

## 0. TL;DR — what to do first

1. Determine where the repo lives (see §2). It is on a DigitalOcean droplet at
   `~/tfsm` under user `tfsm`, OR the operator will point you at a local clone.
2. Get Phase 1 to a **green state**: every file compiles, `pytest` passes,
   `tfsm run-once` works against the live football-data.org API.
3. Initialise git properly and make the first commit if not already done.
4. **Set up a GitHub remote and push** — this ends the nano-over-SSH era.
5. Continue with Phases 2–9 (see §6), one commit per phase, pausing for operator
   review at each boundary.

Do not skip the verification gates in §5. The operator has already hit several
bugs that those gates would have caught (a `dateFrom`/`dateTo` API bug, two
fused-line paste errors). You have the tools to never repeat those.

---

## 1. What this project is

A bot that:
- Pulls upcoming football fixtures from **football-data.org** (free tier, 10 req/min).
- Computes each team's recent form (last 5 finished matches, recency-weighted,
  adjusted for opponent strength via league standings).
- Converts form into win/draw/loss probabilities via softmax.
- Looks up the matching market on **Polymarket** and **Limitless** prediction
  exchanges, picks the best price.
- Bets only when the model's probability exceeds the market's implied
  probability by a configurable edge (default 5%).
- Logs everything to SQLite, settles bets after matches finish, computes P&L,
  and trips a kill switch on excessive drawdown.
- Has a backtest harness and a live-mode "gate" that must pass before real money
  moves.
- Will get a FastAPI + React frontend (Phases 7–8) and a systemd/nginx
  deployment (Phase 9).

**Critical naming note:** the product is "The Football Smart Manager", the CLI is
`tfsm` (with `betbot` as a working alias), and the **Python package stays named
`betbot`** (`src/betbot/...`). Do NOT rename the package — it's the internal
codename and renaming it mid-build is pure churn. User-facing strings say "The
Football Smart Manager".

---

## 2. Environment & where things live

- **VPS**: DigitalOcean droplet, region Amsterdam (AMS3), Ubuntu 24.04.
  - IP referenced in conversation: `209.38.32.34` (confirm before relying on it).
  - User: `tfsm` (has passwordless or password sudo). Home: `/home/tfsm`.
  - Project root: `/home/tfsm/tfsm`.
  - Python: `python3.12` (3.12.3). Virtualenv at `~/tfsm/.venv`.
  - Installed system packages: `git`, `nginx`, `ufw` (active: allows OpenSSH +
    Nginx Full), `sqlite3`, `tmux`.
- **Why Amsterdam**: both Polymarket and Limitless geo-block many jurisdictions
  (Germany, France, US, etc.). Amsterdam/Netherlands is permitted. Do NOT move
  the deployment to a blocked region. (Hetzner was rejected because its main
  regions are in Germany, which is blocked.)
- **Activate the venv** before any Python work: `source ~/tfsm/.venv/bin/activate`.
- The operator works from a Mac and SSHes in. SSH keepalives are configured.
  Long-running processes should run under `tmux` (session name `tfsm`) until
  systemd is set up in Phase 9.

If you are operating on a local clone instead of the droplet, the same layout
applies relative to the repo root.

---

## 3. Current state (as of handover)

**Phase 1 is mostly done but NOT yet verified green.** Here's the precise state:

### Files that exist (created via a `setup_phase1.sh` bootstrap script)
```
pyproject.toml                       # hatchling build; deps; tfsm+betbot scripts
.gitignore
.env.example
.env                                 # operator created; has real FOOTBALL_DATA_API_KEY
README.md
src/betbot/__init__.py
src/betbot/config.py                 # Settings (pydantic-settings), LEAGUE_CODES
src/betbot/logging.py                # structlog
src/betbot/main.py                   # Typer CLI: run-once, run-daemon, bets list, init-db
src/betbot/data/__init__.py
src/betbot/data/models.py            # frozen dataclasses: Team, MatchResult, Fixture, etc.
src/betbot/data/football_data.py     # async httpx client + sliding-window rate limiter
src/betbot/data/form.py              # FormService (last-5 + opponent strength)
src/betbot/strategy/__init__.py
src/betbot/strategy/probabilities.py # pure math: softmax, edge, opponent_strength_factor
src/betbot/strategy/engine.py        # StrategyEngine, Prediction, BetDecision, Outcome
src/betbot/storage/__init__.py
src/betbot/storage/db.py             # SQLAlchemy engine + session_scope
src/betbot/storage/models.py         # ORM: PredictionRow, PaperBet
src/betbot/storage/repos.py          # upsert_prediction, insert_paper_bet*, etc.
src/betbot/exchanges/__init__.py
src/betbot/exchanges/base.py         # ExchangeAdapter Protocol + types (no impls yet)
src/betbot/utils/__init__.py
src/betbot/utils/cache.py            # TTLCache
tests/__init__.py
tests/conftest.py                    # `settings` fixture
tests/test_probabilities.py
tests/test_cache.py
```

### KNOWN ISSUES to resolve immediately (Phase 1 cleanup)

1. **`src/betbot/data/football_data.py` is likely BROKEN.** During a manual nano
   patch, two lines got fused. The operator's last error was:
   ```
   File "src/betbot/data/football_data.py", line 177
       return sorted_m[:limit]        standings = data.get("standings") or []
   SyntaxError: invalid syntax
   ```
   The `list_team_recent_matches` method's final line got merged with the first
   line of the next method (`get_standings`). **You must open this file, find the
   damage, and restore both methods cleanly.** The canonical correct version of
   `list_team_recent_matches` is:
   ```python
   async def list_team_recent_matches(
       self, team_id: int, limit: int = 5, before: str | None = None
   ) -> list[dict[str, Any]]:
       # football-data rejects dateTo without dateFrom; pass either both
       # (a generous 365-day lookback) or neither. We always sort + slice
       # on our side so the API's natural ordering doesn't matter.
       from datetime import date as _date, timedelta as _td

       params: dict[str, Any] = {"status": "FINISHED", "limit": max(limit, 10)}
       if before:
           params["dateTo"] = before
           try:
               params["dateFrom"] = (
                   _date.fromisoformat(before) - _td(days=365)
               ).isoformat()
           except ValueError:
               params.pop("dateTo", None)

       data = await self._get(f"/teams/{team_id}/matches", params=params)
       matches = data.get("matches") or []
       sorted_m = sorted(
           (m for m in matches if isinstance(m, dict)),
           key=lambda m: m.get("utcDate", ""),
           reverse=True,
       )
       return sorted_m[:limit]
   ```
   And `get_standings` (the method that follows) must be intact:
   ```python
   async def get_standings(
       self, competition_code: str
   ) -> list[dict[str, Any]]:
       data = await self._get(f"/competitions/{competition_code}/standings")
       standings = data.get("standings") or []
       for s in standings:
           if s.get("type") == "TOTAL":
               table = s.get("table") or []
               return [row for row in table if isinstance(row, dict)]
       return []
   ```
   The reason for the patch in the first place: the **original** code passed
   `dateTo` to `/teams/{id}/matches` WITHOUT `dateFrom`, which the API rejects
   with HTTP 400 (`"Argument dateFrom must be used in conjunction with dateTo
   and vice versa."`). The fix above sends both bounds or neither.

2. **`config.py` WC patch** may or may not be applied. The intended state of the
   top of `config.py`:
   ```python
   LEAGUE_CODES: tuple[str, ...] = ("PL", "PD", "BL1", "SA", "FL1", "CL", "WC")

   # Competitions where the strategy's club-football assumptions don't hold:
   # no league standings, "last 5" spans 12+ months, home advantage misleading.
   # Predictions flow through (for calibration data) but live ordering on these
   # stays OFF in Phase 5 regardless of mode.
   INTERNATIONAL_COMPETITIONS: frozenset[str] = frozenset({"WC"})
   ```
   The `WC` (FIFA World Cup 2026, runs June 11 – July 19 2026) competition IS in
   football-data.org's free tier (code `WC`). Note: the strategy is NOT
   well-suited to international football (see §7 "World Cup caveat") — for now we
   only collect prediction data, we do not bet real money on WC fixtures.

3. **Verify `INTERNATIONAL_COMPETITIONS` is actually referenced** where it should
   be (Phase 5 live-order gating). In Phase 1 it's just defined.

### Design parameters (locked — do not change without operator sign-off)
- Leagues: PL, PD, BL1, SA, FL1, CL, WC.
- Form: last 5 finished matches, recency weights `(1.5, 1.3, 1.1, 1.0, 0.9)`,
  W=3/D=1/L=0, `+0.3` home advantage, opponent-strength via league position,
  softmax with `draw_score=2.4`, temperature `1.0`.
- Edge filter: bet only when `our_prob − market_implied_prob ≥ 0.05`. On
  Polymarket V2 this comparison must be computed **net of the expected taker
  fee** (we place market orders, so we are always the taker). Subtract the
  estimated fee from the edge before applying the 0.05 gate; do NOT change the
  0.05 threshold itself. See §7 "Polymarket V2 fees".
- Sizing: `$10`/bet, `$50` max, `$200` daily cap, one position per fixture,
  idempotent inserts.
- DB: SQLite at `./data/betbot.sqlite` (configurable via `BETBOT_DB_PATH`).

---

## 4. How to work (operator's hard-won preferences)

- **Stop using nano-over-SSH for file edits.** That's why you're here. Edit files
  directly.
- **One git commit per phase**, with a descriptive message. The operator reviews
  at phase boundaries.
- **Set up GitHub early** (do it right after Phase 1 goes green) so the operator
  can `git pull` on the droplet instead of pasting. Ask the operator to create an
  empty private repo and give you the URL; or use `gh repo create` if `gh` is
  authed.
- The operator is not a Python expert. Explain what you're doing in plain terms
  at phase boundaries; don't dump code walls. Keep them oriented.
- **Always run the verification gates (§5) before claiming a phase is done.** The
  operator has been burned by "it should work" — show them green output.

---

## 5. Verification gates (run these; never skip)

After any change, in the venv:

```bash
# 1. Everything compiles
python -m compileall -q src tests

# 2. Internal imports resolve (catches renamed/missing symbols).
#    Use this ast-based check — it doesn't require deps to be installed:
python - <<'EOF'
import ast, pathlib
src = pathlib.Path("src"); mods={}
for py in src.rglob("*.py"):
    rel = py.relative_to(src).with_suffix(""); name=".".join(rel.parts).removesuffix(".__init__")
    t=ast.parse(py.read_text()); names=set()
    for n in t.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): names.add(n.name)
        elif isinstance(n,ast.Assign):
            for tg in n.targets:
                if isinstance(tg,ast.Name): names.add(tg.id)
        elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name): names.add(n.target.id)
        elif isinstance(n,ast.ImportFrom):
            for a in n.names: names.add(a.asname or a.name)
        elif isinstance(n,ast.Import):
            for a in n.names: names.add((a.asname or a.name).split(".")[0])
    mods[name]=names
prob=[]
for base in (src, pathlib.Path("tests")):
    if not base.exists(): continue
    for py in base.rglob("*.py"):
        for n in ast.walk(ast.parse(py.read_text())):
            if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith("betbot"):
                tgt=mods.get(n.module)
                if tgt is None: prob.append(f"{py}: unknown module {n.module}"); continue
                for a in n.names:
                    if a.name!="*" and a.name not in tgt: prob.append(f"{py}: {n.module}.{a.name} missing")
print("PROBLEMS:" if prob else f"All imports resolve ({len(mods)} modules)")
[print(" ",p) for p in prob]
EOF

# 3. Tests pass
pytest -q

# 4. Smoke test against live API (needs FOOTBALL_DATA_API_KEY in .env)
tfsm run-once
tfsm bets list
```

A phase is "green" only when 1–3 are clean and (where relevant) 4 produces sane
output. Note: in late May 2026 most European leagues have finished their season,
so `run-once` may only find EPL fixtures — that's expected, not a bug. Empty
league responses are the season being over.

---

## 6. Remaining phases (the build plan)

Each phase = one commit. Implementations described here are the intended design;
match the §3 locked parameters.

### Phase 2 — Polymarket adapter + routing + edge filter
- `exchanges/polymarket_gamma.py`: async client for Polymarket **Gamma API**
  (`https://gamma-api.polymarket.com`) for market discovery — `/events`,
  `/sports`, `/markets`. Public, no auth. `clobTokenIds` and `outcomes` come
  back as JSON-encoded STRINGS — `json.loads()` them. Football tag id default
  `100381`, but auto-discover via `/sports`.
- `exchanges/matcher.py`: `TeamAliasResolver` using `rapidfuzz.token_set_ratio`
  + a `config/team_aliases.yaml` of manual overrides. `_normalize()` strips
  diacritics + noise tokens (FC, AFC, CF, etc.).
- `exchanges/polymarket.py`: `PolymarketAdapter` implementing the
  `ExchangeAdapter` Protocol. Discovery via Gamma; orderbook via the CLOB SDK.
  **Use `py-clob-client-v2`** — Polymarket migrated to CTF Exchange **V2 on
  April 28, 2026 (~11:00 UTC)**; V1-signed orders are rejected and there is NO
  backward compatibility. (Earlier in this project's history we used V1; go
  straight to V2 in this rebuild.) V2 EIP-712 domain: name
  `"Polymarket CTF Exchange"`, version `"2"`, chainId 137, verifyingContract
  `0xE111180000d2663C0091e4f400237545B87B996B`. Collateral is now **pUSD**,
  1:1 USDC-backed, needs a wrap step. **Confirm the pUSD token, V2 Exchange,
  Neg-Risk Exchange, and Collateral Onramp addresses against
  docs.polymarket.com/resources/contracts before hardcoding any of them — do
  NOT trust addresses pasted from chat history.** The v2 SDK handles V2 order
  signing (new struct drops `taker`/`expiration`/`nonce`/`feeRateBps` and adds
  `timestamp`/`metadata`/`builder`) — do NOT hand-roll Polymarket order signing.
  Handle BOTH market layouts: (A) single 3-way market with 3 token ids; (B) event
  with 3 binary YES/NO markets (this layout is more common for football).
  `place_order` must be DOUBLE-GATED: requires `enable_orders=True` at
  construction AND `mode=live`.
- `exchanges/router.py`: `ExchangeRouter.find_best_quote()` — lowest yes_price
  wins, tie-break on larger size.
- Wire into `main.py`: build router from the run's team set; scoring loop uses a
  tri-state route result (`bet_logged` / `no_edge` / `already_logged` /
  `no_market`). `no_edge` must NOT fall back to favourite-logging (the market
  vetoes). `no_market` falls through to the Phase-1 favourite paper bet.
- `scripts/seed_aliases.py`: refresh `team_aliases.yaml` from football-data +
  Gamma.
- Tests: matcher, polymarket adapter (both layouts, with a fake Gamma + fake
  CLOB client), router.

### Phase 3 — Limitless adapter + dual-venue routing
- Limitless = Base mainnet (chain id 8453), `https://api.limitless.exchange`.
  CTF Exchange is a FORK of Polymarket's; EIP-712 domain name is
  `"Limitless CTF Exchange"`, version `"1"`, chainId 8453, verifyingContract is
  per-market (`market.venue.exchange`).
- `exchanges/limitless_client.py`: read-only REST client. Endpoints:
  `/markets/active`, `/markets/search?query=`, `/markets/<slug>`,
  `/markets/<slug>/orderbook`, and `POST /orders` (Phase 5). Add a distinct
  `LimitlessGeoBlockedError` (US IPs get 451/403) so the router degrades
  gracefully.
- `exchanges/limitless.py`: `LimitlessAdapter`. Binary YES/NO markets only.
  Reuse Polymarket's `_classify_binary_market` for HOME/AWAY/DRAW classification.
  **DRAW is Polymarket-only by design** — Limitless rarely has a draw market, so
  `find_market` returns the draw slug only if present; the router skips Limitless
  for DRAW favourites.
- Add to router via `_build_router` in `main.py`; no router logic change needed.
- Limitless football coverage is SPARSE — most days the router sees only
  Polymarket. This is expected.
- Tests: `test_limitless.py` with a fake client.

### Phase 4 — Settlement + P&L + kill switch
- `storage/models.py`: add `KillSwitch` table (single row, id=1, last-write-wins:
  `tripped_at`, `reason`, `realized_pnl_usd`, `staked_usd`, `updated_at`).
  `PaperBet` already has `settled_at`, `settled_outcome`, `pnl_usd`.
- `settlement.py`: `SettlementWatcher`. Walk unsettled bets where
  `kickoff + grace(150min) <= now`, fetch result via
  `FootballDataClient.get_match(id)`, map `score.winner`
  (HOME_TEAM/AWAY_TEAM/DRAW) to outcome, set settled fields, compute P&L:
  win = `stake * (1/price - 1)`, loss = `-stake`. **No-market (favourite-only)
  bets settle with pnl_usd=0** — they have no real-money equivalent and must not
  pollute the kill-switch signal. Only `FINISHED`/`AWARDED` statuses settle;
  others retry next run.
- Kill switch: after settling, sum P&L over the trailing window
  (`drawdown_window_days`, default 7). Trip if
  `pnl < -drawdown_kill_pct * staked` (default 0.20) AND
  `staked >= drawdown_min_staked_usd` (default 100). Scoring loop refuses to log
  new bets while tripped.
- `repos.py`: add `list_unsettled_bets_due`, `record_settlement`,
  `settled_pnl_window`, `get_kill_switch`, `is_kill_switch_tripped`,
  `trip_kill_switch`, `reset_kill_switch`. **Important SQLAlchemy gotcha:** any
  repo that returns ORM rows for use after the session closes must call
  `session.expunge_all()` before returning, or attribute access raises
  `DetachedInstanceError`. (This already bit us once on `list_recent_paper_bets`.)
- New settings: `BETBOT_DRAWDOWN_KILL_PCT`, `BETBOT_DRAWDOWN_WINDOW_DAYS`,
  `BETBOT_DRAWDOWN_MIN_STAKED_USD`.
- CLI: `tfsm settle`, `tfsm kill-switch status`, `tfsm kill-switch reset`. Daemon
  runs settle on the same cron tick as score.
- `bets list` should show market price, settled outcome, pnl.
- Tests: `test_settlement.py` (P&L math incl. no-market=0 and invalid price,
  watcher win/loss/in-play/too-recent, kill-switch trip + min-staked floor),
  `KillSwitch` CRUD in `test_storage.py`.

### Phase 5 — Backtest + gate + live ordering + approvals
- `backtest.py`: `backtest_stored` (replay settled bets — hit rate, ROI, Brier,
  per-outcome) and `backtest_mock` (synthetic fair-market diagnostic). CLI
  `tfsm backtest --mode stored|mock`.
- `gate.py`: `evaluate_gate` → `GateResult` with reasons. Requires min settled
  bets (with market price), min window days, min hit rate, min ROI, and kill
  switch clear. Excludes no-market bets. CLI `tfsm gate`.
- `exchanges/limitless_signing.py`: EIP-712 signing via `eth_account`
  (`encode_typed_data(full_message=...)`, `Account.sign_message`; the hash is
  `signed.message_hash`, NOT `signable.hash`). Order struct field order — **this
  is the Limitless / V1-fork struct ONLY** (Limitless is a fork of Polymarket's
  pre-V2 ctf-exchange, domain version "1"):
  `salt, maker, signer, taker, tokenId, makerAmount, takerAmount, expiration,
  nonce, feeRateBps, side, signatureType`. **Do NOT reuse this struct for
  Polymarket** — Polymarket V2 dropped `taker`/`expiration`/`nonce`/`feeRateBps`
  and added `timestamp`/`metadata`/`builder`, and Polymarket signing is handled
  by `py-clob-client-v2` (never hand-rolled). `takerAmount = makerAmount / price`,
  6-decimal USDC scaling. Ship pure helpers (`to_usdc_units`, `shares_for`,
  `random_salt`) testable without `eth_account`.
- Wire live `place_order` in both adapters. Polymarket: use the v2 SDK's market-
  order method — **verify the exact method name and signature in the installed
  `py-clob-client-v2`; the v1 name `create_and_post_market_order` may have
  changed, and order creation now takes an options object / `UserMarketOrderV2`
  shape.** Pass `userUSDCBalance` on market buys so the SDK returns fee-adjusted
  fill amounts (V2 takers pay a protocol fee — see §7 "Polymarket V2 fees").
  Limitless: via the signer + `POST /orders`.
- `main.py`: live pre-flight calls `evaluate_gate`, refuses to start if it fails;
  `_build_router` flips `enable_orders` when not paper; `_try_market_route`
  places the order after logging the paper bet (paper row persists even if the
  order fails). **Skip live ordering for `INTERNATIONAL_COMPETITIONS`.**
- `scripts/polymarket_approve.py` (USDC→pUSD wrap + approvals; web3.py) and
  `scripts/limitless_approve.py` (USDC + CTF setApprovalForAll on Base).
- New deps in a `[polymarket]` extra: `py-clob-client-v2`. `eth-account` +
  `web3` for signing/approvals.
- Tests for signing (incl. recover-address roundtrip), gate, backtest.

### Phase 6 — (already partly reflected) ensure naming is "The Football Smart Manager"
- pyproject name, README, CLI docstrings/help. Package stays `betbot`. `tfsm`
  and `betbot` both as console scripts.

### Phase 7 — FastAPI backend (`backend/tfsm_api/`)
- The operator chose **FastAPI + React** and **full control** (monitor, trigger,
  edit settings, kill-switch).
- Async endpoints wrapping the bot's own functions (NOT shelling out): score,
  settle, gate, backtest, bets list, predictions, kill-switch status/reset,
  settings read/write.
- Settings editing writes to `.env` and signals "restart required" for knobs
  that need it; document which knobs are hot-reloadable. (Consider switching
  `get_settings()` from `lru_cache` to a cache you can clear, so the API can
  reload after writing `.env`.)
- Auth: single `TFSM_API_TOKEN` env var; if set, require
  `Authorization: Bearer`; if unset, bind localhost only.
- CORS configured for the React dev server and the production hostname.
- Serve the built React app (`frontend/dist`) as static files.

### Phase 8 — React frontend (`frontend/`)
- Vite + React 18 + TypeScript + Tailwind + TanStack Query. No Next, no Redux.
- Pages: Dashboard (kill-switch/gate/mode pills, P&L, ROI, hit rate, equity
  curve, recent activity), Bets (filterable table + per-outcome breakdown),
  Predictions (upcoming + softmax probs), Backtest (run stored/mock), Settings +
  Actions (edit knobs, trigger runs, reset kill switch behind a confirm).
- Visual identity used before (reuse): dark surface `#0d1117`/`#161b22`, accent
  `#58a6ff`, success `#3fb950`, danger `#f85149`, tabular-numerics on metrics.
- Build to `frontend/dist`, served by FastAPI.

### Phase 9 — Deployment (Amsterdam droplet)
- `systemd` units: `tfsm-daemon.service` (runs `tfsm run-daemon`) and
  `tfsm-api.service` (uvicorn). Restart on failure, start on boot.
- nginx reverse proxy: `/api` → uvicorn, `/` → `frontend/dist`. Let's Encrypt via
  certbot.
- `.env` mode 0600, owned by `tfsm`.
- litestream nightly backup of the SQLite DB to S3/B2.
- ufw already allows 22/80/443. Document the egress allowlist (football-data.org,
  gamma-api.polymarket.com, clob.polymarket.com, api.limitless.exchange, Polygon
  RPC, Base RPC).
- Deploy doc: `git pull && systemctl restart tfsm-daemon tfsm-api`.

---

## 7. Important gotchas & decisions (don't relearn these the hard way)

- **football-data.org `dateTo` requires `dateFrom`** (and the `dateTo` on
  `/competitions/{c}/matches` is EXCLUSIVE — use `today+2` to include tomorrow).
- **Polymarket V2 cutover (April 28 2026, ~11:00 UTC)** — use `py-clob-client-v2`;
  V1 is dead, no backward compatibility, all V1 open orders were wiped at cutover.
  Production URL stays `clob.polymarket.com` after cutover; the V2 sandbox is
  `clob-v2.polymarket.com` if you want to dry-run before funding.
- **Polymarket V2 fees** — fees are protocol-set at match time, NOT embedded in
  the signed order. Makers pay nothing; takers pay `fee = C × feeRate × p × (1−p)`
  (largest near p=0.5). We place market orders, so we always pay the taker fee.
  Query live params via `getClobMarketInfo(conditionID)` (`fd.r` rate, `fd.e`
  exponent, `fd.to` taker-only) and fold the expected fee into the edge calc
  (Phase 2 edge filter). Remove any manual/embedded `feeRateBps` logic for
  Polymarket — that field no longer exists in V2.
- **Polymarket V2 pUSD must sit in the deposit/funder wallet** (`POLYMARKET_FUNDER`),
  NOT the owner EOA. If the CLOB balance cache is stale you get false
  "insufficient balance" errors even on a funded wallet — call the balance-sync
  endpoint with signature type 3 after wrapping/funding.
- **Polymarket V2 addresses are not hardcoded constants yet** — pull the canonical
  pUSD token, V2 Exchange, Neg-Risk Exchange, and Collateral Onramp addresses from
  docs.polymarket.com/resources/contracts. Do not trust addresses from chat history.
- **Gamma `clobTokenIds`/`outcomes` are JSON strings** — `json.loads()`.
- **Polymarket football markets are usually Layout B** (3 binary markets per
  event), not a single 3-way market. Handle both.
- **Limitless football coverage is sparse**; idle days are normal, not a bug.
- **DRAW is Polymarket-only** (Limitless is binary YES/NO).
- **SQLAlchemy DetachedInstanceError**: `expunge_all()` before returning ORM rows
  that outlive the session.
- **No-market (favourite-only) paper bets get pnl_usd=0** and are excluded from
  the gate and kill-switch signal.
- **eth_account**: hash is `signed.message_hash`, not `signable.hash`.
- **Limitless EIP-712 domain** name/version = `"Limitless CTF Exchange"`/`"1"`;
  Polymarket V2 = `"Polymarket CTF Exchange"`/`"2"`. Don't mix them up.
- **World Cup caveat**: the club-football strategy is a poor fit for
  international football (no league standings → opponent-strength collapses to
  1.0; "last 5" can span a year; home advantage is meaningless at neutral
  venues). For the June–July 2026 tournament we LOG predictions for calibration
  but do NOT bet real money on `WC` fixtures (`INTERNATIONAL_COMPETITIONS`
  guards live ordering). A proper Elo-based international model is a possible
  future phase, not in current scope.
- **Geo**: deployment must stay in a permitted region (Amsterdam). Polymarket
  treats VPNs from blocked regions as a ToS violation. The server's jurisdiction
  AND the operator's own jurisdiction both matter — flag this to the operator,
  don't advise on circumvention.
- **Strategic honesty**: a 5-match-form signal is weak against efficient liquid
  markets. The edge, if any, is on lower-liquidity matches. The operator has been
  told to paper-trade ≥2 weeks before funding. Don't oversell expected returns.

---

## 8. Secrets the operator must provide (never commit these)
- `FOOTBALL_DATA_API_KEY` (free; already set in `.env`).
- `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_FUNDER` (Phase 5 live).
- `LIMITLESS_PRIVATE_KEY` (Phase 5 live).
- `POLYGON_RPC_URL`, Base RPC URL (Phase 5 approvals).
- `POLYMARKET_ONRAMP` (Collateral Onramp address — not yet a hardcoded constant).
- `TFSM_API_TOKEN` (Phase 7 API auth).
- USDC on Polygon (Polymarket) and Base (Limitless) for live trading.

`.gitignore` already excludes `.env`, `data/`, `*.sqlite`, the venv, and caches.
Confirm no secret ever lands in git history.

---

## 9. First actions checklist for you (Claude Code)

- [ ] Locate the repo; activate the venv; run the §5 gates to see current state.
- [ ] Fix the broken `football_data.py` (§3 issue 1) and confirm `get_standings`
      is intact.
- [ ] Apply/verify the `config.py` WC patch (§3 issue 2).
- [ ] `pip install -e ".[dev]"` if not already; run `pytest`; run `tfsm run-once`.
- [ ] Get Phase 1 fully green (§5).
- [ ] `git add -A && git commit` Phase 1 if not committed.
- [ ] Set up a GitHub remote and push (ask operator for repo URL or use `gh`).
- [ ] Tell the operator Phase 1 is green, show them the gate output, and ask to
      proceed to Phase 2.
- [ ] Work Phases 2→9, one commit each, pausing at boundaries.

Welcome aboard. The spec is locked; the operator values honesty about what's
verified vs. assumed. Show green output, don't claim untested things work, and
keep the operator oriented in plain language.


# APPENDIX A — Glicko Rating Engine for International / World Cup Fixtures

> **Instruction to Claude Code:** Append this appendix to `CLAUDE.md`. It
> defines an OPTIONAL phase (call it **Phase 5.5**) that adds a Glicko-2
> rating engine used for `INTERNATIONAL_COMPETITIONS` (currently `{"WC"}`).
> Build it only after Phases 1–5 are green, and only when the operator asks.
> Until then, leave it as documented scope.
>
> **Hard rule that overrides everything else in this appendix:** the World
> Cup model is **paper-mode only**. It must NEVER place a real-money order,
> regardless of `BETBOT_MODE`. The existing `INTERNATIONAL_COMPETITIONS`
> guard already blocks live ordering on `WC`; this appendix must not weaken
> that guard. If you find yourself writing code that would let a WC fixture
> reach `place_order`, stop — that's a bug.

---

## A.0 Why this exists, and what it can and cannot do

The base strategy (last-5 form + league-standings opponent strength + home
advantage + softmax) is built for club football. It degrades badly on
international tournaments because:
- national teams have no league table (opponent-strength input collapses to 1.0),
- "last 5 matches" can span 12+ months and mix friendlies with qualifiers,
- home advantage is meaningless at neutral World Cup venues.

Glicko-2 fixes the *rating* problem: it produces a continuous strength
estimate for each national team without needing a league table, AND it
tracks a **rating deviation (RD)** — an explicit measure of how uncertain
that estimate is. RD rises when a team hasn't played in a while (exactly the
international-football situation) and falls after recent matches. This is why
Glicko-2 is a better fit than plain Elo here: it knows when it doesn't know.

**What this does NOT do — be honest with the operator and in code comments:**
- It does not produce "accurate" winner predictions. Match-level outcome
  accuracy for any football model, including this one, tops out around
  50–55% (draws included). That is the ceiling, not a failure of
  implementation.
- It is not an edge source on World Cup markets. WC markets are deep, liquid,
  and over-analysed; the market line is the best available predictor. This
  model will mostly agree with the market on obvious matchups and be noisy on
  close ones.
- Its purpose is **calibration data**, not profit. We log predictions during
  the tournament to measure how well-calibrated the model is (Brier score,
  reliability curve) and to decide whether an international model is ever
  worth taking live in future tournaments. That decision should be made on
  the data, after the fact, not assumed now.

The success metric for this phase is **calibration, not accuracy or P&L**.
Do not let the operator (or yourself) judge it by "how many winners it got."

---

## A.1 Glicko-2: the algorithm (reference)

Glicko-2 (Glickman, 2012) tracks three numbers per team:
- **rating (r)** — strength, conventionally centred at 1500.
- **rating deviation (RD)** — uncertainty of the rating (higher = less sure).
- **volatility (σ)** — how erratic the team's results are over time.

Internally Glicko-2 works in a transformed scale:
- `μ = (r − 1500) / 173.7178`
- `φ = RD / 173.7178`

System constant **τ** constrains volatility change; use **τ = 0.5**
(Glickman's recommended default; smaller = more stable, range 0.3–1.2).

### Per rating period, for a team with current (μ, φ, σ) and a set of games:

For each opponent j with (μ_j, φ_j) and score s_j (1 win / 0.5 draw / 0 loss):
```
g(φ_j) = 1 / sqrt(1 + 3·φ_j² / π²)
E(μ, μ_j, φ_j) = 1 / (1 + exp(−g(φ_j)·(μ − μ_j)))
```

Estimated variance:
```
v = 1 / Σ_j [ g(φ_j)² · E_j · (1 − E_j) ]
```

Estimated improvement:
```
Δ = v · Σ_j [ g(φ_j) · (s_j − E_j) ]
```

New volatility σ′ via the iterative (Illinois algorithm) solution of:
```
f(x) = e^x·(Δ² − φ² − v − e^x) / (2·(φ² + v + e^x)²) − (x − ln(σ²)) / τ²
```
solve f(x) = 0 for x; σ′ = e^(x/2). (Implement the Illinois method exactly as
in Glickman's 2012 paper, §5.1 step 5 — it converges in a handful of
iterations. Do not approximate this; volatility update is where naive
implementations go wrong.)

Pre-rating-period RD bump (accounts for time passing):
```
φ* = sqrt(φ² + σ′²)
```

New φ and μ:
```
φ′ = 1 / sqrt(1/φ*² + 1/v)
μ′ = μ + φ′² · Σ_j [ g(φ_j)·(s_j − E_j) ]
```

Convert back:
```
r′ = 173.7178·μ′ + 1500
RD′ = 173.7178·φ′
```

### Team that did NOT play in a rating period
Only RD increases (uncertainty grows): `φ′ = sqrt(φ² + σ²)`, μ and σ unchanged.

### Match outcome probabilities (what the bot needs)
Glicko-2 is natively a 2-outcome (win/loss) system. Football has draws.
Convert a rating gap into win/draw/loss with a draw-inflation parameter:
```
# expected score for HOME vs AWAY, incorporating BOTH teams' RD:
g_combined = 1 / sqrt(1 + 3·(φ_home² + φ_away²) / π²)
p_home_raw = 1 / (1 + exp(−g_combined·(μ_home − μ_away + home_field_μ)))
```
- `home_field_μ` is a small bump for genuine host-nation home advantage ONLY.
  Default 0.0 (neutral venue). Set a positive value (e.g. +0.2 on the μ scale)
  for matches where the home team is an actual host nation (USA, Canada,
  Mexico in 2026). Detect host status from a small hardcoded set; do not apply
  generic home advantage to non-hosts.

Split the win/loss probability into 3-way using a draw model. Use the simple,
well-tested approach: a draw-propensity parameter `rho` (default ~0.28 for
international knockout-style football; expose as a setting). One workable
parameterisation:
```
p_draw = rho · (1 − abs(p_home_raw − 0.5)·2)   # more draws when teams are even
p_draw = clamp(p_draw, 0.05, 0.40)
p_home = (1 − p_draw) · p_home_raw
p_away = (1 − p_draw) · (1 − p_home_raw)
```
Document clearly that this draw split is a heuristic, not derived from Glicko,
and is a prime calibration target to revisit after seeing tournament data.

---

## A.2 Data source for ratings

Glicko-2 needs historical international results to bootstrap ratings BEFORE
the tournament starts. football-data.org free tier does NOT give deep
international history. Two acceptable paths — pick based on what's available
when building; ask the operator:

**Path 1 (preferred, no new paid API): seed from a public dataset.**
Use the well-known open dataset of international football results
("results.csv" — international matches since 1872, ~45k rows, MIT-ish public
domain, widely mirrored). Bootstrap each national team's (r, RD, σ) by
running Glicko-2 forward over the last ~4 years of that team's results
(weight recent rating periods; older history just establishes the prior).
Store the seeded ratings in a new DB table (see A.4). This is a one-time
offline job: `scripts/seed_glicko.py`.
- The operator must supply the dataset file (a CSV path) or a URL to fetch.
  Do not hardcode a scraped source. Keep the CSV out of git (add to
  `.gitignore`); store only the derived ratings.

**Path 2 (fallback): seed from a static ratings snapshot.**
If no results dataset is available, seed from a snapshot of published
international Elo/Glicko ratings (operator provides a small YAML/CSV of
team → rating). Set RD high (e.g. 200) to reflect that these are imported
priors, not earned. The model will tighten RD as tournament matches come in.

Either way: **rating periods.** Treat each matchday (or each ~3-day window) of
the tournament as one rating period. After each WC matchday settles, run a
Glicko-2 update so ratings sharpen as the tournament progresses.

---

## A.3 Where this plugs into the existing architecture

Do NOT modify the existing `StrategyEngine`. Add a parallel engine and select
between them by competition.

- New module `src/betbot/strategy/glicko.py`:
  - `Glicko2Rating` dataclass (`rating`, `rd`, `volatility`, `last_period`).
  - `Glicko2Engine` with the period-update math from A.1.
  - Pure functions where possible (testable without DB/network), mirroring how
    `probabilities.py` is kept pure.
- New module `src/betbot/strategy/international_engine.py`:
  - `InternationalStrategyEngine` exposing the SAME interface the scoring loop
    expects: a `predict(fixture_form_or_equivalent) -> Prediction` method and a
    `decide_with_market(prediction, outcome, market_price) -> BetDecision|None`.
  - Internally it ignores `FormSnapshot` (no league form for nations) and
    instead pulls each team's current Glicko rating from the ratings store,
    computes (p_home, p_draw, p_away) via A.1, and returns the same
    `Prediction` dataclass the rest of the system already uses. This keeps
    storage, settlement, backtest, and the API all working unchanged.
  - `decide_with_market` reuses the SAME edge filter and threshold as the club
    engine (`edge_threshold`). It must STILL return a decision object for
    logging — but see the live-order guard below.
- Scoring loop selection (in `main.py`):
  - When `fixture.competition_code in INTERNATIONAL_COMPETITIONS`, use
    `InternationalStrategyEngine`; otherwise use the existing `StrategyEngine`.
  - Keep the existing tri-state routing. The bot still queries Polymarket for
    WC market prices (the markets exist and are liquid) and logs edge-filtered
    PAPER bets — this is the calibration data we want.
- **Live-order guard (critical):** in `_try_market_route` (or wherever
  `place_order` is reached), add an explicit check: if
  `prediction.competition_code in INTERNATIONAL_COMPETITIONS`, log the paper
  bet and RETURN before any `place_order` call, even in live mode. Add a test
  that asserts a WC fixture in live mode never calls `place_order`.

---

## A.4 Storage

New ORM table `glicko_ratings` (in `storage/models.py`):
```
id              int pk
team_name       str  (canonical national team name; index)
team_id         int  nullable (football-data team id if known)
rating          float
rd              float
volatility      float
last_period     str  (ISO date of last rating period applied)
updated_at      datetime
```
Single current row per team (upsert on team_name). Keep a separate append-only
`glicko_rating_history` table if the operator wants to chart rating drift over
the tournament — optional, nice for the frontend later.

Repo helpers: `get_rating(team_name)`, `upsert_rating(...)`,
`all_ratings()`, and a `apply_rating_period(results)` that runs the Glicko-2
update for a batch of settled matches.

Settlement integration: after the SettlementWatcher settles WC matches for a
matchday, call `apply_rating_period` with those results so ratings update.
Gate this behind the competition check so club settlements don't touch Glicko.

---

## A.5 Settings (add to config.py / .env.example)

```
BETBOT_GLICKO_TAU=0.5                 # volatility constraint (0.3–1.2)
BETBOT_GLICKO_DEFAULT_RATING=1500
BETBOT_GLICKO_DEFAULT_RD=200          # high = imported prior, low confidence
BETBOT_GLICKO_DEFAULT_VOL=0.06
BETBOT_GLICKO_DRAW_RHO=0.28           # draw-propensity heuristic
BETBOT_GLICKO_HOST_HOME_MU=0.2        # μ-scale bump for actual host nations only
BETBOT_GLICKO_RESULTS_CSV=            # path to international results dataset (Path 1)
```
Host nations for 2026 (hardcode a small set, document it): USA, Canada, Mexico.

---

## A.6 CLI

- `tfsm glicko seed` — run the offline bootstrap (Path 1 or 2). Idempotent;
  overwrites the ratings table.
- `tfsm glicko ratings` — print current ratings sorted by rating desc.
- `tfsm glicko update --since <date>` — manually trigger a rating-period
  update from settled WC matches (also runs automatically post-settlement).

---

## A.7 Tests (required before calling this phase green)

- `test_glicko.py`:
  - Glicko-2 update against the WORKED EXAMPLE in Glickman's paper, VERIFIED
    against his published values and three independent implementations
    (R PlayerRatings, Scala, PL/SQL). Inputs: subject player rating 1500,
    RD 200, vol 0.06, tau 0.5; opponents (1400, RD 30), (1550, RD 100),
    (1700, RD 300); scores 1, 0, 0 (win, loss, loss). Expected output:
        rating  ~ 1464.05   (higher precision 1464.051)
        RD      ~ 151.52     (higher precision 151.51652)
        vol     ~ 0.05999    (higher precision 0.05999583)
    Assert rating/RD within +/-0.1 and vol within +/-0.0001. This is the
    single most important test; it proves the math is right. If it fails, the
    volatility iteration (Illinois method) is almost certainly the culprit --
    that's where naive implementations diverge.
  - RD increases for a team that didn't play a period.
  - 3-way probability split sums to 1.0 and respects ordering (stronger team
    higher p).
  - Draw probability clamps to [0.05, 0.40].
- `test_international_engine.py`:
  - `predict` returns a valid `Prediction` with probabilities summing to 1.
  - host-nation home bump applied only for USA/Canada/Mexico, not others.
  - `decide_with_market` honours the edge threshold.
- `test_live_guard.py` (or extend existing):
  - a `WC` fixture in `mode=live` with `enable_orders=True` logs a paper bet
    and NEVER calls `place_order` (mock the adapter; assert not called).

---

## A.8 Honest framing for the operator (put this in the phase's completion report)

When you finish this phase, tell the operator, in plain language:
- This produces calibration data for World Cup matches in paper mode only.
- It will not be "accurate" — expect ~50–55% match-outcome hit rate at best,
  same as every other football model and the market itself.
- The point is to measure Brier score / calibration over the tournament and
  decide AFTER the fact whether an international model is ever worth taking
  live. Recommend the operator look at `tfsm backtest --mode stored` filtered
  to WC fixtures once the group stage is done.
- Reiterate: do not fund World Cup betting on the strength of this. The edge,
  if any in football betting at all, is in obscure low-liquidity CLUB matches,
  not the most-watched tournament on earth.
- The single biggest accuracy improvement available is not a fancier model —
  it's blending toward the market line. If the operator wants to pursue real
  edge later, that (a market-aware ensemble) is the higher-value project than
  refining Glicko.

---

## A.9 Build order for this phase

1. `strategy/glicko.py` (pure math) + `test_glicko.py` with the paper's worked
   example. Get this green FIRST — everything depends on the math being right.
2. `storage` table + repos + migration (create_all handles it).
3. `scripts/seed_glicko.py` (Path 1 or 2) + `tfsm glicko seed/ratings`.
4. `strategy/international_engine.py` + tests.
5. Scoring-loop selection by competition + the live-order guard + guard test.
6. Settlement hook to update ratings after WC matchdays.
7. Completion report to operator with the A.8 framing.

One commit for the phase, or split math/storage from wiring if it's large.
Pause for operator review before and after.

---

# HANDOVER — Current state (2026-06-11)

Live status overlay on top of the locked 9-phase plan above. **Read this first** —
it reflects what's actually built/deployed since the original plan. When you finish
meaningful work, update THIS section (date it) so the next session inherits accurate
state.

## Done & on `main` (droplet + GitHub, head 8b2778d)
- Phases 1–9 + 5.5 Glicko + FastAPI backend + React frontend + Telegram bot
  (@FootballNutmegbot, operator = TG user 1533981578).
- Agent EVM wallet `0x608F1144C409E7de0d8164F5e942A390d3a53c0a` (same address on
  Polygon + Base; key at `.secrets/agent_wallet.key`, 0600, gitignored). Funded
  ~100 USDC on Base + a little gas.
- Cross-venue arb scanner (Polymarket + Limitless + SX Bet) with Telegram alerts.
- **Multi-tenant, NON-CUSTODIAL per-user trading** (commit 8b2778d): each registered
  user gets an isolated wallet (`.secrets/users/<id>.key`); the bot places each
  decided bet on EVERY active user's own wallet, sized to that user's balance
  (`main._place_live_for_users`), skipping users below `min_user_stake_usd`. Funds
  are never pooled. Falls back to the single agent wallet when no users registered.
- 120 tests pass on the droplet (`python -m pytest`), ruff clean.

## Take a user live on Polymarket (runbook)
1. User does `/start` on the bot → gets their own wallet address.
2. User funds it with USDC on **Polygon** (Polymarket's chain).
3. Operator: `python scripts/polymarket_approve_users.py` (dry-run), then
   `... --confirm` (sends txs; each user wallet needs a little MATIC for gas).
4. Set `BETBOT_MODE=live` and clear/skip the gate. The daemon then trades each
   funded+approved user wallet automatically. Currently `mode=paper`.

## Limitless live orders — PARKED behind 2 blockers
Plumbing proven (API auth via `LIMITLESS_API_KEY/SECRET`, on-chain USDC approval,
EIP-712 signing, validated FOK schema — all in `limitless.py`). Blocked by:
- `feeRateBps=0` rejected ("out of user's band"); required value not exposed by the
  API → get it from Limitless docs/support, set `LIMITLESS_FEE_RATE_BPS`.
- `minSize` = 100 USDC on WC group markets → no sub-$100 test possible.
Also Limitless has NO per-match 1X2 WC markets (futures/props only), so the match
strategy has nothing to trade there. **Polymarket is the live venue.**

## Still open
- **#30**: install systemd units (`deploy/install-systemd.sh`) for reboot survival —
  needs operator sudo. Until then the bot runs under setsid/tmux and does NOT
  survive a droplet reboot.
- Onboard users: add TG ids to `TELEGRAM_ALLOWED_USER_IDS` or enable
  `TELEGRAM_OPEN_REGISTRATION`.

## Workflow notes
- Droplet is source of truth; edit on Mac, scp to droplet, run there over SSH.
- Pushing the default branch / scp-overwriting droplet code / destructive git are
  harness-gated — do them only with explicit operator say-so; branch + PR for code.
- No `gh` and no API token on the droplet (SSH deploy key only) → merge via git on
  the droplet, not the GitHub API.
- **Strategy honesty:** the Qatar-2022 backtest favourite-hit rate (54.7%) is
  accuracy, NOT a proven edge over market prices. Don't claim the strategy is ±EV.
- **Testing in a git worktree:** a plain `pytest` inside a `git worktree`
  imports the LIVE `~/tfsm` tree (the editable install `.pth` points at
  `/home/tfsm/tfsm/src`), so worktree-only test files fail collection and a
  review can get a FALSE pass. Always run the worktree suite with its own src on
  the path:
  `cd <worktree> && PYTHONPATH=$PWD/src ~/tfsm/.venv/bin/python -m pytest -q`
  (same for `ruff`). This shadows the editable install with the branch code.
