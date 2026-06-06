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
- Edge filter: bet only when `our_prob − market_implied_prob ≥ 0.05`.
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
  April 22, 2026**; V1-signed orders are now rejected. (Earlier in this project's
  history we used V1; go straight to V2 in this rebuild.) V2 EIP-712 domain:
  name `"Polymarket CTF Exchange"`, version `"2"`, chainId 137,
  verifyingContract `0xE111180000d2663C0091e4f400237545B87B996B`. Collateral is
  now **pUSD** (`0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB`), 1:1 USDC-backed,
  needs a wrap step. Handle BOTH market layouts: (A) single 3-way market with 3
  token ids; (B) event with 3 binary YES/NO markets (this layout is more common
  for football). `place_order` must be DOUBLE-GATED: requires
  `enable_orders=True` at construction AND `mode=live`.
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
  `signed.message_hash`, NOT `signable.hash`). Order struct field order (verified
  against the Polymarket/Limitless ctf-exchange contract source):
  `salt, maker, signer, taker, tokenId, makerAmount, takerAmount, expiration,
  nonce, feeRateBps, side, signatureType`. `takerAmount = makerAmount / price`,
  6-decimal USDC scaling. Ship pure helpers (`to_usdc_units`, `shares_for`,
  `random_salt`) testable without `eth_account`.
- Wire live `place_order` in both adapters (Polymarket via `py-clob-client-v2`'s
  `create_and_post_market_order`; Limitless via the signer + `POST /orders`).
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
- **Polymarket V2 cutover (April 22 2026)** — use `py-clob-client-v2`; V1 is dead.
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
