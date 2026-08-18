# RONALDO — Roadmap to best-in-class club match prediction (incl. Champions League)

Drafted 2026-07-31 from a literature + industry sweep. This is the execution
plan for making the club engine the most accurate 1X2 predictor we can build.
Owner: the "Ronaldo" agent. Every phase ships ONLY through the walk-forward
backtest gate (`scripts/backtest_club.py`), extended as noted.

## Where we stand (measured, held-out 2025-26 season, n=1752)

| model                    | acc % | mean RPS | log-loss |
|--------------------------|-------|----------|----------|
| naive form engine (old)  | 32.8  | 0.2399   | 1.2130   |
| club ensemble (current)  | 50.5  | 0.2043   | 0.9986   |
| market closing line      | 53.4  | 0.1982   | 0.9795   |

Gap to market: **0.0061 RPS**. The literature is unanimous: closing odds are
the strongest public predictor; models that "beat" them do so selectively,
not across the board. So the objective function is:

1. close the RPS gap to the closing line (accuracy goal), and
2. maximise **CLV** (closing-line value) on the bets we actually place
   (edge goal) — not raw accuracy. 75–85% accuracy claims online are
   marketing; honest 3-way accuracy for top leagues tops out ≈53–55%
   (the market itself). EA's famous World-Cup record = picking the
   pre-tournament favourite; it never beat a price.

## What the best public systems use (and we don't, yet)

* **Goal-difference-aware cross-league Elo** (ClubElo): trains on domestic
  AND European cup results, giving calibrated cross-league strengths —
  proven to outperform UEFA coefficients for CL prediction.
* **xG instead of raw goals** (538 SPI, Wilkens 2026 Bundesliga study):
  finishing luck regresses; xG-based team strength predicts future results
  better than goals. The Wilkens paper found xG signal not fully priced by
  bookmakers (profitable on home-win subset, 11 seasons).
* **Time-decay** in the goal-model fit (Dixon-Coles 1997, penaltyblog):
  down-weight old matches; we currently fit unweighted.
* **Margin-of-victory ratings** (Hvattum & Arntzen goal-based Elo): we
  already built MOV-Glicko for internationals (flag-gated challenger).
* **Player/lineup layer** (EA-style): squad strength from player ratings,
  injury/suspension adjustments, and CONFIRMED LINEUPS ~60 min pre-kickoff.
  Lineup news is the single biggest source of model-vs-market divergence in
  the hour before kickoff.
* **Meta-blend + per-league isotonic calibration + Hedge selection**: we
  have the machinery (ensemble.py, model_select.py) — wired only to WC.

## STRIP (2026-08-06) — club-focused core
Operator-directed cleanup before merge: removed ALL World-Cup machinery
(intl engine, availability/injuries, fundamentals, simulator, dispersion +
MOV challengers, model-select/Hedge, calibration, api-football client, WC
scripts/tests/config) and the cross-venue ARB scanner/executor/SX Bet (+ its
digest/telegram opt-in). Deposit bridge, Telegram bot, multi-tenant trading
KEPT. R1 sweep harness removed with MOV (result recorded above; all
recoverable from git history). ADDED: weekly club data refresh (Mon 06:00 UTC
daemon job: fetch results -> re-seed Glicko -> refit DC) fixing frozen-ratings
bug — settlement previously updated ratings only for WC fixtures. ADDED:
expected-goals readout (DC lambdas) on every club/CL prediction, persisted +
shown in the daily report xG column. Alias fixes: Espanyol was wrongly
receiving Barcelona's rating (fuzzy-match bug, now pinned by alias).

## Phases (each gated; ship = beats incumbent on walk-forward CI)

### R1 — Quick wins from existing code (no new data)
* Port MOV-Glicko to clubs (`seed_glicko_mov` on club_results.csv; margin
  carries signal — Hvattum & Arntzen).
* Add time-decay to the club DC fit (`decay_weight` already exists in
  dixon_coles.py; half-life ≈ 390 days per literature).
* Grid-sweep club ensemble weights + draw_rho/home_mu on the train split.
* Extend dual-logging (`model_predictions`) + Hedge selection to clubs.
* Fit per-league isotonic calibration from walk-forward preds
  (`ensemble_calibration_club.json` — file supported, not yet fit).

### R1 — RESULT (2026-08-05): GATE FAILED, incumbent stands
1944-config sweep (MOV-Glicko, home_mu, draw_rho, DC half-life, weights,
isotonic calibration) on an inner val season (2024-25), final gate ONCE on
2025-26 test: best-on-val config improved test RPS by only +0.00046,
CI95 [-0.00129, +0.00218] includes 0 -> val-season noise, not signal.
Shipped config unchanged. Directional notes for later re-sweeps: ALL top-10
val configs used the MOV variant and home_mu 0.20 (vs shipped 0.30), form
weight 0.25 (vs 0.5); DC half-life was flat 390-720. Re-run
scripts/backtest_club_r1.py once 2026-27 season data accumulates.

### R2 — Cross-league Elo → unlock Champions League
* Ingest European results (CL/EL/ECL) from football-data.org (already a
  venue we poll) + our domestic CSV into one result stream.
* Build ClubElo-style Elo: home advantage + goal-diff multiplier, k tuned
  on train years. Optionally cross-check/bootstrap vs clubelo.com's free
  API (`api.clubelo.com/<club>`) — but own the replay so it's reproducible.
* New `CL` routing: ensemble = cross-league Elo + (existing per-league DC
  attack/defence, which transfer since they're goal rates) + market anchor.
* Gate: walk-forward on the last 2 CL seasons (football-data.org history).

### R2 — RESULT (2026-08-05): GATE PASSED, SHIPPED (default ON, paper)
ClubElo cross-league Elo engine (EuropeanStrategyEngine) now prices CL.
Walk-forward gate (scripts/backtest_cl.py): tuned HA=65, draw_rho=0.26 on
seasons 2023+2024; Elo+DC blend (dc_weight=1.0) beat pure Elo on train.
Held-out CL season 2025 (n=173 scorable): naive RPS 0.2566 (acc 26.6%%) ->
Elo+DC 0.2001 (acc 59.5%%); bootstrap 95%% CI on per-match RPS improvement
[+0.032, +0.081], excludes 0. Wired in main.py (CL route), config
cl_* fields, 5 tests. Data: data/cl_results.csv (football-data.org),
data/clubelo/*.csv monthly + data/clubelo_latest.csv (fetch_clubelo.py).

OPEN follow-ups: (1) schedule `python scripts/fetch_clubelo.py --latest`
daily (daemon/cron) before CL season — the live engine reads that file;
(2) ClubElo summer snapshots transiently omit some clubs (e.g. Bayern
absent from the 2026-07 snapshot) -> those ties fall back to naive; in-season
snapshots are complete (187/189 resolved in backtest), and CL starts mid-Sep;
(3) 2 name bridges added (Athletic Club->Bilbao, Union SG->St Gillis).

### R3 — xG layer
* Pull Understat shot/xG history, big-5 leagues 2014→now (no official API;
  scrape respectfully + cache; FBref/StatsBomb open data as fallback).
* Refit DC on blended goals: `g_eff = α·xG + (1-α)·goals` (α tuned; lit
  suggests α≈0.6–0.7). Add SPI-style xG offense/defence ratings as an
  ensemble component.
* Gate + also track the home-win subset specifically (Wilkens effect).

### R3 — RESULT (2026-08-06): real xG fetched; does NOT beat goals (gate not passed)
Bought real Understat match xG via Apify (constructive_calm actor), 5 seasons
2021/22-2024/25 complete for all big-5 (~8k matches, ~$27 of $29; partial
2025/26 for PL/PD/SA — truncated at cost cap, immaterial to the gate).
scripts/fetch_understat_xg.py + data/club_xg.csv.

Gate (scripts/backtest_club_xg.py, held-out 2024/25, n=1752): swapping the
Dixon-Coles input goals->xG, or blending both, did NOT beat the goals baseline:
baseline RPS 0.2030 (acc 52.9%), xG 0.2031, blend 0.2028; bootstrap CIs on the
per-match improvement include 0 (not distinguishable). At one season (n=1752)
the effect is below detection; not shipped into the probability model (same
gate discipline as R1). Re-test as seasons accumulate.

TRAP FOUND: Understat's built-in match `forecast` looked market-beating (RPS
0.163, acc 61%) but is POST-HOC — computed from the shots taken IN that match,
so it has lookahead. NOT a usable pre-match predictor; excluded. Kept only as a
labelled reference.

STILL TO DELIVER (operator asked for it): surface predicted expected-goals
(xG-DC lambda) on predictions as a DISPLAY readout (e.g. "Man City 2.07 - 1.23
Liverpool"). This is information, not a probability change, so it needs no gate.
Needs an Understat->football-data.org name bridge + a Prediction xG field.

### R-ODDS — Free pre-match odds anchoring on ALL club fixtures
Problem: `anchor_to_market` only fired when Polymarket happened to list the
fixture, so most big-5 matches shipped raw, UNANCHORED model output.
Fix: a pluggable free odds provider (football-data.co.uk `fixtures.csv` —
no key, no signup, no quota, no cost) + `anchor_triple`, applied to every
scored club fixture. Flag `BETBOT_ODDS_ANCHOR`, DEFAULT OFF.

### R-ODDS — RESULT (2026-08-18): GATE PASSED on pre-match prices
Walk-forward, held-out 2025-26 season, n=1752 (scripts/backtest_club.py):

| model                       | acc %  | mean RPS | log-loss |
|-----------------------------|--------|----------|----------|
| ensemble (incumbent)        | 50.74  | 0.2033   | 0.9995   |
| **anchored_pre (this)**     | 51.71  | 0.2005   | 0.9893   |
| anchored_close (optimistic) | 52.00  | 0.2003   | 0.9885   |
| market_pre (early price)    | 53.42  | 0.19776  | 0.9788   |
| market_close (closing line) | 53.37  | 0.19778  | 0.9783   |

Paired per-match RPS improvement (ensemble - anchored_pre): **+0.00277/match,
bootstrap 95% CI [+0.00200, +0.00357] — excludes 0.** Coverage 1752/1752
(100%), so the anchored-subset and all-fixture effects are identical.

ANTI-LOOKAHEAD: the gate uses the EARLY-WEEK price columns (PSH/B365H), the
same family the live fixtures.csv feed publishes, NOT the closing columns
(PSCH/B365CH). Closing-odds optimism measured directly on this sample:
market_close RPS 0.19778 vs market_pre 0.19776 (+0.00002) — i.e. essentially
nil for big-5 1X2 at the RPS level in 2025-26. Anchoring to closing instead of
pre-match would have added only +0.00015/match.

CEILING CHECK PASSED: anchored 51.71% sits strictly between the ensemble
(50.74%) and the market line (53.42%). Anchoring is SHRINKAGE toward the
price — it moves us toward market-level accuracy and CANNOT exceed it. It is
NOT an edge and must never be described as one.

NAME SAFETY: explicit alias table only, no fuzzy matching (the Espanyol /
Barcelona incident). Season audit 2025-26: 1752/1752 in-scope fixtures
resolved, 0 skipped, 0 mis-resolutions, 0 clubs appearing in two leagues.
Live 2026-27 football-data.org namespace: 88/96 club names resolve (91.7%);
the 8 misses are newly promoted sides (Hull, Coventry, Malaga, Deportivo,
Racing Santander, Paderborn, Elversberg, Le Mans) that have no Glicko rating
either, so those fixtures already take the naive path. Unresolvable = SKIP +
log, never a guess.

PRE-REGISTERED LIVE GATE (before BETBOT_ODDS_ANCHOR may be turned on):
>= 200 paired settled matches, anchored vs unanchored on the SAME fixtures,
paired bootstrap 95% CI on per-match RPS excluding 0. The live ledger held
n=7 as of 2026-08-18, and everything before 2026-08-17 is excluded (poisoned
by the degenerate 0/0/100 AWAY bug). Flag stays OFF until then.

OPEN follow-up: `python scripts/audit_odds_aliases.py` after each matchday and
add aliases ONLY from its unresolved list; `python scripts/fetch_club_odds.py`
regenerates data/club_odds.csv for the backtest.

### R4 — Player & lineup layer (the "EA-style" part)
* Squad strength: aggregate player ratings (Sofascore/FotMob public
  ratings; EA FC ratings dataset as seasonal prior) weighted by expected
  minutes → team attack/defence adjustments.
* Extend the existing api-football injuries integration (currently WC-only
  in `_availability_adjustments`) to club leagues.
* Confirmed-lineup re-price: api-football lineups endpoint ~20–60 min
  pre-kickoff → re-run prediction → only then compare vs market. This is
  where genuine +CLV lives.
* Gate: does the lineup-adjusted model move predictions TOWARD the closing
  line faster than the un-adjusted one? (CLV proxy.)

### R5 — Meta-blend + selective betting policy
* Stacked meta-learner (logistic regression first; GBM only if it gates)
  over component probabilities + context features (rest days, congestion,
  Europe-midweek travel, promoted-team flag).
* CLV instrumentation: log our price AND the market price at prediction
  time and at close; report weekly CLV per league. Bet only where
  divergence > threshold AND model confidence high (systematic-review
  finding: profits come from selective, confident bets).

## Data sources
| need | source | status |
|---|---|---|
| results+closing odds, big-5 | football-data.co.uk | DONE (club_results.csv) |
| CL/EL fixtures+results | football-data.org (have key) | poll now |
| cross-league Elo sanity | api.clubelo.com (free CSV) | new |
| xG/shots | Understat (scrape), FBref, StatsBomb open | new |
| injuries/lineups | api-football (have key) | extend WC→clubs |
| player ratings | Sofascore/FotMob public, EA FC dataset | new |

## Success criteria (honest)
* R1–R3: club RPS ≤ 0.200 on held-out season; CL engine beats naive on CL
  holdout with CI excluding 0.
* R4–R5: positive average CLV on placed paper bets over ≥300 bets before
  any live-stake increase. "Most accurate" operationally = at/above the
  closing line's RPS on our covered leagues; claims beyond that require
  the CLV record, not backtests.
