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

### R2 — Cross-league Elo → unlock Champions League
* Ingest European results (CL/EL/ECL) from football-data.org (already a
  venue we poll) + our domestic CSV into one result stream.
* Build ClubElo-style Elo: home advantage + goal-diff multiplier, k tuned
  on train years. Optionally cross-check/bootstrap vs clubelo.com's free
  API (`api.clubelo.com/<club>`) — but own the replay so it's reproducible.
* New `CL` routing: ensemble = cross-league Elo + (existing per-league DC
  attack/defence, which transfer since they're goal rates) + market anchor.
* Gate: walk-forward on the last 2 CL seasons (football-data.org history).

### R3 — xG layer
* Pull Understat shot/xG history, big-5 leagues 2014→now (no official API;
  scrape respectfully + cache; FBref/StatsBomb open data as fallback).
* Refit DC on blended goals: `g_eff = α·xG + (1-α)·goals` (α tuned; lit
  suggests α≈0.6–0.7). Add SPI-style xG offense/defence ratings as an
  ensemble component.
* Gate + also track the home-win subset specifically (Wilkens effect).

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
