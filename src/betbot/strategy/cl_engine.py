"""EuropeanStrategyEngine — cross-league Elo predictions for the Champions League.

Same public interface as the naive :class:`StrategyEngine`, the
:class:`InternationalStrategyEngine` and the :class:`ClubStrategyEngine`
(``predict`` -> :class:`Prediction`, ``decide_with_market`` -> :class:`BetDecision`),
so storage, settlement, backtest and the API keep working unchanged. Internally
it prices a Champions League match from a single **cross-league ClubElo rating
scale** rather than the per-league-seeded Glicko club ratings:

* the club Glicko/Dixon-Coles engine calibrates ratings WITHIN a domestic
  league, so a Serie A 1600 and a Bundesliga 1600 are not comparable — useless
  for the CL, which mixes leagues in every tie;
* ClubElo (``data/clubelo_latest.csv``, monthly snapshots under
  ``data/clubelo/``) is a single Europe-wide Elo scale, so ``elo_home - elo_away``
  is meaningful across leagues. That is what unlocks CL predictions (R2).

Pricing (tuned by ``scripts/backtest_cl.py``, walk-forward, gate CI>0):

* **Elo** — ``d = elo_home + home_adv - elo_away``; ``p_home_raw`` from the Elo
  logistic; the draw split mirrors :func:`glicko.match_probabilities` exactly
  (same floor/cap), with a tuned ``draw_rho``;
* **Dixon-Coles** club goal model, log-pooled in when BOTH clubs are present in
  the club DC params AND the tuned blend won the gate (``cl_weight_dc`` > 0);
* **the market line**, blended in logit space at decide time, so only genuine
  model-vs-venue divergence gets bet.

Two safety guards (mirroring ClubStrategyEngine):

* **Unresolved teams** — a club not found in the ClubElo snapshot (name
  bridging via ``normalize`` + ``TeamAliasResolver``) has no cross-league
  rating, so we defer to the form-based naive engine, exactly like the club
  engine's unknown-team guard.
* **Stale/missing snapshot** — the two cases differ and it matters:
  - a snapshot that is *present but old* (older than 14 days by its own ``From``
    column, not ``To`` and not file mtime) is STILL used: ``predict`` prices
    every resolvable club off those aged ratings. Staleness here is a
    read-only signal (``snapshot_stale``/``snapshot_age_days``, logged once at
    ERROR) with no hard cutoff — there is intentionally no code that drops an
    old-but-present file to naive, because a month-old cross-league snapshot is
    a smaller error than pricing off nothing (or off a mis-scaled substitute).
    The honest description of the degradation is "CL priced off N-day-old
    ratings", and it worsens with age.
  - a snapshot that is *absent or unparseable* leaves the club list empty, so
    every team is unresolved and every fixture takes the naive path via the
    unresolved-team guard below — degraded, but never wrong.

Scope: Champions League only; the caller routes domestic leagues to the club
engine and the World Cup to the international engine.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

from betbot.data.clubelo import MAX_ELO, MIN_ELO
from betbot.data.models import FixtureForm
from betbot.exchanges.matcher import TeamAliasResolver, normalize
from betbot.logging import get_logger
from betbot.strategy import dixon_coles as dc
from betbot.strategy.engine import BetDecision, Outcome, Prediction, StrategyEngine
from betbot.strategy.ensemble import anchor_to_market, log_pool
from betbot.strategy.glicko import DRAW_CAP, DRAW_FLOOR

log = get_logger(__name__)

STALE_AFTER_DAYS = 14

#: Incumbent (api.clubelo-scale) constants, kept for the dual-log SHADOW only.
#: When the engine switched to the compressed site scale (home_adv 65->51,
#: scale 400->312), these froze the api-scale config so the shadow can price
#: each fixture the way the old engine would have. They are NOT settings on
#: purpose — the shadow is the fixed baseline we are measuring the switch
#: against, not a knob to tune.
INCUMBENT_ELO_HOME_ADV = 65.0
INCUMBENT_ELO_SCALE = 400.0

#: Degenerate-country guard thresholds (see _drop_degenerate_countries).
#: ClubElo emits one identical placeholder Elo for every club of a league it
#: has stopped rating; we drop a whole country only when a single Elo value is
#: shared by at least DEGENERATE_MIN_CLUSTER of its clubs AND those clubs are a
#: majority (>= DEGENERATE_MIN_FRACTION) of the country -- the stopped-league
#: signature, distinct from the 3-way bottom-table promotion floor a healthy
#: country shows.
DEGENERATE_MIN_CLUSTER = 3
DEGENERATE_MIN_FRACTION = 0.5


def _elo_probs(elo_home: float, elo_away: float, home_adv: float, draw_rho: float,
               scale: float = 400.0):
    """Elo 1X2 probabilities; draw split mirrors glicko.match_probabilities.

    ``scale`` is the logistic divisor (classic Elo = 400). A smaller divisor
    sharpens the same rating gap, needed when the snapshot ratings are on a
    compressed scale (e.g. ClubElo site-scale vs api.clubelo scale).
    """
    d = elo_home + home_adv - elo_away
    p_home_raw = 1.0 / (1.0 + 10.0 ** (-d / scale))
    p_draw = draw_rho * (1.0 - abs(p_home_raw - 0.5) * 2.0)
    p_draw = min(DRAW_CAP, max(DRAW_FLOOR, p_draw))
    p_home = (1.0 - p_draw) * p_home_raw
    p_away = (1.0 - p_draw) * (1.0 - p_home_raw)
    return p_home, p_draw, p_away


def _load_snapshot(path: Path) -> tuple[dict[str, float], date | None]:
    """Read a ClubElo CSV -> ({club: elo}, newest ``From`` date seen).

    The date is the newest **From**, not **To**. ClubElo's ``To`` is the end of
    a rating's *validity window* and therefore sits in the future for a current
    snapshot, so an age computed from it is negative and the 14-day staleness
    guard below could never fire when it mattered: the live file measured -10
    days old while actually being 3 days stale. ``From`` is when the rating was
    last recomputed, which is the honest measure.

    Elo values outside the sanity band are dropped. A truncated final row such
    as ``2,Bayern,GER,1,20`` otherwise loads Bayern at Elo 20.0 and prices them
    as the worst team in Europe — silently wrong, which is the worst kind.

    Degenerate-country guard: ClubElo emits one identical placeholder Elo for
    every club of a league it has stopped rating -- the whole Ukrainian top
    flight sat at a single 1241.82 in the 2026-08-31 snapshot, Shakhtar and
    Dynamo Kyiv included, though their real API-scale ratings were 1587/1441.
    Such a value is inside the sanity band and its clubs resolve by name, so
    neither existing guard catches it and a CL side is priced off a bogus
    rating. ``_drop_degenerate_countries`` drops every club of such a country
    so its fixtures fall to the naive form path.
    """
    snap: dict[str, float] = {}
    by_country: dict[str, list[tuple[str, float]]] = {}
    newest: date | None = None
    dropped = 0
    try:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                club = (row.get("Club") or "").strip()
                try:
                    elo = float(row["Elo"])
                except (KeyError, ValueError, TypeError):
                    continue
                if not (MIN_ELO <= elo <= MAX_ELO):
                    dropped += 1
                    continue
                if club:
                    snap[club] = elo
                    by_country.setdefault(
                        (row.get("Country") or "").strip(), []
                    ).append((club, elo))
                try:
                    dt = date.fromisoformat((row.get("From") or "").strip())
                except ValueError:
                    continue
                if newest is None or dt > newest:
                    newest = dt
    except OSError:
        return {}, None
    # Run the degenerate-country guard FIRST so kept= below is the final count.
    _drop_degenerate_countries(snap, by_country)
    if dropped:
        log.error(
            "clubelo_snapshot_rows_dropped",
            path=str(path), dropped=dropped, kept=len(snap),
            reason="elo_out_of_sanity_band",
        )
    return snap, newest


def _drop_degenerate_countries(
    snap: dict[str, float],
    by_country: dict[str, list[tuple[str, float]]],
) -> None:
    """Drop every club of a country whose ratings collapsed to one value.

    Mutates ``snap`` in place. A country is dropped only when a single Elo
    value is shared by ``>= DEGENERATE_MIN_CLUSTER`` of its clubs AND those
    clubs are ``>= DEGENERATE_MIN_FRACTION`` of the country (the stopped-league
    placeholder signature). The majority gate is deliberate: a literal
    any-three-identical rule would also drop the real, distinct top ratings of
    Belgium, the Netherlands and Romania -- whose three bottom clubs share a
    promotion-floor value -- silently pushing live CL sides such as PSV and
    Club Brugge to the naive path. Fires at ERROR (a whole country lost its
    cross-league signal); a sub-majority identical cluster is logged at WARNING
    and kept, so a later degenerate league stays answerable from the logs.
    """
    for country, entries in by_country.items():
        if len(entries) < DEGENERATE_MIN_CLUSTER:
            continue
        value, n = Counter(elo for _club, elo in entries).most_common(1)[0]
        if n < DEGENERATE_MIN_CLUSTER:
            continue
        if n >= DEGENERATE_MIN_FRACTION * len(entries):
            clubs = sorted(club for club, _elo in entries)
            for club in clubs:
                snap.pop(club, None)
            log.error(
                "clubelo_degenerate_country_dropped",
                country=country, shared_elo=value,
                cluster=n, country_clubs=len(entries), clubs=clubs,
                reason="identical_elo_majority_placeholder",
                impact="all_country_fixtures_fall_back_to_naive",
            )
        else:
            log.warning(
                "clubelo_identical_elo_cluster",
                country=country, shared_elo=value,
                cluster=n, country_clubs=len(entries),
                reason="benign_bottom_table_floor_not_dropped",
            )

def _load_dc_params(path: Path) -> dc.DCParams | None:
    try:
        return dc.DCParams.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return None


def _load_name_map(path: Path) -> dict[str, str]:
    """football-data.org normalised name -> dataset normalised name (DC keys)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in d.items()}
    except (OSError, ValueError):
        return {}


def _newest_clubelo_dir(directory: Path) -> Path | None:
    try:
        files = sorted(p for p in directory.glob("*.csv") if p.name[:4].isdigit())
    except OSError:
        return None
    return files[-1] if files else None


class EuropeanStrategyEngine:
    def __init__(
        self,
        settings,
        *,
        snapshot: dict[str, float] | None = None,
        dc_params: dc.DCParams | None = None,
        name_map: dict[str, str] | None = None,
        resolver: TeamAliasResolver | None = None,
    ) -> None:
        self._settings = settings
        self._base = StrategyEngine(settings)  # naive form engine: form + edge filter + fallback
        self._resolver = (
            resolver if resolver is not None
            else TeamAliasResolver.from_yaml("config/team_aliases.yaml")
        )
        self._dc_params = dc_params or _load_dc_params(Path(settings.dc_params_club_path))
        self._name_map = name_map if name_map is not None else _load_name_map(
            Path(settings.club_name_map_path)
        )
        self._warned = False
        # Read-only freshness seam (see _check_freshness).
        self.snapshot_stale = False
        self.snapshot_age_days: int | None = None
        self.snapshot_reason = "injected"

        # Dual-log shadow state (populated only when loading from files).
        self._shadow_snapshot: dict[str, float] = {}
        self._shadow_clubs: list[str] = []
        self._shadow_res_cache: dict[str, str | None] = {}

        if snapshot is not None:
            self._snapshot = snapshot
            self._clubs = list(snapshot.keys())
            self._snapshot_date: date | None = None
        else:
            # Primary = the fresh site-scale feed. Fallback stays site-scale
            # (newest dated file under data/clubelo_site/) — never the api-scale
            # pin, whose scale would not match the tuned site constants.
            path = Path(settings.cl_snapshot_path)
            if not path.exists():
                alt = _newest_clubelo_dir(Path("data/clubelo_site"))
                if alt is not None:
                    path = alt
            snap, snap_date = _load_snapshot(path)
            self._snapshot = snap
            self._clubs = list(snap.keys())
            self._snapshot_date = snap_date
            self._check_freshness(path)
            self._load_shadow(Path(settings.cl_shadow_snapshot_path))

        self._res_cache: dict[str, str | None] = {}

    def _check_freshness(self, path: Path) -> None:
        """Record and announce whether the snapshot behind us is usable.

        Logged at ERROR, not WARNING. Be precise about the two failure modes:
        a *missing/empty* snapshot leaves every team unresolved and drops CL to
        the naive form engine (~50.7% vs the CL engine's ~58.7% held-out
        accuracy); a *present-but-stale* snapshot is still used and CL is priced
        off those N-day-old ratings (a smaller, age-dependent error — there is
        no hard cutoff). Either way the degradation must not pass unnoticed.
        ``snapshot_stale``/``snapshot_age_days`` are the read-only seam an
        operator notifier can poll; this module does not page anyone itself.
        """
        stale = False
        reason = "fresh"
        age: int | None = None
        if not self._snapshot:
            stale, reason = True, "missing_or_empty"
        elif self._snapshot_date is not None:
            age = (date.today() - self._snapshot_date).days
            if age > STALE_AFTER_DAYS:
                stale, reason = True, f"stale_{age}d"
        else:
            stale, reason = True, "undated"

        self.snapshot_stale = stale
        self.snapshot_age_days = age
        self.snapshot_reason = reason

        if stale and not self._warned:
            # Distinguish the two: an empty snapshot really does fall back to
            # naive; a present-but-old one is still priced, just off aged ratings.
            impact = (
                "cl_predictions_fall_back_to_naive"
                if not self._snapshot
                else f"cl_priced_off_{age}d_old_ratings"
            )
            log.error(
                "clubelo_snapshot_stale",
                path=str(path), reason=reason, age_days=age,
                clubs=len(self._snapshot),
                snapshot_date=None if self._snapshot_date is None else str(self._snapshot_date),
                impact=impact,
            )
            self._warned = True

    def _resolve(self, name: str) -> str | None:
        if name in self._res_cache:
            return self._res_cache[name]
        hit = self._resolver.match(name, self._clubs) if self._clubs else None
        self._res_cache[name] = hit
        return hit

    def _dc_key(self, name: str) -> str:
        n = normalize(name)
        return self._name_map.get(n, n)

    def _load_shadow(self, path: Path) -> None:
        """Load the api.clubelo pin as the dual-log shadow. Best-effort: a
        missing or unparseable pin simply means no shadow rows (dual_triples
        returns None), never an error into the engine."""
        try:
            snap, _ = _load_snapshot(path)
        except Exception:  # noqa: BLE001 — shadow load must never break the engine
            snap = {}
        self._shadow_snapshot = snap or {}
        self._shadow_clubs = list(self._shadow_snapshot.keys())

    def _resolve_shadow(self, name: str) -> str | None:
        if name in self._shadow_res_cache:
            return self._shadow_res_cache[name]
        hit = (
            self._resolver.match(name, self._shadow_clubs)
            if self._shadow_clubs else None
        )
        self._shadow_res_cache[name] = hit
        return hit

    def _model_triple(
        self, snapshot: dict[str, float], hit_h: str, hit_a: str,
        home_name: str, away_name: str, home_adv: float, scale: float,
    ) -> tuple[float, float, float]:
        """Elo(+DC) triple for one (snapshot, constants) pair — the SAME blend
        predict() serves, so the shadow is a like-for-like comparison."""
        s = self._settings
        elo_probs = _elo_probs(
            snapshot[hit_h], snapshot[hit_a],
            home_adv, s.cl_elo_draw_rho, scale,
        )
        components: list[tuple[float, tuple[float, float, float]]] = [
            (s.cl_weight_elo, elo_probs)
        ]
        if s.cl_weight_dc > 0 and self._dc_params is not None:
            kh, ka = self._dc_key(home_name), self._dc_key(away_name)
            if kh in self._dc_params.teams and ka in self._dc_params.teams:
                components.append((s.cl_weight_dc, dc.match_probabilities(
                    self._dc_params, kh, ka, home_field=True)))
        return log_pool(components)

    def dual_triples(
        self, home_name: str, away_name: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """(shadow, served) 1X2 triples for the ``model_predictions`` dual-log.

        * ``served`` = the SITE-scale config the engine now prices off, stored
          in the ensemble slot (``e_*``);
        * ``shadow`` = the frozen api.clubelo incumbent (INCUMBENT_* constants
          on the ageing pin), stored in the glicko slot (``g_*``).

        Returns ``None`` unless BOTH feeds resolve BOTH clubs, so the forward
        ledger only records fixtures with a genuine head-to-head. Purely
        observational: it never changes what ``predict()`` serves.

        NOTE for analysts: these CL rows reuse the club dual-log's columns with
        different meaning (glicko slot = api shadow, ensemble slot = site
        served). Separate them from club rows by joining ``fixture_id`` to
        ``predictions.competition_code == 'CL'``.
        """
        hit_h = self._resolve(home_name)
        hit_a = self._resolve(away_name)
        if hit_h is None or hit_a is None:
            return None
        sh_h = self._resolve_shadow(home_name)
        sh_a = self._resolve_shadow(away_name)
        if sh_h is None or sh_a is None:
            return None
        s = self._settings
        served = self._model_triple(
            self._snapshot, hit_h, hit_a, home_name, away_name,
            s.cl_elo_home_adv, s.cl_elo_scale,
        )
        shadow = self._model_triple(
            self._shadow_snapshot, sh_h, sh_a, home_name, away_name,
            INCUMBENT_ELO_HOME_ADV, INCUMBENT_ELO_SCALE,
        )
        return shadow, served

    def model_weight(self) -> float:
        """Summed log-pool weight of the ACTIVE model components.

        The Elo component is always on; DC only when it is weighted AND both
        clubs are in the params — but that is per-fixture, so this returns the
        weight the CL blend carries in the common (DC-available) case, which
        is what the odds anchor uses as its model-side denominator.
        """
        s = self._settings
        return s.cl_weight_elo + (
            s.cl_weight_dc if self._dc_params is not None else 0.0
        )

    def predict(
        self,
        fixture_form: FixtureForm,
        *,
        home_rating_adj: float = 0.0,
        away_rating_adj: float = 0.0,
    ) -> Prediction:
        s = self._settings
        fx = fixture_form.fixture
        home_name, away_name = fx.home_team.name, fx.away_team.name

        # Unresolved-team / missing-snapshot guard: without a cross-league Elo
        # rating for BOTH sides the Elo model is guessing, so defer to the
        # form-based naive engine (mirrors club_engine's unknown-team guard).
        hit_h = self._resolve(home_name)
        hit_a = self._resolve(away_name)
        if hit_h is None or hit_a is None:
            return self._base.predict(fixture_form)

        # Optional lineup-adjusted rating shift (R4a). ClubElo ratings are plain
        # floats, so the adjustment is added directly to eh/ea. Default 0.0
        # leaves the Elo edge — and every downstream number — byte-identical.
        eh = self._snapshot[hit_h] + home_rating_adj
        ea = self._snapshot[hit_a] + away_rating_adj

        elo_probs = _elo_probs(
            eh, ea,
            s.cl_elo_home_adv, s.cl_elo_draw_rho, s.cl_elo_scale,
        )
        components: list[tuple[float, tuple[float, float, float]]] = [
            (s.cl_weight_elo, elo_probs)
        ]
        home_xg: float | None = None
        away_xg: float | None = None
        if s.cl_weight_dc > 0 and self._dc_params is not None:
            kh, ka = self._dc_key(home_name), self._dc_key(away_name)
            if kh in self._dc_params.teams and ka in self._dc_params.teams:
                dc_probs = dc.match_probabilities(
                    self._dc_params, kh, ka, home_field=True
                )
                components.append((s.cl_weight_dc, dc_probs))
                lam_h, lam_a = dc.expected_goals(
                    self._dc_params, kh, ka, home_field=True
                )
                home_xg, away_xg = round(lam_h, 2), round(lam_a, 2)

        p_home, p_draw, p_away = log_pool(components)
        return Prediction(
            fixture_id=fx.id,
            competition_code=fx.competition_code,
            home_team=home_name,
            away_team=away_name,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            home_score=eh,  # store (possibly lineup-adjusted) Elo for transparency
            away_score=ea,
            draw_score=0.0,
            home_xg=home_xg,
            away_xg=away_xg,
        )

    def decide_with_market(
        self, prediction: Prediction, outcome: Outcome, market_price: float,
        *, require_edge: bool = True,
    ) -> BetDecision | None:
        s = self._settings
        # w_model = sum of the active model components (DC only when it fired).
        home_name, away_name = prediction.home_team, prediction.away_team
        dc_active = s.cl_weight_dc > 0 and self._dc_params is not None and (
            self._dc_key(home_name) in self._dc_params.teams
            and self._dc_key(away_name) in self._dc_params.teams
        )
        w_model = s.cl_weight_elo + (s.cl_weight_dc if dc_active else 0.0)
        # Single-anchor invariant — see ClubEnsembleEngine.decide_with_market.
        p_model = prediction.model_probability(outcome)
        p_final = anchor_to_market(p_model, market_price, w_model, s.cl_weight_market)
        anchored = prediction.anchored_to_market(outcome, p_final)
        return self._base.decide_with_market(
            anchored, outcome, market_price, require_edge=require_edge
        )
