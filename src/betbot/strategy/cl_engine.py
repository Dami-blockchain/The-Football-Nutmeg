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
* **Stale/missing snapshot** — if ``clubelo_latest.csv`` is absent or older than
  14 days (by its own From/To date columns, not file mtime), the Elo edge is
  untrustworthy; we log a warning once and fall back to naive when a rating is
  missing.

Scope: Champions League only; the caller routes domestic leagues to the club
engine and the World Cup to the international engine.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from datetime import date
from pathlib import Path

from betbot.data.models import FixtureForm
from betbot.exchanges.matcher import TeamAliasResolver, normalize
from betbot.logging import get_logger
from betbot.strategy import dixon_coles as dc
from betbot.strategy.engine import BetDecision, Outcome, Prediction, StrategyEngine
from betbot.strategy.ensemble import anchor_to_market, log_pool
from betbot.strategy.glicko import DRAW_CAP, DRAW_FLOOR

log = get_logger(__name__)

STALE_AFTER_DAYS = 14


def _elo_probs(elo_home: float, elo_away: float, home_adv: float, draw_rho: float):
    """Elo 1X2 probabilities; draw split mirrors glicko.match_probabilities."""
    d = elo_home + home_adv - elo_away
    p_home_raw = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
    p_draw = draw_rho * (1.0 - abs(p_home_raw - 0.5) * 2.0)
    p_draw = min(DRAW_CAP, max(DRAW_FLOOR, p_draw))
    p_home = (1.0 - p_draw) * p_home_raw
    p_away = (1.0 - p_draw) * (1.0 - p_home_raw)
    return p_home, p_draw, p_away


def _load_snapshot(path: Path) -> tuple[dict[str, float], date | None]:
    """Read a ClubElo CSV -> ({club: elo}, newest 'To' date seen)."""
    snap: dict[str, float] = {}
    newest: date | None = None
    try:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                club = (row.get("Club") or "").strip()
                try:
                    elo = float(row["Elo"])
                except (KeyError, ValueError, TypeError):
                    continue
                if club:
                    snap[club] = elo
                for col in ("To", "From"):
                    try:
                        dt = date.fromisoformat((row.get(col) or "").strip())
                    except ValueError:
                        continue
                    if newest is None or dt > newest:
                        newest = dt
    except OSError:
        return {}, None
    return snap, newest


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

        if snapshot is not None:
            self._snapshot = snapshot
            self._clubs = list(snapshot.keys())
            self._snapshot_date: date | None = None
        else:
            path = Path(settings.clubelo_latest_path)
            if not path.exists():
                alt = _newest_clubelo_dir(Path("data/clubelo"))
                if alt is not None:
                    path = alt
            snap, snap_date = _load_snapshot(path)
            self._snapshot = snap
            self._clubs = list(snap.keys())
            self._snapshot_date = snap_date
            self._check_freshness(path)

        self._res_cache: dict[str, str | None] = {}

    def _check_freshness(self, path: Path) -> None:
        if self._warned:
            return
        stale = False
        reason = ""
        if not self._snapshot:
            stale, reason = True, "missing_or_empty"
        elif self._snapshot_date is not None:
            age = (date.today() - self._snapshot_date).days
            if age > STALE_AFTER_DAYS:
                stale, reason = True, f"stale_{age}d"
        if stale:
            log.warning(
                "clubelo_snapshot_stale", path=str(path), reason=reason,
                snapshot_date=str(self._snapshot_date),
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
            s.cl_elo_home_adv, s.cl_elo_draw_rho,
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
        p_model = {
            Outcome.HOME: prediction.p_home,
            Outcome.DRAW: prediction.p_draw,
            Outcome.AWAY: prediction.p_away,
        }[outcome]
        p_final = anchor_to_market(p_model, market_price, w_model, s.cl_weight_market)
        field_name = {
            Outcome.HOME: "p_home",
            Outcome.DRAW: "p_draw",
            Outcome.AWAY: "p_away",
        }[outcome]
        anchored = dataclasses.replace(prediction, **{field_name: p_final})
        return self._base.decide_with_market(
            anchored, outcome, market_price, require_edge=require_edge
        )
