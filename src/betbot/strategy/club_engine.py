"""ClubStrategyEngine — ensemble predictions for domestic club fixtures.

Same public interface as the naive :class:`StrategyEngine` and the
:class:`InternationalStrategyEngine` (``predict`` -> :class:`Prediction`,
``decide_with_market`` -> :class:`BetDecision`), so storage, settlement,
backtest and the API keep working unchanged. Internally it prices a club match
from an ensemble instead of the form-only softmax:

* **Glicko-2** club ratings (``scripts/seed_glicko_club.py``), with an
  always-on home-field boost — real domestic home advantage, unlike the
  neutral-venue World Cup;
* **Dixon-Coles** club goal model (``data/dc_params_club.json``, fit by
  ``scripts/fit_dixon_coles_club.py``) — loaded when present, else skipped;
* **recent form** (the naive engine's own softmax) folded in as a small third
  component, since form genuinely carries signal for clubs (a nation plays too
  rarely for it to matter, which is why the WC engine ignores it);
* **the market line**, blended in logit space at decide time, so only genuine
  model-vs-venue divergence gets bet.

Two safety guards:

* **Unknown teams** (newly promoted, no rating history) have a Glicko RD at/above
  the default; for those the ensemble would be guessing, so we defer to the
  form-based naive engine.
* Dixon-Coles keys are dataset-named ("man city"); live fixtures are
  football-data.org-named ("Manchester City FC"). ``data/club_name_map.json``
  (written by the seeder) bridges the two.

Scope: domestic leagues only. Champions League mixes teams from different
leagues, whose ratings aren't calibrated against each other in the seeding
data, so CL is left on its previous path by the caller.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

from betbot.data.models import FixtureForm
from betbot.exchanges.matcher import normalize
from betbot.strategy import dixon_coles as dc
from betbot.strategy.engine import BetDecision, Outcome, Prediction, StrategyEngine
from betbot.strategy.ensemble import (
    IsotonicCalibrator,
    anchor_to_market,
    calibrate,
    log_pool,
)
from betbot.strategy.glicko import Glicko2Rating, match_probabilities


def _load_dc_params(path: Path) -> dc.DCParams | None:
    try:
        return dc.DCParams.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return None


def _load_calibrators(path: Path) -> tuple[IsotonicCalibrator, ...] | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            IsotonicCalibrator(xs=d[k]["xs"], ys=d[k]["ys"])
            for k in ("home", "draw", "away")
        )
    except (OSError, ValueError, KeyError):
        return None


def _load_name_map(path: Path) -> dict[str, str]:
    """football-data.org normalised name -> dataset normalised name (DC keys)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in d.items()}
    except (OSError, ValueError):
        return {}


class ClubStrategyEngine:
    def __init__(
        self,
        settings,
        get_rating: Callable[[str], Glicko2Rating] | None = None,
        *,
        dc_params: dc.DCParams | None = None,
        calibrators: tuple[IsotonicCalibrator, ...] | None = None,
        name_map: dict[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._base = StrategyEngine(settings)  # naive form engine: form signal + edge filter + fallback
        self._dc_params = dc_params or _load_dc_params(Path(settings.dc_params_club_path))
        self._calibrators = calibrators or _load_calibrators(
            Path(settings.ensemble_calibration_club_path)
        )
        self._name_map = name_map if name_map is not None else _load_name_map(
            Path(settings.club_name_map_path)
        )
        if get_rating is not None:
            self._get_rating = get_rating
        else:
            from betbot.storage.repos import get_rating as _gr

            self._get_rating = lambda name: _gr(
                name,
                default_rating=settings.glicko_default_rating,
                default_rd=settings.glicko_default_rd,
                default_vol=settings.glicko_default_vol,
            )

    def _dc_key(self, name: str) -> str:
        n = normalize(name)
        return self._name_map.get(n, n)

    def predict(self, fixture_form: FixtureForm) -> Prediction:
        s = self._settings
        fx = fixture_form.fixture
        home_name, away_name = fx.home_team.name, fx.away_team.name
        rh = self._get_rating(home_name)
        ra = self._get_rating(away_name)

        # Unknown-team guard: an RD at/above the default means we have no real
        # rating history for this side (e.g. a just-promoted club). The ensemble
        # would be guessing, so defer to the form-based naive engine.
        if rh.rd >= s.glicko_default_rd or ra.rd >= s.glicko_default_rd:
            return self._base.predict(fixture_form)

        glicko_probs = match_probabilities(
            rh, ra, home_field_mu=s.glicko_club_home_mu, draw_rho=s.glicko_club_draw_rho
        )
        components: list[tuple[float, tuple[float, float, float]]] = [
            (s.club_weight_glicko, glicko_probs)
        ]
        if self._dc_params is not None:
            dc_probs = dc.match_probabilities(
                self._dc_params, self._dc_key(home_name), self._dc_key(away_name),
                home_field=True,
            )
            components.append((s.club_weight_dc, dc_probs))
        if s.club_weight_form > 0:
            fp = self._base.predict(fixture_form)
            components.append((s.club_weight_form, (fp.p_home, fp.p_draw, fp.p_away)))

        probs = log_pool(components)
        p_home, p_draw, p_away = calibrate(probs, self._calibrators)
        return Prediction(
            fixture_id=fx.id,
            competition_code=fx.competition_code,
            home_team=home_name,
            away_team=away_name,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            home_score=rh.rating,   # store ratings for transparency
            away_score=ra.rating,
            draw_score=0.0,
        )

    def decide_with_market(
        self, prediction: Prediction, outcome: Outcome, market_price: float,
        *, require_edge: bool = True,
    ) -> BetDecision | None:
        s = self._settings
        w_model = s.club_weight_glicko + s.club_weight_form + (
            s.club_weight_dc if self._dc_params is not None else 0.0
        )
        p_model = {
            Outcome.HOME: prediction.p_home,
            Outcome.DRAW: prediction.p_draw,
            Outcome.AWAY: prediction.p_away,
        }[outcome]
        p_final = anchor_to_market(p_model, market_price, w_model, s.club_weight_market)
        field_name = {
            Outcome.HOME: "p_home",
            Outcome.DRAW: "p_draw",
            Outcome.AWAY: "p_away",
        }[outcome]
        anchored = dataclasses.replace(prediction, **{field_name: p_final})
        return self._base.decide_with_market(
            anchored, outcome, market_price, require_edge=require_edge
        )
