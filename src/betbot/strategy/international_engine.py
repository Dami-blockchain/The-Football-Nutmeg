"""InternationalStrategyEngine — ensemble predictions for WC fixtures.

Exposes the SAME interface the scoring loop expects (``predict`` →
:class:`Prediction`, ``decide_with_market`` → :class:`BetDecision`), so
storage, settlement, backtest, and the API keep working unchanged. Internally
it ignores the club-football form snapshot (meaningless for nations) and
prices the match from an ensemble:

* **Glicko-2** stored ratings (win/draw/loss signal), as before;
* **Dixon-Coles** goal model (``data/dc_params.json``, fit by
  ``scripts/fit_dixon_coles.py`` with Klement-fundamentals priors) — loaded
  when present, silently skipped when not (pure-Glicko fallback);
* **the market line**, blended in logit space at decide time, which shrinks
  marginal edges so only genuine model-vs-venue divergence gets bet;
* optional isotonic calibration (``data/ensemble_calibration.json``, fit by
  the backtest script).

Host-nation home advantage is applied ONLY for the 2026 hosts, in both
component models. The live-order guard for INTERNATIONAL_COMPETITIONS still
applies regardless of anything this engine outputs.
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
    blend,
    calibrate,
    log_pool,
    EnsembleWeights,
)
from betbot.strategy.dispersion import apply_dispersion
from betbot.strategy.glicko import Glicko2Rating, match_probabilities
from betbot.strategy.model_select import hedge_weights

# 2026 World Cup hosts — genuine home advantage applies only to these.
HOST_NATIONS_2026 = frozenset({"united states", "usa", "canada", "mexico"})


def _is_host(name: str) -> bool:
    return normalize(name) in HOST_NATIONS_2026


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


def _safe_model_losses() -> tuple[float, float, int]:
    """Default loss source: the model_predictions table. Degrades to
    no-evidence (50/50 Hedge) rather than crashing when the DB isn't up."""
    try:
        from betbot.storage.repos import model_select_losses

        return model_select_losses()
    except Exception:  # noqa: BLE001 — prediction must never die on telemetry
        return (0.0, 0.0, 0)


def _safe_record_model_prediction(*args) -> None:
    """Default recorder; same never-crash contract as _safe_model_losses."""
    try:
        from betbot.storage.repos import upsert_model_prediction

        upsert_model_prediction(*args)
    except Exception:  # noqa: BLE001
        pass


class InternationalStrategyEngine:
    def __init__(
        self,
        settings,
        get_rating: Callable[[str], Glicko2Rating] | None = None,
        *,
        dc_params: dc.DCParams | None = None,
        calibrators: tuple[IsotonicCalibrator, ...] | None = None,
        get_model_losses: Callable[[], tuple[float, float, int]] | None = None,
        record_model_prediction: Callable[..., None] | None = None,
    ) -> None:
        self._settings = settings
        self._base = StrategyEngine(settings)  # reuse the edge filter unchanged
        self._dc_params = dc_params or _load_dc_params(Path(settings.dc_params_path))
        self._calibrators = calibrators or _load_calibrators(
            Path(settings.ensemble_calibration_path)
        )
        self._weights = EnsembleWeights(
            glicko=settings.ensemble_weight_glicko,
            dixon_coles=settings.ensemble_weight_dc,
            market=settings.ensemble_weight_market,
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
        if get_model_losses is not None:
            self._get_model_losses = get_model_losses
        else:
            self._get_model_losses = _safe_model_losses
        if record_model_prediction is not None:
            self._record_model_prediction = record_model_prediction
        else:
            self._record_model_prediction = _safe_record_model_prediction

    def predict(
        self,
        fixture_form: FixtureForm,
        *,
        home_rating_adj: float = 0.0,
        away_rating_adj: float = 0.0,
    ) -> Prediction:
        """Price a WC fixture from the Glicko/DC ensemble.

        ``home_rating_adj`` / ``away_rating_adj`` shift each team's Glicko
        rating by that many points BEFORE ``match_probabilities``. They are how
        the optional injury/availability signal feeds in: a negative value for
        the depleted side lowers its win probability (and lifts draw/opponent).
        Default ``0.0`` reproduces the prior behaviour EXACTLY (no branch taken,
        identical ratings used). The stored ``home_score``/``away_score``
        reflect the ADJUSTED ratings so the adjustment is transparent in the
        logged prediction row. The DC component is rating-independent and is
        left untouched.
        """
        s = self._settings
        fx = fixture_form.fixture
        home_name, away_name = fx.home_team.name, fx.away_team.name
        rh = self._get_rating(home_name)
        ra = self._get_rating(away_name)
        if home_rating_adj:
            rh = dataclasses.replace(rh, rating=rh.rating + home_rating_adj)
        if away_rating_adj:
            ra = dataclasses.replace(ra, rating=ra.rating + away_rating_adj)
        home_field = s.glicko_host_home_mu if _is_host(home_name) else 0.0
        glicko_probs = match_probabilities(
            rh, ra, home_field_mu=home_field, draw_rho=s.glicko_draw_rho
        )
        recorded: tuple[tuple[float, float, float], tuple[float, float, float],
                         tuple[float, float]] | None = None
        if self._dc_params is not None:
            dc_probs = dc.match_probabilities(
                self._dc_params,
                normalize(home_name),
                normalize(away_name),
                home_field=_is_host(home_name),
            )
            ens_probs = blend(glicko_probs, dc_probs, weights=self._weights)
            if s.model_select_enabled:
                # Online selection: weight glicko vs ensemble by Hedge over
                # their settled-match RPS this tournament, and dual-log both
                # views so settlement can keep score.
                loss_g, loss_e, _n = self._get_model_losses()
                w_g, w_e = hedge_weights(loss_g, loss_e, eta=s.model_select_eta)
                probs = log_pool([(w_g, glicko_probs), (w_e, ens_probs)])
                recorded = (glicko_probs, ens_probs, (w_g, w_e))
            else:
                probs = ens_probs
        else:
            probs = glicko_probs  # pure-Glicko fallback (no DC artifact)
        # Dispersion challenger: sharpen the home/away split (draw untouched).
        # Always computed and dual-logged; only USED live when the flag is on,
        # so flag-off behaviour is byte-identical to before.
        disp = apply_dispersion(probs, s.dispersion_kappa)
        if recorded is not None:
            self._record_model_prediction(
                fx.id, home_name, away_name, recorded[0], recorded[1], recorded[2], disp
            )
        chosen = disp if s.dispersion_fix_enabled else probs
        p_home, p_draw, p_away = calibrate(chosen, self._calibrators)
        return Prediction(
            fixture_id=fx.id,
            competition_code=fx.competition_code,
            home_team=home_name,
            away_team=away_name,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            home_score=rh.rating,   # store the ratings for transparency
            away_score=ra.rating,
            draw_score=0.0,
        )

    def decide_with_market(
        self, prediction: Prediction, outcome: Outcome, market_price: float,
        *, require_edge: bool = True,
    ) -> BetDecision | None:
        w = self._weights
        w_model = w.glicko + (w.dixon_coles if self._dc_params is not None else 0.0)
        p_model = {
            Outcome.HOME: prediction.p_home,
            Outcome.DRAW: prediction.p_draw,
            Outcome.AWAY: prediction.p_away,
        }[outcome]
        p_final = anchor_to_market(p_model, market_price, w_model, w.market)
        field_name = {
            Outcome.HOME: "p_home",
            Outcome.DRAW: "p_draw",
            Outcome.AWAY: "p_away",
        }[outcome]
        anchored = dataclasses.replace(prediction, **{field_name: p_final})
        return self._base.decide_with_market(
            anchored, outcome, market_price, require_edge=require_edge
        )
