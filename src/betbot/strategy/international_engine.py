"""InternationalStrategyEngine — Glicko-2 predictions for WC fixtures (Phase 5.5).

Exposes the SAME interface the scoring loop expects (``predict`` →
:class:`Prediction`, ``decide_with_market`` → :class:`BetDecision`), so storage,
settlement, backtest, and the API keep working unchanged. Internally it ignores
the club-football form snapshot (meaningless for nations) and prices the match
from each team's stored Glicko rating. Host-nation home advantage is applied
ONLY for the 2026 hosts. Paper-mode only — the live-order guard for
INTERNATIONAL_COMPETITIONS keeps WC off real money regardless.
"""

from __future__ import annotations

from collections.abc import Callable

from betbot.data.models import FixtureForm
from betbot.exchanges.matcher import normalize
from betbot.strategy.engine import BetDecision, Outcome, Prediction, StrategyEngine
from betbot.strategy.glicko import Glicko2Rating, match_probabilities

# 2026 World Cup hosts — genuine home advantage applies only to these.
HOST_NATIONS_2026 = frozenset({"united states", "usa", "canada", "mexico"})


def _is_host(name: str) -> bool:
    return normalize(name) in HOST_NATIONS_2026


class InternationalStrategyEngine:
    def __init__(self, settings, get_rating: Callable[[str], Glicko2Rating] | None = None) -> None:
        self._settings = settings
        self._base = StrategyEngine(settings)  # reuse the edge filter unchanged
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

    def predict(self, fixture_form: FixtureForm) -> Prediction:
        s = self._settings
        fx = fixture_form.fixture
        home_name, away_name = fx.home_team.name, fx.away_team.name
        rh = self._get_rating(home_name)
        ra = self._get_rating(away_name)
        home_field = s.glicko_host_home_mu if _is_host(home_name) else 0.0
        p_home, p_draw, p_away = match_probabilities(
            rh, ra, home_field_mu=home_field, draw_rho=s.glicko_draw_rho
        )
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
        self, prediction: Prediction, outcome: Outcome, market_price: float
    ) -> BetDecision | None:
        return self._base.decide_with_market(prediction, outcome, market_price)
