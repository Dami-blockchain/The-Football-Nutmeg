"""StrategyEngine — converts FixtureForm into probabilities + bet decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from betbot.data.models import FixtureForm, MatchOutcome
from betbot.logging import get_logger
from betbot.strategy.probabilities import edge, softmax

if TYPE_CHECKING:
    from betbot.config import Settings

log = get_logger(__name__)


# Re-export so adapter code can import a single Outcome name.
Outcome = MatchOutcome


@dataclass(frozen=True, slots=True)
class Prediction:
    """The model's view of one fixture."""

    fixture_id: int
    competition_code: str
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    home_score: float
    away_score: float
    draw_score: float

    @property
    def best_outcome(self) -> Outcome:
        triples = [
            (Outcome.HOME, self.p_home),
            (Outcome.DRAW, self.p_draw),
            (Outcome.AWAY, self.p_away),
        ]
        return max(triples, key=lambda kv: kv[1])[0]


@dataclass(frozen=True, slots=True)
class BetDecision:
    """Edge-filtered decision suitable for logging or live placement."""

    fixture_id: int
    competition_code: str
    home_team: str
    away_team: str
    outcome: Outcome
    our_probability: float
    market_price: float
    edge: float
    stake_usd: float
    rationale: str


class StrategyEngine:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    def predict(self, fixture_form: FixtureForm) -> Prediction:
        s = self._settings
        home_score = fixture_form.home_form.weighted_points + s.home_advantage
        away_score = fixture_form.away_form.weighted_points
        draw_score = s.draw_score

        probs = softmax([home_score, draw_score, away_score], s.softmax_temp)
        return Prediction(
            fixture_id=fixture_form.fixture.id,
            competition_code=fixture_form.fixture.competition_code,
            home_team=fixture_form.fixture.home_team.name,
            away_team=fixture_form.fixture.away_team.name,
            p_home=probs[0],
            p_draw=probs[1],
            p_away=probs[2],
            home_score=home_score,
            away_score=away_score,
            draw_score=draw_score,
        )

    # ------------------------------------------------------------------
    def decide_with_market(
        self,
        prediction: Prediction,
        outcome: Outcome,
        market_price: float,
    ) -> BetDecision | None:
        """Apply the edge filter; return a BetDecision or None.

        ``None`` means the market quote vetoed the bet — don't fall back
        to favourite-only logging in this case.
        """
        s = self._settings
        our_p = {
            Outcome.HOME: prediction.p_home,
            Outcome.DRAW: prediction.p_draw,
            Outcome.AWAY: prediction.p_away,
        }[outcome]
        e = edge(our_p, market_price)
        if e < s.edge_threshold:
            return None
        stake = min(s.fixed_stake_usd, s.max_bet_usd)
        rationale = (
            f"edge {e:+.3f} ({our_p:.3f} - {market_price:.3f}) at "
            f"≥{s.edge_threshold:.3f} threshold; stake ${stake:.0f}"
        )
        return BetDecision(
            fixture_id=prediction.fixture_id,
            competition_code=prediction.competition_code,
            home_team=prediction.home_team,
            away_team=prediction.away_team,
            outcome=outcome,
            our_probability=our_p,
            market_price=market_price,
            edge=e,
            stake_usd=stake,
            rationale=rationale,
        )
