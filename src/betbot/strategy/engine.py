"""StrategyEngine — converts FixtureForm into probabilities + bet decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from betbot.data.models import FixtureForm, MatchOutcome
from betbot.logging import get_logger
from betbot.data.form import recency_weight_sum
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
    # Expected-goals readout (display-only): Dixon-Coles lambdas for the two
    # sides. None on fallback paths where no DC component is available.
    home_xg: float | None = None
    away_xg: float | None = None

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
    def _per_game(self, form) -> float:
        """Collapse FormService's summed weighted_points onto a per-game (0-3ish)
        scale so it is commensurate with ``draw_score`` (~2.4) and ``softmax_temp``.

        ``weighted_points`` is a recency-weighted SUM over the team's last N
        finished matches (verified live at ~11-12 for N=5), which — fed raw into
        the softmax against draw_score=2.4 — collapses the draw to ~0 and flips
        home/away. Dividing by the number of matches considered restores a clean
        points-per-game scale. When no matches are available (season start,
        matches_considered=0) we return 0.0 so both sides sit near draw_score and
        the softmax yields a sane near-uniform prior (home nudged by
        home_advantage) rather than H0/D0/A100 garbage.

        Sample-size shrinkage: a per-game score computed from ONE game is far
        noisier than one from five, yet the raw value trusts them equally and so
        over-reacts at season start (Fable flag). We shrink the per-game score
        toward the neutral prior (0.0 — the SAME value the n=0 no-form case maps
        to) with a data weight ``n / (n + K)`` where K = ``form_shrinkage_k``
        (default 4): n=1 => 20% data / 80% prior, n=5 => 56% data, n large =>
        approaches the un-shrunk value. n=0 already returns the prior (unchanged).
        Shrinkage is applied BEFORE the clamp so the [0, 3] bound still holds.
        """
        n = getattr(form, "matches_considered", 0)
        if n <= 0:
            return 0.0
        pg = form.weighted_points / recency_weight_sum(n)
        # Shrink toward the neutral prior (0.0) by the sample-size data weight.
        # K>=0 is enforced defensively so a mis-set config can't invert the pull.
        k = max(getattr(self._settings, "form_shrinkage_k", 4.0), 0.0)
        data_weight = n / (n + k) if (n + k) > 0 else 1.0
        pg *= data_weight
        # Hard-bound to the true points-per-game scale. weighted_points folds in
        # an opponent-strength factor of up to 1.5x, so the normalised value can
        # still exceed 3.0 (max 4.5); left unclamped that re-opens sub-1% tails
        # (H95/D4/A<1) the whole per-game fix exists to prevent. Clamping caps
        # the softmax spread at 3.0 + home_advantage => min outcome prob ~3%.
        return min(max(pg, 0.0), 3.0)

    def predict(self, fixture_form: FixtureForm) -> Prediction:
        s = self._settings
        home_pg = self._per_game(fixture_form.home_form)
        away_pg = self._per_game(fixture_form.away_form)
        home_score = home_pg + s.home_advantage
        away_score = away_pg

        # Anchor the draw on the SAME per-game scale as the two sides. The stored
        # ``draw_score`` (default 2.4) was tuned for the old summed form scale
        # (~11-12); on a 0-3 per-game scale a fixed 2.4 anchor would swamp the
        # softmax and hand every match to the draw. We instead float the draw
        # anchor at the fixture's mean per-game level plus a bias derived from
        # ``draw_score`` (``draw_score - 2.9`` => a modest -0.5 by default). This
        # keeps ``draw_score`` as the draw-propensity tuning knob, stays
        # scale-invariant, and — crucially for the season-start / unrated-team
        # fallback where both sides are 0 — can NEVER produce H0/D0/A100.
        mean_pg = (home_pg + away_pg) / 2.0
        draw_score = mean_pg + (s.draw_score - 2.9)

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
        *,
        require_edge: bool = True,
    ) -> BetDecision | None:
        """Apply the edge filter; return a BetDecision or None.

        ``None`` means the market quote vetoed the bet — don't fall back
        to favourite-only logging in this case. With ``require_edge=False``
        the edge gate is skipped and a decision is always returned (used by
        "bet every match" mode, which knowingly bets at negative edge).
        """
        s = self._settings
        our_p = {
            Outcome.HOME: prediction.p_home,
            Outcome.DRAW: prediction.p_draw,
            Outcome.AWAY: prediction.p_away,
        }[outcome]
        e = edge(our_p, market_price)
        if require_edge and e < s.edge_threshold:
            return None
        stake = min(s.fixed_stake_usd, s.max_bet_usd)
        gate = "forced (bet-every-match)" if not require_edge else f"≥{s.edge_threshold:.3f} threshold"
        rationale = (
            f"edge {e:+.3f} ({our_p:.3f} - {market_price:.3f}); {gate}; "
            f"stake ${stake:.0f}"
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
