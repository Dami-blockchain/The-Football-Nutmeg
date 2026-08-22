"""The wiring regression: a fixture scored by the daily run must actually be
anchored, and the anchored probabilities must be what gets STORED.

This is the bug the feature exists to fix — anchoring that only happens when
Polymarket lists the fixture is anchoring that mostly does not happen. So the
test drives the real ``_score_and_log_one`` with an offline odds service and
asserts on the prediction handed to storage.

It deliberately does NOT assert anything about the bet/no-bet decision: that
path is owned elsewhere and this feature must not move it.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from betbot.config import Settings
from betbot.data.models import Fixture, FixtureForm, FormSnapshot
from betbot.data.odds import MatchOdds, OddsService
from betbot.data.odds_names import OddsNameResolver
from betbot.storage.db import init_engine
from betbot.strategy.engine import StrategyEngine

KICKOFF_ISO = "2026-06-20T15:00:00Z"
MATCH = {
    "id": 9101,
    "utcDate": KICKOFF_ISO,
    "homeTeam": {"id": 1, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS"},
    "awayTeam": {"id": 2, "name": "Chelsea FC", "shortName": "Chelsea", "tla": "CHE"},
}


class FakeForm:
    async def fixture_form(self, fixture_id, competition_code, kickoff, home_team, away_team):
        fixture = Fixture(
            id=fixture_id, home_team=home_team, away_team=away_team,
            kickoff=kickoff, competition_code=competition_code,
        )
        return FixtureForm(
            fixture=fixture,
            home_form=FormSnapshot(team=home_team, weighted_points=6.0,
                                   raw_points=6, matches_considered=5),
            away_form=FormSnapshot(team=away_team, weighted_points=1.0,
                                   raw_points=1, matches_considered=5),
        )


class NoRouter:
    async def find_best_route(self, home, away, kickoff, outcome):
        return None

    async def find_best_quote(self, home, away, kickoff, outcome):
        return None


@pytest.fixture
def fresh_db(tmp_path):
    init_engine(tmp_path / "wiring.sqlite")
    yield


def _settings(**over) -> Settings:
    return Settings(
        _env_file=None,
        FOOTBALL_DATA_API_KEY="fake",
        BETBOT_MODE="paper",
        BETBOT_DC_PARAMS_PATH="./tests/_no_such_dc_params.json",
        BETBOT_ENSEMBLE_CALIBRATION_PATH="./tests/_no_such_calibration.json",
        **over,
    )


def _hermetic_resolver() -> OddsNameResolver:
    """A resolver that does NOT read data/club_name_map.json.

    That file is gitignored — the droplet generates it via
    scripts/seed_glicko_club.py — so on a fresh checkout the canonical set is
    empty, every fixture name resolves to None, and anchoring silently skips
    with no_quote. Injecting the two names this test uses makes the anchor
    path exercisable in CI, which is the only place it is currently guarded.
    "Arsenal FC"/"Chelsea FC" normalise to "arsenal"/"chelsea", matching the
    injected rows below.
    """
    return OddsNameResolver(canonical=["arsenal", "chelsea"])


def _service(settings) -> OddsService:
    """Offline service: no provider, rows injected, explicit resolver. Zero I/O."""
    svc = OddsService(settings, providers=[], resolver=_hermetic_resolver())
    svc.load_rows([
        MatchOdds(
            league="PL", match_date=date(2026, 6, 20), home="arsenal", away="chelsea",
            # A price that disagrees hard with the form model (which has the
            # home side as a runaway favourite): the market makes it close.
            price_home=3.20, price_draw=3.40, price_away=2.30,
            source="test", book="B365H",
        )
    ])
    return svc


def _capture(monkeypatch) -> list:
    import betbot.main as main_mod

    seen: list = []
    real = main_mod.upsert_prediction

    def spy(prediction, *, kickoff):
        seen.append(prediction)
        return real(prediction, kickoff=kickoff)

    monkeypatch.setattr(main_mod, "upsert_prediction", spy)
    return seen


async def test_scored_fixture_is_anchored_when_the_flag_is_on(fresh_db, monkeypatch):
    from betbot.main import _score_and_log_one

    s = _settings(BETBOT_ODDS_ANCHOR="true")
    seen = _capture(monkeypatch)
    await _score_and_log_one(
        MATCH, "PL", FakeForm(), StrategyEngine(s), NoRouter(), s,
        odds_service=_service(s),
    )
    assert seen, "the scoring path must have stored a prediction"
    stored = seen[0]
    # De-vigged market for 3.20/3.40/2.30 puts the AWAY side ahead; the form
    # model had HOME far ahead. The stored probability must have moved toward
    # the price, and must still be a distribution.
    assert stored.p_home + stored.p_draw + stored.p_away == pytest.approx(1.0)
    unanchored = StrategyEngine(s).predict(
        await FakeForm().fixture_form(
            MATCH["id"], "PL", datetime(2026, 6, 20, 15, tzinfo=timezone.utc),
            _team(MATCH["homeTeam"]), _team(MATCH["awayTeam"]),
        )
    )
    market_home = (1 / 3.20) / (1 / 3.20 + 1 / 3.40 + 1 / 2.30)
    assert abs(stored.p_home - market_home) < abs(unanchored.p_home - market_home)


async def test_flag_off_stores_the_unanchored_prediction(fresh_db, monkeypatch):
    from betbot.main import _score_and_log_one

    s = _settings(BETBOT_ODDS_ANCHOR="false")
    seen = _capture(monkeypatch)
    await _score_and_log_one(
        MATCH, "PL", FakeForm(), StrategyEngine(s), NoRouter(), s,
        odds_service=_service(s),
    )
    stored = seen[0]
    unanchored = StrategyEngine(s).predict(
        await FakeForm().fixture_form(
            MATCH["id"], "PL", datetime(2026, 6, 20, 15, tzinfo=timezone.utc),
            _team(MATCH["homeTeam"]), _team(MATCH["awayTeam"]),
        )
    )
    assert stored.p_home == pytest.approx(unanchored.p_home)


async def test_missing_odds_row_still_scores_the_fixture(fresh_db, monkeypatch):
    """Graceful degradation end-to-end: no quote must not lose the fixture."""
    from betbot.main import _score_and_log_one

    s = _settings(BETBOT_ODDS_ANCHOR="true")
    empty = OddsService(s, providers=[], resolver=_hermetic_resolver())
    empty.load_rows([])
    seen = _capture(monkeypatch)
    n = await _score_and_log_one(
        MATCH, "PL", FakeForm(), StrategyEngine(s), NoRouter(), s,
        odds_service=empty,
    )
    assert seen, "an unquoted fixture must still be predicted and stored"
    assert n == 1, "and must still produce its usual paper reco"


def test_anchor_flag_defaults_off():
    """Same discipline as BETBOT_DISPERSION_FIX / BETBOT_MOV_FIX."""
    assert Settings.model_fields["odds_anchor_enabled"].default is False


def _team(raw: dict):
    from betbot.data.form import _parse_team

    return _parse_team(raw)
