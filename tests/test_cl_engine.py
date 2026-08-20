"""Unit tests for the cross-league Elo engine (EuropeanStrategyEngine).

No DB / network: the ClubElo snapshot, DC params and name map are all injected,
and resolver=None means the engine loads config/team_aliases.yaml but resolves
against the injected snapshot's club list (the fixture team names below are used
verbatim as snapshot keys, so exact-normalised matching resolves them).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from betbot.config import get_settings
from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.strategy import dixon_coles as dc
from betbot.strategy.cl_engine import EuropeanStrategyEngine, _load_snapshot
from betbot.strategy.engine import Outcome


def _ff(home: str, away: str, hp: float = 1.5, ap: float = 1.5) -> FixtureForm:
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=7, home_team=ht, away_team=at,
                 kickoff=datetime.now(timezone.utc), competition_code="CL")
    return FixtureForm(
        fixture=fx,
        home_form=FormSnapshot(team=ht, weighted_points=hp, raw_points=0, matches_considered=5),
        away_form=FormSnapshot(team=at, weighted_points=ap, raw_points=0, matches_considered=5),
    )


def _engine(snapshot: dict[str, float], *, dc_params=None, name_map=None):
    s = get_settings()
    return EuropeanStrategyEngine(
        s,
        snapshot=snapshot,
        dc_params=dc_params,
        name_map=name_map if name_map is not None else {},
        resolver=None,
    )


def test_probs_sum_to_one_and_favour_stronger_home():
    snap = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    eng = _engine(snap)
    p = eng.predict(_ff("Strong FC", "Weak FC"))
    assert abs(p.p_home + p.p_draw + p.p_away - 1.0) < 1e-9
    assert p.p_home > p.p_away
    assert p.best_outcome is Outcome.HOME


def test_home_advantage_shifts_equal_elo_toward_home():
    snap = {"A FC": 1700.0, "B FC": 1700.0}
    eng = _engine(snap)
    # Equal Elo: the home side must still be favoured by the Elo home-advantage.
    p = eng.predict(_ff("A FC", "B FC"))
    assert p.p_home > p.p_away


def test_unresolved_team_falls_back_to_naive():
    # Away side not in the snapshot -> naive fallback, byte-identical to naive.
    s = get_settings()
    snap = {"Known FC": 1800.0}  # "Missing FC" absent from snapshot
    eng = _engine(snap)
    ff = _ff("Known FC", "Missing FC", hp=2.5, ap=0.4)
    from betbot.strategy.engine import StrategyEngine
    naive = StrategyEngine(s).predict(ff)
    got = eng.predict(ff)
    assert got.p_home == naive.p_home
    assert got.p_draw == naive.p_draw
    assert got.p_away == naive.p_away


def test_decide_with_market_no_edge_returns_none():
    snap = {"Strong FC": 1950.0, "Weak FC": 1450.0}
    eng = _engine(snap)
    pred = eng.predict(_ff("Strong FC", "Weak FC"))
    # Market price == model prob: after anchoring there is no edge -> veto.
    d = eng.decide_with_market(pred, Outcome.HOME, pred.p_home, require_edge=True)
    assert d is None


def test_dc_component_lifts_strong_team_when_enabled():
    # cl_weight_dc defaults to 1.0 (blend shipped), so a DC-strong team must
    # lift its home probability vs the Elo-only (no DC params) prediction.
    # Team names carry no noise tokens ("FC"/"CF"), so normalize() maps them
    # straight to the DC keys — the engine only blends DC when BOTH clubs are
    # actually in the goal model.
    snap = {"Ajax": 1700.0, "Porto": 1700.0}
    params = dc.DCParams(
        base_mu=0.1, home_adv=0.25, rho=0.0,
        teams={"ajax": dc.DCTeam(attack=0.9, defence=0.5),
               "porto": dc.DCTeam(attack=-0.4, defence=-0.3)},
    )
    with_dc = _engine(snap, dc_params=params).predict(_ff("Ajax", "Porto"))
    without_dc = _engine(snap, dc_params=None).predict(_ff("Ajax", "Porto"))
    assert with_dc.p_home > without_dc.p_home


def test_lineup_adjustment_default_reproduces_current_output():
    snap = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    eng = _engine(snap)
    base = eng.predict(_ff("Strong FC", "Weak FC"))
    zeroed = eng.predict(
        _ff("Strong FC", "Weak FC"), home_rating_adj=0.0, away_rating_adj=0.0
    )
    assert zeroed.p_home == base.p_home
    assert zeroed.p_draw == base.p_draw
    assert zeroed.p_away == base.p_away
    assert zeroed.home_score == base.home_score


def test_negative_home_rating_adj_lowers_p_home():
    snap = {"Strong FC": 1800.0, "Weak FC": 1600.0}
    eng = _engine(snap)
    base = eng.predict(_ff("Strong FC", "Weak FC"))
    weakened = eng.predict(_ff("Strong FC", "Weak FC"), home_rating_adj=-120.0)
    assert weakened.p_home < base.p_home
    # eh is stored (adjusted) for transparency.
    assert weakened.home_score == base.home_score - 120.0


# ---------------------------------------------------------------------------
# Snapshot freshness + degradation safety (fix/clubelo-refresh).
#
# The CL engine is the most accurate component in the system, and it silently
# reverts to the naive form engine when its ClubElo snapshot is bad. These
# tests pin BOTH halves of that: the revert must be safe, and it must be loud.
# ---------------------------------------------------------------------------

_HEADER = "Rank,Club,Country,Level,Elo,From,To"


def _csv(frm: date, rows: int = 3) -> str:
    to = frm + timedelta(days=10)  # ClubElo's To is a FUTURE horizon
    out = [_HEADER]
    for i in range(rows):
        out.append(f"{i + 1},Club {i},ENG,1,{1900 - i * 50}.0,{frm.isoformat()},{to.isoformat()}")
    return "\n".join(out) + "\n"


def _engine_on_file(path, monkeypatch, tmp_path):
    # chdir so the data/clubelo/ directory fallback cannot reach the real repo
    monkeypatch.chdir(tmp_path)
    s = get_settings().model_copy(update={"clubelo_latest_path": Path(path)})
    return EuropeanStrategyEngine(s, dc_params=None, name_map={}, resolver=None)


def test_snapshot_age_uses_From_not_the_future_To_column(tmp_path):
    """Regression: max(To, From) made a 20-day-old snapshot look -10 days old."""
    frm = date.today() - timedelta(days=20)
    p = tmp_path / "c.csv"
    p.write_text(_csv(frm))
    snap, snap_date = _load_snapshot(p)
    assert snap_date == frm
    assert (date.today() - snap_date).days == 20  # positive, and the real age


def test_stale_snapshot_is_flagged_on_the_engine(tmp_path, monkeypatch):
    p = tmp_path / "c.csv"
    p.write_text(_csv(date.today() - timedelta(days=40)))
    eng = _engine_on_file(p, monkeypatch, tmp_path)
    assert eng.snapshot_stale is True
    assert eng.snapshot_age_days == 40
    assert eng.snapshot_reason == "stale_40d"


def test_fresh_snapshot_is_not_flagged(tmp_path, monkeypatch):
    p = tmp_path / "c.csv"
    p.write_text(_csv(date.today()))
    eng = _engine_on_file(p, monkeypatch, tmp_path)
    assert eng.snapshot_stale is False
    assert eng.snapshot_age_days == 0


def test_truncated_row_is_dropped_instead_of_loading_a_nonsense_rating(tmp_path):
    """A half-written CSV used to load 'Bayern' at Elo 20.0 — silently wrong."""
    p = tmp_path / "c.csv"
    p.write_text(_csv(date.today()) + "4,Bayern,GER,1,20")
    snap, _ = _load_snapshot(p)
    assert "Bayern" not in snap          # dropped, not believed
    assert snap["Club 0"] == 1900.0      # the good rows survive


def test_missing_snapshot_degrades_to_naive_and_is_flagged(tmp_path, monkeypatch):
    eng = _engine_on_file(tmp_path / "does_not_exist.csv", monkeypatch, tmp_path)
    assert eng.snapshot_stale is True
    assert eng.snapshot_reason == "missing_or_empty"

    ff = _ff("Barcelona", "Espanyol", hp=2.5, ap=0.4)
    from betbot.strategy.engine import StrategyEngine
    naive = StrategyEngine(get_settings()).predict(ff)
    got = eng.predict(ff)
    # Degraded, but never wrong: identical to naive, not a garbage Elo price.
    assert (got.p_home, got.p_draw, got.p_away) == (naive.p_home, naive.p_draw, naive.p_away)


def test_unparseable_snapshot_degrades_to_naive(tmp_path, monkeypatch):
    p = tmp_path / "c.csv"
    p.write_text("total garbage, not a csv at all\n")
    eng = _engine_on_file(p, monkeypatch, tmp_path)
    assert eng.snapshot_stale is True

    ff = _ff("Barcelona", "Espanyol", hp=2.5, ap=0.4)
    from betbot.strategy.engine import StrategyEngine
    naive = StrategyEngine(get_settings()).predict(ff)
    got = eng.predict(ff)
    assert (got.p_home, got.p_draw, got.p_away) == (naive.p_home, naive.p_draw, naive.p_away)
