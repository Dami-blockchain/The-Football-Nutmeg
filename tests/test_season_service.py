"""Tests for the season-service parsing + match_prob_fn wiring (no network)."""

from __future__ import annotations

from betbot.season_service import (
    build_match_prob_fn,
    parse_season_matches,
    run_season_sim,
)


def _m(home, away, status, hg=None, ag=None, md=None):
    return {
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "status": status,
        "score": {"fullTime": {"home": hg, "away": ag}},
        "matchday": md,
    }


def test_parse_splits_finished_and_upcoming():
    matches = [
        _m("A", "B", "FINISHED", 2, 1, md=1),
        _m("C", "D", "FINISHED", 0, 0, md=1),
        _m("A", "C", "SCHEDULED", md=2),
        _m("B", "D", "TIMED", md=2),
        _m("A", "D", "LIVE", md=2),  # in-progress counts as remaining
    ]
    inp = parse_season_matches(matches)
    assert inp.teams == {"A", "B", "C", "D"}
    assert inp.played_count == 2
    assert ("A", "B", 2, 1) in inp.played
    assert len(inp.remaining) == 3  # scheduled + timed + live
    assert inp.matchday == 1  # max finished matchday


def test_parse_skips_placeholder_and_scoreless_finished():
    matches = [
        _m("", "B", "SCHEDULED"),           # placeholder team -> skipped
        _m("A", "B", "FINISHED", None, None),  # finished but no score -> skipped
    ]
    inp = parse_season_matches(matches)
    assert inp.played_count == 0
    assert inp.remaining == []


class _FakeEngine:
    """Minimal ClubStrategyEngine stand-in for the wiring test."""

    def __init__(self, rated_teams):
        self._rated = set(rated_teams)

    def is_rated(self, home, away):
        return home in self._rated and away in self._rated

    def probability_triple(self, home, away):
        return (0.6, 0.25, 0.15), 1.5, 0.9


class _S:
    glicko_default_rating = 1500.0
    glicko_default_rd = 200.0
    glicko_default_vol = 0.06
    glicko_club_home_mu = 0.30
    glicko_club_draw_rho = 0.28


def test_match_prob_fn_flags_unrated():
    engine = _FakeEngine({"A", "B"})
    fn, unrated = build_match_prob_fn(engine, _S())
    # rated tie -> engine triple
    assert fn("A", "B") == (0.6, 0.25, 0.15)
    # unrated tie -> neutral fallback + flagged
    triple = fn("A", "Z")
    assert abs(sum(triple) - 1.0) < 1e-9
    assert "Z" in unrated


def test_run_season_sim_shapes_result():
    engine = _FakeEngine({"A", "B", "C", "D"})
    matches = [
        _m("A", "B", "FINISHED", 1, 0, md=1),
        _m("C", "D", "SCHEDULED", md=1),
        _m("A", "C", "SCHEDULED", md=2),
    ]
    inp = parse_season_matches(matches)
    res = run_season_sim(inp, engine, _S(), n_sims=500, seed=1)
    assert res["table"]
    assert res["n_sims"] == 500
    # rows carry the four projection fields
    row = res["table"][0]
    assert {"team", "p_title", "p_top4", "exp_points"} <= set(row)
    # p_title over the table sums to ~1
    assert abs(sum(r["p_title"] for r in res["table"]) - 1.0) < 0.02
