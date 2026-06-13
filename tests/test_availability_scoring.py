"""Scoring-path wiring for the availability adjustment (main._availability_adjustments).

Asserts the contract: NO-OP when disabled/keyless, real adjustment when on,
and a fetch error falls back to (0, 0) without breaking — never raises.
"""

from __future__ import annotations

import pytest

from betbot.main import _availability_adjustments


def _clone(base, over):
    # Settings is a pydantic model; mutate copies for the few knobs we need.
    s = base.model_copy()
    for k, v in over.items():
        setattr(s, k, v)
    return s


class FakeAF:
    """Stand-in for ApiFootballClient with a controllable has_key + responses."""

    def __init__(self, *, has_key=True, ids=None, injuries=None, raise_on=None):
        self.has_key = has_key
        self._ids = ids or {}
        self._injuries = injuries or {}
        self._raise_on = raise_on  # "resolve" | "injuries" | None
        self.resolve_calls = 0
        self.injury_calls = 0

    async def resolve_team_id(self, name):
        self.resolve_calls += 1
        if self._raise_on == "resolve":
            raise RuntimeError("boom")
        return self._ids.get(name)

    async def get_injuries(self, team_id, season):
        self.injury_calls += 1
        if self._raise_on == "injuries":
            raise RuntimeError("boom")
        return self._injuries.get(team_id, [])


@pytest.mark.asyncio
async def test_noop_when_client_none(settings):
    s = _clone(settings, {"lineup_adjust_enabled": True})
    assert await _availability_adjustments(None, s, "Spain", "Croatia") == (0.0, 0.0)


@pytest.mark.asyncio
async def test_noop_when_keyless(settings):
    af = FakeAF(has_key=False)
    s = _clone(settings, {"lineup_adjust_enabled": True})
    assert await _availability_adjustments(af, s, "Spain", "Croatia") == (0.0, 0.0)
    assert af.resolve_calls == 0  # never touched the API


@pytest.mark.asyncio
async def test_real_adjustment_from_injuries(settings):
    af = FakeAF(
        ids={"Spain": 9, "Croatia": 3},
        injuries={9: [{"name": "a"}, {"name": "b"}, {"name": "c"}], 3: []},
    )
    s = _clone(settings, {
        "lineup_adjust_enabled": True,
        "injury_penalty_per_player": 8.0,
        "injury_penalty_cap": 40.0,
        "api_football_season": 2026,
    })
    home_adj, away_adj = await _availability_adjustments(af, s, "Spain", "Croatia")
    assert home_adj == pytest.approx(-24.0)  # 3 * 8
    assert away_adj == 0.0


@pytest.mark.asyncio
async def test_unresolved_team_yields_zero(settings):
    af = FakeAF(ids={}, injuries={})  # nothing resolves
    s = _clone(settings, {"lineup_adjust_enabled": True})
    assert await _availability_adjustments(af, s, "Atlantis", "Wakanda") == (0.0, 0.0)
    assert af.injury_calls == 0  # no id -> no injury fetch


@pytest.mark.asyncio
async def test_fetch_error_falls_back_to_zero(settings):
    af = FakeAF(ids={"Spain": 9, "Croatia": 3}, raise_on="injuries")
    s = _clone(settings, {"lineup_adjust_enabled": True})
    # Must NOT raise; falls back to (0, 0).
    assert await _availability_adjustments(af, s, "Spain", "Croatia") == (0.0, 0.0)


@pytest.mark.asyncio
async def test_resolve_error_falls_back_to_zero(settings):
    af = FakeAF(raise_on="resolve")
    s = _clone(settings, {"lineup_adjust_enabled": True})
    assert await _availability_adjustments(af, s, "Spain", "Croatia") == (0.0, 0.0)
