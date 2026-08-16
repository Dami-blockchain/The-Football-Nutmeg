"""LineupService: Highlightly XI resolution + api-football-minutes adjustments.

All network mocked (fake Highlightly + api-football clients); the on-disk
player-minutes cache is written to a tmp dir and monkeypatched in. Pins:
  * resolve a fixture to a Highlightly match id via list_matches + aliases,
  * a NONZERO adjustment when a top-minutes regular is missing from the XI,
  * (0.0, 0.0) when the XI is not posted / the fixture is unresolved.
"""

from __future__ import annotations

import json

from betbot.config import Settings
from betbot.data import lineup_service as ls_mod
from betbot.data.lineup_service import LineupService


def _settings() -> Settings:
    return Settings(
        FOOTBALL_DATA_API_KEY="x",
        HIGHLIGHTLY_API_KEY="hl-key",
        API_FOOTBALL_KEY="af-key",
        BETBOT_AF_SEASON=2026,
        BETBOT_LINEUP_MAX_PENALTY=120.0,
    )


class _FakeHighlightly:
    """Records calls; returns canned matches + lineup."""

    def __init__(self, matches, lineup):
        self._matches = matches
        self._lineup = lineup
        self.match_calls: list[tuple[str, str]] = []
        self.lineup_calls: list[object] = []

    async def list_matches(self, league_name, date):
        self.match_calls.append((league_name, date))
        return self._matches

    async def get_lineup(self, match_id):
        self.lineup_calls.append(match_id)
        return self._lineup

    async def close(self):
        pass


class _FakeAf:
    async def close(self):
        pass


def _svc(settings, highlightly) -> LineupService:
    return LineupService(settings, client=_FakeAf(), highlightly=highlightly)


# ----------------------------------------------------------------------
async def test_resolve_match_id_uses_aliases(monkeypatch):
    s = _settings()
    matches = [
        {"match_id": 900, "home_name": "Real Madrid", "away_name": "Barcelona",
         "state": "Not started"},
    ]
    hl = _FakeHighlightly(matches, lineup=None)
    svc = _svc(s, hl)

    # Our football-data names differ ("Real Madrid CF" / "FC Barcelona") ->
    # the alias resolver must still match.
    mid = await svc.resolve_match_id(
        "PD", "Real Madrid CF", "FC Barcelona", "2026-08-16"
    )
    assert mid == 900
    # Cached: a second call issues no new /matches request.
    mid2 = await svc.resolve_match_id(
        "PD", "Real Madrid CF", "FC Barcelona", "2026-08-16"
    )
    assert mid2 == 900
    assert len(hl.match_calls) == 1
    assert hl.match_calls[0] == ("La Liga", "2026-08-16")


async def test_resolve_unmapped_league_is_none():
    s = _settings()
    hl = _FakeHighlightly([], lineup=None)
    svc = _svc(s, hl)
    assert await svc.resolve_match_id("XX", "A", "B", "2026-08-16") is None


async def test_nonzero_adjustment_when_top_regular_missing(monkeypatch, tmp_path):
    """A top-minutes regular absent from the confirmed XI -> negative home adj."""
    s = _settings()
    # 2024 minutes cache: Haaland is by far the top regular for the home team.
    cache = {
        "Man City": {
            "Erling Haaland": 3000, "Rodri": 2900, "Ederson": 2800,
            "Bernardo Silva": 2000, "Kyle Walker": 1800, "Ruben Dias": 1700,
            "Phil Foden": 1600, "John Stones": 1500, "Jack Grealish": 1400,
            "Kevin De Bruyne": 1300, "Nathan Ake": 1200,
        },
    }
    minutes_dir = tmp_path / "af_player_minutes"
    minutes_dir.mkdir()
    (minutes_dir / "PD_2024.json").write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr(ls_mod, "PLAYER_MINUTES_DIR", minutes_dir)

    # Confirmed XI omits Haaland (the top regular) for the home side; away has
    # every regular present (use the same names so it's "complete").
    lineup = {
        "home": {"formation": "4-3-3", "xi": [
            "Rodri", "Ederson", "Bernardo Silva", "Kyle Walker", "Ruben Dias",
            "Phil Foden", "John Stones", "Jack Grealish", "Kevin De Bruyne",
            "Nathan Ake", "Julian Alvarez",  # Haaland replaced
        ]},
        "away": {"formation": "4-4-2", "xi": [
            "Erling Haaland", "Rodri", "Ederson", "Bernardo Silva", "Kyle Walker",
            "Ruben Dias", "Phil Foden", "John Stones", "Jack Grealish",
            "Kevin De Bruyne", "Nathan Ake",
        ]},
    }
    matches = [{"match_id": 42, "home_name": "Man City", "away_name": "Everton",
                "state": "Not started"}]
    hl = _FakeHighlightly(matches, lineup=lineup)
    svc = _svc(s, hl)

    home_adj, away_adj = await svc.adjustments_for_fixture(
        "PD", "Man City", "Everton", "2026-08-16"
    )
    assert home_adj < 0.0  # Haaland (top minutes) missing -> penalty
    # Away side has a full Man City regulars XI matched against Everton's minutes
    # (no Everton cache) -> 0.0 (no minutes data for Everton).
    assert away_adj == 0.0


async def test_zero_adjustment_when_xi_not_posted(tmp_path, monkeypatch):
    s = _settings()
    minutes_dir = tmp_path / "af_player_minutes"
    minutes_dir.mkdir()
    monkeypatch.setattr(ls_mod, "PLAYER_MINUTES_DIR", minutes_dir)
    matches = [{"match_id": 7, "home_name": "Real Madrid", "away_name": "Barcelona",
                "state": "Not started"}]
    hl = _FakeHighlightly(matches, lineup=None)  # get_lineup -> None (not posted)
    svc = _svc(s, hl)

    adj = await svc.adjustments_for_fixture(
        "PD", "Real Madrid CF", "FC Barcelona", "2026-08-16"
    )
    assert adj == (0.0, 0.0)


async def test_get_confirmed_xi_returns_lineup_shape(tmp_path, monkeypatch):
    s = _settings()
    lineup = {"home": {"formation": "4-3-3", "xi": ["A"]},
              "away": {"formation": "4-4-2", "xi": ["B"]}}
    matches = [{"match_id": 3, "home_name": "Real Madrid", "away_name": "Barcelona",
                "state": "Not started"}]
    hl = _FakeHighlightly(matches, lineup=lineup)
    svc = _svc(s, hl)
    got = await svc.get_confirmed_xi(
        "PD", "Real Madrid CF", "FC Barcelona", "2026-08-16"
    )
    assert got == lineup
