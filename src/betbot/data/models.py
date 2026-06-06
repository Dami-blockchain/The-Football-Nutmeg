"""Pure domain models — frozen dataclasses, no ORM, no IO."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MatchOutcome(str, Enum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"


class Side(str, Enum):
    HOME = "HOME"
    AWAY = "AWAY"


@dataclass(frozen=True, slots=True)
class Team:
    id: int
    name: str
    short_name: str | None = None
    tla: str | None = None  # three-letter abbreviation


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One historical match used in form calculation."""
    home_team: Team
    away_team: Team
    home_goals: int
    away_goals: int
    kickoff: datetime
    competition_code: str

    @property
    def outcome(self) -> MatchOutcome:
        if self.home_goals > self.away_goals:
            return MatchOutcome.HOME
        if self.away_goals > self.home_goals:
            return MatchOutcome.AWAY
        return MatchOutcome.DRAW

    def points_for(self, side: Side) -> int:
        outcome = self.outcome
        if outcome is MatchOutcome.DRAW:
            return 1
        if side is Side.HOME and outcome is MatchOutcome.HOME:
            return 3
        if side is Side.AWAY and outcome is MatchOutcome.AWAY:
            return 3
        return 0


@dataclass(frozen=True, slots=True)
class Fixture:
    """An upcoming match we need to score."""
    id: int
    home_team: Team
    away_team: Team
    kickoff: datetime
    competition_code: str


@dataclass(frozen=True, slots=True)
class FormSnapshot:
    """One team's last-5 form. Used by FormService output."""
    team: Team
    weighted_points: float
    raw_points: int
    matches_considered: int


@dataclass(frozen=True, slots=True)
class FixtureForm:
    """Both teams' form alongside the fixture they apply to."""
    fixture: Fixture
    home_form: FormSnapshot
    away_form: FormSnapshot
