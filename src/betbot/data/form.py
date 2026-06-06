"""FormService — pulls last-5 form from football-data.org and weights it."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from betbot.data.football_data import FootballDataClient
from betbot.data.models import (
    Fixture,
    FixtureForm,
    FormSnapshot,
    MatchResult,
    Side,
    Team,
)
from betbot.logging import get_logger

if TYPE_CHECKING:
    from betbot.config import Settings

log = get_logger(__name__)


# Recency weights for the last 5 matches, most-recent first.
_RECENCY_WEIGHTS: tuple[float, ...] = (1.5, 1.3, 1.1, 1.0, 0.9)


def _parse_kickoff(s: str) -> datetime:
    """Parse an ISO-8601 string from football-data into an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _parse_team(raw: dict[str, Any] | None) -> Team:
    if raw is None:
        return Team(id=0, name="Unknown")
    return Team(
        id=int(raw.get("id") or 0),
        name=str(raw.get("name") or "Unknown"),
        short_name=raw.get("shortName"),
        tla=raw.get("tla"),
    )


def _parse_match_result(raw: dict[str, Any]) -> MatchResult | None:
    home = _parse_team(raw.get("homeTeam"))
    away = _parse_team(raw.get("awayTeam"))
    score = (raw.get("score") or {}).get("fullTime") or {}
    home_goals = score.get("home")
    away_goals = score.get("away")
    if home_goals is None or away_goals is None:
        return None
    return MatchResult(
        home_team=home,
        away_team=away,
        home_goals=int(home_goals),
        away_goals=int(away_goals),
        kickoff=_parse_kickoff(raw.get("utcDate") or ""),
        competition_code=str((raw.get("competition") or {}).get("code") or ""),
    )


class FormService:
    """Computes weighted-points form for a team over its last 5 finished matches.

    Opponent-strength adjustment: each match's points contribution is
    multiplied by an opponent-strength factor derived from the opponent's
    current league position (top of table => higher factor, bottom =>
    lower).
    """

    def __init__(self, client: FootballDataClient, settings: "Settings") -> None:
        self._client = client
        self._settings = settings
        # Per-competition cache of {team_id: position} to avoid repeated
        # standings calls within a single scoring run.
        self._standings_cache: dict[str, dict[int, int]] = {}
        self._standings_size: dict[str, int] = {}

    # ------------------------------------------------------------------
    async def _get_positions(self, competition_code: str) -> tuple[dict[int, int], int]:
        if competition_code in self._standings_cache:
            return (
                self._standings_cache[competition_code],
                self._standings_size[competition_code],
            )
        table = await self._client.get_standings(competition_code)
        positions: dict[int, int] = {}
        for row in table:
            tid = (row.get("team") or {}).get("id")
            pos = row.get("position")
            if isinstance(tid, int) and isinstance(pos, int):
                positions[tid] = pos
        size = len(positions) or 20  # sane default for top-5 leagues
        self._standings_cache[competition_code] = positions
        self._standings_size[competition_code] = size
        return positions, size

    # ------------------------------------------------------------------
    async def _team_snapshot(
        self,
        team: Team,
        before_kickoff: datetime,
        competition_code: str,
    ) -> FormSnapshot:
        # Pull a few extra so we can drop matches that haven't happened yet.
        raw = await self._client.list_team_recent_matches(
            team.id, limit=10, before=before_kickoff.date().isoformat()
        )
        positions, league_size = await self._get_positions(competition_code)

        weight_w = self._settings.opp_strength_weight
        results: list[tuple[MatchResult, Side]] = []
        for m in raw:
            mr = _parse_match_result(m)
            if mr is None or mr.kickoff >= before_kickoff:
                continue
            if mr.home_team.id == team.id:
                side = Side.HOME
            elif mr.away_team.id == team.id:
                side = Side.AWAY
            else:
                continue
            results.append((mr, side))
            if len(results) >= 5:
                break

        weighted = 0.0
        raw_pts = 0
        for i, (mr, side) in enumerate(results):
            w = _RECENCY_WEIGHTS[i] if i < len(_RECENCY_WEIGHTS) else 1.0
            opponent = mr.away_team if side is Side.HOME else mr.home_team
            opp_pos = positions.get(opponent.id)
            factor = _opp_strength_factor(opp_pos, league_size, weight_w)
            pts = mr.points_for(side)
            weighted += w * pts * factor
            raw_pts += pts

        return FormSnapshot(
            team=team,
            weighted_points=weighted,
            raw_points=raw_pts,
            matches_considered=len(results),
        )

    # ------------------------------------------------------------------
    async def fixture_form(
        self,
        fixture_id: int,
        competition_code: str,
        kickoff: datetime,
        home_team: Team,
        away_team: Team,
    ) -> FixtureForm:
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        home_form = await self._team_snapshot(home_team, kickoff, competition_code)
        away_form = await self._team_snapshot(away_team, kickoff, competition_code)
        return FixtureForm(
            fixture=Fixture(
                id=fixture_id,
                home_team=home_team,
                away_team=away_team,
                kickoff=kickoff,
                competition_code=competition_code,
            ),
            home_form=home_form,
            away_form=away_form,
        )


def _opp_strength_factor(
    opp_position: int | None, league_size: int, weight: float
) -> float:
    # Local re-implementation to avoid circular import with probabilities.
    if opp_position is None or league_size <= 1:
        return 1.0
    norm = (opp_position - 1) / (league_size - 1)
    norm = max(0.0, min(1.0, norm))
    return 1.0 + weight * (1.0 - 2.0 * norm)
