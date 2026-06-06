"""Refresh config/team_aliases.yaml from football-data team metadata.

For each configured league this pulls the official team list and records each
team's ``shortName`` and ``tla`` as aliases for its canonical name. Existing
entries in the file (your hand-curated nicknames like "PSG", "Barca") are
preserved — we take the UNION, never overwriting. Matching still falls back to
fuzzy for anything not listed, so this only needs running occasionally.

Usage:
    python scripts/seed_aliases.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from betbot.config import get_settings
from betbot.data.football_data import FootballDataClient
from betbot.logging import configure_logging, get_logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ALIASES_PATH = _REPO_ROOT / "config" / "team_aliases.yaml"


async def _collect_from_football_data() -> dict[str, set[str]]:
    settings = get_settings()
    log = get_logger("seed_aliases")
    out: dict[str, set[str]] = {}
    async with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
        rate_limit_per_min=settings.football_data_rate_limit_per_min,
    ) as client:
        for league in settings.leagues:
            try:
                teams = await client.list_competition_teams(league)
            except Exception as e:  # noqa: BLE001
                log.warning("seed_league_failed", league=league, error=str(e))
                continue
            for t in teams:
                name = t.get("name")
                if not name:
                    continue
                aliases = {
                    v for v in (t.get("shortName"), t.get("tla")) if v and v != name
                }
                out.setdefault(name, set()).update(aliases)
            log.info("seed_league_done", league=league, teams=len(teams))
    return out


def _load_existing() -> dict[str, set[str]]:
    if not _ALIASES_PATH.exists():
        return {}
    data = yaml.safe_load(_ALIASES_PATH.read_text()) or {}
    return {k: set(v or []) for k, v in (data.get("aliases") or {}).items()}


def _write_merged(merged: dict[str, list[str]]) -> None:
    header = (
        "# Manual + auto-seeded team-name aliases (scripts/seed_aliases.py).\n"
        "# canonical football-data name -> alternative spellings markets use.\n"
        "# Hand edits are preserved on re-seed (union merge).\n\n"
    )
    body = yaml.safe_dump(
        {"aliases": merged}, allow_unicode=True, sort_keys=True, default_flow_style=False
    )
    _ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ALIASES_PATH.write_text(header + body)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    discovered = asyncio.run(_collect_from_football_data())

    merged_sets = _load_existing()
    for name, aliases in discovered.items():
        merged_sets.setdefault(name, set()).update(aliases)

    merged = {k: sorted(v) for k, v in sorted(merged_sets.items()) if v}
    _write_merged(merged)
    print(f"Wrote {len(merged)} teams to {_ALIASES_PATH}")


if __name__ == "__main__":
    main()
