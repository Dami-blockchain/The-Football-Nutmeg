"""The ONE place that turns an internal competition code into a human name.

The DB stores a compact ``competition_code`` (``PL``, ``PD``, ``BL1``, ``SA``,
``FL1``, ``CL`` and historically ``WC``). Users should never see the code —
they should see "Premier League", "La Liga", etc. Every user-facing surface
that names a fixture routes its league label through :func:`league_label` so
the mapping lives in exactly one place and stays consistent across the morning
notice, the prematch/confirmed-XI alerts, the result alert, the paywall
teasers, the group broadcast and the chat assistant context.

Codes are football-data.org competition IDs (see ``LEAGUE_CODES`` in
:mod:`betbot.config`). Unknown or missing codes degrade GRACEFULLY to the raw
code (never a crash, never the string "None"): there are historical World Cup
rows and new competitions will appear.
"""

from __future__ import annotations

#: Canonical code -> display name. The single source of truth for league names
#: shown to users. ``betbot.season_service.LEAGUE_NAMES`` is derived from this.
LEAGUE_DISPLAY_NAMES: dict[str, str] = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "Champions League",
    "WC": "World Cup",
}


def league_label(code: str | None) -> str:
    """Human-readable league name for a competition code.

    * A known code -> its display name ("PD" -> "La Liga").
    * An unknown but non-empty code -> the raw code, upper-cased (so a brand new
      competition still identifies the match instead of crashing).
    * ``None`` / empty / whitespace -> ``""`` (the caller shows no label, and
      the message never renders the literal "None").
    """
    if not code or not str(code).strip():
        return ""
    key = str(code).strip().upper()
    return LEAGUE_DISPLAY_NAMES.get(key, key)
