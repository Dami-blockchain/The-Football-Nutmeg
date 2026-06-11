"""TeamAliasResolver — reconcile football-data team names with market titles.

Prediction-market titles are terse and inconsistent ("PSG", "Man City",
"Bayern"), while football-data.org names are formal ("Paris Saint-Germain FC",
"FC Bayern München"). We bridge the gap with two layers:

1. A manual alias table (``config/team_aliases.yaml``) for cases fuzzy matching
   gets wrong (e.g. "PSG" ↔ "Paris Saint-Germain FC").
2. ``rapidfuzz.token_set_ratio`` over diacritic-stripped, noise-token-stripped
   normalised forms for everything else.

Everything here is pure (no network, no DB) so it is cheaply unit-testable.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from betbot.data.models import MatchOutcome

# Club-name noise tokens stripped before fuzzy matching. These are the
# corporate/legal suffixes and filler words that carry no discriminative
# signal ("Arsenal FC" and "Arsenal" are the same club).
_NOISE_TOKENS: frozenset[str] = frozenset(
    {
        "fc", "afc", "cf", "sc", "ac", "ssc", "bsc", "vfl", "vfb", "cp",
        "ud", "rc", "ss", "us", "as", "fk", "sk", "cd", "sd", "rcd", "club",
        "de", "futbol", "football", "calcio", "the",
    }
)

# Default similarity threshold (0–100). token_set_ratio is lenient about word
# order and subsets, so 80 keeps "Man City" ↔ "Manchester City" while still
# rejecting different clubs.
DEFAULT_THRESHOLD: float = 80.0


def _strip_diacritics(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize(name: str) -> str:
    """Lowercase, strip diacritics + punctuation, drop club-noise tokens.

    ``"FC Bayern München"`` → ``"bayern munchen"``;
    ``"Paris Saint-Germain FC"`` → ``"paris saint germain"``.
    """
    folded = _strip_diacritics(name).lower()
    cleaned = "".join(c if c.isalnum() else " " for c in folded)
    tokens = [t for t in cleaned.split() if t and t not in _NOISE_TOKENS]
    # Guard: if a name is *entirely* noise tokens (unlikely), keep the raw
    # tokens so we never collapse to an empty string.
    if not tokens:
        tokens = [t for t in cleaned.split() if t]
    return " ".join(tokens)


class TeamAliasResolver:
    """Match a team name against candidate market labels.

    The alias table maps a canonical football-data name to the alternative
    spellings a market might use. Internally we work in normalised space and
    fold every alias back to the canonical normalised form, so an exact alias
    hit short-circuits fuzzy matching entirely.
    """

    def __init__(self, aliases: Mapping[str, Iterable[str]] | None = None) -> None:
        # normalised alias/canonical  ->  canonical normalised form
        self._alias_to_canon: dict[str, str] = {}
        for canon, alts in (aliases or {}).items():
            canon_norm = normalize(canon)
            self._alias_to_canon[canon_norm] = canon_norm
            for alt in alts:
                self._alias_to_canon[normalize(alt)] = canon_norm

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeamAliasResolver":
        """Load from a YAML file with a top-level ``aliases:`` mapping.

        A missing file yields an empty (fuzzy-only) resolver rather than an
        error — the alias table is an optional override, not a requirement.
        """
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text()) or {}
        return cls(aliases=data.get("aliases") or {})

    def _canonical_norm(self, name: str) -> str:
        """Normalised form, folded through the alias table when known."""
        norm = normalize(name)
        return self._alias_to_canon.get(norm, norm)

    def match(
        self,
        name: str,
        candidates: Iterable[str],
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> str | None:
        """Return the candidate string best matching ``name``, or ``None``.

        Resolution order: exact normalised/alias hit first, then fuzzy
        ``token_set_ratio``. Returns the *original* candidate string (not its
        normalised form) so callers can use it as a key.
        """
        cand_list = [c for c in candidates if c and c.strip()]
        if not cand_list:
            return None

        target = self._canonical_norm(name)

        # Map each candidate's canonical-normalised form back to the first
        # original spelling that produced it.
        norm_to_original: dict[str, str] = {}
        for c in cand_list:
            norm_to_original.setdefault(self._canonical_norm(c), c)

        # 1) Exact hit in normalised/alias space.
        if target in norm_to_original:
            return norm_to_original[target]

        # 2) Fuzzy match over the normalised candidate forms.
        best = process.extractOne(
            target,
            list(norm_to_original.keys()),
            scorer=fuzz.token_set_ratio,
        )
        if best is not None and best[1] >= threshold:
            return norm_to_original[best[0]]
        return None

    def same_team(
        self, a: str, b: str, *, threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """True if two names refer to the same team."""
        return self.match(a, [b], threshold=threshold) is not None


# ----------------------------------------------------------------------
# Market-identity verification (matcher hardening)
# ----------------------------------------------------------------------
# Prop / non-1X2 markets we must NEVER route as a fixture's HOME/AWAY/DRAW.
# These are exact-scoreline, spread/handicap, totals, BTTS, cards, corners,
# "to win by N", "first goal", substitutions, etc. A market whose title trips
# any of these — OR whose structured metadata carries a non-zero prop threshold
# — is excluded from 1X2 routing. (The Mexico/SA 0.014 "edge" was a longshot
# prop the matcher paired to a 1X2 fixture.)
_PROP_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwin by\b",
        r"\bby\s+\d+\+?\s*(?:goal|point)",
        r"\b\d+\s*\+?\s*(?:total\s+)?goals?\b",
        r"\bover\b|\bunder\b",
        r"\bo/?u\b",
        r"\bbtts\b",
        r"both\s+teams?\s+to\s+score",
        r"\bhandicap\b",
        r"\bspread\b",
        r"[+-]\d+(?:\.\d+)?\s*(?:goal|handicap|spread)",  # +1.5 goals / -2 handicap
        r"\bcards?\b",
        r"\bcorners?\b",
        r"\bbooking",
        r"\bred\s+card\b|\byellow\s+card\b",
        r"first\s+(?:goal|to\s+score|scorer)",
        r"\banytime\b|\bscorer\b",
        r"\bsubstitution",
        r"\bclean\s+sheet\b",
        r"\bcorrect\s+score\b|exact\s+score|\bscoreline\b",
        r"\bhalf[\s-]?time\b|\bhalftime\b|\b1st\s+half\b|\b2nd\s+half\b",
        r"\bto\s+qualify\b|\bto\s+advance\b|\bto\s+reach\b|\bto\s+lift\b|\bto\s+win\s+the\b",
        r"\bpenalt",
        r"\bextra\s+time\b",
    )
)

# Limitless metadata keys that, when present and non-zero, mark a prop market.
_PROP_THRESHOLD_KEYS: tuple[str, ...] = (
    "goalsThreshold",
    "spreadThreshold",
    "cardsThreshold",
    "cornersThreshold",
)

# Limitless metadata.marketType values that ARE 1X2 match-result markets.
_MATCH_RESULT_TYPES: frozenset[str] = frozenset(
    {"match_result", "matchresult", "1x2", "moneyline", "winner", "result", "outcome"}
)


def _title_is_prop(title: str) -> bool:
    t = title or ""
    return any(p.search(t) for p in _PROP_TITLE_PATTERNS)


def is_match_result_market(
    metadata: Mapping[str, object] | None,
    title: str = "",
) -> bool:
    """True only if this market is a 1X2 / match-result market (not a prop).

    Used to gate fixture→market routing so a longshot prop (exact scoreline,
    spread, totals, BTTS, cards, corners, "to win by N", first goal, …) can
    never be paired to a fixture as its HOME/AWAY/DRAW.

    Decision order:
      1. A non-zero prop threshold in metadata (Limitless ``goalsThreshold`` /
         ``spreadThreshold`` / ``cardsThreshold`` / ``cornersThreshold``) → prop.
      2. A ``marketType`` that is a known match-result type → match-result
         (subject to title check); an explicit non-result ``marketType`` → prop.
      3. A prop-shaped title → prop.
      4. Otherwise assume match-result (the discovery layer already required
         both team names and a 3-way/binary YES-NO structure).
    """
    md = metadata or {}

    for key in _PROP_THRESHOLD_KEYS:
        raw = md.get(key)
        if raw in (None, "", 0, 0.0):
            continue
        try:
            if float(raw) != 0.0:
                return False
        except (TypeError, ValueError):
            # A non-numeric, non-empty threshold value is still a prop signal.
            return False

    market_type = str(md.get("marketType") or md.get("market_type") or "").strip().lower()
    if market_type:
        normalized_type = market_type.replace("-", "_").replace(" ", "_")
        if normalized_type.replace("_", "") in _MATCH_RESULT_TYPES:
            return not _title_is_prop(title)
        # An explicit, recognised-but-non-result type is a prop.
        return False

    # No structured type signal: lean on the title.
    return not _title_is_prop(title)


def normalized_team_pair(home: str, away: str) -> frozenset[str]:
    """Orientation-agnostic identity key for a fixture's two teams.

    Returns the normalised team names as a frozenset so HOME/AWAY orientation
    doesn't matter when comparing two venues' markets. Empty names are dropped.
    """
    return frozenset(n for n in (normalize(home), normalize(away)) if n)


def classify_binary_outcome(
    label: str,
    home_team: str,
    away_team: str,
    resolver: TeamAliasResolver,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> MatchOutcome | None:
    """Map a binary-market outcome label to HOME / DRAW / AWAY (or None).

    Shared by both adapters: on Polymarket ``label`` is the team extracted from a
    "Will <team> win?" question; on Limitless it is the child market's title
    ("Liechtenstein", "Cyprus", "Draw"). Returns ``None`` when the label matches
    neither team nor "draw" — keeping outcome classification in one place so the
    two venues can't drift apart.
    """
    if not label:
        return None
    if "draw" in label.lower():
        return MatchOutcome.DRAW
    if resolver.same_team(label, home_team, threshold=threshold):
        return MatchOutcome.HOME
    if resolver.same_team(label, away_team, threshold=threshold):
        return MatchOutcome.AWAY
    return None
