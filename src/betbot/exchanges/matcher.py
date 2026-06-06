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

import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

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
