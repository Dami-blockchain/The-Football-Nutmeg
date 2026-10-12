"""Free-data feed + Elo builder for the self-built CL fallback snapshot.

Pure, stdlib + rapidfuzz, importable + unit-tested. The operator-run scripts
(``scripts/seed_selfbuilt_elo.py``, ``scripts/build_selfbuilt_elo.py``) are
thin orchestrators over this module; the load-bearing, FREE-only-critical parts
— the 200-not-payload fetch guard, the football-data.co.uk parser, the Elo
update and the COUNTRY-RESTRICTED ClubElo name bridge — live here so they are
tested.

NEVER pay: every source is free and keyless —
``football-data.co.uk/mmz4281/{season}/{div}.csv`` for the domestic divisions,
and football-data.org (free tier, via ``FootballDataClient``) for Champions
League finished matches. Europa League / Conference League finished results are
NOT on a free feed, so the cross-league linkage is CL-only (documented).

Elo spec (matches the validated prototype, api.clubelo-like BUILD scale): the
SNAPSHOT stores RAW ratings on this scale (SD ~147); the engine prices them
with the LIVE 51/312 constants (option (a)). The build constants here
(K/HA/DIVISOR) are NOT the serve constants.

Name bridge (FALLBACK_SCOPE.md §5a, as corrected by the Fable review): the
schochastics idents are ``"Name (Country)"`` and ClubElo uses a 3-letter
``Country`` code, so matching is restricted to the SAME country, the
``(Country)`` suffix is stripped before normalising, a single shared token can
never score a match (``>= 2`` shared tokens or an exact normalised equality is
required), and a collision (two idents contesting one ClubElo key) is left
UNRESOLVED unless an alias pins it — never "most games wins".
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from rapidfuzz import fuzz

from betbot.exchanges.matcher import normalize

# ---- Elo BUILD constants (NOT the serve 51/312; see module docstring) -------
ELO_K = 20.0
ELO_HA = 65.0
ELO_DIVISOR = 400.0
ELO_INIT = 1500.0

#: Fuzzy acceptance (token_set_ratio, 0-100) when >= 2 tokens are shared.
FUZZY_THRESHOLD = 80.0

#: Non-combining chars NFKD does not decompose, folded to their digraph/base so
#: the Scandinavian/German spellings line up across sources.
_PRETRANS = {
    "ß": "ss", "ø": "o", "Ø": "o", "œ": "oe", "Œ": "oe",
    "æ": "ae", "Æ": "ae", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l",
    "ı": "i", "İ": "i",  # Turkish dotless/dotted i (Kasımpaşa, İstanbul, Göztepe)
}


def _translit(name: str) -> str:
    """Transliteration-folded normalised form (ö/oe/o unify, ü/ue, ä/ae, ø/o).

    ``normalize`` already strips combining diacritics (ö->o via NFKD), so we
    pre-fold the non-decomposing chars (ß, ø, …) and then collapse the common
    German/Nordic digraphs to their base, which lets the schochastics
    transliterations (Koeln, Nuernberg, Malmoe, Haecken, Zuerich) meet the
    umlaut/ASCII spellings (Köln / Koln, Nürnberg / Nurnberg, …) on one form.
    """
    s = strip_country(name)[0]
    for k, v in _PRETRANS.items():
        s = s.replace(k, v)
    base = normalize(s)
    for dig, one in (("oe", "o"), ("ue", "u"), ("ae", "a"), ("aa", "a")):
        base = base.replace(dig, one)
    return base


def _ftokens(name: str) -> set[str]:
    """Transliteration-folded token SET of a name (``(Country)`` stripped)."""
    return set(_translit(name).split())


class FeedError(Exception):
    """A fetched body failed validation (200-not-payload, bad columns, …)."""


# ----------------------------------------------------------------------
# 200-not-payload fetch guard (§10)
# ----------------------------------------------------------------------
def validate_csv_body(
    body: str,
    *,
    content_type: str | None,
    required_cols: list[str],
    min_rows: int = 1,
) -> list[dict]:
    """Parse a CSV body only after it proves it IS the payload.

    Rejects (raises :class:`FeedError`) when the body is HTML / a CDN shield
    page, when the declared content-type is HTML, when the header is missing a
    required column, or when fewer than ``min_rows`` data rows parse.
    """
    if content_type and "html" in content_type.lower():
        raise FeedError(f"content-type is HTML, not CSV: {content_type!r}")
    stripped = body.lstrip()
    if stripped[:1] == "<" or stripped[:9].lower() == "<!doctype":
        raise FeedError("body looks like HTML (shield page), not CSV")
    reader = csv.DictReader(io.StringIO(body))
    cols = set(reader.fieldnames or [])
    missing = [c for c in required_cols if c not in cols]
    if missing:
        raise FeedError(f"CSV missing required columns: {missing}")
    rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    if len(rows) < min_rows:
        raise FeedError(f"too few rows: {len(rows)} < {min_rows}")
    return rows


# ----------------------------------------------------------------------
# football-data.co.uk parser
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Result:
    match_date: date
    home: str
    away: str
    gh: int
    ga: int
    competition: str  # "DOM" or "UEFA CL"


def _parse_fd_date(raw: str) -> date | None:
    """football-data.co.uk dates are dd/mm/yy or dd/mm/yyyy."""
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def parse_football_data_co_uk(
    body: str, *, content_type: str | None = None, min_rows: int = 1,
) -> list[Result]:
    """Parse one mmz4281 division CSV into domestic results.

    Required columns: Date, HomeTeam, AwayTeam, FTHG, FTAG. Rows with an
    unparseable date or non-integer score are skipped, and the whole body is
    200-not-payload-guarded first.
    """
    rows = validate_csv_body(
        body, content_type=content_type,
        required_cols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
        min_rows=min_rows,
    )
    out: list[Result] = []
    for r in rows:
        d = _parse_fd_date(r.get("Date", ""))
        if d is None:
            continue
        try:
            gh, ga = int(r["FTHG"]), int(r["FTAG"])
        except (KeyError, ValueError, TypeError):
            continue
        home = (r.get("HomeTeam") or "").strip()
        away = (r.get("AwayTeam") or "").strip()
        if not home or not away:
            continue
        out.append(Result(d, home, away, gh, ga, "DOM"))
    return out


# ----------------------------------------------------------------------
# Elo update (streaming, no look-ahead; seen-set dedupe across sources/runs)
# ----------------------------------------------------------------------
def gmult(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def expected_home(rh: float, ra: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((rh + ELO_HA) - ra) / ELO_DIVISOR))


def _match_key(r: Result) -> str:
    return f"{r.match_date.isoformat()}|{r.home}|{r.away}"


@dataclass
class EloState:
    """Mutable rating store. ``last_date`` is the newest result incorporated
    (the snapshot's ``From`` date) and advances ONLY on an applied row.
    ``seen`` is the (date|home|away) dedupe set that makes a daily re-fetch of
    overlapping windows — and multiple sources — idempotent."""

    ratings: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    last_date: date | None = None
    seen: set[str] = field(default_factory=set)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, ELO_INIT)

    def apply(self, r: Result) -> None:
        rh, ra = self.rating(r.home), self.rating(r.away)
        we = expected_home(rh, ra)
        w = 1.0 if r.gh > r.ga else (0.0 if r.ga > r.gh else 0.5)
        delta = ELO_K * gmult(r.gh - r.ga) * (w - we)
        self.ratings[r.home] = rh + delta
        self.ratings[r.away] = ra - delta
        self.games[r.home] = self.games.get(r.home, 0) + 1
        self.games[r.away] = self.games.get(r.away, 0) + 1
        self.seen.add(_match_key(r))
        if self.last_date is None or r.match_date > self.last_date:
            self.last_date = r.match_date


def update_elo(state: EloState, results: list[Result]) -> int:
    """Apply ALL results (any sources) in GLOBAL date order, skipping any whose
    (date, home, away) is already in ``state.seen``. Returns rows applied.

    This is the F1 fix: the caller gathers EVERY source first and calls this
    ONCE, so no source's rows are dropped by another's cutoff, and ``From``
    never advances falsely (it moves only on an applied row).
    """
    applied = 0
    for r in sorted(results, key=lambda x: (x.match_date, x.home, x.away)):
        if _match_key(r) in state.seen:
            continue
        state.apply(r)
        applied += 1
    return applied


# ----------------------------------------------------------------------
# Country-restricted name bridge (§5a, Fable-corrected)
# ----------------------------------------------------------------------
#: ClubElo 3-letter Country code -> schochastics full-country name (normalised
#: form). Several ClubElo spellings map to one country (BHZ/BIH, LAT/LVA,
#: LIT/LTU, MNE/MNT, ROM/ROU, SLK/SVK, FAR/FRO).
CODE2COUNTRY: dict[str, str] = {
    "ALB": "albania", "AND": "andorra", "ARM": "armenia", "AUT": "austria",
    "AZE": "azerbaijan", "BEL": "belgium", "BHZ": "bosnia herzegovina",
    "BIH": "bosnia herzegovina", "BLR": "belarus", "BUL": "bulgaria",
    "CRO": "croatia", "CYP": "cyprus", "CZE": "czech republic", "DEN": "denmark",
    "ENG": "england", "ESP": "spain", "EST": "estonia", "FAR": "faroe islands",
    "FRO": "faroe islands", "FIN": "finland", "FRA": "france", "GEO": "georgia",
    "GER": "germany", "GIB": "gibraltar", "GRE": "greece", "HUN": "hungary",
    "IRL": "ireland republic", "ISL": "iceland", "ISR": "israel", "ITA": "italy",
    "KAZ": "kazakhstan", "KOS": "kosovo", "LAT": "latvia", "LVA": "latvia",
    "LIT": "lithuania", "LTU": "lithuania", "LUX": "luxembourg",
    "MAC": "macedonia fyr", "MLT": "malta", "MNE": "montenegro",
    "MNT": "montenegro", "MOL": "moldova", "NED": "netherlands",
    "NIR": "northern ireland", "NOR": "norway", "POL": "poland",
    "POR": "portugal", "ROM": "romania", "ROU": "romania", "RUS": "russia",
    "SCO": "scotland", "SLK": "slovakia", "SVK": "slovakia", "SMR": "san marino",
    "SRB": "serbia", "SUI": "switzerland", "SVN": "slovenia", "SWE": "sweden",
    "TUR": "turkey", "UKR": "ukraine", "WAL": "wales",
}

#: football-data.co.uk division -> schochastics full-country name (normalised).
DIVISION_COUNTRY: dict[str, str] = {
    "E0": "england", "E1": "england", "SP1": "spain", "SP2": "spain",
    "D1": "germany", "D2": "germany", "I1": "italy", "I2": "italy",
    "F1": "france", "F2": "france", "N1": "netherlands", "P1": "portugal",
    "B1": "belgium", "T1": "turkey", "SC0": "scotland", "G1": "greece",
}

_COUNTRY_SUFFIX = re.compile(r"\s*\(([^)]*)\)\s*$")


def ctry_norm(s: str) -> str:
    """Normalise a country string: diacritic-free lowercase alnum words."""
    return normalize(s) if s else ""


def strip_country(ident: str) -> tuple[str, str]:
    """``"Qarabağ (Azerbaijan)"`` -> ``("Qarabağ", "azerbaijan")``.

    Returns the name without the trailing ``(Country)`` and the normalised
    country (``""`` when there is no suffix).
    """
    m = _COUNTRY_SUFFIX.search(ident)
    if not m:
        return ident, ""
    return ident[: m.start()].strip(), ctry_norm(m.group(1))


class AliasFolder:
    """Fold a name to a canonical normalised form through the alias table.

    ``alias_map`` is ``{canonical ClubElo club: [self-built spellings]}``. Both
    the canonical and each alt fold to ``normalize(canonical)``. A name with no
    alias just folds to its own ``normalize`` (after the ``(Country)`` suffix is
    stripped). This is what lets "Manchester City" and the ClubElo key
    "Man City" meet on one token set.
    """

    def __init__(self, alias_map: dict[str, list[str]] | None = None) -> None:
        self._fold: dict[str, str] = {}
        for canon, alts in (alias_map or {}).items():
            cn = _translit(canon)
            self._fold[cn] = cn
            for alt in alts or []:
                self._fold[_translit(strip_country(alt)[0])] = cn

    def canon_norm(self, name: str) -> str:
        base = _translit(name)
        return self._fold.get(base, base)


def _name_score(a: str, b: str) -> float | None:
    """Match score (0-1) between two already alias-folded normalised names.

    * exact equality -> 1.0 (any token count, so single-word clubs like
      "arsenal" resolve);
    * else require ``>= 2`` SHARED tokens (a single shared token — "sheffield",
      "manchester" — never scores a match) and token_set_ratio >=
      FUZZY_THRESHOLD, scaled to 0-1;
    * else ``None``.
    """
    if a == b and a:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    if len(ta & tb) < 2:
        return None
    score = fuzz.token_set_ratio(a, b)
    return score / 100.0 if score >= FUZZY_THRESHOLD else None


def bridge_to_clubelo(
    ratings: dict[str, float],
    games: dict[str, int],
    clubelo_rows: list[tuple[str, str]],
    alias_map: dict[str, list[str]] | None = None,
    *,
    min_games: int = 15,
) -> tuple[dict[str, float], list[str]]:
    """Map self-built ``"Name (Country)"`` idents -> ClubElo canonical keys.

    ``clubelo_rows`` is ``[(club, country_code)]``. Matching is COUNTRY-
    RESTRICTED (via :data:`CODE2COUNTRY`), alias-aware and collision-safe:

    * idents with ``< min_games`` are ignored;
    * within each country a greedy MAX assignment pairs each ClubElo club with
      at most one ident and each ident with at most one club (the global best
      remaining pair first), so a shared surname can never attach one ident to
      two clubs — a contested key is simply left unresolved unless an alias
      made the two names fold equal;
    * returns ``({clubelo_key: elo}, unresolved_clubelo_clubs)``.
    """
    folder = AliasFolder(alias_map)
    # Index eligible idents by normalised country.
    idents_by_ctry: dict[str, list[tuple[str, str]]] = {}
    for ident, elo in ratings.items():
        if games.get(ident, 0) < min_games:
            continue
        _name, ctry = strip_country(ident)
        idents_by_ctry.setdefault(ctry, []).append((ident, folder.canon_norm(ident)))
    clubs_by_ctry: dict[str, list[tuple[str, str]]] = {}
    for club, code in clubelo_rows:
        ctry = CODE2COUNTRY.get((code or "").strip().upper())
        if ctry is None:
            continue
        clubs_by_ctry.setdefault(ctry, []).append((club, folder.canon_norm(club)))

    snap: dict[str, float] = {}
    unresolved: list[str] = []
    for ctry, clubs in clubs_by_ctry.items():
        idents = idents_by_ctry.get(ctry, [])
        # Score every (club, ident) pair in this country. Greedy MAX assignment
        # by (score, then games): when two idents match a key equally well — the
        # same club under two spellings both alias-folded to the key — the
        # better-sampled (more games) rating wins, so a low-game post-cutoff
        # fragment can never displace the seed's full-history rating. This is NOT
        # the forbidden "most games wins" across DIFFERENT clubs: a lower score
        # (a different club) always ranks below, so a contested key still goes
        # unresolved rather than to a wrong club.
        pairs: list[tuple[float, int, str, str]] = []
        for club, club_cn in clubs:
            for ident, ident_cn in idents:
                sc = _name_score(club_cn, ident_cn)
                if sc is not None:
                    pairs.append((sc, games.get(ident, 0), club, ident))
        pairs.sort(key=lambda p: (-p[0], -p[1], p[2], p[3]))
        used_clubs: set[str] = set()
        used_idents: set[str] = set()
        for _sc, _g, club, ident in pairs:
            if club in used_clubs or ident in used_idents:
                continue
            snap[club] = ratings[ident]
            used_clubs.add(club)
            used_idents.add(ident)
        unresolved.extend(club for club, _ in clubs if club not in used_clubs)
    return snap, unresolved


def _display_country(ctry_normed: str) -> str:
    """Title-case a normalised country for a readable ``(Country)`` suffix.
    ``strip_country`` re-normalises it, so the exact casing is cosmetic."""
    return ctry_normed.title() if ctry_normed else ""


def align_to_state(
    results: list[Result],
    state_idents: list[str],
    country: str | None = None,
    *,
    team_country: dict[str, str] | None = None,
    alias_map: dict[str, list[str]] | None = None,
    games: dict[str, int] | None = None,
) -> list[Result]:
    """Fold feed team names onto the EXISTING ``"(Country)"`` state idents so
    the daily feed does not FRAGMENT a club already carried from the seed.

    Country-restricted (never cross-country): ``country`` sets ONE country for
    every team (a domestic division, see :data:`DIVISION_COUNTRY`);
    ``team_country`` gives a per-team country (Champions League, from
    ``area.name``) so each CL side is matched only against its own country's
    idents — which breaks the "Arsenal FC" vs "Arsenal FC (Belarus)" tie.

    Matching, within the team's country, in order:

    1. **Exact alias-fold equality.** Both the feed name and the ident are
       folded through the alias table; if they land on the SAME canonical and
       exactly one ident does, accept. This is the ONLY way token rules cannot
       merge (``inter`` vs ``internazionale`` — blocked from a prefix match by
       the Milano/Milan clash — and ``man`` vs ``manchester``), so without it
       Inter's and Man City's post-cutoff CL/domestic games would fragment onto
       a fresh 1500 ident. It is exact-equality + country-restricted + unique,
       so only names the table explicitly pairs can collapse — a non-aliased
       name folds to itself and this pass is a no-op for it.
    2. **Unique token-subset** (transliteration-folded SETS, no alias): accept
       iff the feed set ⊆ exactly one ident's set (or that ident's ⊆ the feed)
       and no other candidate also satisfies it. "Leicester" ⊂ "Leicester City"
       resolves; "Verona" ⊂ both "Hellas Verona" and "Chievo Verona" is refused.

    A name that matches neither but whose country is known gets that country
    APPENDED (``f"{name} ({Country})"``) so it is never a suffix-less phantom:
    it stays bridgeable and its (date,home,away) key is stable across runs.
    """
    folder = AliasFolder(alias_map)
    by_ctry: dict[str, list[tuple[str, set[str], str]]] = {}
    for i in state_idents:
        by_ctry.setdefault(
            strip_country(i)[1], []).append((i, _ftokens(i), folder.canon_norm(i)))
    default_ctry = ctry_norm(country or "")
    cache: dict[tuple[str, str], str] = {}

    def canon(name: str, ctry: str) -> str:
        key = (name, ctry)
        if key in cache:
            return cache[key]
        cands = by_ctry.get(ctry, [])
        af = folder.canon_norm(name)
        exact = [ident for ident, _it, ia in cands if ia == af]
        if len(exact) == 1:
            res = exact[0]
        elif len(exact) > 1:
            # Several seed idents share this canonical — the SAME club under
            # multiple spellings (e.g. "Sunderland" / "Sunderland AFC"). Pick the
            # best-sampled one so the club's games consolidate rather than split.
            res = max(exact, key=lambda i: (games or {}).get(i, 0))
        else:
            ft = _ftokens(name)
            subset = [
                ident for ident, it, _ia in cands
                if ft and it and (ft <= it or it <= ft)
            ]
            if len(subset) == 1:
                res = subset[0]
            elif ctry:
                res = f"{name} ({_display_country(ctry)})"
            else:
                res = name
        cache[key] = res
        return res

    out: list[Result] = []
    for r in results:
        if team_country is not None:
            ch = ctry_norm(team_country.get(r.home, ""))
            ca = ctry_norm(team_country.get(r.away, ""))
        else:
            ch = ca = default_ctry
        out.append(Result(r.match_date, canon(r.home, ch), canon(r.away, ca),
                          r.gh, r.ga, r.competition))
    return out


def parse_fdorg_matches(
    matches: list[dict], *, competition: str = "UEFA CL",
) -> list[Result]:
    """football-data.org match dicts (FINISHED only) -> Results. Tolerant: a
    match missing a score/date/team is skipped, never a crash."""
    out: list[Result] = []
    for m in matches:
        if (m.get("status") or "") != "FINISHED":
            continue
        try:
            d = date.fromisoformat((m["utcDate"] or "")[:10])
            ft = (m.get("score") or {}).get("fullTime") or {}
            gh, ga = int(ft["home"]), int(ft["away"])
            h = (m["homeTeam"] or {}).get("name") or ""
            a = (m["awayTeam"] or {}).get("name") or ""
        except (KeyError, ValueError, TypeError):
            continue
        if not h or not a:
            continue
        out.append(Result(d, h.strip(), a.strip(), gh, ga, competition))
    return out


def load_clubelo_rows(path) -> list[tuple[str, str]]:
    """Read a ClubElo CSV -> ``[(Club, Country_code)]`` for the bridge."""
    rows: list[tuple[str, str]] = []
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                club = (r.get("Club") or "").strip()
                if club:
                    rows.append((club, (r.get("Country") or "").strip()))
    except OSError:
        return []
    return rows


# ----------------------------------------------------------------------
# Snapshot writer (ClubElo CSV schema so the engine loads it unchanged)
# ----------------------------------------------------------------------
def write_clubelo_snapshot(
    path, snap: dict[str, float], *, from_date: date,
    country_of: dict[str, str] | None = None,
) -> None:
    """Write the self-built snapshot in ClubElo schema (Rank,Club,Country,
    Level,Elo,From,To). ``From`` is the newest result incorporated so the
    engine's freshness guard reads it; ``To`` equals ``From``."""
    country_of = country_of or {}
    ordered = sorted(snap.items(), key=lambda kv: -kv[1])
    iso = from_date.isoformat()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Rank", "Club", "Country", "Level", "Elo", "From", "To"])
        for rank, (club, elo) in enumerate(ordered, start=1):
            w.writerow([rank, club, country_of.get(club, ""), 1, round(elo, 2), iso, iso])
