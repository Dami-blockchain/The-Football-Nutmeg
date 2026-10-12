"""Free-data feed + Elo builder for the self-built CL fallback snapshot.

Pure, stdlib-only, importable + unit-tested. The operator-run scripts
(``scripts/seed_selfbuilt_elo.py``, ``scripts/build_selfbuilt_elo.py``) are
thin orchestrators over this module; the load-bearing, FREE-only-critical parts
— the 200-not-payload fetch guard, the football-data.co.uk parser, the Elo
update and the ClubElo name bridge — live here so they are tested.

NEVER pay: every source is free and keyless —
``football-data.co.uk/mmz4281/{season}/{div}.csv`` (same source odds.py uses)
and ``openfootball/europe`` (weekly-fresh UEFA results). No API key, no
subscription.

Elo spec (matches the validated prototype, api.clubelo-like BUILD scale): the
SNAPSHOT stores RAW ratings on this scale (SD ~147); the engine prices them
with the LIVE 51/312 constants (option (a)). The build constants here
(K/HA/DIVISOR) are NOT the serve constants and must not be confused with them.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date

# ---- Elo BUILD constants (NOT the serve 51/312; see module docstring) -------
ELO_K = 20.0
ELO_HA = 65.0
ELO_DIVISOR = 400.0
ELO_INIT = 1500.0


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
    required column, or when fewer than ``min_rows`` data rows parse. This is
    the hazard memory flags for football-data.co.uk (HTTP 200 whose body is not
    the CSV). On success returns the rows as dicts.
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
    competition: str  # "DOM" (domestic) or "UEFA CL" etc.


def _parse_fd_date(raw: str) -> date | None:
    """football-data.co.uk dates are dd/mm/yy or dd/mm/yyyy."""
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            from datetime import datetime
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def parse_football_data_co_uk(
    body: str, *, content_type: str | None = None, min_rows: int = 1,
) -> list[Result]:
    """Parse one mmz4281 division CSV into domestic results.

    Required columns: Date, HomeTeam, AwayTeam, FTHG, FTAG. Rows with an
    unparseable date or non-integer score are skipped (never crash the build),
    and the whole body is 200-not-payload-guarded first.
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
# openfootball/europe tolerant .txt parser (UEFA cross-league results)
# ----------------------------------------------------------------------
_SCORE_RE = re.compile(r"^(?P<home>.+?)\s{2,}(?P<gh>\d+)\s*[-:]\s*(?P<ga>\d+)\s+(?P<away>.+?)\s*$")
_DATE_RE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_DOW_DATE_RE = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?P<mon>[A-Z][a-z]{2})/(?P<d>\d{1,2})"
)
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def parse_openfootball_txt(
    text: str, *, competition: str = "UEFA CL", default_year: int | None = None,
) -> list[Result]:
    """Best-effort parse of an openfootball match .txt into UEFA results.

    openfootball files carry date headers (either ``2026-09-16`` or
    ``Tue Sep/16``) followed by score lines ``Home   3-1  Away``. Lines that do
    not match are ignored. A ``Sep/16`` header needs a year: ``default_year``
    (the file's season) supplies it. Tolerant by design — a format drift
    degrades to "fewer rows", never a crash.
    """
    results: list[Result] = []
    cur: date | None = None
    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            continue
        m = _DATE_RE.search(s)
        if m:
            try:
                cur = date(int(m["y"]), int(m["m"]), int(m["d"]))
            except ValueError:
                pass
            continue
        m = _DOW_DATE_RE.search(s)
        if m and default_year is not None:
            mon = _MONTHS.get(m["mon"])
            if mon:
                try:
                    cur = date(default_year, mon, int(m["d"]))
                except ValueError:
                    pass
            continue
        m = _SCORE_RE.match(s)
        if m and cur is not None:
            home = m["home"].strip().lstrip("@").strip()
            away = m["away"].strip()
            # Drop trailing annotations like "(5-4 pen.)" already excluded by \s{2,}.
            try:
                results.append(Result(cur, home, away,
                                      int(m["gh"]), int(m["ga"]), competition))
            except ValueError:
                continue
    return results


# ----------------------------------------------------------------------
# Elo update (streaming, no look-ahead when fed date-ordered results)
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


@dataclass
class EloState:
    """Mutable rating store. ``last_date`` is the newest result incorporated
    (the snapshot's ``From`` date). ``games`` gates the bridge (ignore clubs
    with too few games)."""

    ratings: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    last_date: date | None = None

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
        if self.last_date is None or r.match_date > self.last_date:
            self.last_date = r.match_date


def align_to_state(
    results: list[Result], existing_keys: list[str], resolver, *, threshold: float = 70.0,
) -> list[Result]:
    """Rewrite each result's team names onto an EXISTING state key when the
    resolver resolves them, so the daily incremental (football-data.co.uk short
    names + openfootball long names) does not FRAGMENT a club already carried
    under the seed's ident. Unresolved names keep their raw spelling and become
    new keys (bridged — or dropped — at snapshot time). Resolution uses the same
    TeamAliasResolver + ambiguity rejection as everywhere else.

    A tiny per-call cache avoids re-resolving the same name across a season CSV.
    """
    if not existing_keys:
        return results
    cache: dict[str, str] = {}

    def canon(name: str) -> str:
        if name in cache:
            return cache[name]
        hit = resolver.match(name, existing_keys, threshold=threshold)
        cache[name] = hit or name
        return cache[name]

    return [
        Result(r.match_date, canon(r.home), canon(r.away), r.gh, r.ga, r.competition)
        for r in results
    ]


def update_elo(state: EloState, results: list[Result]) -> int:
    """Apply results IN DATE ORDER (no look-ahead). Returns rows applied.

    Only results strictly AFTER ``state.last_date`` are applied, so a daily run
    that re-fetches an overlapping window never double-counts a match.
    """
    applied = 0
    cutoff = state.last_date
    for r in sorted(results, key=lambda x: (x.match_date, x.home, x.away)):
        if cutoff is not None and r.match_date <= cutoff:
            continue
        state.apply(r)
        applied += 1
    return applied


# ----------------------------------------------------------------------
# ClubElo name bridge (§5a) — uses TeamAliasResolver, never an ad-hoc matcher
# ----------------------------------------------------------------------
def bridge_to_clubelo(
    ratings: dict[str, float],
    games: dict[str, int],
    clubelo_keys: list[str],
    resolver,
    *,
    min_games: int = 15,
    threshold: float = 70.0,
) -> tuple[dict[str, float], list[str]]:
    """Map self-built team names -> ClubElo canonical keys.

    Resolution is delegated to the SAME ``TeamAliasResolver`` the rest of
    betbot uses: exact alias table first, then exact-normalised, then fuzzy
    (``threshold`` = 70 per §5a), and it REFUSES ambiguous top-ties (returns
    None) rather than mis-mapping. An unresolved club is logged in the returned
    list and DROPPED (never priced off a wrong rating) — same contract as the
    engine's unresolved-team guard.

    Returns ``({clubelo_key: elo}, unresolved_names)``. On a key collision
    (two self-built names mapping to one ClubElo key) the better-rated-games
    entry wins deterministically.
    """
    snap: dict[str, float] = {}
    best_games: dict[str, int] = {}
    unresolved: list[str] = []
    for name, elo in ratings.items():
        if games.get(name, 0) < min_games:
            continue
        key = resolver.match(name, clubelo_keys, threshold=threshold)
        if key is None:
            unresolved.append(name)
            continue
        g = games.get(name, 0)
        if key not in snap or g > best_games.get(key, -1):
            snap[key] = elo
            best_games[key] = g
    return snap, unresolved


# ----------------------------------------------------------------------
# Snapshot writer (ClubElo CSV schema so the engine loads it unchanged)
# ----------------------------------------------------------------------
def write_clubelo_snapshot(
    path, snap: dict[str, float], *, from_date: date, country_of: dict[str, str] | None = None,
) -> None:
    """Write the self-built snapshot in ClubElo schema (Rank,Club,Country,
    Level,Elo,From,To). ``From`` is the newest result incorporated so the
    engine's own freshness guard reads it. ``To`` is left equal to ``From``
    (the engine measures age off ``From``)."""
    country_of = country_of or {}
    ordered = sorted(snap.items(), key=lambda kv: -kv[1])
    iso = from_date.isoformat()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Rank", "Club", "Country", "Level", "Elo", "From", "To"])
        for rank, (club, elo) in enumerate(ordered, start=1):
            w.writerow([rank, club, country_of.get(club, ""), 1, round(elo, 2), iso, iso])
