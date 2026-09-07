"""Tests for the clubelo.com HTML scrape fallback (no network — fixture/mocked).

The CSV API (api.clubelo.com) was deactivated upstream; the website still
serves the same ratings as HTML. These tests pin the fallback that rebuilds an
API-compatible snapshot from a *captured* page (``fixtures/clubelo_home.html``),
and the wiring that keeps the API primary and the scrape a fallback:

* parse the fixture into clubs + the snapshot date;
* rebuild a CSV whose column contract is byte-identical to the API's, with club
  names CANONICALISED back to the previous snapshot so nothing downstream moves;
* non-European clubs (the website is now worldwide) are filtered out, killing
  the Ecuador-"Barcelona" / Uruguay-"Liverpool" name collisions;
* coverage gaps (previous clubs we could not refresh) are surfaced, never
  silently stale;
* API-works / API-fails-scrape-works / both-fail through ``refresh_latest``.
"""

from __future__ import annotations

import io
import socket
from datetime import date
from pathlib import Path

import betbot.data.clubelo as clubelo

FIXTURE = Path(__file__).parent / "fixtures" / "clubelo_home.html"
HEADER = "Rank,Club,Country,Level,Elo,From,To"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# A previous (last-known-good) snapshot with the CANONICAL API short names the
# website spells differently, plus two clubs absent from the fixture.
_PREV = "\n".join(
    [
        HEADER,
        "1,Arsenal,ENG,1,2063.75,2026-08-31,2026-12-31",
        "2,Bayern,GER,1,2000.87,2026-08-31,2026-12-31",
        "3,Man City,ENG,1,1970.85,2026-08-31,2026-12-31",
        "6,Real Madrid,ESP,1,1922.90,2026-08-31,2026-12-31",
        "10,Inter,ITA,1,1888.60,2026-08-31,2026-12-31",
        "18,Atletico,ESP,1,1827.70,2026-08-31,2026-12-31",
        "62,Bilbao,ESP,1,1677.90,2026-08-31,2026-12-31",
        "65,Sociedad,ESP,1,1669.50,2026-08-31,2026-12-31",
        "4,Paris SG,FRA,1,1967.87,2026-08-31,2026-12-31",  # FRA: not in fixture
        "144,Alkmaar,NED,1,1539.00,2026-08-31,2026-12-31",  # NED: not in fixture
    ]
) + "\n"


def _write_prev(tmp_path: Path) -> Path:
    dest = tmp_path / "clubelo_latest.csv"
    dest.write_text(_PREV, encoding="utf-8")
    return dest


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parse_fixture_yields_clubs_and_snapshot_date():
    rows, snap = clubelo._parse_clubelo_html(_fixture_html())
    assert snap == date(2026, 9, 6)
    names = {r.name for r in rows}
    # linked top clubs AND unlinked deep rows both parsed
    assert "Bayern München" in names
    assert "Internazionale" in names
    assert "Atlético" in names
    # the eloData JS widget must NOT leak rows (it has no Ast cell)
    assert all(r.elo > 0 for r in rows)


def test_parser_ignores_the_elodata_widget():
    # eloData rows use the display name without a NonAst/Ast cell; the widget's
    # three entries must not double-count Bayern/Arsenal/Man City.
    rows, _ = clubelo._parse_clubelo_html(_fixture_html())
    assert sum(1 for r in rows if r.name == "Bayern München") == 1


# --------------------------------------------------------------------------
# CSV rebuild — shape, canonicalisation, filtering
# --------------------------------------------------------------------------


def test_scrape_builds_api_compatible_csv(tmp_path):
    dest = _write_prev(tmp_path)
    assert clubelo.scrape_latest(dest, html=_fixture_html()) is True

    text = dest.read_text()
    assert text.splitlines()[0] == HEADER
    # validates against the SAME gate the API path uses
    assert clubelo._validate(text) is None
    st = clubelo.snapshot_status(dest)
    assert st.snapshot_date == date(2026, 9, 6)
    assert not st.stale


def test_names_are_canonicalised_to_the_previous_snapshot(tmp_path):
    dest = _write_prev(tmp_path)
    clubelo.scrape_latest(dest, html=_fixture_html())
    rows = {c["Club"]: c for c in _read_csv(dest)}
    # website spells these differently; output must keep the API short names
    assert "Inter" in rows and "Internazionale" not in rows
    assert "Atletico" in rows and "Atlético" not in rows
    assert "Bilbao" in rows and "Athletic Club" not in rows
    assert "Sociedad" in rows
    # and they carry the FRESH website Elo (integers), From = ratings date
    assert rows["Inter"]["Elo"] == "1948"
    assert rows["Atletico"]["Elo"] == "1891"
    assert rows["Inter"]["From"] == "2026-09-06"


def test_non_european_clubs_are_filtered_out(tmp_path):
    dest = _write_prev(tmp_path)
    clubelo.scrape_latest(dest, html=_fixture_html())
    countries = {c["Country"] for c in _read_csv(dest)}
    assert "BRA" not in countries
    names = {c["Club"] for c in _read_csv(dest)}
    assert "Flamengo" not in names and "Palmeiras" not in names


def test_unrefreshed_previous_clubs_are_surfaced_not_silently_stale():
    rows, snap = clubelo._parse_clubelo_html(_fixture_html())
    import csv as _csv

    prev = list(_csv.DictReader(io.StringIO(_PREV)))
    text, report = clubelo._build_csv_from_scrape(rows, prev, snap_date=snap)
    # FRA (Paris SG) and NED (Alkmaar) have no rows in the fixture
    assert "Paris SG" in report["unrefreshed"]
    assert "Alkmaar" in report["unrefreshed"]
    # ...and they are kept OUT of the CSV (they take the naive path, not a
    # silently frozen rating)
    out_names = {c["Club"] for c in _csv.DictReader(io.StringIO(text))}
    assert "Paris SG" not in out_names
    assert "Alkmaar" not in out_names


def test_cold_start_without_previous_still_emits_a_valid_snapshot(tmp_path):
    dest = tmp_path / "clubelo_latest.csv"  # no previous file
    assert clubelo.scrape_latest(dest, html=_fixture_html()) is True
    text = dest.read_text()
    assert clubelo._validate(text) is None  # display names, still valid


# --------------------------------------------------------------------------
# refresh_latest wiring: API primary, scrape fallback
# --------------------------------------------------------------------------


def _urlopen_api_fails_web_serves(html: str):
    """API host times out; website host serves the fixture."""

    def _f(target, *a, **k):
        url = target if isinstance(target, str) else target.full_url
        if "api.clubelo.com" in url:
            raise socket.timeout("timed out")
        return _Resp(html.encode("utf-8"))

    return _f


def test_api_success_does_not_invoke_the_scrape(tmp_path, monkeypatch):
    today = date.today().isoformat()
    lines = [HEADER, f"1,Inter,ITA,1,1888.6,{today},{today}"]
    lines += [f"{i + 2},Club {i},ENG,1,{1800 - i}.0,{today},{today}" for i in range(60)]
    good = ("\n".join(lines) + "\n").encode()

    def _api_ok(target, *a, **k):
        url = target if isinstance(target, str) else target.full_url
        # only the API host may be hit; touching the website means we fell back
        assert "api.clubelo.com" in url, f"scrape was invoked: {url}"
        return _Resp(good)

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _api_ok)
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest, sleep=lambda _s: None) is True
    assert "Inter" in dest.read_text()


def test_api_fails_then_scrape_fallback_serves(tmp_path, monkeypatch):
    dest = _write_prev(tmp_path)
    monkeypatch.setattr(
        clubelo.urllib.request, "urlopen", _urlopen_api_fails_web_serves(_fixture_html())
    )
    assert clubelo.refresh_latest(dest, retries=2, sleep=lambda _s: None) is True
    # served from the website: fresh integer Elo under the canonical name
    rows = {c["Club"]: c for c in _read_csv(dest)}
    assert rows["Inter"]["Elo"] == "1948"
    assert clubelo.snapshot_status(dest).snapshot_date == date(2026, 9, 6)


def test_both_paths_fail_leaves_previous_and_raises_the_stale_alarm(tmp_path, monkeypatch):
    dest = tmp_path / "clubelo_latest.csv"
    # an old snapshot on disk so the staleness alarm has something to flag
    old = _PREV.replace("2026-08-31", "2026-08-01").replace("2026-12-31", "2026-08-11")
    dest.write_text(old)

    def _all_fail(*a, **k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _all_fail)
    seen: list[str] = []
    monkeypatch.setattr(clubelo.log, "error", lambda ev, **kw: seen.append(ev))
    monkeypatch.setattr(clubelo.log, "warning", lambda ev, **kw: None)

    assert clubelo.refresh_latest(dest, retries=2, sleep=lambda _s: None) is False
    assert dest.read_text() == old  # previous snapshot untouched
    assert "clubelo_refresh_failed" in seen
    assert "clubelo_snapshot_stale" in seen  # alarm fires only when BOTH fail


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _read_csv(path: Path):
    import csv as _csv

    with path.open() as fh:
        return list(_csv.DictReader(fh))
