"""Tests for the clubelo.com HTML scrape fallback (no network — fixture/mocked).

The CSV API (api.clubelo.com) was deactivated upstream. The website still serves
ratings as HTML, BUT on a different scale ("Elo +/- Golo"), so it is NOT fed to
the API-tuned CL engine. These tests pin:

* the parser (per-row country, the eloData widget ignored, flagless rows);
* the site-scale monitoring CSV the scrape rebuilds, canonicalised to a pinned
  API reference — with COUNTRY AGREEMENT required for every match (no rating
  theft) and the nine UEFA nations the two sources code differently recovered;
* the wiring: the API stays primary; on API failure the engine snapshot is left
  UNTOUCHED (so the staleness alarm still fires) and only a separate site-scale
  monitoring file is refreshed.
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


# A pinned API-scale reference with the CANONICAL short names the website spells
# differently, the nine-nation code mismatch (ROM/SLK for Romania/Slovakia), a
# rating-theft target (Dinamo Bucuresti ROM, with no in-country scrape match but
# a same-name-ish Dinamo Brest in BLR), and two clubs absent from the fixture.
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
        "120,FCSB,ROM,1,1590.00,2026-08-31,2026-12-31",  # ROM: site uses ROU (B3)
        "160,Slovan Bratislava,SLK,1,1510.00,2026-08-31,2026-12-31",  # SLK->SVK (B3)
        "210,Dinamo Bucuresti,ROM,1,1440.00,2026-08-31,2026-12-31",  # B2 theft target
        "170,BATE,BLR,1,1460.00,2026-08-31,2026-12-31",  # anchors BLR into scope
        "4,Paris SG,FRA,1,1967.87,2026-08-31,2026-12-31",  # FRA: not in fixture
        "144,Alkmaar,NED,1,1539.00,2026-08-31,2026-12-31",  # NED: not in fixture
    ]
) + "\n"


def _write_ref(tmp_path: Path) -> Path:
    """Write the pinned API reference; return the (separate) monitor dest path."""
    ref = tmp_path / "clubelo_latest.csv"
    ref.write_text(_PREV, encoding="utf-8")
    return tmp_path / "clubelo_scrape_latest.csv"


def _read_csv(path: Path):
    import csv as _csv

    with path.open() as fh:
        return list(_csv.DictReader(fh))


def _prev_rows():
    import csv as _csv

    return list(_csv.DictReader(io.StringIO(_PREV)))


# --------------------------------------------------------------------------
# parsing (B5: per-row country, flagless rows, eloData widget ignored)
# --------------------------------------------------------------------------


def test_parse_fixture_yields_clubs_and_snapshot_date():
    rows, snap = clubelo._parse_clubelo_html(_fixture_html())
    assert snap == date(2026, 9, 6)
    names = {r.name for r in rows}
    assert "Bayern München" in names  # linked top club
    assert "Internazionale" in names  # website display name (canon later)
    assert "Atlético" in names
    assert all(r.elo > 0 for r in rows)


def test_parser_ignores_the_elodata_widget():
    rows, _ = clubelo._parse_clubelo_html(_fixture_html())
    assert sum(1 for r in rows if r.name == "Bayern München") == 1


def test_flagless_row_gets_no_country_and_is_not_carried_forward():
    # 'Young Africans' sits in the flagless Calculation section AFTER a BRA block;
    # it must NOT inherit BRA (or any) country — its country is None (B5).
    rows, _ = clubelo._parse_clubelo_html(_fixture_html())
    ya = [r for r in rows if r.name == "Young Africans"]
    assert ya and ya[0].country is None


# --------------------------------------------------------------------------
# CSV rebuild — shape, canonicalisation, scope, country agreement
# --------------------------------------------------------------------------


def test_scrape_builds_valid_site_scale_csv(tmp_path):
    dest = _write_ref(tmp_path)
    ref = tmp_path / "clubelo_latest.csv"
    assert clubelo.scrape_latest(dest, reference=ref, html=_fixture_html()) is True
    text = dest.read_text()
    assert text.splitlines()[0] == HEADER
    assert clubelo._validate(text) is None
    st = clubelo.snapshot_status(dest)
    assert st.snapshot_date == date(2026, 9, 6)


def test_scrape_writes_dated_site_archive(tmp_path):
    # The monitor file is overwritten every run; a dated copy under
    # data/clubelo_site/<snap_date>.csv must ALSO be written so a real
    # site-scale history accrues (byte-identical to the monitor snapshot).
    dest = _write_ref(tmp_path)
    ref = tmp_path / "clubelo_latest.csv"
    assert clubelo.scrape_latest(dest, reference=ref, html=_fixture_html()) is True
    archive = tmp_path / "clubelo_site" / "2026-09-06.csv"
    assert archive.exists()
    assert archive.read_text() == dest.read_text()


def test_scrape_archive_failure_does_not_fail_the_scrape(tmp_path):
    # If the archive dir cannot be created (here: a regular FILE sits where the
    # clubelo_site/ directory should be), the scrape must STILL succeed — dest
    # written, returns True — and log clubelo_scrape_archive_failed. The monitor
    # snapshot is the contract; the dated archive is best-effort.
    import structlog

    (tmp_path / "clubelo_site").write_text("i am a file, not a dir\n")
    dest = _write_ref(tmp_path)
    ref = tmp_path / "clubelo_latest.csv"
    with structlog.testing.capture_logs() as logs:
        assert clubelo.scrape_latest(dest, reference=ref, html=_fixture_html()) is True
    assert dest.read_text().splitlines()[0] == HEADER
    assert not (tmp_path / "clubelo_site" / "2026-09-06.csv").exists()
    assert any(e.get("event") == "clubelo_scrape_archive_failed" for e in logs)


def test_names_are_canonicalised_to_the_reference(tmp_path):
    dest = _write_ref(tmp_path)
    ref = tmp_path / "clubelo_latest.csv"
    clubelo.scrape_latest(dest, reference=ref, html=_fixture_html())
    rows = {c["Club"]: c for c in _read_csv(dest)}
    assert "Inter" in rows and "Internazionale" not in rows
    assert "Atletico" in rows and "Atlético" not in rows
    assert "Bilbao" in rows and "Athletic Club" not in rows
    # website Elo (integers), From = ratings date
    assert rows["Inter"]["Elo"] == "1948"
    assert rows["Atletico"]["Elo"] == "1891"
    assert rows["Inter"]["From"] == "2026-09-06"


def test_non_european_clubs_are_filtered_out(tmp_path):
    dest = _write_ref(tmp_path)
    ref = tmp_path / "clubelo_latest.csv"
    clubelo.scrape_latest(dest, reference=ref, html=_fixture_html())
    names = {c["Club"] for c in _read_csv(dest)}
    assert "Flamengo" not in names and "Palmeiras" not in names
    assert "Young Africans" not in names  # flagless => no country => dropped


# --------------------------------------------------------------------------
# B3: nine-nation country-code mismatch recovered
# --------------------------------------------------------------------------


def test_code_mismatch_nations_are_recovered_not_dropped(tmp_path):
    # Reference codes ROM/SLK; site codes ROU/SVK. Without the map both would be
    # out of scope and dropped. FCSB and Slovan Bratislava are recent CL sides.
    dest = _write_ref(tmp_path)
    ref = tmp_path / "clubelo_latest.csv"
    clubelo.scrape_latest(dest, reference=ref, html=_fixture_html())
    rows = {c["Club"]: c for c in _read_csv(dest)}
    assert "FCSB" in rows and rows["FCSB"]["Elo"] == "1600"
    assert "Slovan Bratislava" in rows and rows["Slovan Bratislava"]["Elo"] == "1520"


# --------------------------------------------------------------------------
# B2: rating theft prevented by country agreement
# --------------------------------------------------------------------------


def test_cross_country_fuzzy_match_does_not_steal_a_rating():
    rows, snap = clubelo._parse_clubelo_html(_fixture_html())
    text, report = clubelo._build_csv_from_scrape(rows, _prev_rows(), snap_date=snap)
    out = {c["Club"]: c for c in __import__("csv").DictReader(io.StringIO(text))}
    # Dinamo Bucuresti (ROM/ROU) has NO in-country scrape row; the only near-name
    # is Dinamo Brest (BLR). Country agreement must keep them apart.
    assert "Dinamo Bucuresti" not in out
    assert "Dinamo Bucuresti" in report["unrefreshed"]
    # Dinamo Brest keeps its OWN identity (emitted under its own name, not stolen)
    assert "Dinamo Brest" in out
    assert out["Dinamo Brest"]["Country"] == "BLR"


# --------------------------------------------------------------------------
# B5: duplicate name+country with differing Elo is surfaced
# --------------------------------------------------------------------------


def test_duplicate_name_country_with_differing_elo_is_logged():
    rows, snap = clubelo._parse_clubelo_html(_fixture_html())
    _text, report = clubelo._build_csv_from_scrape(rows, _prev_rows(), snap_date=snap)
    # Real Sociedad appears twice in ESP with 1780 and 1770.
    assert report["duplicate_count"] >= 1


def test_unrefreshed_previous_clubs_are_surfaced_not_silently_stale():
    rows, snap = clubelo._parse_clubelo_html(_fixture_html())
    text, report = clubelo._build_csv_from_scrape(rows, _prev_rows(), snap_date=snap)
    assert "Paris SG" in report["unrefreshed"]  # FRA, absent from fixture
    assert "Alkmaar" in report["unrefreshed"]  # NED, absent from fixture
    out_names = {c["Club"] for c in __import__("csv").DictReader(io.StringIO(text))}
    assert "Paris SG" not in out_names and "Alkmaar" not in out_names


def test_cold_start_without_reference_still_emits_a_valid_snapshot(tmp_path):
    dest = tmp_path / "clubelo_scrape_latest.csv"  # no reference file present
    ref = tmp_path / "clubelo_latest.csv"
    assert clubelo.scrape_latest(dest, reference=ref, html=_fixture_html()) is True
    assert clubelo._validate(dest.read_text()) is None


# --------------------------------------------------------------------------
# refresh_latest wiring: API primary; scrape is monitoring-only (B1/B4)
# --------------------------------------------------------------------------


def _urlopen_api_fails_web_serves(html: str):
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
        assert "api.clubelo.com" in url, f"scrape was invoked: {url}"
        return _Resp(good)

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _api_ok)
    dest = tmp_path / "clubelo_latest.csv"
    assert clubelo.refresh_latest(dest, sleep=lambda _s: None) is True
    assert "Inter" in dest.read_text()


def test_api_failure_leaves_engine_snapshot_untouched_and_writes_monitor(tmp_path, monkeypatch):
    # The engine snapshot (clubelo_latest.csv) is the API-scale, pinned input.
    dest = tmp_path / "clubelo_latest.csv"
    dest.write_text(_PREV, encoding="utf-8")

    monkeypatch.setattr(
        clubelo.urllib.request, "urlopen", _urlopen_api_fails_web_serves(_fixture_html())
    )
    # refresh reports FAILURE (the API-scale feed did not refresh); it does NOT
    # pretend the mis-scaled site data is a fresh API snapshot.
    assert clubelo.refresh_latest(dest, retries=2, sleep=lambda _s: None) is False
    # engine input is byte-for-byte unchanged (staleness alarm can still fire)
    assert dest.read_text() == _PREV
    # ...but a separate site-scale monitoring file was written
    monitor = tmp_path / "clubelo_scrape_latest.csv"
    assert monitor.exists()
    rows = {c["Club"]: c for c in _read_csv(monitor)}
    assert rows["Inter"]["Elo"] == "1948"  # site scale, monitoring only


def test_scrape_monitor_reference_is_the_pinned_snapshot_not_its_own_output(tmp_path, monkeypatch):
    # B4: a transient scrape miss must not permanently delete a club. The
    # reference is always the pinned API snapshot, so a club absent from one
    # scrape is still looked for on the next.
    dest = tmp_path / "clubelo_latest.csv"
    dest.write_text(_PREV, encoding="utf-8")
    monitor = tmp_path / "clubelo_scrape_latest.csv"
    # First cycle: a truncated page (only Inter present) writes a thin monitor.
    thin = (
        '<h1><a href="/2026-09-06/"></a></h1><table class="ast">'
        '<tr><td class="l"><a href="/ITA"><img alt="ITA" src="x"/></a> '
        '<small> 10 </small><a href="/Inter"><span class="NonAst">INT</span>'
        '<span class="Ast">Internazionale</span></a></td><td class="r">1948</td></tr>'
        "</table>"
    )
    assert clubelo.scrape_latest(monitor, reference=dest, html=thin) is False  # too few rows
    # Second cycle: the full page restores coverage — proving we never narrowed
    # the reference to the previous (thin) scrape output.
    assert clubelo.scrape_latest(monitor, reference=dest, html=_fixture_html()) is True
    names = {c["Club"] for c in _read_csv(monitor)}
    assert "FCSB" in names and "Bilbao" in names


def test_both_paths_fail_leaves_previous_and_raises_the_stale_alarm(tmp_path, monkeypatch):
    dest = tmp_path / "clubelo_latest.csv"
    old = _PREV.replace("2026-08-31", "2026-08-01").replace("2026-12-31", "2026-08-11")
    dest.write_text(old)

    def _all_fail(*a, **k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _all_fail)
    seen: list[str] = []
    monkeypatch.setattr(clubelo.log, "error", lambda ev, **kw: seen.append(ev))
    monkeypatch.setattr(clubelo.log, "warning", lambda ev, **kw: None)
    monkeypatch.setattr(clubelo.log, "info", lambda ev, **kw: None)

    assert clubelo.refresh_latest(dest, retries=2, sleep=lambda _s: None) is False
    assert dest.read_text() == old  # previous snapshot untouched
    assert "clubelo_refresh_failed" in seen
    assert "clubelo_snapshot_stale" in seen  # alarm fires when the feed is down
