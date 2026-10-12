"""Tests for the self-built-Elo CL fallback (FALLBACK_SCOPE.md rev 2).

No network. Snapshots are injected; the DB tests use a tmp sqlite. Covers:
flag-off regression (byte-identical), engage/disengage, flag-on-fresh
dual-logs-without-using, missing-site, both-stale (fresher-by-From / naive),
SD sanity band, the production name bridge + ambiguity rejection, the operator
alert transition + failure isolation + the sync-in-loop no-op, and the
cl_rating_shadow dual-log + its failure isolation.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from betbot.cl_fallback_alert import (
    FallbackEpisode,
    decide_alert,
    maybe_alert_cl_fallback,
)
from betbot.data.models import Fixture, FixtureForm, FormSnapshot, Team
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.main import log_cl_rating_shadow
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import ClRatingShadow
from betbot.storage.repos import upsert_cl_rating_shadow
from betbot.strategy.cl_engine import EuropeanStrategyEngine
from betbot.strategy.engine import StrategyEngine
from betbot.strategy.selfbuilt_elo import (
    SELFBUILT_SD_MAX,
    SELFBUILT_SD_MIN,
    assess_selfbuilt,
    assess_site,
    decide_active,
    elo_sd,
)

TODAY = date.today()
FRESH = TODAY - timedelta(days=2)
STALE_SITE = TODAY - timedelta(days=40)
STALE_SELF = TODAY - timedelta(days=40)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _ff(home: str, away: str, hp: float = 1.5, ap: float = 1.5) -> FixtureForm:
    ht, at = Team(id=1, name=home), Team(id=2, name=away)
    fx = Fixture(id=7, home_team=ht, away_team=at,
                 kickoff=datetime.now(timezone.utc), competition_code="CL")
    return FixtureForm(
        fixture=fx,
        home_form=FormSnapshot(team=ht, weighted_points=hp, raw_points=0, matches_considered=5),
        away_form=FormSnapshot(team=at, weighted_points=ap, raw_points=0, matches_considered=5),
    )


#: Filler clubs that push a 2-club test snapshot's Elo SD into the sanity band
#: [120, 180] so assess_selfbuilt treats it as usable.
_FILLERS = {
    "Filler0 FC": 1800.0, "Filler1 FC": 1740.0, "Filler2 FC": 1680.0,
    "Filler3 FC": 1620.0, "Filler4 FC": 1560.0, "Filler5 FC": 1500.0,
    "Filler6 FC": 1440.0, "Filler7 FC": 1380.0,
}


def _band_snap(home: str, away: str, he: float, ae: float) -> dict[str, float]:
    snap = {home: he, away: ae}
    snap.update(_FILLERS)
    return snap


def _eng(settings, *, site, site_date, selfb=None, selfb_date=None, flag=False):
    s = settings.model_copy(update={"cl_selfbuilt_fallback_enabled": flag})
    return EuropeanStrategyEngine(
        s, snapshot=site, snapshot_date=site_date,
        selfbuilt_snapshot=selfb, selfbuilt_date=selfb_date,
        dc_params=None, name_map={}, resolver=None,
    )


# ----------------------------------------------------------------------
# pure selfbuilt_elo helpers
# ----------------------------------------------------------------------
def test_band_snap_sd_is_in_sanity_band():
    snap = _band_snap("Strong FC", "Weak FC", 1700.0, 1600.0)
    sd = elo_sd(snap)
    assert SELFBUILT_SD_MIN <= sd <= SELFBUILT_SD_MAX


def test_assess_selfbuilt_rejects_sd_out_of_band():
    # Two clubs only -> SD ~50, below the band -> unusable, stale by construction.
    h = assess_selfbuilt({"A": 1700.0, "B": 1600.0}, FRESH, today=TODAY)
    assert h.usable is False
    assert h.stale is True
    assert "sd_out_of_band" in h.reason


def test_assess_selfbuilt_fresh_usable():
    snap = _band_snap("A FC", "B FC", 1700.0, 1600.0)
    h = assess_selfbuilt(snap, FRESH, today=TODAY)
    assert h.usable is True and h.stale is False


def test_decide_active_flag_off_always_site():
    site = assess_site({"A": 1700.0}, STALE_SITE, today=TODAY, stale_after=14)
    selfb = assess_selfbuilt(_band_snap("A FC", "B FC", 1700, 1600), FRESH, today=TODAY)
    d = decide_active(site, selfb, flag_on=False)
    assert d.source == "site_scrape" and d.engaged is False


def test_decide_active_both_stale_serves_fresher_by_from():
    older = _band_snap("A FC", "B FC", 1700, 1600)
    newer = _band_snap("A FC", "B FC", 1500, 1500)
    site = assess_site(older, TODAY - timedelta(days=50), today=TODAY, stale_after=14)
    selfb = assess_selfbuilt(newer, TODAY - timedelta(days=30), today=TODAY)
    d = decide_active(site, selfb, flag_on=True)
    assert d.engaged is True and d.source == "self_built"  # self-built From is newer


# ----------------------------------------------------------------------
# engine: flag OFF regression (byte-identical)
# ----------------------------------------------------------------------
def test_flag_off_stale_site_byte_identical(settings):
    site = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    selfb = _band_snap("Strong FC", "Weak FC", 1600.0, 1700.0)  # disagrees hard
    eng = _eng(settings, site=site, site_date=STALE_SITE,
               selfb=selfb, selfb_date=FRESH, flag=False)
    ref = _eng(settings, site=site, site_date=STALE_SITE, flag=False)
    ff = _ff("Strong FC", "Weak FC")
    got, want = eng.predict(ff), ref.predict(ff)
    assert (got.p_home, got.p_draw, got.p_away) == (want.p_home, want.p_draw, want.p_away)
    assert got.home_score == 1900.0  # priced off the SITE elo, not self-built
    assert eng.fallback_engaged is False and eng.fallback_source == "site_scrape"


# ----------------------------------------------------------------------
# engine: engage / fresh-site / missing / both-stale
# ----------------------------------------------------------------------
def test_flag_on_stale_site_prices_off_selfbuilt(settings):
    site = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    selfb = _band_snap("Strong FC", "Weak FC", 1550.0, 1800.0)  # away now stronger
    eng = _eng(settings, site=site, site_date=STALE_SITE,
               selfb=selfb, selfb_date=FRESH, flag=True)
    p = eng.predict(_ff("Strong FC", "Weak FC"))
    assert eng.fallback_engaged is True and eng.fallback_source == "self_built"
    assert p.home_score == 1550.0 and p.away_score == 1800.0
    assert p.p_away > p.p_home  # self-built ordering (away stronger) is served


def test_flag_on_fresh_site_not_used_but_dual_logged(settings):
    site = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    selfb = _band_snap("Strong FC", "Weak FC", 1550.0, 1800.0)
    eng = _eng(settings, site=site, site_date=FRESH,
               selfb=selfb, selfb_date=FRESH, flag=True)
    p = eng.predict(_ff("Strong FC", "Weak FC"))
    assert eng.fallback_engaged is False and eng.fallback_source == "site_scrape"
    assert p.home_score == 1900.0  # site served
    rows = eng.cl_shadow_rows("Strong FC", "Weak FC")
    sources = {r["source"] for r in rows}
    assert sources == {"site_scrape", "self_built"}
    assert all(r["fallback_engaged"] is False for r in rows)


def test_flag_on_missing_site_engages_selfbuilt_not_naive(settings):
    selfb = _band_snap("Strong FC", "Weak FC", 1900.0, 1500.0)
    eng = _eng(settings, site={}, site_date=None,
               selfb=selfb, selfb_date=FRESH, flag=True)
    ff = _ff("Strong FC", "Weak FC", hp=0.4, ap=2.5)  # form says away-strong
    p = eng.predict(ff)
    naive = StrategyEngine(settings).predict(ff)
    assert eng.fallback_source == "self_built" and eng.fallback_engaged is True
    # Self-built (home 1900 >> away) must dominate the naive form read.
    assert p.p_home > p.p_away
    assert (p.p_home, p.p_draw, p.p_away) != (naive.p_home, naive.p_draw, naive.p_away)


def test_flag_on_both_unusable_falls_to_naive(settings):
    eng = _eng(settings, site={}, site_date=None,
               selfb={}, selfb_date=None, flag=True)
    ff = _ff("Strong FC", "Weak FC", hp=2.5, ap=0.4)
    p = eng.predict(ff)
    naive = StrategyEngine(settings).predict(ff)
    assert eng.fallback_source == "naive" and eng.fallback_engaged is True
    assert (p.p_home, p.p_draw, p.p_away) == (naive.p_home, naive.p_draw, naive.p_away)


def test_flag_on_both_stale_serves_fresher_site_when_selfbuilt_bad(settings):
    # Self-built SD out of band (unusable) + site present-but-stale -> serve site.
    site = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    selfb = {"Strong FC": 1600.0, "Weak FC": 1590.0}  # SD ~5 -> unusable
    eng = _eng(settings, site=site, site_date=STALE_SITE,
               selfb=selfb, selfb_date=FRESH, flag=True)
    p = eng.predict(_ff("Strong FC", "Weak FC"))
    assert eng.fallback_engaged is True and eng.fallback_source == "site_scrape"
    assert p.home_score == 1900.0  # aged site ratings, never the corrupted build


# ----------------------------------------------------------------------
# production name bridge + ambiguity rejection (§5a)
# ----------------------------------------------------------------------
_BRIDGE_ALIASES = {
    "Man City": ["Manchester City", "Manchester City FC"],
    "Paris SG": ["Paris Saint-Germain", "Paris Saint-Germain FC", "PSG"],
    "Inter": ["Internazionale", "Internazionale Milano", "Inter Milan"],
    "Karabakh Agdam": ["Qarabag Agdam", "Qarabag", "Qarabağ"],
    "Bodoe Glimt": ["Bodo/Glimt", "Bodo Glimt"],
}


def test_bridge_alias_resolves_known_cases():
    r = TeamAliasResolver(aliases=_BRIDGE_ALIASES)
    cands = ["Man City", "Paris SG", "Paris FC", "Inter", "Milan",
             "Karabakh Agdam", "Bodoe Glimt"]
    assert r.match("Manchester City", cands) == "Man City"
    assert r.match("Paris Saint-Germain", cands) == "Paris SG"  # NOT Paris FC
    assert r.match("Internazionale Milano", cands) == "Inter"   # NOT Milan
    assert r.match("Qarabag Agdam", cands) == "Karabakh Agdam"
    assert r.match("Bodo/Glimt", cands) == "Bodoe Glimt"


def test_bridge_rejects_ambiguous_multi_hit():
    # No alias; "United" is a token-subset of BOTH distinct clubs (each scores
    # 100) -> ambiguous tie -> refuse to guess, leave unresolved (not mis-mapped).
    r = TeamAliasResolver(aliases={})
    assert r.match("United", ["Manchester United", "Newcastle United"], threshold=70) is None


# ----------------------------------------------------------------------
# operator alert (§7): pure transition + failure isolation + sync no-op
# ----------------------------------------------------------------------
def test_decide_alert_engage_then_steady_then_disengage():
    off = FallbackEpisode(engaged=False)
    on = FallbackEpisode(engaged=True, source="self_built", snapshot_from="2026-09-01")
    eng_msg = decide_alert(off, on, site_age_days=40)
    assert eng_msg is not None and "ENGAGED" in eng_msg[1]
    assert decide_alert(on, on, site_age_days=40) is None  # steady, no spam
    dis = decide_alert(on, off, site_age_days=None)
    assert dis is not None and "DISENGAGED" in dis[1]


def test_decide_alert_new_episode_fires_again():
    ep1 = FallbackEpisode(engaged=True, source="self_built", snapshot_from="2026-09-01")
    ep2 = FallbackEpisode(engaged=True, source="self_built", snapshot_from="2026-09-15")
    assert decide_alert(ep1, ep2, site_age_days=14) is not None


def test_decide_alert_naive_message_flags_naive():
    off = FallbackEpisode(engaged=False)
    naive = FallbackEpisode(engaged=True, source="naive", snapshot_from="2026-09-01")
    key, msg = decide_alert(off, naive, site_age_days=40)
    assert "NAIVE" in msg.upper()


class _FakeEng:
    def __init__(self, engaged, source, snap_from, age):
        self.fallback_engaged = engaged
        self.fallback_source = source
        self._snapshot_date = None if snap_from is None else date.fromisoformat(snap_from)
        self.snapshot_age_days = age


@pytest.mark.asyncio
async def test_maybe_alert_fires_once_then_disengage(settings, tmp_path):
    sent = []

    async def _notify(s, text, **kw):
        sent.append((text, kw.get("dedupe_key")))
        return True

    sp = tmp_path / "state.json"
    eng_on = _FakeEng(True, "self_built", "2026-09-01", 40)
    assert await maybe_alert_cl_fallback(settings, eng_on, state_path=sp, notify=_notify) is True
    # Second identical run: steady episode, no second message.
    assert await maybe_alert_cl_fallback(settings, eng_on, state_path=sp, notify=_notify) is False
    assert len(sent) == 1
    # Fresh scrape returns -> disengage fires once.
    eng_off = _FakeEng(False, "site_scrape", None, None)
    assert await maybe_alert_cl_fallback(settings, eng_off, state_path=sp, notify=_notify) is True
    assert len(sent) == 2 and "DISENGAGED" in sent[1][0]


@pytest.mark.asyncio
async def test_maybe_alert_swallows_notify_failure(settings, tmp_path):
    async def _boom(s, text, **kw):
        raise RuntimeError("telegram down")

    eng_on = _FakeEng(True, "self_built", "2026-09-01", 40)
    # Must not raise; returns False.
    assert await maybe_alert_cl_fallback(
        settings, eng_on, state_path=tmp_path / "s.json", notify=_boom
    ) is False


@pytest.mark.asyncio
async def test_notify_operator_sync_refuses_inside_running_loop(settings):
    # Regression guard for the sync-lambda outage class: the sync wrapper must
    # NOT send from inside a running loop (the alert MUST be awaited instead).
    from betbot import notify as _n
    assert _n.notify_operator_sync(settings, "x") is False


# ----------------------------------------------------------------------
# cl_rating_shadow dual-log + failure isolation (§7a)
# ----------------------------------------------------------------------
@pytest.fixture()
def db(tmp_path):
    init_engine(tmp_path / "shadow.sqlite")
    yield


def _rows(fixture_id: int):
    with session_scope() as s:
        return s.execute(
            select(ClRatingShadow).where(ClRatingShadow.fixture_id == fixture_id)
        ).scalars().all()


def test_log_cl_rating_shadow_writes_both_sources(settings, db):
    site = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    selfb = _band_snap("Strong FC", "Weak FC", 1600.0, 1700.0)
    eng = _eng(settings, site=site, site_date=FRESH,
               selfb=selfb, selfb_date=FRESH, flag=False)
    pred = eng.predict(_ff("Strong FC", "Weak FC"))
    n = log_cl_rating_shadow(eng, "CL", pred, fixture_id=99)
    assert n == 2
    rows = _rows(99)
    assert {r.source for r in rows} == {"site_scrape", "self_built"}
    # Re-run refreshes in place (still 2 rows).
    assert log_cl_rating_shadow(eng, "CL", pred, fixture_id=99) == 2
    assert len(_rows(99)) == 2


def test_log_cl_rating_shadow_skips_non_cl(settings, db):
    site = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    eng = _eng(settings, site=site, site_date=FRESH, flag=False)
    pred = eng.predict(_ff("Strong FC", "Weak FC"))
    assert log_cl_rating_shadow(eng, "PL", pred, fixture_id=5) == 0
    assert _rows(5) == []


def test_log_cl_rating_shadow_swallows_db_error(settings, db, monkeypatch):
    site = {"Strong FC": 1900.0, "Weak FC": 1500.0}
    eng = _eng(settings, site=site, site_date=FRESH, flag=False)
    pred = eng.predict(_ff("Strong FC", "Weak FC"))

    def _boom(**kw):
        raise RuntimeError("db exploded")

    monkeypatch.setattr("betbot.main.upsert_cl_rating_shadow", _boom)
    # Must not raise and must return 0 — logging never breaks scoring.
    assert log_cl_rating_shadow(eng, "CL", pred, fixture_id=7) == 0


def test_upsert_cl_rating_shadow_roundtrip(db):
    rows = [
        {"source": "site_scrape", "elo_home": 1900.0, "elo_away": 1500.0,
         "p_home": 0.6, "p_draw": 0.25, "p_away": 0.15, "favourite": "HOME",
         "snapshot_age_days": 3, "fallback_engaged": False},
        {"source": "self_built", "elo_home": 1600.0, "elo_away": 1700.0,
         "p_home": 0.3, "p_draw": 0.3, "p_away": 0.4, "favourite": "AWAY",
         "snapshot_age_days": 5, "fallback_engaged": False},
    ]
    assert upsert_cl_rating_shadow(
        fixture_id=42, run_date="2026-10-12",
        home_team="Strong FC", away_team="Weak FC", rows=rows,
    ) == 2
    got = {r.source: r for r in _rows(42)}
    assert got["site_scrape"].favourite == "HOME"
    assert got["self_built"].favourite == "AWAY"
