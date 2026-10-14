"""Branch A — CL Polymarket per-match anchor.

Covers the orderbook-midpoint provider (discovery parse, de-vig, every reject
guard), the anchor glue (flag/league gating, single-anchor invariant,
market_p_draw capture), and the alias overlay merge. All network is injected,
so the suite stays hermetic.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.config import Settings
from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.polymarket_cl_odds import (
    PolymarketCLOddsProvider,
    _load_merged_aliases,
    _parse_event,
    reset_shared_cl_pm_provider,
    shared_cl_pm_provider,
)
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import PredictionRow
from betbot.storage.repos import upsert_prediction
from betbot.strategy.cl_pm_anchor import anchor_prediction_cl
from betbot.strategy.engine import Prediction

KO = datetime(2026, 10, 14, 16, 45, tzinfo=timezone.utc)


def _event(home="FK Shakhtar Donetsk", away="AEK", date="2026-10-14T16:45:00Z",
           toks=("TOKH", "TOKD", "TOKA")):
    return {
        "title": f"{home} vs. {away}",
        "markets": [
            {"question": f"Will {home} win on 2026-10-14?", "groupItemTitle": home,
             "clobTokenIds": f'["{toks[0]}","n"]', "gameStartTime": date},
            {"question": f"Will {home} vs. {away} end in a draw?", "groupItemTitle": "Draw",
             "clobTokenIds": f'["{toks[1]}","n"]', "gameStartTime": date},
            {"question": f"Will {away} win on 2026-10-14?", "groupItemTitle": away,
             "clobTokenIds": f'["{toks[2]}","n"]', "gameStartTime": date},
        ],
    }


def _provider(events, mids):
    async def gamma(path, params):
        return events if params.get("offset", 0) == 0 else []

    async def clob(tok):
        return mids.get(tok)

    return PolymarketCLOddsProvider(
        resolver=TeamAliasResolver(aliases=_load_merged_aliases()),
        gamma_get=gamma, clob_midpoint=clob,
    )


class _Eng:
    def model_weight(self):
        return 2.0


def _cl_pred(fixture_id=1, home="Shakhtar Donetsk", away="AEK Athens FC"):
    return Prediction(
        fixture_id=fixture_id, competition_code="CL", home_team=home, away_team=away,
        p_home=0.70, p_draw=0.12, p_away=0.18,
        home_score=0.0, away_score=0.0, draw_score=2.4,
    )


# ---- parsing ----------------------------------------------------------
def test_parse_valid_1x2_event():
    p = _parse_event(_event())
    assert p is not None
    assert (p.tok_home, p.tok_draw, p.tok_away) == ("TOKH", "TOKD", "TOKA")
    assert p.match_date == "2026-10-14"


def test_parse_rejects_derivative_event():
    deriv = {"title": "A vs. B", "markets": [
        {"question": "A to score first vs. B?", "clobTokenIds": '["1"]', "gameStartTime": "x"},
        {"question": "A vs. B: Neither team to score first?", "clobTokenIds": '["2"]', "gameStartTime": "x"},
        {"question": "B to score first vs. A?", "clobTokenIds": '["3"]', "gameStartTime": "x"},
    ]}
    assert _parse_event(deriv) is None


def test_parse_rejects_wrong_market_count():
    ev = _event()
    ev["markets"] = ev["markets"][:2]
    assert _parse_event(ev) is None


# ---- pricing guards ---------------------------------------------------
@pytest.mark.asyncio
async def test_quote_devigs_in_band_triple():
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    q = await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-14")
    assert q is not None
    assert abs(sum(q.probabilities) - 1.0) < 1e-9
    assert abs(q.probabilities[1] - 0.26 / 1.05) < 1e-9
    assert q.source == "pm"


@pytest.mark.asyncio
async def test_quote_rejects_one_sided_leg():
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.999, "TOKA": 0.24})
    assert await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-14") is None


@pytest.mark.asyncio
async def test_quote_rejects_missing_leg():
    prov = _provider([_event()], {"TOKH": 0.55, "TOKA": 0.24})  # draw missing
    assert await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-14") is None


@pytest.mark.asyncio
async def test_quote_rejects_sum_out_of_band():
    prov = _provider([_event()], {"TOKH": 0.40, "TOKD": 0.20, "TOKA": 0.20})  # sum 0.80
    assert await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-14") is None


@pytest.mark.asyncio
async def test_quote_none_when_fixture_not_listed():
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    assert await prov.quote("Real Madrid", "FC Barcelona", "2026-10-14") is None


@pytest.mark.asyncio
async def test_quote_none_on_date_mismatch():
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    assert await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-15") is None


# ---- anchor glue ------------------------------------------------------
@pytest.mark.asyncio
async def test_anchor_noop_when_flag_off():
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    pred = _cl_pred()
    out = await anchor_prediction_cl(pred, league="CL", kickoff=KO,
                                     settings=Settings(), provider=prov, engine=_Eng())
    assert out is pred


@pytest.mark.asyncio
async def test_anchor_noop_for_non_cl_league(monkeypatch):
    monkeypatch.setenv("BETBOT_CL_PM_ANCHOR", "true")
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    pred = _cl_pred()
    out = await anchor_prediction_cl(pred, league="PL", kickoff=KO,
                                     settings=Settings(), provider=prov, engine=_Eng())
    assert out is pred


@pytest.mark.asyncio
async def test_anchor_applies_and_sets_source_and_draw(monkeypatch):
    monkeypatch.setenv("BETBOT_CL_PM_ANCHOR", "true")
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    pred = _cl_pred()
    out = await anchor_prediction_cl(pred, league="CL", kickoff=KO,
                                     settings=Settings(), provider=prov, engine=_Eng())
    assert out.anchor_source == "pm"
    assert out.market_p_draw is not None
    assert abs(out.market_p_draw - 0.26 / 1.05) < 1e-9
    # single-anchor invariant: raw model triple preserved, displayed is a dist.
    assert out.model_probs == (0.70, 0.12, 0.18)
    assert abs(out.p_home + out.p_draw + out.p_away - 1.0) < 1e-9
    # shrunk toward the market draw (0.12 -> higher)
    assert out.p_draw > pred.p_draw


@pytest.mark.asyncio
async def test_anchor_noop_when_already_anchored(monkeypatch):
    monkeypatch.setenv("BETBOT_CL_PM_ANCHOR", "true")
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    import dataclasses
    pre = dataclasses.replace(_cl_pred(), anchor_source="odds")
    out = await anchor_prediction_cl(pre, league="CL", kickoff=KO,
                                     settings=Settings(), provider=prov, engine=_Eng())
    assert out is pre


@pytest.mark.asyncio
async def test_anchor_noop_when_fixture_unlisted(monkeypatch):
    monkeypatch.setenv("BETBOT_CL_PM_ANCHOR", "true")
    prov = _provider([_event()], {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24})
    pred = _cl_pred(home="Paris Saint-Germain FC", away="Arsenal FC")
    out = await anchor_prediction_cl(pred, league="CL", kickoff=KO,
                                     settings=Settings(), provider=prov, engine=_Eng())
    assert out is pred
    assert out.anchor_source is None


# ---- shared instance + persistence -----------------------------------
def test_shared_provider_none_when_flag_off():
    reset_shared_cl_pm_provider()
    assert shared_cl_pm_provider(Settings()) is None


def test_shared_provider_built_when_flag_on(monkeypatch):
    reset_shared_cl_pm_provider()
    monkeypatch.setenv("BETBOT_CL_PM_ANCHOR", "true")
    prov = shared_cl_pm_provider(Settings())
    assert prov is not None
    reset_shared_cl_pm_provider()


def test_alias_overlay_merged():
    al = _load_merged_aliases()
    assert "FK Shakhtar Donetsk" in al
    assert "Qarabag Agdam FK" in al
    # domestic table still present
    assert any("Bayern" in k for k in al)


def test_market_p_draw_persists_through_upsert(tmp_path):
    init_engine(tmp_path / "cl_pm.sqlite")
    import dataclasses
    pred = dataclasses.replace(
        _cl_pred(5), anchor_source="pm",
        model_probs=(0.70, 0.12, 0.18), market_p_draw=0.2476,
    )
    upsert_prediction(pred, kickoff=KO, stage="LEAGUE_STAGE")
    with session_scope() as s:
        row = s.query(PredictionRow).filter_by(fixture_id=5).one()
        assert row.market_p_draw is not None
        assert abs(row.market_p_draw - 0.2476) < 1e-9
        assert row.anchor_source == "pm"


def test_market_p_draw_null_for_unanchored(tmp_path):
    init_engine(tmp_path / "cl_pm2.sqlite")
    upsert_prediction(_cl_pred(6), kickoff=KO)
    with session_scope() as s:
        row = s.query(PredictionRow).filter_by(fixture_id=6).one()
        assert row.market_p_draw is None


# ---- cache TTL (long-lived daemon) -----------------------------------
async def test_prices_refetched_after_ttl_expiry():
    """The shared provider lives for the daemon's lifetime, so a frozen
    midpoint cache would anchor day-2 fixtures to day-1 prices. On TTL expiry
    both the event index and the midpoint cache must be rebuilt."""
    fake_clock = {"t": 1000.0}
    mids = {"TOKH": 0.55, "TOKD": 0.26, "TOKA": 0.24}
    calls = {"n": 0}

    async def gamma(path, params):
        return [_event()] if params.get("offset", 0) == 0 else []

    async def clob(tok):
        calls["n"] += 1
        return mids.get(tok)

    prov = PolymarketCLOddsProvider(
        resolver=TeamAliasResolver(aliases=_load_merged_aliases()),
        gamma_get=gamma, clob_midpoint=clob,
        ttl_seconds=100.0, clock=lambda: fake_clock["t"],
    )
    q1 = await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-14")
    assert q1 is not None and calls["n"] == 3
    await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-14")
    assert calls["n"] == 3  # within TTL: no refetch
    fake_clock["t"] += 200.0  # past TTL
    mids["TOKD"] = 0.28
    q2 = await prov.quote("Shakhtar Donetsk", "AEK Athens FC", "2026-10-14")
    assert calls["n"] == 6
    assert q2.probabilities[1] != q1.probabilities[1]
