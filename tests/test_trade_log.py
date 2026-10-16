"""Operator Polymarket trade-log mirror — parsing, matching, P&L, rendering.

Covers :mod:`betbot.exchanges.pm_activity` (fee reconciliation, field decode,
pagination/date-filtering) and :mod:`betbot.trade_log` (title parsing, fill
classification vs the model's call, the REDEEM-has-no-asset grouping fix,
position P&L, the hit-rate-vs-price edge summary, and the Telegram-safe
renderer), plus the additive-table repo helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from betbot.config import Settings
from betbot.exchanges.pm_activity import PmActivity, PmPosition, taker_fee
from betbot.storage.db import init_engine, session_scope
from betbot.storage.models import OperatorTrade, PredictionRow
from betbot.storage.repos import (
    actual_outcome_for,
    freshest_predictions_on_date,
    list_operator_trades,
    record_prediction_outcome,
    record_reveal,
    upsert_operator_trade,
)
from betbot import trade_log as tl

KO = datetime(2026, 10, 14, 19, 0, tzinfo=timezone.utc)
KO_TS = int(KO.timestamp())


@pytest.fixture
def db(tmp_path, monkeypatch):
    init_engine(tmp_path / "tradelog.sqlite")
    from betbot.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        BETBOT_TRADE_LOG="true",
        BETBOT_OPERATOR_PM_WALLET="0xabc",
        BETBOT_HIGH_CONF_ALERT_MIN_P="0.65",
    )


# --------------------------------------------------------------------------
# Fee reconciliation (the "don't guess" requirement)
# --------------------------------------------------------------------------


def test_taker_fee_matches_known_fill():
    # Operator's real BUY: 600.08 Stuttgart shares @0.71, usdcSize 432.2346.
    # usdcSize == size*price + fee, fee == size*0.05*p*(1-p).
    size, price = 600.08, 0.71
    fee = taker_fee(size, price)
    assert fee == pytest.approx(600.08 * 0.05 * 0.71 * 0.29, rel=1e-9)
    assert size * price + fee == pytest.approx(432.2346, abs=0.01)


def test_taker_fee_degenerate_prices_are_zero():
    assert taker_fee(100, 0.0) == 0.0
    assert taker_fee(100, 1.0) == 0.0
    assert taker_fee(0, 0.5) == 0.0


def test_activity_fee_only_on_trades():
    buy = PmActivity.from_api(
        {"type": "TRADE", "side": "BUY", "size": 600.08, "price": 0.71,
         "usdcSize": 432.2346, "outcome": "Yes", "outcomeIndex": 0,
         "conditionId": "c", "asset": "a", "transactionHash": "t",
         "title": "T", "slug": "s", "eventSlug": "e", "timestamp": 1}
    )
    assert buy.fee_usdc == pytest.approx(abs(432.2346 - 600.08 * 0.71))
    assert buy.fee_usdc == pytest.approx(taker_fee(600.08, 0.71), abs=0.01)
    redeem = PmActivity.from_api(
        {"type": "REDEEM", "side": "", "size": 600.08, "price": 0.0,
         "usdcSize": 600.08, "outcome": "Yes", "outcomeIndex": 0,
         "conditionId": "c", "asset": "", "transactionHash": "t2",
         "title": "T", "slug": "s", "eventSlug": "e", "timestamp": 2}
    )
    assert redeem.fee_usdc == 0.0


# --------------------------------------------------------------------------
# Market-title parsing
# --------------------------------------------------------------------------


def test_parse_win_market():
    ref = tl.parse_market("Will VfB Stuttgart win on 2026-10-14?")
    assert ref and ref.kind == "WIN"
    assert ref.team == "VfB Stuttgart"
    assert ref.date == "2026-10-14"


def test_parse_draw_market_date_from_slug():
    ref = tl.parse_market(
        "Will Aston Villa FC vs. Fenerbahce SK end in a draw?",
        slug="ucl-ast-fen-2026-10-14-draw",
        event_slug="ucl-ast-fen-2026-10-14",
    )
    assert ref and ref.kind == "DRAW"
    assert ref.date == "2026-10-14"


def test_parse_non_market_returns_none():
    assert tl.parse_market("Exact Score: Atletico 2 - 0 Osasuna?") is None
    assert tl.parse_market("Who wins the Champions League?") is None


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_classify_yes_on_pick_is_win():
    assert tl._classify("AWAY", "Yes", "AWAY") == tl.WIN


def test_classify_yes_on_non_pick_is_contrarian():
    assert tl._classify("HOME", "Yes", "AWAY") == tl.CONTRARIAN


def test_classify_no_on_opponent_is_no_opp():
    # pick HOME; a NO on the AWAY-win market is the safer draw-risk play.
    assert tl._classify("AWAY", "No", "HOME") == tl.NO_OPP


def test_classify_no_on_own_pick_is_contrarian():
    assert tl._classify("HOME", "No", "HOME") == tl.CONTRARIAN


def test_exit_timing_buckets():
    assert tl.exit_timing(KO_TS - 600, KO) == "pre-KO"
    assert tl.exit_timing(KO_TS + 10 * 60, KO) == "1st half"
    assert tl.exit_timing(KO_TS + 50 * 60, KO) == "at HT"
    assert tl.exit_timing(KO_TS + 80 * 60, KO) == "2nd half"
    assert tl.exit_timing(KO_TS + 200 * 60, KO) == "post-FT"
    assert tl.exit_timing(KO_TS, None) is None


# --------------------------------------------------------------------------
# Matching against seeded predictions
# --------------------------------------------------------------------------


def _seed_prediction(fid, home, away, p, *, code="CL", anchor=None, mpd=None):
    with session_scope() as s:
        s.add(PredictionRow(
            fixture_id=fid, competition_code=code, kickoff=KO,
            run_date=KO.date().isoformat(), home_team=home, away_team=away,
            p_home=p[0], p_draw=p[1], p_away=p[2],
            home_score=0.0, away_score=0.0, draw_score=0.0,
            anchor_source=anchor, market_p_draw=mpd,
        ))


def _activity(title, outcome, *, side="BUY", typ="TRADE", size=100.0, price=0.7,
              usdc=None, cond="c1", asset="a1", oidx=0, ts=KO_TS, txh="tx1"):
    usdc = usdc if usdc is not None else size * price
    return PmActivity.from_api({
        "type": typ, "side": side, "size": size, "price": price,
        "usdcSize": usdc, "outcome": outcome, "outcomeIndex": oidx,
        "conditionId": cond, "asset": asset, "transactionHash": txh,
        "title": title, "slug": "sl", "eventSlug": "ev", "timestamp": ts,
    })


def test_match_win_bet_on_away_pick(db, settings):
    # LASK vs Liverpool, model picks AWAY (0.76); operator buys YES Liverpool.
    _seed_prediction(575350, "LASK Linz", "Liverpool FC", (0.12, 0.12, 0.76))
    m = tl.PredictionMatcher(settings)
    res = m.match(_activity("Will Liverpool FC win on 2026-10-14?", "Yes"))
    assert res.fixture_id == 575350
    assert res.market_side == "AWAY"
    assert res.pick == "AWAY"
    assert res.classification == tl.WIN
    assert res.high_conf is True
    assert res.competition_code == "CL"


def test_match_unmatched_when_no_prediction(db, settings):
    m = tl.PredictionMatcher(settings)
    res = m.match(_activity("Will Celta de Vigo win on 2026-10-14?", "Yes"))
    assert res.classification == tl.UNMATCHED
    assert res.fixture_id is None


def test_match_high_conf_uses_sold_reveal_triple(db, settings):
    # Prediction stored high (0.76) but the SOLD reveal was sub-0.65 -> not hi.
    _seed_prediction(1, "A", "B", (0.76, 0.12, 0.12))
    record_reveal(1, 1, True, 0.60, 0.22, 0.18, anchor_source=None)
    m = tl.PredictionMatcher(settings)
    res = m.match(_activity("Will A win on 2026-10-14?", "Yes"))
    assert res.high_conf is False


def test_match_draw_risk_requires_anchor(db, settings):
    # Seed BOTH before building the matcher (it caches predictions per date).
    _seed_prediction(2, "A", "B", (0.70, 0.12, 0.18), anchor="odds", mpd=0.28)
    _seed_prediction(3, "C", "D", (0.70, 0.12, 0.18), anchor=None, mpd=0.28)
    m = tl.PredictionMatcher(settings)
    res = m.match(_activity("Will A win on 2026-10-14?", "Yes"))
    assert res.draw_risk is True
    # Same market_p_draw but unanchored -> no draw-risk tag.
    res2 = m.match(_activity("Will C win on 2026-10-14?", "Yes", cond="c3"))
    assert res2.draw_risk is False


def test_match_cash_type(db, settings):
    m = tl.PredictionMatcher(settings)
    res = m.match(_activity("whatever", "", typ="YIELD", side=""))
    assert res.classification == tl.CASH


# --------------------------------------------------------------------------
# Repo idempotency + helpers
# --------------------------------------------------------------------------


def test_upsert_is_idempotent_and_refreshes_derived(db):
    fields = dict(
        transaction_hash="tx", asset="a", activity_type="TRADE", side="BUY",
        condition_id="c", size=10.0, price=0.5, usdc_size=5.0, fee_usdc=0.1,
        outcome="Yes", outcome_index=0, title="T", slug="s", event_slug="e",
        traded_at=KO, classification=tl.UNMATCHED,
    )
    assert upsert_operator_trade(dict(fields)) is True
    # Re-run: no new row, derived fields refresh (now matched).
    fields["classification"] = tl.WIN
    fields["matched_fixture_id"] = 42
    assert upsert_operator_trade(dict(fields)) is False
    rows = list_operator_trades()
    assert len(rows) == 1
    assert rows[0].classification == tl.WIN
    assert rows[0].matched_fixture_id == 42
    # Raw facts preserved.
    assert rows[0].usdc_size == 5.0


def test_freshest_prediction_wins(db):
    with session_scope() as s:
        s.add(PredictionRow(
            fixture_id=9, competition_code="PL", kickoff=KO, run_date="2026-10-10",
            home_team="A", away_team="B", p_home=0.4, p_draw=0.3, p_away=0.3,
            home_score=0, away_score=0, draw_score=0))
        s.add(PredictionRow(
            fixture_id=9, competition_code="PL", kickoff=KO, run_date="2026-10-14",
            home_team="A", away_team="B", p_home=0.6, p_draw=0.2, p_away=0.2,
            home_score=0, away_score=0, draw_score=0))
    rows = freshest_predictions_on_date("2026-10-14")
    assert len(rows) == 1
    assert rows[0].p_home == 0.6  # the fresher run_date


# --------------------------------------------------------------------------
# Position aggregation + the REDEEM-has-no-asset grouping fix
# --------------------------------------------------------------------------


def test_redeem_groups_with_buy_despite_empty_asset(db, settings):
    # The live-data bug: a REDEEM row carries asset="" but the same
    # (conditionId, outcomeIndex). It MUST merge with its BUY, not split.
    _seed_prediction(10, "A", "B", (0.8, 0.12, 0.08), code="BL1")
    buy = _activity("Will A win on 2026-10-14?", "Yes", side="BUY",
                    size=435.65, price=0.83, usdc=364.66, cond="cx",
                    asset="951381", oidx=0, txh="b1")
    redeem = _activity("Will A win on 2026-10-14?", "Yes", side="",
                       typ="REDEEM", size=435.65, price=0.0, usdc=435.65,
                       cond="cx", asset="", oidx=0, txh="r1")
    m = tl.PredictionMatcher(settings)
    trades = []
    for act in (buy, redeem):
        res = m.match(act)
        trades.append(_store(act, res))
    positions = tl.build_positions(trades)
    assert len(positions) == 1
    p = positions[0]
    assert p.bought_usdc == pytest.approx(364.66)
    assert p.redeem_usdc == pytest.approx(435.65)
    assert p.realized_pnl == pytest.approx(70.99)
    assert p.classification == tl.WIN


def _store(act: PmActivity, res) -> OperatorTrade:
    """Persist one fill the way sync() does and return the ORM row."""
    et = (tl.exit_timing(act.timestamp, res.kickoff)
          if act.activity_type == "TRADE" and act.side == "SELL" else None)
    upsert_operator_trade(dict(
        transaction_hash=act.transaction_hash, condition_id=act.condition_id,
        asset=act.asset, activity_type=act.activity_type, side=act.side,
        size=act.size, price=act.price, usdc_size=act.usdc_size,
        fee_usdc=act.fee_usdc, outcome=act.outcome, outcome_index=act.outcome_index,
        title=act.title, slug=act.slug, event_slug=act.event_slug,
        traded_at=datetime.fromtimestamp(act.timestamp, tz=timezone.utc),
        competition_code=res.competition_code, matched_fixture_id=res.fixture_id,
        prediction_pick=res.pick, market_side=res.market_side,
        classification=res.classification, high_conf=res.high_conf,
        draw_risk=res.draw_risk, anchor_source=res.anchor_source, exit_timing=et,
    ))
    return [r for r in list_operator_trades()
            if r.transaction_hash == act.transaction_hash][0]


def test_open_position_marked_unrealized(db, settings):
    _seed_prediction(11, "A", "B", (0.7, 0.2, 0.1), code="PL")
    buy = _activity("Will A win on 2026-10-14?", "Yes", side="BUY",
                    size=100, price=0.5, usdc=50.0, cond="co", asset="tok", oidx=0)
    m = tl.PredictionMatcher(settings)
    row = _store(buy, m.match(buy))
    feed = [PmPosition.from_api({
        "conditionId": "co", "asset": "tok", "size": 100, "avgPrice": 0.5,
        "currentValue": 65.0, "curPrice": 0.65, "redeemable": False,
        "outcomeIndex": 0, "title": "A", "endDate": "2026-10-14",
    })]
    positions = tl.build_positions([row], feed)
    p = positions[0]
    assert p.is_open is True
    assert p.current_value == 65.0
    assert p.unrealized_pnl == pytest.approx(15.0)  # 65 - (50 - 0)


# --------------------------------------------------------------------------
# Summary edge check + rendering
# --------------------------------------------------------------------------


def _won_position(cls, pick, actual, entry, cost):
    p = tl.Position(
        condition_id="c", asset="a", outcome_index=0, title="t",
        competition_code="PL", fixture_id=1, classification=cls, pick=pick,
        market_side=pick, high_conf=True, draw_risk=False, anchor_source=None,
    )
    p.bought_usdc = cost
    p.bought_shares = cost / entry
    p.entry_notional = entry * p.bought_shares
    p.redeem_usdc = cost / entry if actual == pick else 0.0
    p.actual_outcome = actual
    p.first_ts = KO
    return p


def test_summary_hit_rate_and_edge():
    positions = [
        _won_position(tl.WIN, "HOME", "HOME", 0.60, 60.0),   # hit
        _won_position(tl.WIN, "AWAY", "HOME", 0.50, 50.0),   # miss
        _won_position(tl.WIN, "HOME", "HOME", 0.70, 70.0),   # hit
    ]
    s = tl.summarize(positions, label="All")
    assert s.n_positions == 3
    assert s.n_settled_directional == 3
    assert s.hit_rate == pytest.approx(2 / 3)
    assert s.avg_entry_price == pytest.approx((0.60 + 0.50 + 0.70) / 3)
    assert s.edge == pytest.approx(2 / 3 - 0.60)
    assert s.by_type[tl.WIN] == 3


def test_summary_month_to_date_filter():
    old = _won_position(tl.WIN, "HOME", "HOME", 0.6, 60.0)
    old.first_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = _won_position(tl.WIN, "HOME", "HOME", 0.6, 60.0)
    new.first_ts = datetime(2026, 10, 15, tzinfo=timezone.utc)
    s = tl.summarize([old, new], label="MTD",
                     since=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert s.n_positions == 1


def test_no_opp_hits_on_draw():
    # pick HOME, NO on AWAY: wins if HOME wins OR DRAW.
    p = _won_position(tl.NO_OPP, "HOME", "DRAW", 0.3, 30.0)
    assert p.won is True
    p2 = _won_position(tl.NO_OPP, "HOME", "AWAY", 0.3, 30.0)
    assert p2.won is False


def test_render_is_fenced_and_plain_safe():
    positions = [_won_position(tl.WIN, "HOME", "HOME", 0.6, 60.0)]
    body = tl.render_report(positions, now=datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert body.startswith("```")
    assert body.count("```") == 2  # exactly one fenced block
    assert "Operator trade log" in body
    assert "edge =" in body


# --------------------------------------------------------------------------
# sync() end-to-end with a fake client
# --------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, acts, positions=None):
        self._acts = acts
        self._pos = positions or []

    def fetch_activity(self, address, *, start=None, end=None):
        return [a for a in self._acts if start is None or a.timestamp >= start]

    def fetch_positions(self, address):
        return self._pos

    def close(self):
        pass


def test_sync_stores_and_matches(db, settings):
    _seed_prediction(575358, "SK Slovan Bratislava", "VfB Stuttgart",
                     (0.14, 0.12, 0.74))
    record_prediction_outcome(
        fixture_id=575358, competition_code="CL", p_home=0.14, p_draw=0.12,
        p_away=0.74, actual_outcome="AWAY", home_goals=0, away_goals=2,
        settled_at=KO, kickoff=KO, anchor_source=None)
    acts = [
        _activity("Will VfB Stuttgart win on 2026-10-14?", "Yes", side="BUY",
                  size=600.08, price=0.71, usdc=432.23, cond="cstu",
                  asset="tokstu", txh="buy1"),
        _activity("Will VfB Stuttgart win on 2026-10-14?", "Yes", side="",
                  typ="REDEEM", size=600.08, price=0.0, usdc=600.08,
                  cond="cstu", asset="", txh="red1"),
    ]
    stats = tl.sync(settings, client=_FakeClient(acts))
    assert stats["pulled"] == 2
    assert stats["inserted"] == 2
    trades = list_operator_trades()
    positions = tl.build_positions(trades)
    assert len(positions) == 1
    p = positions[0]
    assert p.classification == tl.WIN
    assert p.won is True
    assert p.realized_pnl == pytest.approx(600.08 - 432.23)
    assert actual_outcome_for(575358) == "AWAY"
    # Re-sync is idempotent.
    stats2 = tl.sync(settings, client=_FakeClient(acts))
    assert stats2["inserted"] == 0
    assert len(list_operator_trades()) == 2


def test_sync_requires_wallet(db):
    s = Settings(_env_file=None, BETBOT_TRADE_LOG="true")
    with pytest.raises(ValueError):
        tl.sync(s, client=_FakeClient([]))


# --------------------------------------------------------------------------
# Fable review fixes
# --------------------------------------------------------------------------


def test_fee_zero_for_maker_fill():
    # Maker fill: usdcSize == size*price exactly -> fee 0.
    maker = PmActivity.from_api({
        "type": "TRADE", "side": "BUY", "size": 100.0, "price": 0.50,
        "usdcSize": 50.0, "outcome": "Yes", "outcomeIndex": 0,
        "conditionId": "c", "asset": "a", "transactionHash": "t",
        "title": "T", "slug": "s", "eventSlug": "e", "timestamp": 1})
    assert maker.fee_usdc == pytest.approx(0.0)
    # Taker fill: fee is the cash gap.
    taker = PmActivity.from_api({
        "type": "TRADE", "side": "BUY", "size": 100.0, "price": 0.50,
        "usdcSize": 51.25, "outcome": "Yes", "outcomeIndex": 0,
        "conditionId": "c", "asset": "a", "transactionHash": "t2",
        "title": "T", "slug": "s", "eventSlug": "e", "timestamp": 2})
    assert taker.fee_usdc == pytest.approx(1.25)


def test_won_none_when_unresolved():
    # Directional WIN, no outcome row, no redeem, no feed -> UNRESOLVED (None),
    # NOT a miss.
    p = tl.Position(
        condition_id="c", asset="a", outcome_index=0, title="t",
        competition_code="PL", fixture_id=1, classification=tl.WIN, pick="HOME",
        market_side="HOME", high_conf=True, draw_risk=False, anchor_source=None)
    p.bought_usdc = 50.0
    p.bought_shares = 100.0
    p.entry_notional = 50.0
    p.first_ts = KO
    assert p.won is None
    s = tl.summarize([p], label="x")
    assert s.n_settled_directional == 0  # excluded, not counted as a miss
    assert s.hit_rate is None


def test_pending_payout_excluded_from_pnl(db, settings):
    _seed_prediction(12, "A", "B", (0.8, 0.12, 0.08), code="PL")
    buy = _activity("Will A win on 2026-10-14?", "Yes", side="BUY",
                    size=100, price=0.6, usdc=60.0, cond="cp", asset="tp", oidx=0)
    m = tl.PredictionMatcher(settings)
    row = _store(buy, m.match(buy))
    # Settled winner, redeemable, value > 0, NOT yet redeemed.
    feed = [PmPosition.from_api({
        "conditionId": "cp", "asset": "tp", "size": 100, "avgPrice": 0.6,
        "currentValue": 100.0, "curPrice": 1.0, "redeemable": True,
        "outcomeIndex": 0, "title": "A", "endDate": "2026-10-14"})]
    positions = tl.build_positions([row], feed)
    p = positions[0]
    assert p.is_pending is True
    assert p.is_open is False
    s = tl.summarize(positions, label="x")
    assert s.n_pending == 1
    assert s.pending_claimable == pytest.approx(100.0)
    assert s.pending_cost == pytest.approx(60.0)
    # NOT counted as a realised loss.
    assert s.realized_pnl == pytest.approx(0.0)
    assert s.total_cost == pytest.approx(0.0)


def test_monthly_report_uses_previous_month():
    start = datetime(2026, 9, 10, tzinfo=timezone.utc)
    sep = _won_position(tl.WIN, "HOME", "HOME", 0.6, 60.0)
    sep.first_ts = start
    oct1 = _won_position(tl.WIN, "HOME", "HOME", 0.6, 60.0)
    oct1.first_ts = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)  # 08:00 on the 1st
    body = tl.render_monthly_report([sep, oct1], now=now)
    assert "September 2026" in body
    assert "monthly report" in body
    # The Sept position is in-window; the Oct-1 one is not.
    s = tl.summarize([sep, oct1], label="Sep",
                     since=tl.prev_month_bounds(now)[0],
                     until=tl.prev_month_bounds(now)[1])
    assert s.n_positions == 1


def test_prev_month_bounds():
    start, end = tl.prev_month_bounds(datetime(2026, 1, 15, tzinfo=timezone.utc))
    assert start == datetime(2025, 12, 1, tzinfo=timezone.utc)
    assert end == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_idempotency_key_includes_ts_and_size(db):
    base = dict(
        transaction_hash="tx", asset="a", activity_type="TRADE", side="BUY",
        condition_id="c", price=0.5, usdc_size=5.0, fee_usdc=0.0,
        outcome="Yes", outcome_index=0, title="T", slug="s", event_slug="e",
        classification=tl.WIN)
    # Two partial fills in one tx: same tx+asset+type+side but different size/ts.
    f1 = dict(base, size=10.0, traded_at=KO)
    f2 = dict(base, size=20.0, traded_at=datetime(2026, 10, 14, 19, 5, tzinfo=timezone.utc))
    assert upsert_operator_trade(f1) is True
    assert upsert_operator_trade(f2) is True
    assert len(list_operator_trades()) == 2


def test_live_marks_unavailable_note():
    positions = [_won_position(tl.WIN, "HOME", "HOME", 0.6, 60.0)]
    body = tl.render_report(positions, now=KO, live_marks_ok=False)
    assert "live marks unavailable" in body
    ok_body = tl.render_report(positions, now=KO, live_marks_ok=True)
    assert "live marks unavailable" not in ok_body
