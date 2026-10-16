"""Read-only mirror + P&L report of the OPERATOR's own Polymarket trading.

This is a reporting layer, nothing more. It pulls the operator's public on-chain
activity from ``data-api.polymarket.com`` (:mod:`betbot.exchanges.pm_activity`),
links each fill to the Ronaldo prediction for that fixture, classifies the
position relative to the model's call, and computes realised P&L after fees.
It signs nothing and moves no money. Everything it needs is public:

* the operator's PROXY wallet address — a SETTING
  (:data:`betbot.config.Settings.operator_pm_wallet`), never hardcoded, never
  logged to a user-facing channel;
* the public /activity + /positions feeds;
* the local predictions ledger already in the database.

Fee treatment: ``usdcSize`` from the API is the actual cash moved INCLUSIVE of
the 5% * p(1-p) sports taker fee (BUY pays more, SELL receives less, REDEEM is
a fee-free $1/share payout). So realised P&L built from ``usdcSize`` is already
net of fees; see :mod:`betbot.exchanges.pm_activity`.

Flag-gated: the sync job, the monthly report and the ``/trades`` command are
all gated on ``BETBOT_TRADE_LOG`` (default OFF). With the flag off this module
is never imported by a running service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from betbot.exchanges.matcher import TeamAliasResolver
from betbot.exchanges.pm_activity import PmActivity, PmDataClient, PmPosition
from betbot.exchanges.polymarket_cl_odds import _load_merged_aliases
from betbot.logging import get_logger
from betbot.storage import repos

log = get_logger(__name__)

# Backfill origin (operator directive): mirror from this date forward.
DEFAULT_BACKFILL_START = datetime(2026, 8, 1, tzinfo=timezone.utc)

# Classification labels.
WIN = "WIN"              # YES on the team the model picked to win
NO_OPP = "NO_OPP"        # NO on the OPPONENT-win market (the draw-risk safer play)
DRAW_BET = "DRAW_BET"    # YES on the draw market (never the model's pick)
CONTRARIAN = "CONTRARIAN"  # aligned with neither the pick nor its safer play
UNMATCHED = "UNMATCHED"  # no Ronaldo prediction found for the fixture
CASH = "CASH"            # YIELD / rebate / other non-position cash flow

_DIRECTIONAL = {WIN, NO_OPP, DRAW_BET}

# /activity types that represent a position (vs. pure cash movements).
_POSITION_TYPES = {"TRADE", "REDEEM"}


# --------------------------------------------------------------------------
# Market-title parsing
# --------------------------------------------------------------------------

_WIN_RE = re.compile(r"^Will\s+(.+?)\s+win\s+on\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_DRAW_RE = re.compile(r"^Will\s+(.+?)\s+vs\.?\s+(.+?)\s+end in a draw", re.IGNORECASE)
_SLUG_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class MarketRef:
    """What a market title/slug says: kind, the team(s), and the match date."""

    kind: str                 # "WIN" | "DRAW"
    team: str | None          # the subject team for a WIN market
    date: str | None          # ISO yyyy-mm-dd


def parse_market(title: str, slug: str = "", event_slug: str = "") -> MarketRef | None:
    """Parse a Polymarket football market into a :class:`MarketRef`.

    Handles the two layouts the matcher sees:
      * ``"Will <Team> win on <date>?"`` -> kind WIN, subject team, date
      * ``"Will <A> vs. <B> end in a draw?"`` -> kind DRAW (date from the slug)
    Returns ``None`` for anything else (a prop / outright / unknown shape).
    """
    m = _WIN_RE.match(title.strip())
    if m:
        return MarketRef(kind="WIN", team=m.group(1).strip(), date=m.group(2))
    d = _DRAW_RE.match(title.strip())
    if d:
        return MarketRef(kind="DRAW", team=None, date=_date_from_slug(slug, event_slug))
    return None


def _date_from_slug(*slugs: str) -> str | None:
    for s in slugs:
        if not s:
            continue
        m = _SLUG_DATE_RE.search(s)
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------
# Matching one fill to a Ronaldo prediction
# --------------------------------------------------------------------------

_HIGH_CONF_DEFAULT = 0.65
_DRAW_RISK_DEFAULT = 0.20


@dataclass(frozen=True)
class MatchResult:
    fixture_id: int | None
    competition_code: str | None
    pick: str | None            # HOME / DRAW / AWAY (model argmax, as sold)
    market_side: str | None     # HOME / DRAW / AWAY the market is about
    classification: str
    high_conf: bool | None
    draw_risk: bool | None
    anchor_source: str | None
    kickoff: datetime | None


class PredictionMatcher:
    """Resolves a Polymarket fill to its fixture + the model's call.

    Loads the merged domestic + CL alias table once. Caches freshest
    predictions per date to avoid re-querying for every fill.
    """

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._resolver = TeamAliasResolver(aliases=_load_merged_aliases())
        self._by_date: dict[str, list] = {}
        self._high_conf_min = float(
            getattr(settings, "high_conf_alert_min_p", _HIGH_CONF_DEFAULT)
        )
        self._draw_risk_min = float(
            getattr(settings, "draw_risk_threshold", _DRAW_RISK_DEFAULT)
        )

    def _predictions_on(self, date_str: str) -> list:
        if date_str not in self._by_date:
            self._by_date[date_str] = repos.freshest_predictions_on_date(date_str)
        return self._by_date[date_str]

    def _find_fixture(self, ref: MarketRef):
        """Return the PredictionRow for this market, or None."""
        if not ref.date:
            return None
        rows = self._predictions_on(ref.date)
        if ref.kind == "WIN" and ref.team:
            for r in rows:
                if self._resolver.same_team(
                    ref.team, r.home_team
                ) or self._resolver.same_team(ref.team, r.away_team):
                    return r
            return None
        # DRAW market: the title names no single team; match on the pair having
        # both sides present is impossible from the title alone, so we cannot
        # reliably bind a DRAW market to a fixture by title. Fall back to the
        # slug team codes being unavailable -> leave unmatched. (Draw bets are
        # rare in the operator's flow; they surface as DRAW_BET/UNMATCHED.)
        return None

    def match(self, act: PmActivity) -> MatchResult:
        if act.activity_type not in _POSITION_TYPES:
            return MatchResult(None, None, None, None, CASH, None, None, None, None)
        ref = parse_market(act.title, act.slug, act.event_slug)
        # Draw-market YES is a DRAW bet regardless of fixture binding.
        if ref is not None and ref.kind == "DRAW":
            cls = DRAW_BET if _is_yes(act.outcome) else CONTRARIAN
            return MatchResult(
                None, None, "DRAW", "DRAW", cls, None, None, None, None
            )
        if ref is None:
            return MatchResult(None, None, None, None, UNMATCHED, None, None, None, None)
        row = self._find_fixture(ref)
        if row is None:
            return MatchResult(
                None, None, None, None, UNMATCHED, None, None, None, None
            )

        market_side = (
            "HOME" if self._resolver.same_team(ref.team or "", row.home_team) else "AWAY"
        )
        sold_triple, anchor_source, pick = self._sold_pick(row)
        high_conf = max(sold_triple) >= self._high_conf_min
        draw_risk = self._draw_risk(row, anchor_source)
        cls = _classify(market_side, act.outcome, pick)
        return MatchResult(
            fixture_id=row.fixture_id,
            competition_code=row.competition_code,
            pick=pick,
            market_side=market_side,
            classification=cls,
            high_conf=high_conf,
            draw_risk=draw_risk,
            anchor_source=anchor_source,
            kickoff=row.kickoff,
        )

    def _sold_pick(self, row) -> tuple[tuple[float, float, float], str | None, str]:
        """The triple ACTUALLY SOLD to the operator at alert time + its pick.

        Prefers the reveal ledger (what was on screen when the call shipped);
        falls back to the freshest prediction triple when a fixture was never
        revealed.
        """
        triple, anchor, known = repos.sold_call(row.fixture_id)
        if triple is None:
            triple = (row.p_home, row.p_draw, row.p_away)
            anchor = row.anchor_source
        pick = _argmax_pick(triple)
        return triple, anchor, pick

    def _draw_risk(self, row, anchor_source: str | None) -> bool:
        if not anchor_source:  # draw-risk tag is only valid under an anchor
            return False
        mpd = getattr(row, "market_p_draw", None)
        if mpd is None:
            return False
        return float(mpd) >= self._draw_risk_min


def _is_yes(outcome: str) -> bool:
    return (outcome or "").strip().lower() == "yes"


def _aware(dt: datetime | None) -> datetime | None:
    """Coerce a possibly tz-naive datetime (SQLite drops tzinfo) to UTC-aware."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _argmax_pick(triple: tuple[float, float, float]) -> str:
    labels = ("HOME", "DRAW", "AWAY")
    return labels[max(range(3), key=lambda i: triple[i])]


def _classify(market_side: str, outcome: str, pick: str) -> str:
    """Classify a WIN-market fill relative to the model's pick."""
    yes = _is_yes(outcome)
    if pick == "DRAW":
        # Model picked the draw (vanishingly rare). A YES on either team's win
        # is contrarian; a NO is a hedge toward the draw -> treat as NO_OPP-like
        # only if it is a straight team bet, else contrarian.
        return CONTRARIAN
    if yes:
        return WIN if market_side == pick else CONTRARIAN
    # outcome == No -> betting that `market_side` does NOT win.
    if market_side != pick:
        # NO on the OPPONENT-win market: pays if our pick wins OR the match
        # draws — the sanctioned draw-risk "safer play".
        return NO_OPP
    # NO on our own pick's win market -> betting against the call.
    return CONTRARIAN


def exit_timing(sell_ts: int, kickoff: datetime | None) -> str | None:
    """Bucket a pre-settlement SELL by when it happened relative to kickoff."""
    if kickoff is None:
        return None
    ko = kickoff if kickoff.tzinfo else kickoff.replace(tzinfo=timezone.utc)
    delta_min = (sell_ts - ko.timestamp()) / 60.0
    if delta_min < 0:
        return "pre-KO"
    if delta_min < 45:
        return "1st half"
    if delta_min < 60:
        return "at HT"
    if delta_min < 105:
        return "2nd half"
    return "post-FT"


# --------------------------------------------------------------------------
# Sync: pull the feed and persist the mirror
# --------------------------------------------------------------------------


def sync(
    settings,
    *,
    start: datetime | None = None,
    client: PmDataClient | None = None,
) -> dict:
    """Pull /activity since ``start`` and upsert the mirror. No Telegram.

    Returns a small stats dict. Raises :class:`ValueError` only when the wallet
    setting is missing (so a flag-on-but-unconfigured deploy is loud).
    """
    address = (getattr(settings, "operator_pm_wallet", "") or "").strip()
    if not address:
        raise ValueError("BETBOT_OPERATOR_PM_WALLET is not set")
    start = start or DEFAULT_BACKFILL_START
    start_ts = int(start.timestamp())
    matcher = PredictionMatcher(settings)

    own = client is None
    c = client or PmDataClient()
    try:
        acts = c.fetch_activity(address, start=start_ts)
    finally:
        if own:
            c.close()

    inserted = 0
    for act in acts:
        m = matcher.match(act)
        et = (
            exit_timing(act.timestamp, m.kickoff)
            if (act.activity_type == "TRADE" and act.side == "SELL")
            else None
        )
        fields = {
            "transaction_hash": act.transaction_hash,
            "condition_id": act.condition_id,
            "asset": act.asset,
            "activity_type": act.activity_type,
            "side": act.side,
            "size": act.size,
            "price": act.price,
            "usdc_size": act.usdc_size,
            "fee_usdc": act.fee_usdc,
            "outcome": act.outcome,
            "outcome_index": act.outcome_index,
            "title": act.title,
            "slug": act.slug,
            "event_slug": act.event_slug,
            "traded_at": datetime.fromtimestamp(act.timestamp, tz=timezone.utc),
            "competition_code": m.competition_code,
            "matched_fixture_id": m.fixture_id,
            "prediction_pick": m.pick,
            "market_side": m.market_side,
            "classification": m.classification,
            "high_conf": m.high_conf,
            "draw_risk": m.draw_risk,
            "anchor_source": m.anchor_source,
            "exit_timing": et,
        }
        if repos.upsert_operator_trade(fields):
            inserted += 1
    log.info(
        "operator_trade_sync",
        pulled=len(acts),
        inserted=inserted,
        since=start.date().isoformat(),
    )
    return {"pulled": len(acts), "inserted": inserted}


# --------------------------------------------------------------------------
# Position aggregation + summary
# --------------------------------------------------------------------------


@dataclass
class Position:
    """One token position (all fills for a condition_id + asset), with P&L."""

    condition_id: str
    asset: str
    outcome_index: int
    title: str
    competition_code: str | None
    fixture_id: int | None
    classification: str
    pick: str | None
    market_side: str | None
    high_conf: bool | None
    draw_risk: bool | None
    anchor_source: str | None
    bought_usdc: float = 0.0
    sold_usdc: float = 0.0
    redeem_usdc: float = 0.0
    bought_shares: float = 0.0
    sold_shares: float = 0.0
    redeem_shares: float = 0.0
    fee_usdc: float = 0.0
    entry_notional: float = 0.0   # sum(price*size) over BUYs -> avg price
    exit_timings: list[str] = field(default_factory=list)
    first_ts: datetime | None = None

    # filled from /positions + outcomes
    current_value: float = 0.0
    is_open: bool = False
    actual_outcome: str | None = None

    @property
    def realized_pnl(self) -> float:
        return self.sold_usdc + self.redeem_usdc - self.bought_usdc

    @property
    def roi(self) -> float:
        return self.realized_pnl / self.bought_usdc if self.bought_usdc > 0 else 0.0

    @property
    def avg_entry_price(self) -> float:
        return self.entry_notional / self.bought_shares if self.bought_shares > 0 else 0.0

    @property
    def net_shares(self) -> float:
        return self.bought_shares - self.sold_shares - self.redeem_shares

    @property
    def sold_early(self) -> bool:
        """Sold (any shares) before settlement redemption."""
        return self.sold_usdc > 0.0

    @property
    def unrealized_pnl(self) -> float:
        # Mark the still-held shares: current value minus their remaining cost.
        remaining_cost = self.bought_usdc - self.sold_usdc
        return self.current_value - remaining_cost

    @property
    def won(self) -> bool | None:
        """Did the side held resolve in the money? None if undetermined."""
        if self.classification == DRAW_BET:
            if self.actual_outcome is None:
                return self.redeem_usdc > 0 if self._is_settled else None
            return self.actual_outcome == "DRAW"
        if self.actual_outcome is not None and self.pick is not None:
            if self.classification == WIN:
                return self.actual_outcome == self.pick
            if self.classification == NO_OPP:
                # NO on opponent -> wins unless the opponent won.
                opp = "AWAY" if self.pick == "HOME" else "HOME"
                return self.actual_outcome != opp
        # Fallback: redemption only happens on the winning side.
        if self._is_settled:
            return self.redeem_usdc > 0
        return None

    @property
    def _is_settled(self) -> bool:
        return not self.is_open

    @property
    def is_directional(self) -> bool:
        return self.classification in _DIRECTIONAL


def _blank_position(row, title: str) -> Position:
    return Position(
        condition_id=row.condition_id,
        asset=row.asset,
        outcome_index=row.outcome_index,
        title=title,
        competition_code=row.competition_code,
        fixture_id=row.matched_fixture_id,
        classification=row.classification,
        pick=row.prediction_pick,
        market_side=row.market_side,
        high_conf=row.high_conf,
        draw_risk=row.draw_risk,
        anchor_source=row.anchor_source,
    )


def build_positions(
    trades: list,
    positions_feed: list[PmPosition] | None = None,
) -> list[Position]:
    """Aggregate fills into positions, overlay open /positions marks + outcomes."""
    by_key: dict[tuple[str, int], Position] = {}
    for t in trades:
        key = (t.condition_id, t.outcome_index)
        pos = by_key.get(key)
        if pos is None:
            pos = _blank_position(t, t.title)
            by_key[key] = pos
        # Keep the most informative (non-CASH, matched) classification.
        if pos.classification in (CASH, UNMATCHED) and t.classification not in (
            CASH,
            UNMATCHED,
        ):
            pos.classification = t.classification
            pos.fixture_id = t.matched_fixture_id
            pos.pick = t.prediction_pick
            pos.market_side = t.market_side
            pos.high_conf = t.high_conf
            pos.draw_risk = t.draw_risk
            pos.anchor_source = t.anchor_source
            pos.competition_code = t.competition_code
        if not pos.asset and t.asset:
            pos.asset = t.asset
        pos.fee_usdc += t.fee_usdc or 0.0
        tts = _aware(t.traded_at)
        if pos.first_ts is None or (tts is not None and tts < pos.first_ts):
            pos.first_ts = tts
        if t.activity_type == "TRADE" and t.side == "BUY":
            pos.bought_usdc += t.usdc_size
            pos.bought_shares += t.size
            pos.entry_notional += t.price * t.size
        elif t.activity_type == "TRADE" and t.side == "SELL":
            pos.sold_usdc += t.usdc_size
            pos.sold_shares += t.size
            if t.exit_timing:
                pos.exit_timings.append(t.exit_timing)
        elif t.activity_type == "REDEEM":
            pos.redeem_usdc += t.usdc_size
            pos.redeem_shares += t.size

    feed_by_key = {(p.condition_id, p.outcome_index): p for p in (positions_feed or [])}
    for key, pos in by_key.items():
        feed = feed_by_key.get(key)
        # Open = still holding shares, not redeemed, and the live feed marks it
        # with a non-zero current value (an unsettled market). A settled loser
        # has current_value 0 and is NOT open.
        if feed is not None:
            pos.current_value = feed.current_value
            pos.is_open = (
                pos.net_shares > 1e-6
                and pos.redeem_usdc <= 0.0
                and feed.current_value > 0.0
                and not feed.redeemable
            )
        if pos.fixture_id is not None:
            pos.actual_outcome = repos.actual_outcome_for(pos.fixture_id)
    # Exclude pure-cash rows (YIELD etc.) — they have no condition position.
    return [p for p in by_key.values() if p.classification != CASH]


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


@dataclass
class Summary:
    label: str
    n_positions: int = 0
    n_open: int = 0
    realized_pnl: float = 0.0
    total_cost: float = 0.0
    fees: float = 0.0
    open_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    # directional settled subset for the edge check
    n_settled_directional: int = 0
    n_hits: int = 0
    sum_entry_price: float = 0.0
    # splits
    by_type: dict[str, int] = field(default_factory=dict)
    pnl_by_type: dict[str, float] = field(default_factory=dict)
    n_domestic: int = 0
    n_cl: int = 0
    pnl_domestic: float = 0.0
    pnl_cl: float = 0.0
    n_early_sold: int = 0

    @property
    def roi(self) -> float:
        return self.realized_pnl / self.total_cost if self.total_cost > 0 else 0.0

    @property
    def hit_rate(self) -> float | None:
        if self.n_settled_directional == 0:
            return None
        return self.n_hits / self.n_settled_directional

    @property
    def avg_entry_price(self) -> float | None:
        if self.n_settled_directional == 0:
            return None
        return self.sum_entry_price / self.n_settled_directional

    @property
    def edge(self) -> float | None:
        hr, ap = self.hit_rate, self.avg_entry_price
        if hr is None or ap is None:
            return None
        return hr - ap


def summarize(positions: list[Position], *, label: str, since: datetime | None = None) -> Summary:
    s = Summary(label=label)
    for p in positions:
        if since is not None and (p.first_ts is None or p.first_ts < since):
            continue
        s.n_positions += 1
        s.realized_pnl += p.realized_pnl
        s.total_cost += p.bought_usdc
        s.fees += p.fee_usdc
        s.by_type[p.classification] = s.by_type.get(p.classification, 0) + 1
        s.pnl_by_type[p.classification] = (
            s.pnl_by_type.get(p.classification, 0.0) + p.realized_pnl
        )
        if p.competition_code == "CL":
            s.n_cl += 1
            s.pnl_cl += p.realized_pnl
        else:
            s.n_domestic += 1
            s.pnl_domestic += p.realized_pnl
        if p.sold_early:
            s.n_early_sold += 1
        if p.is_open:
            s.n_open += 1
            s.open_exposure += p.current_value
            s.unrealized_pnl += p.unrealized_pnl
        if p.is_directional and not p.is_open:
            won = p.won
            if won is not None:
                s.n_settled_directional += 1
                s.sum_entry_price += p.avg_entry_price
                if won:
                    s.n_hits += 1
    return s


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------
# Rendering (Telegram-safe: everything goes inside a fenced code block so a
# team name with a Markdown metachar can never trigger the legacy 400).
# --------------------------------------------------------------------------

_TYPE_LABEL = {
    WIN: "WIN",
    NO_OPP: "NO-opp",
    DRAW_BET: "draw",
    CONTRARIAN: "contrarian",
    UNMATCHED: "unmatched",
}


def _money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def _pct(x: float) -> str:
    return f"{'+' if x >= 0 else ''}{x * 100:.1f}%"


def _render_summary_block(s: Summary) -> list[str]:
    lines = [f"— {s.label} —"]
    lines.append(f"Positions: {s.n_positions} ({s.n_open} open)")
    lines.append(
        f"Realized P&L: {_money(s.realized_pnl)} (ROI {_pct(s.roi)})  "
        f"fees ${s.fees:,.2f}"
    )
    hr, ap, edge = s.hit_rate, s.avg_entry_price, s.edge
    if hr is not None and ap is not None and edge is not None:
        lines.append(
            f"Hit rate: {hr * 100:.0f}% ({s.n_hits}/{s.n_settled_directional})  "
            f"avg entry {ap:.2f}  edge {edge * 100:+.1f}pts"
        )
    else:
        lines.append("Hit rate: n/a (no settled directional positions yet)")
    tparts = []
    for key in (WIN, NO_OPP, DRAW_BET, CONTRARIAN, UNMATCHED):
        n = s.by_type.get(key, 0)
        if n:
            tparts.append(f"{_TYPE_LABEL[key]} {n}")
    tparts.append(f"early-sold {s.n_early_sold}")
    lines.append("By type: " + "  ".join(tparts))
    lines.append(
        f"Domestic: {s.n_domestic} ({_money(s.pnl_domestic)})   "
        f"CL: {s.n_cl} ({_money(s.pnl_cl)})"
    )
    if s.n_open:
        lines.append(
            f"Open exposure: ${s.open_exposure:,.2f} "
            f"(unreal {_money(s.unrealized_pnl)})"
        )
    return lines


def render_report(
    positions: list[Position],
    *,
    now: datetime | None = None,
    heading: str = "Operator trade log — Polymarket",
) -> str:
    now = now or datetime.now(timezone.utc)
    mstart = month_start(now)
    mtd = summarize(positions, label=now.strftime("Month to date (%b %Y)"), since=mstart)
    alltime = summarize(positions, label="All time (since 2026-08-01)")
    body = [heading, ""]
    body += _render_summary_block(mtd)
    body.append("")
    body += _render_summary_block(alltime)
    note = (
        "\nRead-only mirror of your own on-chain trades. P&L is net of the "
        "5% p(1-p) taker fee. Edge = hit rate - avg entry price paid."
    )
    return "```\n" + "\n".join(body) + "\n```" + note


def render_report_from_db(settings, *, now: datetime | None = None) -> str:
    """Build the /trades body from the stored mirror + a live /positions pull."""
    trades = repos.list_operator_trades(since=DEFAULT_BACKFILL_START)
    positions_feed: list[PmPosition] = []
    address = (getattr(settings, "operator_pm_wallet", "") or "").strip()
    if address:
        try:
            with PmDataClient() as c:
                positions_feed = c.fetch_positions(address)
        except Exception as e:  # noqa: BLE001 — a live-mark failure must not break the report
            log.warning("operator_positions_fetch_failed", error_type=type(e).__name__)
    positions = build_positions(trades, positions_feed)
    return render_report(positions, now=now)
