"""Quantify the double-anchoring defect on the REAL held-out season.

The defect (fixed in this branch): with ``BETBOT_ODDS_ANCHOR=true`` the
prediction reaching ``decide_with_market`` had already been anchored toward a
bookmaker line, and ``decide_with_market`` anchored it a SECOND time toward the
exchange price. This script measures what that did to real bet/no-bet calls.

Everything here is FREE and EARLY-WEEK. No closing prices are read anywhere.

  model     the real walk-forward club ensemble (Glicko + Dixon-Coles), fit on
            < --test-from and replayed forward, identical to the gate.
  BOOK      Pinnacle's early-week 1X2 line from data/club_odds.csv — the rows
            the live odds anchor actually anchors to.
  EXCHANGE  Bet365's early-week 1X2 line, a second REAL and independent venue
            on the same fixtures, standing in for the Polymarket price. There
            is no free archive of historical Polymarket club-match prices and
            no big-5 1X2 market was open when this was written, so a second
            real venue is the closest honest substitute. It stands in for the
            PRICE only; the arithmetic under test does not care which venue it
            is. A --delta sweep then displaces the exchange price to show how
            much venue disagreement the defect needs to place real money.

Honesty: anchoring shrinks the model toward the price. It moves us toward
market-level accuracy and cannot exceed it. Nothing here is an edge.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import betbot.data.odds as odds_mod  # noqa: E402
from backtest_club import (  # noqa: E402
    Form, Glicko, _fixture_form, _load, _load_odds, _lookup_odds, _outcome,
)
from betbot.config import get_settings  # noqa: E402
from betbot.data.odds import FootballDataCoUkProvider, _build_resolver  # noqa: E402
from betbot.exchanges.matcher import normalize  # noqa: E402
from betbot.strategy import dixon_coles as dc  # noqa: E402
from betbot.strategy.engine import StrategyEngine  # noqa: E402
from betbot.strategy.ensemble import anchor_to_market, anchor_triple, log_pool  # noqa: E402
from betbot.strategy.glicko import match_probabilities  # noqa: E402

OUTS = ("HOME", "DRAW", "AWAY")
LEAGUES = ("PL", "PD", "SA", "BL1", "FL1")


def _de_vig(triple: tuple[float, float, float]) -> tuple[float, float, float]:
    inv = [1.0 / p for p in triple]
    z = sum(inv)
    return (inv[0] / z, inv[1] / z, inv[2] / z)


def second_venue(season: str) -> dict:
    """Bet365 EARLY-WEEK prices for the held-out season. Never the C-columns."""
    saved = odds_mod.PREMATCH_BOOKS
    odds_mod.PREMATCH_BOOKS = (("B365H", "B365D", "B365A"),)
    try:
        prov = FootballDataCoUkProvider(_build_resolver(get_settings()))
        idx: dict = defaultdict(list)
        for lg in LEAGUES:
            for r in prov.fetch_season(season, lg, kind="prematch"):
                idx[(r.league, r.home, r.away)].append(
                    (r.match_date,
                     _de_vig((r.price_home, r.price_draw, r.price_away)))
                )
        return idx
    finally:
        odds_mod.PREMATCH_BOOKS = saved


def _nearest(idx, key, d: date, slack: int = 3):
    best, gap = None, None
    for rd, tri in idx.get(key, ()):
        g = abs((rd - d).days)
        if g <= slack and (gap is None or g < gap):
            best, gap = tri, g
    return best


def walk_forward(csv_path: Path, cutoff: date, iterations: int, book_idx, exch_idx):
    """Replay the gate's walk-forward, yielding one record per outcome leg."""
    s = get_settings()
    rows = _load(csv_path)
    train = [r for r in rows if r["date"] < cutoff]
    test = [r for r in rows if r["date"] >= cutoff]
    dcp = dc.fit(
        [dc.DCMatch(date=r["date"], home=normalize(r["home"]),
                    away=normalize(r["away"]), home_goals=r["hs"],
                    away_goals=r["as"], neutral=False, friendly=False)
         for r in train],
        priors={}, iterations=iterations,
    )
    glk, form = Glicko(s), Form()
    by_date: dict = defaultdict(list)
    for r in train:
        by_date[r["date"]].append(r)
    for d in sorted(by_date):
        glk.update_day(
            [(r["home"], r["away"], _outcome(r["hs"], r["as"])) for r in by_date[d]],
            d.isoformat(),
        )
        for r in by_date[d]:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))

    naive = StrategyEngine(s)
    weights = [(s.club_weight_glicko, "glicko"), (s.club_weight_dc, "dc"),
               (s.club_weight_form, "form")]
    test_by_date: dict = defaultdict(list)
    for r in test:
        test_by_date[r["date"]].append(r)

    out = []
    for d in sorted(test_by_date):
        day = test_by_date[d]
        for r in day:
            h, a = r["home"], r["away"]
            fp = naive.predict(_fixture_form(h, a, form))
            comp = {
                "glicko": match_probabilities(
                    glk.get(h), glk.get(a), home_field_mu=s.glicko_club_home_mu,
                    draw_rho=s.glicko_club_draw_rho),
                "dc": dc.match_probabilities(dcp, normalize(h), normalize(a),
                                             home_field=True),
                "form": (fp.p_home, fp.p_draw, fp.p_away),
            }
            model = log_pool([(w, comp[n]) for w, n in weights if w > 0])
            key = (r["league"], normalize(h), normalize(a))
            book, _close = _lookup_odds(book_idx, r["league"], normalize(h),
                                        normalize(a), d)
            exch = _nearest(exch_idx, key, d)
            if not book or not exch:
                continue
            for i, o in enumerate(OUTS):
                out.append((d, r["league"], h, a, o,
                            _outcome(r["hs"], r["as"]), model, book, exch, i))
        glk.update_day(
            [(r["home"], r["away"], _outcome(r["hs"], r["as"])) for r in day],
            d.isoformat(),
        )
        for r in day:
            form.push(r["home"], r["away"], _outcome(r["hs"], r["as"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "data/club_results.csv")
    ap.add_argument("--odds", type=Path, default=ROOT / "data/club_odds.csv")
    ap.add_argument("--test-from", default="2025-07-01")
    ap.add_argument("--season", default="2526", help="football-data.co.uk season code")
    ap.add_argument("--dc-iterations", type=int, default=200)
    ap.add_argument("--delta", type=float, nargs="*",
                    default=[0.02, 0.05, 0.10, 0.15],
                    help="displace the exchange price this far BELOW the book")
    args = ap.parse_args()

    s = get_settings()
    w_model = s.club_weight_glicko + s.club_weight_dc + s.club_weight_form
    w_book, w_mkt, thresh = (s.odds_anchor_market_weight, s.club_weight_market,
                             s.edge_threshold)
    f_book = w_book / (w_model + w_book)
    f_keep = w_model / (w_model + w_mkt)
    print(f"w_model={w_model} w_book={w_book} w_exchange={w_mkt} "
          f"edge_threshold={thresh}")
    print(f"LEAKAGE ALGEBRA (logit space): the odds anchor moves the model "
          f"{f_book:.4f} of the way to the BOOK, then the exchange anchor "
          f"retains {f_keep:.4f} of that displacement. So {f_book * f_keep:.4f} "
          f"of the book-vs-exchange logit gap used to land straight in the "
          f"decided edge, with no model opinion behind it.")

    legs = walk_forward(args.csv, date.fromisoformat(args.test_from),
                        args.dc_iterations, _load_odds(args.odds),
                        second_venue(args.season))
    print(f"\nlegs {len(legs)}  fixtures {len(legs) // 3}")
    gaps = [abs(b[i] - e[i]) for *_, b, e, i in legs]
    gaps_sorted = sorted(gaps)
    print(f"REAL venue disagreement |book - exchange|: mean "
          f"{sum(gaps) / len(gaps):.4f}  p90 {gaps_sorted[int(0.9 * len(gaps))]:.4f}"
          f"  max {max(gaps):.4f}")

    def decide(model, book, i, ex):
        anch = anchor_triple(model, book, w_model, w_book)
        return (anchor_to_market(anch[i], ex, w_model, w_mkt) - ex,   # OLD, twice
                anchor_to_market(model[i], ex, w_model, w_mkt) - ex,  # NEW, once
                anch[i])

    def report(label, price_of):
        old_bets = new_bets = manu = supp = 0
        recs_m, recs_s = [], []
        for d, lg, h, a, o, res, model, book, exch, i in legs:
            ex = price_of(book, exch, i)
            if not 0.02 < ex < 0.98:
                continue
            e_old, e_new, anch = decide(model, book, i, ex)
            old_bets += e_old >= thresh
            new_bets += e_new >= thresh
            rec = (d, lg, h, a, o, res, model[i], book[i], ex, anch, e_old, e_new)
            if e_old >= thresh > e_new:
                manu += 1
                recs_m.append((e_old - e_new, rec))
            elif e_new >= thresh > e_old:
                supp += 1
                recs_s.append((e_new - e_old, rec))
        print(f"\n=== {label} ===")
        print(f"  BETs placed  OLD (anchored twice) {old_bets:5d}   "
              f"NEW (anchored once) {new_bets:5d}")
        print(f"  decisions changed by the fix: {manu + supp} "
              f"({(manu + supp) / len(legs):.1%} of legs) — "
              f"{manu} MANUFACTURED by the venue gap, {supp} over-shrunk away")
        for title, recs in (("MANUFACTURED (old bet, fix does not)", recs_m),
                            ("SUPPRESSED (old passed, fix bets)", recs_s)):
            for _shift, rec in sorted(recs, key=lambda x: -x[0])[:3]:
                d, lg, h, a, o, res, m, b, ex, anch, e_old, e_new = rec
                print(f"\n  [{title}]")
                print(f"  {d} {lg}  {h} (HOME) vs {a} (AWAY)  leg {o}  actual {res}")
                print(f"     model {m:.4f} | BOOK {b:.4f} | EXCHANGE {ex:.4f}"
                      f"  (venues differ {b - ex:+.4f})")
                print(f"     OLD  model->book {anch:.4f} ->then-> exchange "
                      f"{ex + e_old:.4f}   edge {e_old:+.4f}  "
                      f"{'BET' if e_old >= thresh else 'NO BET'}")
                print(f"     NEW  model ---------------------> exchange "
                      f"{ex + e_new:.4f}   edge {e_new:+.4f}  "
                      f"{'BET' if e_new >= thresh else 'NO BET'}")

    report("REAL PAIRING — Pinnacle book vs Bet365 exchange",
           lambda book, exch, i: exch[i])
    for delta in args.delta:
        report(f"SENSITIVITY — exchange priced {delta:.2f} BELOW the book",
               lambda book, exch, i, _d=delta: book[i] - _d)


if __name__ == "__main__":
    main()
