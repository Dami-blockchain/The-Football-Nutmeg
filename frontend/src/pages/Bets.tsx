import { useState } from "react";
import { useBets, Bet } from "../api";
import { Card, ErrorBox, usd, num } from "../ui";

function breakdown(bets: Bet[]) {
  const by: Record<string, { n: number; pnl: number; settled: number; wins: number }> = {};
  for (const b of bets) {
    const k = b.outcome;
    by[k] ??= { n: 0, pnl: 0, settled: 0, wins: 0 };
    by[k].n++;
    if (b.pnl_usd != null) {
      by[k].pnl += b.pnl_usd;
      by[k].settled++;
      if (b.settled_outcome === b.outcome) by[k].wins++;
    }
  }
  return by;
}

export default function Bets() {
  const [days, setDays] = useState(30);
  const { data, error } = useBets(days);
  if (error) return <ErrorBox error={error} />;
  const bets = data?.bets ?? [];
  const by = breakdown(bets);

  return (
    <div className="grid gap-4">
      <Card title="Per-outcome breakdown">
        <div className="flex gap-2 mb-3 text-sm">
          {[7, 30, 90].map((d) => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className={`px-2 py-1 rounded border ${days === d ? "border-accent text-accent" : "border-border text-muted"}`}
            >
              {d}d
            </button>
          ))}
        </div>
        <div className="grid grid-cols-3 gap-3">
          {["HOME", "DRAW", "AWAY"].map((o) => (
            <div key={o} className="bg-bg border border-border rounded-lg p-3 text-sm">
              <div className="text-muted">{o}</div>
              <div className="tabnum">{by[o]?.n ?? 0} bets</div>
              <div className={`tabnum ${(by[o]?.pnl ?? 0) >= 0 ? "text-success" : "text-danger"}`}>
                {usd(by[o]?.pnl ?? 0)}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title={`Bets (${bets.length})`}>
        {bets.length === 0 ? (
          <div className="text-muted text-sm">No bets in this window.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm tabnum">
              <thead className="text-muted">
                <tr className="text-left">
                  <th className="py-1.5 pr-3">fixture</th><th>out</th><th>p</th>
                  <th>mkt</th><th>stake</th><th>result</th><th>pnl</th>
                </tr>
              </thead>
              <tbody>
                {bets.map((b, i) => (
                  <tr key={i} className="border-t border-[#21262d]">
                    <td className="py-1.5 pr-3">{b.fixture_id}</td>
                    <td>{b.outcome}</td>
                    <td>{num(b.our_probability, 2)}</td>
                    <td>{b.market_price == null ? "—" : num(b.market_price, 2)}</td>
                    <td>{usd(b.stake_usd, 0)}</td>
                    <td>{b.settled_outcome ?? "—"}</td>
                    <td className={b.pnl_usd == null ? "" : b.pnl_usd >= 0 ? "text-success" : "text-danger"}>
                      {b.pnl_usd == null ? "—" : usd(b.pnl_usd)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
