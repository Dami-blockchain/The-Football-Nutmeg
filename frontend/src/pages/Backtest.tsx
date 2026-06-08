import { useState } from "react";
import { useBacktest } from "../api";
import { Card, Metric, ErrorBox, pct, usd } from "../ui";

export default function Backtest() {
  const [mode, setMode] = useState<"stored" | "mock">("stored");
  const { data: r, error, isLoading } = useBacktest(mode);

  return (
    <div className="grid gap-4">
      <div className="flex gap-2 text-sm">
        {(["stored", "mock"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-3 py-1.5 rounded border ${mode === m ? "border-accent text-accent" : "border-border text-muted"}`}
          >
            {m}
          </button>
        ))}
      </div>
      {error && <ErrorBox error={error} />}
      {isLoading && <div className="text-muted">loading…</div>}
      {r && (
        <>
          <Card title={`${mode} backtest`}>
            <Metric k="Bets" v={r.n} />
            <Metric k="Hit rate" v={pct(r.hit_rate)} />
            <Metric k="ROI" v={pct(r.roi)} tone={r.roi >= 0 ? "pos" : "neg"} />
            <Metric k="Brier" v={r.brier.toFixed(3)} />
            <Metric k="P&L" v={usd(r.pnl_usd)} tone={r.pnl_usd >= 0 ? "pos" : "neg"} />
            <Metric k="Staked" v={usd(r.staked_usd)} />
          </Card>
          {Object.keys(r.per_outcome).length > 0 && (
            <Card title="Per outcome">
              {Object.entries(r.per_outcome).map(([o, st]) => (
                <Metric key={o} k={o} v={`${st.n} bets · ${pct(st.hit_rate)} · ${pct(st.roi)} ROI`} />
              ))}
            </Card>
          )}
          {mode === "mock" && (
            <p className="text-muted text-xs">
              Mock is a fair-market diagnostic: ROI should hover near 0 (no informational edge).
            </p>
          )}
        </>
      )}
    </div>
  );
}
