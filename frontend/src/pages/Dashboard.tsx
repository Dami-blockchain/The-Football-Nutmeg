import { useState } from "react";
import { useStatus, useActions } from "../api";
import { Card, Metric, ErrorBox, pct, usd } from "../ui";

export default function Dashboard() {
  const { data: s, isLoading, error } = useStatus();
  const { score, settle } = useActions();
  const [copied, setCopied] = useState(false);

  if (isLoading) return <div className="text-muted">loading…</div>;
  if (error || !s) return <ErrorBox error={error} />;

  const copy = () => {
    navigator.clipboard.writeText(s.wallet.address);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card title="Deposit USDC">
        <div
          onClick={copy}
          className="font-mono text-sm bg-bg border border-border rounded-lg p-2.5 break-all cursor-pointer hover:border-accent"
          title="click to copy"
        >
          {s.wallet.address} {copied && <span className="text-success">✓ copied</span>}
        </div>
        <div className="mt-3">
          {s.wallet.balances.map((b) => (
            <Metric key={b.chain} k={b.label} v={`${b.usdc.toFixed(2)} USDC${b.ok ? "" : " ⚠"}`} />
          ))}
          <Metric k="Total" v={`${s.wallet.total_usdc.toFixed(2)} USDC`} />
        </div>
        <p className="text-muted text-xs mt-2 leading-relaxed">
          Same address on Polygon (Polymarket) &amp; Base (Limitless). Depositing does not start
          live trading — that stays gated until the paper record passes.
        </p>
      </Card>

      <Card title="Performance">
        <Metric k="Settled bets" v={s.performance.n} />
        <Metric k="Hit rate" v={pct(s.performance.hit_rate)} />
        <Metric k="ROI" v={pct(s.performance.roi)} />
        <Metric k="Brier" v={s.performance.brier.toFixed(3)} />
        <Metric
          k={`P&L (${s.trailing_window.days}d)`}
          v={usd(s.trailing_window.pnl_usd)}
          tone={s.trailing_window.pnl_usd >= 0 ? "pos" : "neg"}
        />
        <Metric k="Daily exposure" v={usd(s.daily_exposure_usd)} />
      </Card>

      <Card title="Gate" className="md:col-span-2">
        {s.gate.passed ? (
          <div className="text-success text-sm">PASS — paper record clears the live-trading thresholds.</div>
        ) : (
          <div className="text-sm">
            <div className="text-danger mb-1">FAIL</div>
            <ul className="text-muted list-disc ml-5">
              {s.gate.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      <Card title="Actions" className="md:col-span-2">
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => score.mutate()}
            disabled={score.isPending}
            className="px-3 py-1.5 text-sm rounded-lg border border-border hover:border-accent disabled:opacity-50"
          >
            {score.isPending ? "scoring…" : "Run scoring"}
          </button>
          <button
            onClick={() => settle.mutate()}
            disabled={settle.isPending}
            className="px-3 py-1.5 text-sm rounded-lg border border-border hover:border-accent disabled:opacity-50"
          >
            {settle.isPending ? "settling…" : "Run settlement"}
          </button>
        </div>
      </Card>
    </div>
  );
}
