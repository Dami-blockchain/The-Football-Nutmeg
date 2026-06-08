import { useState } from "react";
import { useSettings, useStatus, useActions } from "../api";
import { Card, ErrorBox } from "../ui";

const EDITABLE: { key: string; label: string; env: string }[] = [
  { key: "mode", label: "Mode (paper/live)", env: "BETBOT_MODE" },
  { key: "edge_threshold", label: "Edge threshold", env: "BETBOT_EDGE_THRESHOLD" },
  { key: "fixed_stake_usd", label: "Fixed stake $", env: "BETBOT_FIXED_STAKE_USD" },
  { key: "daily_exposure_cap_usd", label: "Daily cap $", env: "BETBOT_DAILY_EXPOSURE_CAP_USD" },
  { key: "drawdown_kill_pct", label: "Drawdown kill %", env: "BETBOT_DRAWDOWN_KILL_PCT" },
  { key: "gate_min_bets", label: "Gate min bets", env: "BETBOT_GATE_MIN_BETS" },
  { key: "gate_min_roi", label: "Gate min ROI", env: "BETBOT_GATE_MIN_ROI" },
];

export default function SettingsPage() {
  const { data: s, error } = useSettings();
  const { data: status } = useStatus();
  const { setSetting, resetKill } = useActions();
  const [draft, setDraft] = useState<Record<string, string>>({});

  if (error) return <ErrorBox error={error} />;
  if (!s) return <div className="text-muted">loading…</div>;

  return (
    <div className="grid gap-4">
      <Card title="Knobs">
        <div className="grid gap-2">
          {EDITABLE.map((f) => {
            const cur = String((s as unknown as Record<string, unknown>)[f.key] ?? "");
            return (
              <div key={f.key} className="flex items-center gap-2">
                <label className="text-sm text-muted w-40">{f.label}</label>
                <input
                  defaultValue={cur}
                  onChange={(e) => setDraft((d) => ({ ...d, [f.env]: e.target.value }))}
                  className="bg-bg border border-border rounded px-2 py-1 text-sm tabnum w-32"
                />
                <button
                  onClick={() => setSetting.mutate({ key: f.env, value: draft[f.env] ?? cur })}
                  className="px-2 py-1 text-xs rounded border border-border hover:border-accent"
                >
                  save
                </button>
              </div>
            );
          })}
        </div>
        <p className="text-muted text-xs mt-3">
          Writes to .env. Mode changes need a daemon restart to take effect.
        </p>
      </Card>

      <Card title="Kill switch">
        {status?.kill_switch.tripped ? (
          <div className="text-sm">
            <div className="text-danger mb-2">TRIPPED — {status.kill_switch.reason}</div>
            <button
              onClick={() => {
                if (confirm("Reset the kill switch and resume betting?")) resetKill.mutate();
              }}
              className="px-3 py-1.5 text-sm rounded-lg border border-danger/50 text-danger hover:bg-danger/10"
            >
              Reset kill switch
            </button>
          </div>
        ) : (
          <div className="text-success text-sm">clear ✅</div>
        )}
      </Card>
    </div>
  );
}
