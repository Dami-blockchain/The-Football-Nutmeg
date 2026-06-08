import { useState } from "react";
import { useStatus } from "./api";
import { Pill } from "./ui";
import Dashboard from "./pages/Dashboard";
import Bets from "./pages/Bets";
import Predictions from "./pages/Predictions";
import Backtest from "./pages/Backtest";
import SettingsPage from "./pages/Settings";

const TABS = ["Dashboard", "Bets", "Predictions", "Backtest", "Settings"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Dashboard");
  const { data: s } = useStatus();

  return (
    <div className="max-w-5xl mx-auto px-4 py-6 pb-16">
      <header className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h1 className="text-xl font-semibold">⚽ The Football Smart Manager</h1>
        <div className="flex gap-2 flex-wrap">
          {s && <Pill tone="accent">mode: {s.mode}</Pill>}
          {s && (
            <Pill tone={s.kill_switch.tripped ? "bad" : "good"}>
              kill switch: {s.kill_switch.tripped ? "TRIPPED" : "clear"}
            </Pill>
          )}
          {s && <Pill tone={s.gate.passed ? "good" : "bad"}>gate: {s.gate.passed ? "PASS" : "FAIL"}</Pill>}
        </div>
      </header>

      <nav className="flex gap-1 border-b border-border mb-5">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-2 text-sm -mb-px border-b-2 ${
              tab === t ? "border-accent text-text" : "border-transparent text-muted hover:text-text"
            }`}
          >
            {t}
          </button>
        ))}
      </nav>

      {tab === "Dashboard" && <Dashboard />}
      {tab === "Bets" && <Bets />}
      {tab === "Predictions" && <Predictions />}
      {tab === "Backtest" && <Backtest />}
      {tab === "Settings" && <SettingsPage />}
    </div>
  );
}
