import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

const token = () => localStorage.getItem("tfsm_token") || "";
function headers(json = false): HeadersInit {
  const h: Record<string, string> = {};
  if (token()) h["Authorization"] = `Bearer ${token()}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: headers() });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}
async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: headers(true),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

// ---- types ----
export interface Balance { chain: string; label: string; usdc: number; ok: boolean; }
export interface Wallet { address: string; balances: Balance[]; total_usdc: number; }
export interface OutcomeStat { n: number; wins: number; hit_rate: number; roi: number; pnl_usd: number; }
export interface Perf {
  n: number; wins: number; hit_rate: number; roi: number; brier: number;
  pnl_usd: number; staked_usd: number; per_outcome: Record<string, OutcomeStat>;
}
export interface Status {
  mode: string;
  kill_switch: { tripped: boolean; reason: string | null; tripped_at: string | null };
  gate: { passed: boolean; reasons: string[]; window_days: number };
  performance: Perf;
  trailing_window: { days: number; pnl_usd: number; staked_usd: number };
  daily_exposure_usd: number;
  wallet: Wallet;
}
export interface Bet {
  fixture_id: number; outcome: string; our_probability: number;
  market_price: number | null; edge: number | null; stake_usd: number;
  created_at: string | null; settled_outcome: string | null; pnl_usd: number | null;
  rationale: string;
}
export interface Prediction {
  fixture_id: number; competition_code: string; home_team: string; away_team: string;
  kickoff: string | null; p_home: number; p_draw: number; p_away: number;
}
export interface Settings {
  mode: string; edge_threshold: number; fixed_stake_usd: number; max_bet_usd: number;
  daily_exposure_cap_usd: number; drawdown_kill_pct: number; drawdown_window_days: number;
  drawdown_min_staked_usd: number; gate_min_bets: number; gate_min_window_days: number;
  gate_min_hit_rate: number; gate_min_roi: number;
}

// ---- hooks ----
export const useStatus = () => useQuery({ queryKey: ["status"], queryFn: () => get<Status>("/api/status") });
export const useBets = (days = 30) =>
  useQuery({ queryKey: ["bets", days], queryFn: () => get<{ bets: Bet[] }>(`/api/bets?days=${days}`) });
export const usePredictions = (days = 7) =>
  useQuery({ queryKey: ["preds", days], queryFn: () => get<{ predictions: Prediction[] }>(`/api/predictions?days=${days}`) });
export const useBacktest = (mode: "stored" | "mock") =>
  useQuery({ queryKey: ["bt", mode], queryFn: () => get<Perf>(`/api/backtest?mode=${mode}`) });
export const useSettings = () => useQuery({ queryKey: ["settings"], queryFn: () => get<Settings>("/api/settings") });

export function useActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries();
  return {
    score: useMutation({ mutationFn: () => post("/api/score"), onSuccess: invalidate }),
    settle: useMutation({ mutationFn: () => post("/api/settle"), onSuccess: invalidate }),
    resetKill: useMutation({ mutationFn: () => post("/api/kill-switch/reset"), onSuccess: invalidate }),
    setSetting: useMutation({
      mutationFn: (v: { key: string; value: string }) => post("/api/settings", v),
      onSuccess: invalidate,
    }),
  };
}
