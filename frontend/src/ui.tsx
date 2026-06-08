import { ReactNode } from "react";

export const pct = (n?: number | null, d = 1) => (n == null ? "—" : `${(n * 100).toFixed(d)}%`);
export const usd = (n?: number | null, d = 2) => (n == null ? "—" : `$${n.toFixed(d)}`);
export const num = (n?: number | null, d = 2) => (n == null ? "—" : n.toFixed(d));

type Tone = "default" | "good" | "bad" | "accent";
const toneCls: Record<Tone, string> = {
  default: "border-border text-text",
  good: "border-success/40 text-success",
  bad: "border-danger/40 text-danger",
  accent: "border-accent/40 text-accent",
};

export function Pill({ children, tone = "default" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span className={`text-xs px-2.5 py-1 rounded-full border bg-surface ${toneCls[tone]}`}>
      {children}
    </span>
  );
}

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <div className={`bg-surface border border-border rounded-xl p-4 ${className}`}>
      {title && <h2 className="text-xs uppercase tracking-wide text-muted mb-3">{title}</h2>}
      {children}
    </div>
  );
}

export function Metric({ k, v, tone }: { k: string; v: ReactNode; tone?: "pos" | "neg" }) {
  const c = tone === "pos" ? "text-success" : tone === "neg" ? "text-danger" : "text-text";
  return (
    <div className="flex justify-between items-baseline py-1.5 border-b border-[#21262d] last:border-0">
      <span className="text-muted text-sm">{k}</span>
      <span className={`tabnum font-semibold ${c}`}>{v}</span>
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="text-danger text-sm">
      {msg}
      {msg.includes("401") && " — set a token in the browser console: localStorage.tfsm_token = '…'"}
    </div>
  );
}
