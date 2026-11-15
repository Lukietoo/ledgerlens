"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Dashboard } from "@/lib/api";
import { monthLabel, titleCase, usd } from "@/lib/api";

// Validated with the dataviz palette checker against --surface (#f8faf7): all checks pass,
// worst adjacent CVD ΔE 9.2. Green and red are skipped on purpose: in this ledger they mean
// money in / money out. Fixed order; a 6th+ category folds into "Other", never a new hue.
const SERIES = ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#e87ba4"];
const OTHER = "#a3aab3";

type Row = Record<string, number | string>;

export function SpendingChart({ data }: { data: Dashboard }) {
  const top = data.overview.top_categories.slice(0, SERIES.length).map((c) => c.category);
  const rows: Row[] = data.monthly_by_category.map((m) => {
    const row: Row = { month: m.month as string };
    let other = 0;
    for (const [k, v] of Object.entries(m)) {
      if (k === "month") continue;
      if (top.includes(k)) row[k] = v as number;
      else other += v as number;
    }
    row.OTHER = Math.round(other * 100) / 100;
    return row;
  });
  const keys = [...top, "OTHER"];
  const color = (k: string) => (k === "OTHER" ? OTHER : SERIES[top.indexOf(k)]);

  return (
    <figure aria-labelledby="spend-title">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <h2 id="spend-title" className="text-lg font-semibold">Spending by month</h2>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-soft" aria-label="Legend">
          {keys.map((k) => (
            <li key={k} className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: color(k) }} aria-hidden />
              {k === "OTHER" ? "Everything else" : titleCase(k)}
            </li>
          ))}
        </ul>
      </figcaption>
      <div className="mt-4 h-64" role="img" aria-label="Stacked bar chart of monthly spending by category">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 4, right: 0, bottom: 0, left: 0 }} barCategoryGap="28%">
            <CartesianGrid vertical={false} stroke="var(--rule)" strokeDasharray="0" />
            <XAxis dataKey="month" tickFormatter={monthLabel} tickLine={false} axisLine={false}
                   tick={{ fill: "var(--ink-faint)", fontSize: 12 }} />
            <YAxis tickFormatter={(v) => usd(v, false)} tickLine={false} axisLine={false} width={64}
                   tick={{ fill: "var(--ink-faint)", fontSize: 12 }} />
            <Tooltip cursor={{ fill: "rgba(23,33,58,0.05)" }} content={<ChartTooltip keys={keys} color={color} />} />
            {keys.map((k, i) => (
              <Bar key={k} dataKey={k} stackId="s" fill={color(k)} stroke="var(--surface)" strokeWidth={2}
                   radius={i === keys.length - 1 ? [4, 4, 0, 0] : 0} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

function ChartTooltip({ active, payload, label, keys, color }: {
  active?: boolean; payload?: { dataKey: string; value: number }[]; label?: string;
  keys: string[]; color: (k: string) => string;
}) {
  if (!active || !payload?.length) return null;
  const byKey = Object.fromEntries(payload.map((p) => [p.dataKey, p.value]));
  const total = payload.reduce((s, p) => s + (p.value || 0), 0);
  return (
    <div className="min-w-48 rounded-md border border-rule bg-surface px-3 py-2 text-[13px] shadow-sm">
      <p className="mb-1 font-semibold">{label && monthLabel(label)}: {usd(total, false)}</p>
      {[...keys].reverse().filter((k) => byKey[k]).map((k) => (
        <p key={k} className="num flex items-center justify-between gap-4 text-ink-soft">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-[2px]" style={{ background: color(k) }} />
            {k === "OTHER" ? "Everything else" : titleCase(k)}
          </span>
          <span className="text-ink">{usd(byKey[k], false)}</span>
        </p>
      ))}
    </div>
  );
}
