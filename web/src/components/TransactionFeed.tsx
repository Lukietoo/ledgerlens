"use client";

import { useEffect, useState } from "react";
import { api, titleCase, usd, type Txn } from "@/lib/api";

const PAGE = 25;

export function TransactionFeed({ datasetId, categories }: { datasetId: string; categories: string[] }) {
  const [q, setQ] = useState("");
  const [category, setCategory] = useState("");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<{ total: number; rows: Txn[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setOffset(0), [q, category]);
  useEffect(() => {
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api.transactions(datasetId, { q, category, limit: PAGE, offset })
        .then((d) => { if (!ctrl.signal.aborted) { setData(d); setError(null); } })
        .catch((e) => !ctrl.signal.aborted && setError(e.message));
    }, 180);
    return () => { ctrl.abort(); clearTimeout(t); };
  }, [datasetId, q, category, offset]);

  return (
    <section aria-labelledby="feed-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="feed-title" className="text-lg font-semibold">Transactions</h2>
          <p className="text-[13px] text-ink-soft">What your bank wrote, and what it was.</p>
        </div>
        <div className="flex gap-2">
          <label className="sr-only" htmlFor="feed-q">Search transactions</label>
          <input id="feed-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search merchants"
                 className="w-44 rounded-md border border-rule bg-surface px-3 py-1.5 text-sm placeholder:text-ink-faint" />
          <label className="sr-only" htmlFor="feed-cat">Category</label>
          <select id="feed-cat" value={category} onChange={(e) => setCategory(e.target.value)}
                  className="rounded-md border border-rule bg-surface px-2 py-1.5 text-sm">
            <option value="">All categories</option>
            {categories.map((c) => <option key={c} value={c}>{titleCase(c)}</option>)}
          </select>
        </div>
      </div>

      {error && <p role="alert" className="mt-4 text-sm text-red">Couldn&apos;t load transactions: {error}</p>}

      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[520px] table-fixed border-collapse text-sm">
          <thead>
            <tr className="border-b border-rule text-left text-[13px] text-ink-faint">
              <th className="w-14 py-2 pr-3 font-normal">Date</th>
              <th className="py-2 pr-3 font-normal">Merchant</th>
              <th className="w-32 py-2 pr-3 font-normal">Category</th>
              <th className="w-28 py-2 text-right font-normal">Amount</th>
            </tr>
          </thead>
          <tbody>
            {data?.rows.map((r, i) => (
              <tr key={`${r.date}-${i}-${offset}`} className="border-b border-rule/60 align-top">
                <td className="num whitespace-nowrap py-2.5 pr-3 text-ink-soft">{r.date.slice(5)}</td>
                <td className="min-w-0 py-2.5 pr-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="truncate font-medium">{r.merchant}</span>
                    {r.source === "llm" && (
                      <span className="rounded bg-amber-soft px-1.5 text-[11px] text-amber" title="Named by Claude: this merchant wasn't in the directory, or the model was unsure">Claude</span>
                    )}
                    {r.is_recurring && <span className="text-[11px] text-ink-faint" title="Recurring charge">recurring</span>}
                  </div>
                  <div className="raw mt-0.5 truncate text-ink-faint" title={r.description}>{r.description}</div>
                </td>
                <td className="py-2.5 pr-3">
                  <span className="text-ink-soft">{titleCase(r.category)}</span>
                  {r.confidence < 0.8 && r.source !== "llm" && (
                    <span className="ml-1.5 text-[11px] text-ink-faint" title={`Model confidence ${(r.confidence * 100).toFixed(0)}%`}>unsure</span>
                  )}
                </td>
                <td className={`num py-2.5 text-right ${r.amount < 0 ? "text-red" : "text-green"}`}>
                  {r.amount < 0 ? "−" : "+"}{usd(Math.abs(r.amount))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {data && data.rows.length === 0 && (
          <p className="py-8 text-center text-sm text-ink-soft">No transactions match. Try a different search or category.</p>
        )}
      </div>

      {data && data.total > PAGE && (
        <div className="mt-3 flex items-center justify-between text-[13px] text-ink-soft">
          <span className="num">{offset + 1}–{Math.min(offset + PAGE, data.total)} of {data.total.toLocaleString()}</span>
          <div className="flex gap-2">
            <button className="rounded-md border border-rule px-3 py-1 disabled:opacity-40" disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - PAGE))}>Newer</button>
            <button className="rounded-md border border-rule px-3 py-1 disabled:opacity-40" disabled={offset + PAGE >= data.total}
                    onClick={() => setOffset(offset + PAGE)}>Older</button>
          </div>
        </div>
      )}
    </section>
  );
}
