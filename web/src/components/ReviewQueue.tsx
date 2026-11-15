"use client";

import { useEffect, useState } from "react";
import { api, titleCase, type ReviewItem } from "@/lib/api";
import { FixEditor } from "./FixEditor";

// The merchants where a quick human look does the most good: many transactions x an unsure model.
export function ReviewQueue({ datasetId, refreshKey, onChanged }: {
  datasetId: string;
  refreshKey: number;
  onChanged: (message: string) => void;
}) {
  const [items, setItems] = useState<ReviewItem[] | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.review(datasetId, 4)
      .then((r) => { if (live) { setItems(r.items); setError(null); } })
      .catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [datasetId, refreshKey]);

  async function confirm(item: ReviewItem) {
    try {
      const r = await api.correct(datasetId, { merchant_key: item.merchant_key, merchant: item.merchant, category: item.category });
      // merchant first: "Marked 21 76 transactions" was unreadable for the 76 gas station
      onChanged(`Confirmed ${item.merchant} (${r.updated} ${r.updated === 1 ? "transaction" : "transactions"}).`);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (error) return <p role="alert" className="text-sm text-red">Couldn&apos;t load the review list: {error}</p>;
  if (!items) return null;

  return (
    <section aria-labelledby="review-title">
      <h2 id="review-title" className="text-lg font-semibold">Double-check these</h2>
      <p className="text-[13px] text-ink-soft">
        {items.length
          ? "Ledgerlens is least sure about these. One fix applies to every transaction from that merchant."
          : "Nothing left to check. Everything here has been confirmed or fixed."}
      </p>
      <ul className="mt-3 divide-y divide-rule/70">
        {items.map((it) => (
          <li key={it.merchant_key} className="py-2.5">
            <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <div className="min-w-0">
                <p className="text-sm">
                  <span className="font-medium">{it.merchant}</span>
                  <span className="text-ink-soft"> · {titleCase(it.category)} · {it.rows} transactions</span>
                </p>
                <p className="raw truncate text-ink-faint" title={it.example}>{it.example}</p>
              </div>
              {editing !== it.merchant_key && (
                <div className="flex shrink-0 gap-2 text-[13px]">
                  <button onClick={() => confirm(it)} className="rounded border border-rule px-2.5 py-1 hover:border-green">Looks right</button>
                  <button onClick={() => setEditing(it.merchant_key)} className="rounded border border-rule px-2.5 py-1 hover:border-ink/50">Fix</button>
                </div>
              )}
            </div>
            {editing === it.merchant_key && (
              <FixEditor datasetId={datasetId} merchantKey={it.merchant_key} merchant={it.merchant} category={it.category}
                         onCancel={() => setEditing(null)}
                         onDone={(m) => { setEditing(null); onChanged(m); }} />
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
