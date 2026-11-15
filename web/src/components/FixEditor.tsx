"use client";

import { useState } from "react";
import { api, CATEGORIES, titleCase } from "@/lib/api";

// Fix a merchant everywhere it appears: the server applies it to every transaction with this merchant key.
export function FixEditor({ datasetId, merchantKey, merchant, category, onDone, onCancel }: {
  datasetId: string;
  merchantKey: string;
  merchant: string;
  category: string;
  onDone: (message: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(merchant);
  const [cat, setCat] = useState(category);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Give the merchant a name.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await api.correct(datasetId, { merchant_key: merchantKey, merchant: trimmed, category: cat });
      onDone(`Updated ${r.updated} ${r.updated === 1 ? "transaction" : "transactions"} from ${trimmed}.`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <form className="mt-2 flex flex-wrap items-center gap-2 rounded-md bg-paper px-2.5 py-2"
          onSubmit={(e) => { e.preventDefault(); save(); }}>
      <label className="sr-only" htmlFor={`fix-name-${merchantKey}`}>Merchant name</label>
      <input id={`fix-name-${merchantKey}`} value={name} onChange={(e) => setName(e.target.value)} maxLength={80} autoFocus
             className="min-w-0 flex-1 rounded border border-rule bg-surface px-2 py-1 text-sm" />
      <label className="sr-only" htmlFor={`fix-cat-${merchantKey}`}>Category</label>
      <select id={`fix-cat-${merchantKey}`} value={cat} onChange={(e) => setCat(e.target.value)}
              className="rounded border border-rule bg-surface px-1.5 py-1 text-sm">
        {CATEGORIES.map((c) => <option key={c} value={c}>{titleCase(c)}</option>)}
      </select>
      <button type="submit" disabled={busy} className="rounded bg-ink px-3 py-1 text-sm font-medium text-surface disabled:opacity-50">
        {busy ? "Saving…" : "Save for all"}
      </button>
      <button type="button" onClick={onCancel} className="px-1.5 py-1 text-sm text-ink-soft">Cancel</button>
      {error && <p role="alert" className="w-full text-[12px] text-red">{error}</p>}
    </form>
  );
}
