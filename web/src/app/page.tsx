"use client";

import { useEffect, useRef, useState } from "react";
import { Chat } from "@/components/Chat";
import { Decode } from "@/components/Decode";
import { Recurring } from "@/components/Recurring";
import { SpendingChart } from "@/components/SpendingChart";
import { TransactionFeed } from "@/components/TransactionFeed";
import { api, usd, type Dashboard, type DatasetInfo, type Txn } from "@/lib/api";

// Shown before anything is loaded: real-looking lines from the synthetic generator's noise model.
const SAMPLE: Txn[] = [
  { description: "PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BTL COF OAKLAND CA S382074619283746 CARD 4821", merchant: "Blue Bottle Coffee", category: "COFFEE" },
  { description: "CHECKCARD 0412 AMZN MKTP US*2K4AB1CD2 AMZN.COM/BILL WA 30196686274025326268433", merchant: "Amazon", category: "SHOPPING" },
  { description: "ZILLOW RENT DES:RENT ID:63YY09OPN INDN:ANDREW ANDERSON CO ID:2369560690 WEB", merchant: "Zillow Rent", category: "HOUSING" },
  { description: "TST* GOLDEN FOX TAQUERIA BERKELEY CA", merchant: "Golden Fox Taqueria", category: "DINING" },
].map((t) => ({ ...t, date: "", amount: 0, confidence: 1, source: "model", merchant_in_directory: true, is_recurring: false }));

export default function Home() {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [dash, setDash] = useState<Dashboard | null>(null);
  const [examples, setExamples] = useState<Txn[]>(SAMPLE);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  // public demo servers can turn statement uploads off; hide the button rather than fail on click
  const [uploadsEnabled, setUploadsEnabled] = useState(true);
  useEffect(() => {
    api.health().then((h) => setUploadsEnabled(h.uploads_enabled !== false)).catch(() => {});
  }, []);

  async function load(start: () => Promise<DatasetInfo>, label: string) {
    setLoading(label);
    setError(null);
    try {
      const i = await start();
      const [d, feed] = await Promise.all([api.dashboard(i.dataset_id), api.transactions(i.dataset_id, { limit: 200 })]);
      setInfo(i);
      setDash(d);
      setExamples(pickExamples(feed.rows));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(null);
    }
  }

  const o = dash?.overview;
  return (
    <main className="mx-auto max-w-6xl px-4 pb-24 sm:px-8">
      <header className="flex flex-wrap items-center justify-between gap-4 py-6">
        <p className="text-[17px] font-semibold tracking-[-0.01em]">Ledgerlens</p>
        <div className="flex flex-wrap gap-2">
          {uploadsEnabled && (
            <button onClick={() => fileRef.current?.click()} disabled={!!loading}
                    className="rounded-md border border-ink/20 px-3.5 py-2 text-sm font-medium hover:border-ink/50 disabled:opacity-50">
              {loading === "upload" ? "Reading your statement…" : "Upload a bank CSV"}
            </button>
          )}
          <button onClick={() => load(api.demo, "demo")} disabled={!!loading}
                  className="rounded-md bg-ink px-3.5 py-2 text-sm font-medium text-surface hover:bg-ink/90 disabled:opacity-50">
            {loading === "demo" ? "Loading a year of transactions…" : info ? "Reload demo" : "Try it on a demo account"}
          </button>
          <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden"
                 onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) load(() => api.upload(f), "upload"); }} />
        </div>
      </header>

      <section className="border-b border-rule pb-10 pt-6" aria-label="What Ledgerlens does">
        <Decode examples={examples} />
        {o ? (
          <p className="mt-6 text-ink-soft">
            {o.transactions.toLocaleString()} transactions from {fmtDate(o.first_date)} to {fmtDate(o.last_date)}:{" "}
            <span className="num text-ink">{usd(o.total_spending, false)}</span> spent,{" "}
            <span className="num text-ink">{usd(o.total_income, false)}</span> earned.
            {dash && dash.enrichment.llm_routed > 0 && <> {dash.enrichment.llm_routed} lines were named by Claude.</>}
          </p>
        ) : (
          <p className="mt-6 max-w-xl text-ink-soft">
            Bank statements say <span className="raw text-ink">SQ *BLUE BTL COF</span>. Ledgerlens works out who you paid,
            what kind of spending it was, and which charges come back every month. Then you can ask it questions.
            Your statement stays in memory on the server and is never saved.
          </p>
        )}
        {error && <p role="alert" className="mt-4 text-sm text-red">{error}</p>}
      </section>

      {info && dash && (
        <div className="mt-10 grid gap-12 lg:grid-cols-[minmax(0,1fr)_360px]">
          <div className="min-w-0 space-y-12">
            <SpendingChart data={dash} />
            <TransactionFeed datasetId={info.dataset_id} categories={Object.keys(
              dash.monthly_by_category.reduce((acc, m) => ({ ...acc, ...m }), {} as Record<string, unknown>),
            ).filter((k) => k !== "month").sort()} />
          </div>
          <aside className="space-y-10 lg:sticky lg:top-6 lg:self-start">
            <Chat datasetId={info.dataset_id} />
            <Recurring data={dash} />
          </aside>
        </div>
      )}
    </main>
  );
}

function fmtDate(iso: string) {
  return new Date(iso + "T00:00").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

// Hero examples: card purchases where the merchant's name isn't legible in the bank's text,
// i.e. the lines that actually needed decoding. Bank transfers and P2P make poor examples.
const key = (s: string) => s.toUpperCase().replace(/[^A-Z0-9]/g, "");
const ACH = /\b(DES:|PPD|WEB ID|CO ID)\b|^ZELLE|^VENMO/i;

function pickExamples(rows: Txn[]): Txn[] {
  const seen = new Set<string>();
  const decoded = rows.filter((r) =>
    r.amount < 0 && !ACH.test(r.description) && r.category !== "TRANSFER" &&
    !key(r.description).includes(key(r.merchant)));
  const pool = decoded.length >= 3 ? decoded : rows.filter((r) => r.amount < 0);
  return pool
    .filter((r) => (seen.has(r.merchant) ? false : (seen.add(r.merchant), true)))
    .slice(0, 6);
}
