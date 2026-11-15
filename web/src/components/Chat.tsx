"use client";

import { useRef, useState } from "react";
import { api, type ChatMessage, type ChatReply } from "@/lib/api";

const STARTERS = [
  "How much did I spend on dining last month?",
  "Which subscriptions cost the most?",
  "Can I afford a $1,200 trip next month?",
];

type Turn = ChatMessage & { meta?: ChatReply };

export function Chat({ datasetId }: { datasetId: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    const next: Turn[] = [...turns, { role: "user", content: question }];
    setTurns(next);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const r = await api.chat(datasetId, next.map(({ role, content }) => ({ role, content })));
      setTurns([...next, { role: "assistant", content: r.reply, meta: r }]);
    } catch (e) {
      setError((e as Error).message);
      setTurns(turns); // roll back so the question can be retried
      setInput(question);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => endRef.current?.scrollIntoView({ block: "nearest" }));
    }
  }

  return (
    <section aria-labelledby="chat-title" className="rounded-lg border border-rule bg-surface">
      <div className="border-b border-rule px-4 py-3">
        <h2 id="chat-title" className="text-lg font-semibold">Ask about your money</h2>
        <p className="text-[13px] text-ink-soft">Claude answers from your transactions. Every number comes from a calculation, not a guess.</p>
      </div>

      <div className="max-h-[420px] space-y-4 overflow-y-auto px-4 py-4" aria-live="polite">
        {turns.length === 0 && (
          <div className="flex flex-col items-start gap-2">
            {STARTERS.map((s) => (
              <button key={s} onClick={() => send(s)} disabled={busy}
                      className="rounded-full border border-rule px-3 py-1 text-left text-[13px] text-ink-soft hover:border-green hover:text-ink">
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((t, i) =>
          t.role === "user" ? (
            <p key={i} className="ml-8 rounded-lg bg-paper px-3 py-2 text-sm">{t.content}</p>
          ) : (
            <div key={i} className="text-sm">
              <p className="whitespace-pre-wrap">{t.content}</p>
              {t.meta && t.meta.trace.length > 0 && (
                <details className="mt-1.5 text-[12px] text-ink-faint">
                  <summary className="cursor-pointer select-none">
                    Checked {t.meta.trace.length} {t.meta.trace.length === 1 ? "calculation" : "calculations"} in {t.meta.seconds.toFixed(1)}s
                  </summary>
                  <ol className="mt-1 space-y-1 border-l border-rule pl-3">
                    {t.meta.trace.map((s, j) => (
                      <li key={j}>
                        <span className={s.is_error ? "text-red" : "text-ink-soft"}>{s.tool.replaceAll("_", " ")}</span>
                        <span className="raw ml-1">{compactArgs(s.input)}</span>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </div>
          ),
        )}
        {busy && <p className="text-sm text-ink-faint">Working it out…</p>}
        <div ref={endRef} />
      </div>

      {error && <p role="alert" className="px-4 pb-2 text-[13px] text-red">{error}</p>}

      <form className="flex gap-2 border-t border-rule px-3 py-3" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <label htmlFor="chat-in" className="sr-only">Your question</label>
        <input id="chat-in" value={input} onChange={(e) => setInput(e.target.value)} maxLength={4000}
               placeholder="Ask a question" disabled={busy}
               className="min-w-0 flex-1 rounded-md border border-rule bg-paper px-3 py-2 text-sm placeholder:text-ink-faint" />
        <button type="submit" disabled={busy || !input.trim()}
                className="rounded-md bg-ink px-4 py-2 text-sm font-medium text-surface disabled:opacity-40">Ask</button>
      </form>
    </section>
  );
}

function compactArgs(input: Record<string, unknown>) {
  const parts = Object.entries(input).filter(([, v]) => v !== null && v !== undefined && v !== "");
  return parts.length ? parts.map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(" ") : "";
}
