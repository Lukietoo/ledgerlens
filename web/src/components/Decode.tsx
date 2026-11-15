"use client";

import { useEffect, useState } from "react";
import type { Txn } from "@/lib/api";
import { titleCase } from "@/lib/api";

// The hero: a real line from the statement, then what it actually is.
// Cycles through a few examples once loaded; static under reduced motion.
export function Decode({ examples }: { examples: Txn[] }) {
  const [i, setI] = useState(0);
  useEffect(() => {
    if (examples.length < 2) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const t = setInterval(() => setI((x) => (x + 1) % examples.length), 3600);
    return () => clearInterval(t);
  }, [examples.length]);

  const ex = examples[i];
  if (!ex) return null;
  return (
    <div className="max-w-3xl" aria-live="polite">
      <p className="raw truncate text-ink-faint" title={ex.description}>{ex.description}</p>
      <p key={i} className="decode-in mt-2 text-[34px] font-semibold leading-tight tracking-[-0.02em] sm:text-[44px]">
        {ex.merchant}
        <span className="ml-3 align-middle text-base font-normal text-ink-soft">
          {titleCase(ex.category)}
          {ex.source === "llm" && <span className="ml-2 rounded bg-amber-soft px-1.5 py-0.5 text-[12px] text-amber">named by Claude</span>}
        </span>
      </p>
    </div>
  );
}
