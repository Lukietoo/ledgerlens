## Cost per 1M transactions

Model-only: measured **11,496 txns/s** on 4 workers (M1 laptop, batches of 100) → **$0.0039 per 1M** (assumes $0.04/vCPU-hour).

Hybrid routes rows below confidence 0.8 to the LLM: 22.8% of traffic (cold directory) / 15.7% (warm).

| LLM | tokens/txn in / out | LLM-only | hybrid, no cache (cold / warm) | hybrid + unknown merchants + cache (cold / warm) |
|---|---|---|---|---|
| claude-opus-5-5 | 45 / 60 (est.) | $1,381.48 | $314.95 / $217.18 | $61.28 / $31.18 |
| claude-sonnet-5-5 | 45 / 60 (est.) | $690.74 | $157.48 / $108.59 | $30.64 / $15.59 |
| claude-haiku-5-5 | 103 / 31 | $25.95 | $5.92 / $4.08 | $1.15 / $0.59 |

All costs per 1M transactions. Cached LLM calls per 100 txns (one call per new merchant key, measured over a year of test traffic): 4.44 cold / 2.26 warm. Real traffic from more users repeats merchants more, so this is conservative.

(est.) = tokens estimated before any API call (≈4 chars/token, pessimistic output); run `python -m ledgerlens.llm_eval --yes --model <m>` to replace with measured usage.
Standard API rates; the Batch API would halve LLM costs for non-real-time backfills.
