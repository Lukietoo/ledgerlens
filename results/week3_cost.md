## Cost per 1M transactions

Model-only: measured **11,496 txns/s** on 4 workers (M1 laptop, batches of 100) → **$0.0039 per 1M** (assumes $0.04/vCPU-hour).

Hybrid routes rows below confidence 0.8 to the LLM: 21.5% of traffic (cold directory) / 13.9% (warm).

| LLM | tokens/txn in / out | LLM-only per 1M | hybrid per 1M (cold) | hybrid per 1M (warm) | LLM-only ÷ model-only |
|---|---|---|---|---|---|
| claude-opus-5-5 | 45 / 60 (est.) | $1,381 | $298 | $192 | 357,333× |
| claude-sonnet-5-5 | 45 / 60 (est.) | $691 | $149 | $96 | 178,666× |
| claude-haiku-5-5 | 103 / 31 | $26 | $6 | $4 | 6,712× |

(est.) = tokens estimated before any API call (≈4 chars/token, pessimistic output); run `python -m ledgerlens.llm_eval --yes --model <m>` to replace with measured usage.
Standard API rates; the Batch API would halve LLM costs for non-real-time backfills.
