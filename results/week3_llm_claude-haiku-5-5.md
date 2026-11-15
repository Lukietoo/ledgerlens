## LLM fallback eval: claude-haiku-5-5, cold directory, 400 unique test txns

| strategy | category acc | merchant exact | LLM calls (txns or merchant keys) |
|---|---|---|---|
| model only | 76.5% | 40.5% | 0 |
| LLM only | 91.5% | 83.8% | 400 (reused from previous run) |
| hybrid: confidence only (week 3) | 91.2% | 52.8% | 66 keys for 130 rows |
| hybrid: + unknown merchants, LLM names only | 91.2% | 82.2% | 190 keys for 386 rows |
| hybrid: + unknown merchants, LLM names + categories | 89.2% | 82.2% | 190 keys for 386 rows |

Hybrid API usage (all policies share one cache): 8 requests, $0.0060, 0 refusals.
Measured tokens per transaction (LLM-only): 103 in / 31 out. LLM-only refusals: 0 requests.
