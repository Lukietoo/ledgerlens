## LLM fallback eval: claude-haiku-5-5, cold directory, 400 unique test txns

| strategy | category acc | merchant exact | API cost for this sample |
|---|---|---|---|
| model only | 76.5% | 40.5% | $0 |
| LLM only | 91.5% | 83.8% | $0.01 |
| hybrid (LLM on 130 txns below 0.8) | 92.0% | 53.8% | $0.00 |

Measured tokens per transaction (LLM-only): 103 in / 31 out. Refusals: 0 requests.
