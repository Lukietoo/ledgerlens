## Full system on the independent benchmark (1,988 held-out-merchant rows)

| | accuracy | macro-F1 |
|---|---|---|
| model only | 73.8% | 0.743 |
| **model + claude-haiku-5-5 fallback** | **90.3%** | **0.910** |

77% of rows routed to the LLM; 35 requests; measured cost $0.027 ($0.014 per 1K transactions).
