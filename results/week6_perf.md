## Inference latency (in-process, one core, M1 laptop)

| request | path | p50 | p95 | p99 |
|---|---|---|---|---|
| 1 transaction | pandas | 10.42 ms | 12.86 ms | 16.98 ms |
| 1 transaction | fast | 1.33 ms | 2.21 ms | 5.47 ms |
| 16 transactions, one user | pandas | 12.84 ms | 14.19 ms | 16.08 ms |
| 16 transactions, one user | fast | 3.20 ms | 3.82 ms | 4.06 ms |

Speedup at p50: **7.8×** for one transaction, **4.0×** for 16. Batches of 100 use the pandas path: 24.1 ms p50, about 4,150 transactions/s on one core.

The fast path is checked to give identical outputs to the pandas path (tests/test_fastpath.py).
