## Horizontal scaling (Redis-backed state, stateful workload)

160 concurrent simulated users, 45 s per run, 8-core laptop; locust (3 processes) on the same machine. 16 statements are created up front; each user opens one and browses its dashboard, feed and review list, corrects merchants (an atomic, lock-free write in Redis), and enriches single transactions. Every dataset request needs shared state: any worker may serve any user.

| API workers | throughput | vs 1 worker | p50 | p95 | p99 | errors | busy (503, retryable) |
|---|---|---|---|---|---|---|---|
| 1 | 163 req/s | 1.0× | 890 ms | 1300 ms | 1800 ms | 0 / 7,313 | 0 |
| 2 | 244 req/s | 1.5× | 580 ms | 950 ms | 1100 ms | 0 / 11,010 | 0 |
| 4 | 371 req/s | 2.3× | 360 ms | 750 ms | 1100 ms | 0 / 16,704 | 0 |
| 8 | 318 req/s | 2.0× | 260 ms | 1600 ms | 2700 ms | 0 / 14,440 | 0 |

Latency is under saturation: the simulated users send requests as fast as the API answers, so throughput is capacity and p50 includes queueing. Locust (3 processes) and Redis share the same 8 cores, so 4 workers is the useful maximum here; at 8 the machine is oversubscribed and throughput drops. More capacity means more machines, which is the Kubernetes setup.

### Chaos: kill a worker under load

4 workers; worker pid 23183 SIGKILLed at 15.0 s. Failed requests: **34 of 13,346** (0.25%), all in the second of the kill: the requests in flight on that worker. Worker replaced by the supervisor: **yes** (full worker set back in 0.4 s). Statements and corrections live in Redis, so nothing a user saved was lost.
