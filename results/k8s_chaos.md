## Kubernetes chaos test (kind, 3 API replicas behind a Service, Redis-backed state)

75 s of in-cluster load (16 concurrent clients, a new connection per request). Under load, one API
pod was deleted normally (like a deploy or node drain), then 25 s later another was force-killed
(like a crash).

| | |
|---|---|
| requests | 13,938 (185.8 req/s) |
| failed | **0** (0.00%) |
| latency p50 / p99 | 56.1 ms / 403.0 ms |
| back to 8/8 ready after the kill | 10 s |
| correction saved before the chaos, still there after | yes |
| replicas scraped by Prometheus (pod discovery) | 10 |
