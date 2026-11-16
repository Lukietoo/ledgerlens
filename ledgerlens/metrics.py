"""Prometheus metrics, served at GET /metrics.

Route labels are the route *template* (/v1/datasets/{dataset_id}/dashboard), never the raw
path, so a dataset id can't explode label cardinality. Unmatched paths share one label.

Works with several uvicorn workers per process tree: set PROMETHEUS_MULTIPROC_DIR to an empty,
writable directory and every worker writes its samples there; /metrics merges them. In
Kubernetes each pod runs one worker and Prometheus scrapes pods individually, so it's optional.
"""

from __future__ import annotations

import os
import time

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

# latency buckets sized for this service: sub-ms fast path through multi-second LLM/agent calls
_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

REQUESTS = Counter("ledgerlens_http_requests", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("ledgerlens_http_request_duration_seconds", "HTTP request latency", ["method", "route"],
                    buckets=_BUCKETS)
ENRICHED = Counter("ledgerlens_transactions_enriched", "Transactions enriched", ["path"])  # fast | batch
LLM_ROUTED = Counter("ledgerlens_llm_routed_transactions", "Transactions sent to the LLM fallback")
LLM_SPEND = Counter("ledgerlens_llm_spend_usd", "Estimated LLM spend in USD", ["kind"])  # enrich | agent
CORRECTIONS = Counter("ledgerlens_corrections", "User corrections applied")


def _route(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


async def middleware(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = _route(request)
        if route != "/metrics":
            LATENCY.labels(request.method, route).observe(time.perf_counter() - t0)
            REQUESTS.labels(request.method, route, str(status)).inc()


def exposition() -> Response:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        body = generate_latest(registry)
    else:
        body = generate_latest()
    return Response(body, media_type=CONTENT_TYPE_LATEST)
