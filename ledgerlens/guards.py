"""Guards for running the API in public: per-client rate limits and a daily LLM spend cap.

In-process state, so they hold per instance (the dashboard already requires a single
worker). A multi-instance deployment would move both to Redis.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime

from fastapi import HTTPException, Request


def client_id(request: Request) -> str:
    """First hop of X-Forwarded-For when behind a proxy (Render, Fly), else the socket peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding window: at most ``limit`` calls per ``window`` seconds per (client, route)."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                retry = int(self.window - (now - q[0])) + 1
                raise HTTPException(429, f"too many requests; try again in {retry}s",
                                    headers={"Retry-After": str(retry)})
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory under a flood of distinct clients
                for k in [k for k, v in self._hits.items() if not v][:5_000]:
                    del self._hits[k]


class DailyBudget:
    """Caps LLM spend per UTC day. ``check`` before a paid call, ``spend`` after it."""

    def __init__(self, usd_per_day: float, today=lambda: datetime.now(UTC).date()):
        self.limit, self.today = usd_per_day, today
        self._day, self._spent = today(), 0.0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        if self.today() != self._day:
            self._day, self._spent = self.today(), 0.0

    def check(self) -> None:
        with self._lock:
            self._roll()
            if self._spent >= self.limit:
                raise HTTPException(503, "the demo's daily AI budget is used up; it resets at midnight UTC")

    def spend(self, usd: float) -> None:
        with self._lock:
            self._roll()
            self._spent += max(0.0, usd)

    @property
    def spent_today(self) -> float:
        with self._lock:
            self._roll()
            return round(self._spent, 4)
