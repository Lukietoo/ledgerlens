"""Where the API keeps state: in process (default) or in Redis (LEDGERLENS_REDIS_URL).

Four kinds of state, each with an in-memory and a Redis implementation behind the same interface:

  datasets   uploaded statements (Ledgers) and their corrections
  llm cache  LLM answers per merchant key
  limiter    per-client sliding-window rate limits
  budget     daily LLM spend cap

In memory, all of it lives in one process, so the API must run as a single worker. In Redis,
any number of workers (or machines) share it.

Design choices:
- Ledgers are stored as JSON, never pickle: loading a pickle runs code, so a compromised Redis
  must not be able to execute anything here.
- Each worker caches decoded Ledgers locally and checks a version number on every read (one
  cheap round trip), so a correction made on one worker is seen by all of them at once.
- Statements are written once; corrections are stored beside them as hash fields, so a
  correction is an O(1) atomic write with no lock (see RedisDatasets).
- The rate limit is one Lua script (atomic): two workers can't both admit the "last" request.
- Failure modes are deliberate: if Redis is down, rate limiting fails OPEN (keep serving) but
  the spend cap fails CLOSED (never spend money we can't count).
"""

from __future__ import annotations

import io
import json
import logging
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import HTTPException

from ledgerlens.finance import Ledger
from ledgerlens.guards import DailyBudget, RateLimiter
from ledgerlens.llm import Prediction
from ledgerlens.taxonomy import CATEGORIES

log = logging.getLogger("ledgerlens")

MAX_DATASETS = 32
DATASET_TTL_S = 2 * 60 * 60
_DERIVED = ["month", "is_spend", "is_income"]  # Ledger recomputes these
NOT_FOUND = "dataset not found (it may have expired); upload it again"


# ---------------------------------------------------------------- datasets

class MemoryDatasets:
    """One process: bounded LRU of Ledgers."""

    def __init__(self, capacity: int = MAX_DATASETS):
        self.capacity = capacity
        self._items: OrderedDict[str, Ledger] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, ledger: Ledger, key: str | None = None) -> str:
        key = key or secrets.token_urlsafe(12)
        with self._lock:
            self._items[key] = ledger
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return key

    def get(self, key: str) -> Ledger:
        with self._lock:
            if key not in self._items:
                raise HTTPException(404, NOT_FOUND)
            self._items.move_to_end(key)
            return self._items[key]

    def correct(self, key: str, merchant_key: str, merchant: str | None = None, category: str | None = None) -> int:
        return self.get(key).apply_correction(merchant_key, merchant=merchant, category=category)  # thread-safe


def ledger_to_json(ledger: Ledger) -> str:
    df = ledger.df.drop(columns=[c for c in _DERIVED if c in ledger.df])
    return json.dumps({"df": df.to_json(orient="split", date_format="iso"), "corrections": ledger.corrections})


def ledger_from_json(s: str) -> Ledger:
    d = json.loads(s)
    # dtype=False: keep JSON types as-is; Ledger parses the ISO dates itself
    df = pd.read_json(io.StringIO(d["df"]), orient="split", dtype=False)
    return Ledger(df, corrections=d["corrections"])


_SEP = "\x1f"  # merchant keys never contain a unit separator


class RedisDatasets:
    """Shared by every worker; everything expires after 2 h idle.

      ll:ds:{id}       hash {data: the enriched statement as JSON (written once), version: int}
      ll:ds:{id}:fix   hash {merchant_key<US>merchant: name, merchant_key<US>category: category}

    The statement is immutable once stored; corrections live beside it. A correction is a few
    hash fields plus a version bump in one MULTI: O(1), atomic, and lock-free, so writers never
    queue behind each other and a worker that dies mid-request can't leave anything held.
    Corrections to different merchants commute and each field is last-writer-wins, which is
    exactly Ledger.apply_correction's semantics, so the fold order doesn't matter.

    Reads cost one round trip: each worker caches the decoded statement and the corrected view
    by version, and rebuilds the view (base + fixes) only when the version moves.
    """

    def __init__(self, r, ttl_s: int = DATASET_TTL_S, local_capacity: int = MAX_DATASETS):
        self.r, self.ttl = r, ttl_s
        self._base: OrderedDict[str, Ledger] = OrderedDict()            # immutable statements
        self._view: OrderedDict[str, tuple[int, Ledger]] = OrderedDict()  # version -> corrected view
        self._cap = local_capacity
        self._lock = threading.Lock()

    @staticmethod
    def _k(key: str) -> str:
        return f"ll:ds:{key}"

    def _lru(self, d: OrderedDict, key: str, value) -> None:
        with self._lock:
            d[key] = value
            d.move_to_end(key)
            while len(d) > self._cap:
                d.popitem(last=False)

    def put(self, ledger: Ledger, key: str | None = None) -> str:
        key = key or secrets.token_urlsafe(12)
        p = self.r.pipeline()
        p.delete(self._k(key) + ":fix")
        p.hset(self._k(key), mapping={"data": ledger_to_json(ledger), "version": 1})
        p.expire(self._k(key), self.ttl)
        p.execute()
        self._lru(self._base, key, ledger_from_json(ledger_to_json(ledger)))  # private copy: never mutated
        return key

    def _statement(self, key: str) -> Ledger:
        with self._lock:
            base = self._base.get(key)
        if base is None:
            data = self.r.hget(self._k(key), "data")
            if data is None:
                raise HTTPException(404, NOT_FOUND)
            base = ledger_from_json(data)
            self._lru(self._base, key, base)
        return base

    def get(self, key: str) -> Ledger:
        p = self.r.pipeline()  # MULTI: version and fixes are one consistent snapshot
        p.hget(self._k(key), "version")
        p.hgetall(self._k(key) + ":fix")
        p.expire(self._k(key), self.ttl)  # sliding expiry: in use = kept
        p.expire(self._k(key) + ":fix", self.ttl)
        version, fixes, _, _ = p.execute()
        if version is None:
            raise HTTPException(404, NOT_FOUND)
        version = int(version)
        with self._lock:
            hit = self._view.get(key)
        if hit and hit[0] == version:
            return hit[1]
        base = self._statement(key)
        view = Ledger(base.df.drop(columns=_DERIVED), corrections=dict(base.corrections))
        by_key: dict[str, dict] = {}
        for field_name, value in fixes.items():
            mk, _, attr = field_name.rpartition(_SEP)
            by_key.setdefault(mk, {})[attr] = value
        for mk in sorted(by_key):
            view.apply_correction(mk, merchant=by_key[mk].get("merchant"), category=by_key[mk].get("category"))
        self._lru(self._view, key, (version, view))
        return view

    def correct(self, key: str, merchant_key: str, merchant: str | None = None, category: str | None = None) -> int:
        """Validate against the statement, then record the fix and bump the version atomically."""
        base = self._statement(key)
        if merchant is None and category is None:
            raise ValueError("give a merchant name, a category, or both")
        if category is not None and category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        n = int((base.df["merchant_key"] == merchant_key).sum())
        if n == 0:
            raise KeyError(merchant_key)
        fields = {f"{merchant_key}{_SEP}{attr}": v for attr, v in (("merchant", merchant), ("category", category))
                  if v is not None}
        p = self.r.pipeline()
        p.hset(self._k(key) + ":fix", mapping=fields)
        p.hincrby(self._k(key), "version", 1)
        p.expire(self._k(key) + ":fix", self.ttl)
        p.expire(self._k(key), self.ttl)
        p.execute()
        return n


# ---------------------------------------------------------------- rate limits

_SLIDING_WINDOW = """
local key, now, window, limit, member = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
if redis.call('ZCARD', key) >= limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  return {0, oldest[2]}
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.ceil(window * 1000))
return {1, '0'}
"""


class RedisRateLimiter:
    """Same contract as guards.RateLimiter, shared by all workers. Fails open if Redis is down."""

    def __init__(self, r, name: str, limit: int, window: float = 60.0, clock=time.time):
        self.r, self.name, self.limit, self.window, self.clock = r, name, limit, window, clock
        self._script = r.register_script(_SLIDING_WINDOW)

    def check(self, key: str) -> None:
        import redis

        now = self.clock()
        try:
            ok, oldest = self._script(keys=[f"ll:rl:{self.name}:{key}"],
                                      args=[now, self.window, self.limit, f"{now}:{uuid.uuid4().hex}"])
        except redis.RedisError as e:  # availability over strictness for a rate limit
            log.warning("rate limiter unavailable, allowing request: %s", e)
            return
        if int(ok) == 0:
            retry = int(self.window - (now - float(oldest))) + 1
            raise HTTPException(429, f"too many requests; try again in {retry}s", headers={"Retry-After": str(retry)})


# ---------------------------------------------------------------- spend cap

class RedisDailyBudget:
    """Same contract as guards.DailyBudget, shared by all workers. Fails CLOSED if Redis is down."""

    def __init__(self, r, usd_per_day: float, today=lambda: datetime.now(UTC).date()):
        self.r, self.limit, self.today = r, usd_per_day, today

    def _k(self) -> str:
        return f"ll:budget:{self.today().isoformat()}"

    def check(self) -> None:
        import redis

        try:
            spent = float(self.r.get(self._k()) or 0)
        except redis.RedisError as e:
            log.warning("budget store unavailable, refusing paid call: %s", e)
            raise HTTPException(503, "can't verify today's AI budget right now; try again shortly") from e
        if spent >= self.limit:
            raise HTTPException(503, "the demo's daily AI budget is used up; it resets at midnight UTC")

    def spend(self, usd: float) -> None:
        p = self.r.pipeline()
        p.incrbyfloat(self._k(), max(0.0, usd))
        p.expire(self._k(), 2 * 24 * 3600)
        p.execute()

    @property
    def spent_today(self) -> float:
        return round(float(self.r.get(self._k()) or 0), 4)


# ---------------------------------------------------------------- LLM answer cache

class RedisLLMCache(MutableMapping):
    """merchant key -> Prediction, shared by all workers (one LLM call per merchant, fleet-wide)."""

    def __init__(self, r, model: str):
        self.r, self.k = r, f"ll:llmcache:{model}"

    def __getitem__(self, key: str) -> Prediction:
        v = self.r.hget(self.k, key)
        if v is None:
            raise KeyError(key)
        return Prediction(**json.loads(v))

    def __setitem__(self, key: str, value: Prediction) -> None:
        self.r.hset(self.k, key, json.dumps({"merchant": value.merchant, "category": value.category}))

    def __delitem__(self, key: str) -> None:
        self.r.hdel(self.k, key)

    def __contains__(self, key: object) -> bool:
        return bool(self.r.hexists(self.k, key))

    def __iter__(self):
        return iter(self.r.hkeys(self.k))

    def __len__(self) -> int:
        return int(self.r.hlen(self.k))


# ---------------------------------------------------------------- wiring

@dataclass
class State:
    backend: str
    datasets: Any
    budget: Any
    redis: Any = None

    def limiter(self, name: str, limit: int):
        return RedisRateLimiter(self.redis, name, limit) if self.redis is not None else RateLimiter(limit)

    def llm_cache(self, model: str) -> MutableMapping:
        return RedisLLMCache(self.redis, model) if self.redis is not None else {}

    def ping(self) -> bool:
        if self.redis is None:
            return True
        try:
            return bool(self.redis.ping())
        except Exception:
            return False


def build_state(redis_url: str | None, budget_usd: float, client=None) -> State:
    """``client`` lets tests pass a fakeredis instance."""
    if not redis_url and client is None:
        return State("memory", MemoryDatasets(), DailyBudget(budget_usd))
    if client is None:
        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2,
                                      health_check_interval=30)
    return State("redis", RedisDatasets(client), RedisDailyBudget(client, budget_usd), client)
