"""HTTP API.

    uvicorn ledgerlens.service:app --workers 4

POST /v1/enrich                       batch of transactions -> merchant, category, confidence, recurring...
GET  /healthz                         liveness + model metadata
POST /v1/datasets                     upload a bank CSV (held in memory only) -> dataset_id
POST /v1/datasets/demo                load a synthetic demo user
GET  /v1/datasets/{id}/dashboard      totals, monthly by category, recurring, forecast
GET  /v1/datasets/{id}/transactions   enriched feed (search, category filter, paging)
POST /v1/datasets/{id}/chat           finance agent (Claude + tools); needs credentials

LLM fallback is off unless LEDGERLENS_LLM=1, and then only if the Anthropic client
can find credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ledgerlens.agent import DEFAULT_AGENT_MODEL, FinanceAgent
from ledgerlens.finance import Ledger
from ledgerlens.guards import DailyBudget, RateLimiter, client_id
from ledgerlens.llm import DEFAULT_MODEL, LLMClassifier
from ledgerlens.normalize import canon
from ledgerlens.pipeline import DEFAULT_LLM_THRESHOLD, Enricher, RoutingPolicy, apply_llm_fallback
from ledgerlens.statements import StatementError, parse_csv
from ledgerlens.taxonomy import CATEGORIES

log = logging.getLogger("ledgerlens")
MAX_BATCH = 5000


class TxnIn(BaseModel):
    id: str | None = None
    description: str = Field(min_length=1, max_length=512)
    amount: float
    date: str = Field(description="ISO date, or any format pandas can parse")
    user_id: str = "default"


class EnrichRequest(BaseModel):
    transactions: list[TxnIn] = Field(min_length=1, max_length=MAX_BATCH)
    llm_fallback: bool = False


class TxnOut(BaseModel):
    id: str | None
    merchant: str
    merchant_in_directory: bool
    category: Literal[CATEGORIES]  # type: ignore[valid-type]
    confidence: float
    is_recurring: bool
    processor: str | None
    source: Literal["model", "llm"]


class EnrichResponse(BaseModel):
    results: list[TxnOut]
    model_ms: float
    llm_ms: float = 0.0
    llm_routed: int = 0


def _make_llm() -> LLMClassifier | None:
    if os.environ.get("LEDGERLENS_LLM") != "1":
        return None
    try:
        import anthropic

        return LLMClassifier(anthropic.Anthropic(), model=os.environ.get("LEDGERLENS_LLM_MODEL", DEFAULT_MODEL))
    except Exception as e:  # no credentials, etc.: serve model-only rather than fail
        log.warning("LLM fallback disabled: %s", e)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.enricher = Enricher.load(Path(os.environ.get("LEDGERLENS_MODEL", "models/enricher.joblib")))
    app.state.llm = _make_llm()
    app.state.threshold = float(os.environ.get("LEDGERLENS_LLM_THRESHOLD", DEFAULT_LLM_THRESHOLD))
    app.state.datasets = DatasetStore()
    app.state.agent_client = None
    # public-deploy guards (see guards.py); generous defaults for local use
    app.state.limits = {name: RateLimiter(int(os.environ.get(f"LEDGERLENS_{name.upper()}_PER_MIN", default)))
                        for name, default in (("chat", 8), ("upload", 5), ("demo", 20))}
    app.state.budget = DailyBudget(float(os.environ.get("LEDGERLENS_DAILY_LLM_BUDGET_USD", "2.00")))
    app.state.allow_uploads = os.environ.get("LEDGERLENS_ALLOW_UPLOADS", "1") != "0"
    yield


app = FastAPI(title="Ledgerlens", version="0.4.0", lifespan=lifespan)


@app.get("/healthz")
def healthz(request: Request) -> dict:
    s = request.app.state
    return {"ok": True, "llm_fallback_available": s.llm is not None, "llm_threshold": s.threshold,
            "directory_merchants": len(s.enricher.resolver.merchants), "uploads_enabled": s.allow_uploads,
            "llm_spent_today_usd": s.budget.spent_today, "llm_daily_budget_usd": s.budget.limit}


@app.post("/v1/enrich", response_model=EnrichResponse)
def enrich(req: EnrichRequest, request: Request) -> EnrichResponse:
    # sync handler: FastAPI runs it in a threadpool, so CPU work doesn't block the event loop
    s = request.app.state
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    t0 = time.perf_counter()
    out = s.enricher.enrich(df[["description", "amount", "date", "user_id"]])
    model_ms = (time.perf_counter() - t0) * 1000

    llm_ms, routed = 0.0, 0
    if req.llm_fallback and s.llm is not None:
        t1 = time.perf_counter()
        # names-only policy won the eval (results/week3_llm_*.md): route low-confidence rows AND rows
        # whose merchant isn't in the directory; take the LLM's category only on low-confidence rows.
        out = apply_llm_fallback(out, s.llm, RoutingPolicy(threshold=s.threshold))
        llm_ms = (time.perf_counter() - t1) * 1000
        routed = int((out["source"] == "llm").sum())
    else:
        out["source"] = "model"

    results = [
        TxnOut(id=i, merchant=r.merchant, merchant_in_directory=bool(r.merchant_in_directory), category=r.category,
               confidence=float(r.confidence), is_recurring=bool(r.is_recurring),
               processor=r.processor if isinstance(r.processor, str) else None, source=r.source)
        # ids from the request, not the DataFrame: a mix of present/absent ids becomes NaN in pandas
        for i, r in zip((t.id for t in req.transactions), out.itertuples(index=False))
    ]
    return EnrichResponse(results=results, model_ms=round(model_ms, 2), llm_ms=round(llm_ms, 2), llm_routed=routed)


# ---------------------------------------------------------------- week 4: datasets + agent
#
# A "dataset" is one person's statement, enriched and held IN MEMORY ONLY (never written
# to disk or logged). The store is bounded: the oldest datasets are evicted first.

MAX_DATASETS = 32
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 20_000
DEMO_SEED = 2026


class DatasetStore:
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
                raise HTTPException(404, "dataset not found (it may have expired); upload it again")
            self._items.move_to_end(key)
            return self._items[key]


def _enrich_ledger(s, raw: pd.DataFrame) -> Ledger:
    out = s.enricher.enrich(raw[["description", "amount", "date"]])
    if s.llm is not None and _within_budget(s):
        before = s.llm.usage.cost_usd()
        out = apply_llm_fallback(out, s.llm, RoutingPolicy(threshold=s.threshold))
        s.budget.spend(s.llm.usage.cost_usd() - before)
    else:
        out["source"] = "model"
    return Ledger(pd.concat([out, raw[["date"]].reset_index(drop=True)], axis=1))


def _within_budget(s) -> bool:
    """LLM fallback during enrichment is optional: past the daily cap, serve model-only instead of failing."""
    try:
        s.budget.check()
        return True
    except HTTPException:
        return False


def _demo_frame() -> pd.DataFrame:
    """One synthetic user, 12 months. Deterministic, so the demo video is reproducible."""
    from ledgerlens.synth import Simulator

    txns = Simulator(seed=DEMO_SEED, n_users=1, locals_per_category=40).simulate_user(0)
    return pd.DataFrame([{"date": t.date.isoformat(), "description": t.description, "amount": t.amount}
                         for t in txns])


class DatasetInfo(BaseModel):
    dataset_id: str
    transactions: int
    first_date: str
    last_date: str
    llm_fallback: bool


def _info(key: str, ledger: Ledger, s) -> DatasetInfo:
    o = ledger.overview()
    return DatasetInfo(dataset_id=key, transactions=o["transactions"], first_date=o["first_date"],
                       last_date=o["last_date"], llm_fallback=s.llm is not None)


@app.post("/v1/datasets/demo", response_model=DatasetInfo)
def create_demo(request: Request) -> DatasetInfo:
    s = request.app.state
    s.limits["demo"].check(client_id(request))
    if not hasattr(s, "datasets"):
        s.datasets = DatasetStore()
    ledger = _enrich_ledger(s, _demo_frame())
    return _info(s.datasets.put(ledger), ledger, s)


@app.post("/v1/datasets", response_model=DatasetInfo)
async def upload_statement(request: Request, file: UploadFile) -> DatasetInfo:
    s = request.app.state
    if not s.allow_uploads:
        raise HTTPException(403, "uploads are turned off on this demo server; try the demo account, or run Ledgerlens locally")
    s.limits["upload"].check(client_id(request))
    if not hasattr(s, "datasets"):
        s.datasets = DatasetStore()
    body = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB)")
    try:
        rows = parse_csv(body.decode("utf-8-sig", errors="replace"), source="upload")
    except StatementError as e:
        raise HTTPException(422, str(e)) from e
    raw = pd.DataFrame(rows).rename(columns={"raw_description": "description"})
    if raw.empty:
        raise HTTPException(422, "no transactions found in the file")
    raw = raw[raw["amount"] != ""].head(MAX_UPLOAD_ROWS)
    raw["amount"] = raw["amount"].astype(float)
    raw["date"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    raw = raw.dropna(subset=["date"])
    if raw.empty:
        raise HTTPException(422, "no rows with a parseable date and amount")
    raw["date"] = raw["date"].dt.date.astype(str)
    ledger = await run_in_threadpool(_enrich_ledger, s, raw.reset_index(drop=True))
    return _info(s.datasets.put(ledger), ledger, s)


def _ledger(request: Request, dataset_id: str) -> Ledger:
    s = request.app.state
    if not hasattr(s, "datasets"):
        raise HTTPException(404, "dataset not found")
    return s.datasets.get(dataset_id)


@app.get("/v1/datasets/{dataset_id}/dashboard")
def dashboard(dataset_id: str, request: Request) -> dict:
    return _ledger(request, dataset_id).dashboard()


@app.get("/v1/datasets/{dataset_id}/transactions")
def transactions(dataset_id: str, request: Request, q: str | None = None, category: str | None = None,
                 limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict:
    d = _ledger(request, dataset_id).df
    if category:
        d = d[d["category"] == category.upper()]
    if q:
        key = canon(q)
        d = d[d["merchant"].map(canon).str.contains(key, regex=False)
              | d["description"].map(canon).str.contains(key, regex=False)]
    d = d.sort_values("date", ascending=False)
    page = d.iloc[offset:offset + limit]
    cols = ["date", "description", "merchant", "category", "amount", "confidence", "source",
            "merchant_in_directory", "is_recurring"]
    page = page[[c for c in cols if c in page.columns]].copy()
    page["date"] = page["date"].dt.date.astype(str)
    return {"total": int(len(d)), "rows": page.to_dict(orient="records")}


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


def _agent_client(request: Request):
    s = request.app.state
    if getattr(s, "agent_client", None) is None:
        import anthropic

        s.agent_client = anthropic.Anthropic()
    return s.agent_client


@app.post("/v1/datasets/{dataset_id}/chat")
def chat(dataset_id: str, req: ChatRequest, request: Request) -> dict:
    import anthropic

    ledger = _ledger(request, dataset_id)
    s = request.app.state
    s.limits["chat"].check(client_id(request))
    s.budget.check()
    if req.messages[-1].role != "user":
        raise HTTPException(422, "the last message must be from the user")
    agent = FinanceAgent(_agent_client(request), ledger,
                         model=os.environ.get("LEDGERLENS_AGENT_MODEL", DEFAULT_AGENT_MODEL))
    no_creds = "chat needs Anthropic credentials on the server (ANTHROPIC_API_KEY)"
    try:
        r = agent.run([m.model_dump() for m in req.messages])
    except anthropic.AuthenticationError as e:
        raise HTTPException(503, no_creds) from e
    except TypeError as e:  # the SDK raises this before sending when no credentials resolve at all
        if "authentication method" in str(e):
            raise HTTPException(503, no_creds) from e
        raise
    except anthropic.RateLimitError as e:
        raise HTTPException(429, "rate limited by the model provider; try again shortly") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(502, "couldn't reach the model provider") from e
    except anthropic.APIStatusError as e:
        log.warning("agent API error %s (request id %s)", e.status_code, getattr(e, "request_id", None))
        raise HTTPException(502, f"model provider error ({e.status_code})") from e
    s.budget.spend(r.cost_usd())
    return {"reply": r.reply, "stop_reason": r.stop_reason, "model": r.model,
            "trace": [t.__dict__ for t in r.trace], "seconds": r.seconds,
            "usage": {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                      "cost_usd": round(r.cost_usd(), 5)}}
