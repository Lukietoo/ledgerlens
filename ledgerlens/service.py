"""HTTP API.

    uvicorn ledgerlens.service:app --workers 4

POST /v1/enrich   batch of transactions -> merchant, category, confidence, recurring...
GET  /healthz     liveness + model metadata

LLM fallback is off unless LEDGERLENS_LLM=1, and then only if the Anthropic client
can find credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ledgerlens.llm import DEFAULT_MODEL, LLMClassifier
from ledgerlens.pipeline import DEFAULT_LLM_THRESHOLD, Enricher, RoutingPolicy, apply_llm_fallback
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
    yield


app = FastAPI(title="Ledgerlens", version="0.3.0", lifespan=lifespan)


@app.get("/healthz")
def healthz(request: Request) -> dict:
    s = request.app.state
    return {"ok": True, "llm_fallback_available": s.llm is not None, "llm_threshold": s.threshold,
            "directory_merchants": len(s.enricher.resolver.merchants)}


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
