import json
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ledgerlens import cost, llm_eval, routing, train
from ledgerlens.llm import PRICES, RESPONSE_SCHEMA, LLMClassifier, Txn
from ledgerlens.pipeline import Enricher, apply_llm_fallback
from ledgerlens.synth import Simulator, write_csv
from ledgerlens.taxonomy import CATEGORIES


class FakeMessages:
    """Shaped like client.beta.messages: records calls, answers with a fixed category."""

    def __init__(self, category="HOUSING", refuse_on_call=None, model="claude-opus-5-5"):
        self.calls, self.category, self.refuse_on_call, self.model = [], category, refuse_on_call, model

    def create(self, **kw):
        self.calls.append(kw)
        n = sum(1 for line in kw["messages"][0]["content"].splitlines() if line[:1].isdigit())
        usage = types.SimpleNamespace(input_tokens=100 * n, output_tokens=50 * n)
        if self.refuse_on_call == len(self.calls):
            return types.SimpleNamespace(model=self.model, stop_reason="refusal", usage=usage, content=[])
        results = [{"id": i, "merchant": f"M{i}", "category": self.category} for i in range(n)]
        results.append({"id": 99, "merchant": "out of range", "category": "DINING"})  # must be ignored
        return types.SimpleNamespace(model=self.model, stop_reason="end_turn", usage=usage,
                                     content=[types.SimpleNamespace(type="text", text=json.dumps({"results": results}))])


def fake_client(**kw):
    return types.SimpleNamespace(beta=types.SimpleNamespace(messages=FakeMessages(**kw)))


def test_llm_batches_and_request_shape():
    client = fake_client()
    clf = LLMClassifier(client, batch_size=2)
    preds = clf.classify([Txn(f"DESC {i}", -float(i), hint="DINING") for i in range(5)])
    calls = client.beta.messages.calls
    assert len(calls) == 3
    assert [p.category for p in preds] == ["HOUSING"] * 5
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["schema"] == RESPONSE_SCHEMA
    assert "model hint: DINING" in kw["messages"][0]["content"]
    assert clf.usage.requests == 3 and clf.usage.input_tokens == 500
    pin, pout = PRICES["claude-opus-5-5"]
    assert clf.usage.cost_usd() == pytest.approx(500 / 1e6 * pin + 250 / 1e6 * pout)


def test_llm_refusal_keeps_rows_unanswered():
    clf = LLMClassifier(fake_client(refuse_on_call=1), batch_size=2)
    preds = clf.classify([Txn("A", -1), Txn("B", -2), Txn("C", -3)])
    assert preds[:2] == [None, None] and preds[2].category == "HOUSING"
    assert clf.usage.refusals == 1


def test_schema_enum_matches_taxonomy():
    enum = RESPONSE_SCHEMA["properties"]["results"]["items"]["properties"]["category"]["enum"]
    assert enum == list(CATEGORIES)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w3")
    data = tmp / "s.csv"
    write_csv(Simulator(13, 12, 15).run(), data)
    _, artifacts = train.run_seed(train.load(data), seed=0, hide_frac=0.8, history_dropout=0.2, log=lambda *_: None)
    path = tmp / "e.joblib"
    joblib.dump(artifacts, path)
    return path


def test_apply_llm_fallback_routes_only_low_confidence(model_path):
    enriched = Enricher.load(model_path).enrich(pd.DataFrame({
        "description": ["NETFLIX.COM 866-579-7172 CA", "SQ *MYSTERY PLACE OAKLAND CA"],
        "amount": [-15.49, -12.0], "date": ["2026-03-01", "2026-03-02"]}))
    enriched.loc[0, "confidence"], enriched.loc[1, "confidence"] = 0.99, 0.30
    client = fake_client(category="ENTERTAINMENT")
    out = apply_llm_fallback(enriched, LLMClassifier(client), threshold=0.8)
    assert list(out["source"]) == ["model", "llm"]
    assert out.loc[1, "category"] == "ENTERTAINMENT"
    assert out.loc[0, "merchant"] == "Netflix"  # directory merchant never overwritten
    assert out.loc[1, "merchant"] == "M0"  # not in directory: LLM name used
    assert "MYSTERY" in client.beta.messages.calls[0]["messages"][0]["content"]
    assert "NETFLIX" not in client.beta.messages.calls[0]["messages"][0]["content"]


def test_service(model_path, monkeypatch):
    monkeypatch.setenv("LEDGERLENS_MODEL", str(model_path))
    monkeypatch.delenv("LEDGERLENS_LLM", raising=False)
    from ledgerlens.service import app

    with TestClient(app) as c:
        h = c.get("/healthz").json()
        assert h["ok"] and h["llm_fallback_available"] is False
        r = c.post("/v1/enrich", json={"transactions": [
            {"id": "a", "description": "NETFLIX.COM 866-579-7172 CA", "amount": -15.49, "date": "2026-03-02"},
            {"description": "TST* MISSION LOTUS OAKLAND CA", "amount": -23.1, "date": "03/04/2026", "user_id": "u2"},
        ], "llm_fallback": True})  # requested but unavailable -> model only
        assert r.status_code == 200
        body = r.json()
        assert [x["id"] for x in body["results"]] == ["a", None]
        assert body["results"][0]["merchant"] == "Netflix"
        assert body["results"][1]["processor"] == "TOAST"
        assert {x["source"] for x in body["results"]} == {"model"} and body["llm_routed"] == 0
        assert c.post("/v1/enrich", json={"transactions": []}).status_code == 422
        assert c.post("/v1/enrich", json={"transactions": [{"description": "", "amount": 1, "date": "x"}]}
                      ).status_code == 422


def test_coverage_curve_and_calibration():
    conf = np.array([0.95, 0.9, 0.6, 0.4])
    correct = np.array([True, True, False, False])
    curve = {c["threshold"]: c for c in routing.coverage_curve(conf, correct)}
    assert curve[0.0]["routed_to_llm"] == 0 and curve[0.0]["model_acc_on_kept"] == 0.5
    assert curve[0.7]["routed_to_llm"] == 0.5 and curve[0.7]["model_acc_on_kept"] == 1.0
    assert curve[0.7]["hybrid_upper_bound"] == 1.0
    cal = routing.calibration(conf, correct)
    assert sum(r["n"] for r in cal) == 4 and 0 <= routing.ece(cal) <= 1


def test_cost_helpers(tmp_path):
    stats = tmp_path / "s.csv"
    stats.write_text("Type,Name,Request Count,Requests/s\nPOST,/v1/enrich,10,5.0\n,Aggregated,10,5.5\n")
    assert cost.measured_throughput(stats, 100) == 550.0
    tin, tout, measured = cost.tokens_per_txn(tmp_path, "claude-opus-5-5")
    assert not measured and tin > 30 and tout == llm_eval.EST_OUTPUT_TOKENS_PER_TXN
    (tmp_path / "week3_llm_claude-opus-5-5.json").write_text(json.dumps({"tokens_per_txn": {"input": 40, "output": 20}}))
    assert cost.tokens_per_txn(tmp_path, "claude-opus-5-5") == (40, 20, True)
    assert llm_eval.estimate_cost(400, "claude-haiku-5-5", 25) < llm_eval.estimate_cost(400, "claude-opus-5-5", 25)
