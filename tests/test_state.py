import json
import threading
import types
from datetime import date

import fakeredis
import pandas as pd
import pytest
import redis
from fastapi import HTTPException

from ledgerlens.finance import Ledger
from ledgerlens.llm import LLMClassifier, Prediction, Txn
from ledgerlens.state import (MemoryDatasets, RedisDailyBudget, RedisDatasets, RedisLLMCache, RedisRateLimiter,
                              build_state, ledger_from_json, ledger_to_json)


def ledger(n_merchants: int = 3) -> Ledger:
    rows = []
    for m in range(1, 4):
        for k in range(n_merchants):
            rows.append((f"2026-0{m}-{k + 1:02d}", f"SHOP{k} OAKLAND CA", -10.0 - k, f"Shop {k}", "SHOPPING",
                         f"SHOP{k}|out", 0.5, False, False, "model"))
        rows.append((f"2026-0{m}-15", "ACME PAYROLL", 3000.0, "Acme", "INCOME", "ACME|in", 0.99, True, True, "model"))
    return Ledger(pd.DataFrame(rows, columns=["date", "description", "amount", "merchant", "category", "merchant_key",
                                              "confidence", "merchant_in_directory", "is_recurring", "source"]))


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def client(server):
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture(params=["memory", "redis"])
def datasets(request, server):
    return MemoryDatasets() if request.param == "memory" else RedisDatasets(client(server))


def test_dataset_contract(datasets):
    key = datasets.put(ledger())
    assert datasets.get(key).overview()["transactions"] == 12
    assert datasets.correct(key, "SHOP0|out", category="DINING") == 3
    got = datasets.get(key)
    assert set(got.df.loc[got.df["merchant_key"] == "SHOP0|out", "category"]) == {"DINING"}
    assert got.corrections["SHOP0|out"]["category"] == "DINING"
    with pytest.raises(HTTPException) as e:
        datasets.get("nope")
    assert e.value.status_code == 404


def test_json_roundtrip_is_faithful():
    L = ledger()
    L.apply_correction("SHOP1|out", merchant="Fixed", category="TRANSFER")
    back = ledger_from_json(ledger_to_json(L))
    assert back.dashboard() == L.dashboard()  # same totals, chart, recurring, forecast
    assert back.corrections == L.corrections
    assert "pickle" not in ledger_to_json(L) and json.loads(ledger_to_json(L))  # plain JSON


def test_two_workers_share_datasets_and_see_each_others_fixes(server):
    a, b = RedisDatasets(client(server)), RedisDatasets(client(server))  # two processes' views
    key = a.put(ledger())
    assert b.get(key).overview()["transactions"] == 12  # created on A, read on B
    a.get(key)  # A now holds a local decoded copy (version 1)
    b.correct(key, "SHOP2|out", category="DINING")
    fresh = a.get(key)  # version bump invalidates A's local copy
    assert set(fresh.df.loc[fresh.df["merchant_key"] == "SHOP2|out", "category"]) == {"DINING"}


def test_concurrent_corrections_lose_nothing(server):
    workers = [RedisDatasets(client(server)) for _ in range(4)]
    key = workers[0].put(ledger(n_merchants=12))
    errors = []

    def fix(i):
        try:
            workers[i % 4].correct(key, f"SHOP{i}|out", merchant=f"Fixed {i}")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=fix, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    final = workers[1].get(key)
    assert len(final.corrections) == 12  # every concurrent write survived
    assert set(final.df.loc[final.df["merchant_key"] != "ACME|in", "source"]) == {"user"}


def test_rate_limit_is_shared_across_workers(server):
    now = [1000.0]
    a = RedisRateLimiter(client(server), "chat", limit=3, clock=lambda: now[0])
    b = RedisRateLimiter(client(server), "chat", limit=3, clock=lambda: now[0])
    a.check("1.2.3.4"), b.check("1.2.3.4"), a.check("1.2.3.4")
    with pytest.raises(HTTPException) as e:
        b.check("1.2.3.4")  # 4th request in the window, on a different worker
    assert e.value.status_code == 429 and int(e.value.headers["Retry-After"]) >= 1
    b.check("5.6.7.8")  # per client
    now[0] += 61
    a.check("1.2.3.4")  # window slid


def test_budget_is_shared_and_resets_daily(server):
    day = [date(2026, 11, 15)]
    a = RedisDailyBudget(client(server), 1.0, today=lambda: day[0])
    b = RedisDailyBudget(client(server), 1.0, today=lambda: day[0])
    a.spend(0.6)
    b.spend(0.5)
    with pytest.raises(HTTPException):
        a.check()
    assert b.spent_today == pytest.approx(1.1)
    day[0] = date(2026, 11, 16)
    a.check()


class Broken:
    """A Redis client whose server is gone."""

    def __getattr__(self, name):
        def fail(*a, **k):
            raise redis.ConnectionError("down")
        if name == "register_script":
            return lambda src: fail
        return fail


def test_failure_modes_open_limiter_closed_budget():
    RedisRateLimiter(Broken(), "chat", limit=1).check("x")  # fails open: keep serving
    with pytest.raises(HTTPException) as e:
        RedisDailyBudget(Broken(), 5.0).check()  # fails closed: never spend money we can't count
    assert e.value.status_code == 503


def test_llm_cache_shared_across_workers(server):
    calls = []

    def create(**kw):
        calls.append(kw)
        n = sum(1 for line in kw["messages"][0]["content"].splitlines() if line[:1].isdigit())
        res = [{"id": i, "merchant": "Blue Bottle Coffee", "category": "COFFEE"} for i in range(n)]
        return types.SimpleNamespace(model=kw["model"], stop_reason="end_turn",
                                     usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
                                     content=[types.SimpleNamespace(type="text", text=json.dumps({"results": res}))])

    fake = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))
    w1 = LLMClassifier(fake, cache=RedisLLMCache(client(server), "m"))
    w2 = LLMClassifier(fake, cache=RedisLLMCache(client(server), "m"))
    w1.classify_keyed({"BLUEBTLCOF|out": Txn("SQ *BLUE BTL COF", -6.5)})
    got = w2.classify_keyed({"BLUEBTLCOF|out": Txn("SQ *BLUE BTL COF", -6.5)})
    assert len(calls) == 1  # the second worker answered from the shared cache
    assert got["BLUEBTLCOF|out"] == Prediction("Blue Bottle Coffee", "COFFEE")
    assert len(RedisLLMCache(client(server), "m")) == 1


def test_build_state_picks_backend(server):
    assert build_state(None, 2.0).backend == "memory"
    st = build_state("redis://unused", 2.0, client=client(server))
    assert st.backend == "redis" and st.ping()
    assert isinstance(st.limiter("chat", 5), RedisRateLimiter)


def test_api_on_redis_survives_a_restart(monkeypatch, server, tmp_path):
    import joblib

    from ledgerlens import train
    from ledgerlens.synth import Simulator, write_csv

    data = tmp_path / "s.csv"
    write_csv(Simulator(31, 10, 15).run(), data)
    _, artifacts = train.run_seed(train.load(data), seed=0, hide_frac=0.8, history_dropout=0.2, log=lambda *_: None)
    joblib.dump(artifacts, tmp_path / "e.joblib")
    monkeypatch.setenv("LEDGERLENS_MODEL", str(tmp_path / "e.joblib"))
    monkeypatch.setenv("LEDGERLENS_REDIS_URL", "redis://fake")
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: client(server)))
    from fastapi.testclient import TestClient

    from ledgerlens.service import app

    with TestClient(app) as c:  # "worker 1"
        h = c.get("/healthz").json()
        assert h["state_backend"] == "redis" and h["state_ok"] is True
        did = c.post("/v1/datasets/demo").json()["dataset_id"]
        key = c.get(f"/v1/datasets/{did}/review", params={"limit": 1}).json()["items"][0]["merchant_key"]
        assert c.post(f"/v1/datasets/{did}/corrections", json={"merchant_key": key, "merchant": "Saved Fix"}).status_code == 200
    with TestClient(app) as c:  # fresh process state: new caches, same Redis
        feed = c.get(f"/v1/datasets/{did}/transactions", params={"q": "saved fix"}).json()
        assert feed["total"] >= 1 and {r["source"] for r in feed["rows"]} == {"user"}



def test_corrections_merge_across_workers_and_restarts(server):
    """Name fixed on one worker, category on another: both stick, in any order, for any reader."""
    a, b = RedisDatasets(client(server)), RedisDatasets(client(server))
    key = a.put(ledger())
    a.correct(key, "SHOP0|out", merchant="Corner Shop")
    b.correct(key, "SHOP0|out", category="GROCERIES")
    b.correct(key, "SHOP0|out", merchant="Corner Market")  # last writer wins, per field
    fresh = RedisDatasets(client(server))  # a worker that just started: no local cache
    for reader in (a, b, fresh):
        d = reader.get(key).df
        rows = d[d["merchant_key"] == "SHOP0|out"]
        assert set(rows["merchant"]) == {"Corner Market"} and set(rows["category"]) == {"GROCERIES"}
        assert set(rows["source"]) == {"user"}
    assert fresh.get(key).corrections["SHOP0|out"] == {"merchant": "Corner Market", "category": "GROCERIES", "rows": 3}


def test_correction_is_validated_and_never_touches_the_stored_statement(server):
    a = RedisDatasets(client(server))
    key = a.put(ledger())
    stored = client(server).hget(f"ll:ds:{key}", "data")
    with pytest.raises(KeyError):
        a.correct(key, "NOPE|out", merchant="x")
    with pytest.raises(ValueError):
        a.correct(key, "SHOP0|out", category="NOT_A_CATEGORY")
    a.correct(key, "SHOP0|out", merchant="Fixed")
    assert client(server).hget(f"ll:ds:{key}", "data") == stored  # O(1) write: the statement isn't rewritten
    assert client(server).hget(f"ll:ds:{key}", "version") == "2"
