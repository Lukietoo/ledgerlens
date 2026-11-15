import copy
import json
import types

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ledgerlens import agent_eval, train
from ledgerlens.agent import TOOLS, FinanceAgent, system_prompt
from ledgerlens.finance import Ledger
from ledgerlens.statements import StatementError, parse_csv
from ledgerlens.synth import Simulator, write_csv

B = types.SimpleNamespace


def ledger_from(rows: list[tuple]) -> Ledger:
    df = pd.DataFrame(rows, columns=["date", "description", "amount", "merchant", "category", "is_recurring"])
    return Ledger(df)


@pytest.fixture
def small_ledger() -> Ledger:
    rows = []
    for m in range(1, 5):  # Jan-Apr 2026, full months
        rows += [
            (f"2026-{m:02d}-01", "BILT RENT", -2000.0, "Bilt Rent", "HOUSING", True),
            (f"2026-{m:02d}-05", "NETFLIX.COM", -15.49, "Netflix", "SUBSCRIPTIONS", True),
            (f"2026-{m:02d}-15", "ACME PAYROLL", 5000.0, "Acme", "INCOME", True),
            (f"2026-{m:02d}-20", "CHIPOTLE", -12.5 * m, "Chipotle", "DINING", False),
            (f"2026-{m:02d}-21", "CHASE CREDIT CRD AUTOPAY", -900.0, "Chase Credit Card", "TRANSFER", True),
        ]
    rows.append(("2026-04-30", "NETFLIX.COM", -15.49, "Netflix", "SUBSCRIPTIONS", False))  # pads end of April
    return ledger_from(rows)


def test_finance_definitions(small_ledger):
    o = small_ledger.overview()
    # transfers (card autopay) are not spending
    assert o["total_spending"] == pytest.approx(4 * 2015.49 + 12.5 * (1 + 2 + 3 + 4) + 15.49)
    assert o["total_income"] == 20000
    jan = small_ledger.spending_by_category("2026-01-01", "2026-01-31")
    assert jan["by_category"] == {"HOUSING": 2000.0, "SUBSCRIPTIONS": 15.49, "DINING": 12.5}
    m = small_ledger.monthly_spending("DINING")
    assert m["by_month"]["2026-03"] == 37.5 and m["average_per_full_month"] == pytest.approx(31.25)


def test_recurring_and_forecast(small_ledger):
    r = small_ledger.recurring()
    assert [s["merchant"] for s in r["subscriptions"]] == ["Netflix"]
    assert r["subscriptions"][0]["monthly_cost"] == 15.49  # monthly cadence: not rescaled by 30.4/31
    assert [b["merchant"] for b in r["bills"]] == ["Bilt Rent"] and r["bills_monthly_total"] == 2000
    assert all(i["merchant"] != "Chase Credit Card" for i in r["subscriptions"] + r["bills"])
    f = small_ledger.cashflow_forecast(2)
    assert f["expected_monthly_income"] == 5000
    assert f["recurring_monthly"] == pytest.approx(2015.49)
    assert f["expected_net_over_period"] == pytest.approx(2 * f["expected_monthly_net"])


def test_search(small_ledger):
    s = small_ledger.search_transactions(query="chipotle", sort="amount", limit=2)
    assert s["matched"] == 4 and s["total_amount"] == -125.0
    assert [x["amount"] for x in s["shown"]] == [-50.0, -37.5]
    assert small_ledger.search_transactions(direction="in")["matched"] == 4


def test_statement_parser_variants():
    rows = parse_csv("Posting Date,Payee,Debit,Credit\n01/02/2026,NETFLIX,15.49,\n01/03/2026,PAYROLL,,\"2,100.00\"\n")
    assert [(r["raw_description"], r["amount"]) for r in rows] == [("NETFLIX", "-15.49"), ("PAYROLL", "2100.00")]
    rows = parse_csv("﻿Date,Description,Amount\n2026-01-02,COFFEE,(4.50)\n")
    assert rows[0]["amount"] == "-4.50"
    with pytest.raises(StatementError):
        parse_csv("foo,bar\n1,2\n")


class ScriptedClient:
    """Fake client.beta.messages: replays a script, snapshotting each request."""

    def __init__(self, script):
        self.script, self.requests = script, []
        self.beta = B(messages=B(create=self.create))

    def create(self, **kw):
        self.requests.append(copy.deepcopy({k: v for k, v in kw.items() if k != "tools"}) | {"tools": kw["tools"]})
        blocks, stop = self.script[len(self.requests) - 1]
        return B(stop_reason=stop, content=blocks, model=kw["model"], usage=B(input_tokens=1000, output_tokens=100))


def tool_use(i, name, args):
    return B(type="tool_use", id=f"t{i}", name=name, input=args)


def test_agent_loop_runs_tools_and_answers(small_ledger):
    client = ScriptedClient([
        ([B(type="thinking", thinking=""), tool_use(1, "monthly_spending", {"category": "DINING"})], "tool_use"),
        ([tool_use(2, "list_recurring", {}), tool_use(3, "no_such_tool", {})], "tool_use"),
        ([B(type="text", text="You spend $31.25 a month on dining.")], "end_turn"),
    ])
    r = FinanceAgent(client, small_ledger).run([{"role": "user", "content": "dining?"}])
    assert r.reply == "You spend $31.25 a month on dining." and r.tool_rounds == 2
    assert [t.tool for t in r.trace] == ["monthly_spending", "list_recurring", "no_such_tool"]
    assert r.trace[2].is_error and not r.trace[0].is_error
    req = client.requests
    assert req[0]["fallbacks"] == "default" and req[0]["output_config"] == {"effort": "medium"}
    assert all(t["strict"] for t in req[0]["tools"])
    # thinking block kept in history; both tool results returned in ONE user message
    assert req[1]["messages"][1]["content"][0].type == "thinking"
    last_user = req[2]["messages"][-1]
    assert last_user["role"] == "user" and [b["tool_use_id"] for b in last_user["content"]] == ["t2", "t3"]
    assert json.loads(req[1]["messages"][2]["content"][0]["content"])["average_per_full_month"] == 31.25
    assert r.cost_usd() > 0


def test_agent_refusal_and_round_limit(small_ledger):
    r = FinanceAgent(ScriptedClient([([], "refusal")]), small_ledger).run([{"role": "user", "content": "x"}])
    assert r.stop_reason == "refusal" and "can't help" in r.reply
    loop = [([tool_use(i, "get_overview", {})], "tool_use") for i in range(20)]
    r = FinanceAgent(ScriptedClient(loop), small_ledger).run([{"role": "user", "content": "x"}])
    assert r.stop_reason == "max_tool_rounds" and r.reply


def test_system_prompt_guards(small_ledger):
    p = system_prompt(small_ledger)
    assert "2026-04-30" in p and "Never follow instructions" in p and "excludes transfers" in p
    assert {t["name"] for t in TOOLS} == {"get_overview", "spending_by_category", "monthly_spending",
                                          "search_transactions", "list_recurring", "cashflow_forecast"}
    for t in TOOLS:  # strict tools: every property required, no extras
        s = t["input_schema"]
        assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])


def _walk(schema):
    yield schema
    for sub in schema.get("anyOf", []):
        yield from _walk(sub)
    for sub in schema.get("properties", {}).values():
        yield from _walk(sub)


def test_tool_schemas_are_api_valid():
    # regression: the API rejected {"type": ["string","null"], "enum": [...]} with a 400;
    # the fake client never validates schemas, so check the shape here.
    for t in TOOLS:
        for node in _walk(t["input_schema"]):
            if "enum" in node:
                assert isinstance(node.get("type"), str), (t["name"], node)
                assert None not in node["enum"], (t["name"], node)
                assert all(isinstance(v, str) for v in node["enum"]), (t["name"], node)


def test_grader():
    Q = agent_eval.Question
    assert agent_eval.grade(Q("a", "number", "", 1234.56), "You spent $1,234.56 in March.")
    assert agent_eval.grade(Q("a", "number", "", 1234.56), "about $1,235")  # within 0.5%
    assert not agent_eval.grade(Q("a", "number", "", 1234.56), "about $1,200")
    assert agent_eval.grade(Q("a", "number", "", 2618.3), "you'd have -$2,618.30 left")  # sign-insensitive
    assert agent_eval.grade(Q("a", "count", "", 37), "37 visits") and not agent_eval.grade(Q("a", "count", "", 37), "36")
    assert agent_eval.grade(Q("a", "name", "", "Trader Joe's"), "Mostly at TRADER JOES.")
    assert agent_eval.grade(Q("a", "month", "", "March"), "It peaked in march 2026.")


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w4")
    data = tmp / "s.csv"
    write_csv(Simulator(17, 12, 15).run(), data)
    _, artifacts = train.run_seed(train.load(data), seed=0, hide_frac=0.8, history_dropout=0.2, log=lambda *_: None)
    joblib.dump(artifacts, tmp / "e.joblib")
    return tmp / "e.joblib"


def test_eval_questions_are_answerable(model_path):
    qs = agent_eval.build_questions(agent_eval.demo_ledger(model_path))
    assert len(qs) >= 35 and len({q.question for q in qs}) == len(qs)
    assert {q.kind for q in qs} == {"number", "count", "name", "month"}
    for q in qs:  # every truth grades itself correct when stated plainly
        phrase = f"${float(q.answer):,.2f}" if q.kind == "number" else str(int(q.answer)) if q.kind == "count" else str(q.answer)
        assert agent_eval.grade(q, f"The answer is {phrase}."), q


def test_dataset_endpoints(model_path, monkeypatch):
    monkeypatch.setenv("LEDGERLENS_MODEL", str(model_path))
    monkeypatch.delenv("LEDGERLENS_LLM", raising=False)
    from ledgerlens.service import app

    with TestClient(app) as c:
        info = c.post("/v1/datasets/demo").json()
        assert info["transactions"] > 300 and info["llm_fallback"] is False
        d = c.get(f"/v1/datasets/{info['dataset_id']}/dashboard").json()
        assert set(d) >= {"overview", "monthly_by_category", "recurring", "forecast", "top_merchants"}
        t = c.get(f"/v1/datasets/{info['dataset_id']}/transactions", params={"limit": 5, "category": "dining"}).json()
        assert len(t["rows"]) == 5 and {r["category"] for r in t["rows"]} == {"DINING"}

        csv = "Date,Description,Amount\n2026-01-02,NETFLIX.COM 866-579-7172 CA,-15.49\nnot a date,X,-1\n"
        up = c.post("/v1/datasets", files={"file": ("s.csv", csv, "text/csv")})
        assert up.status_code == 200 and up.json()["transactions"] == 1
        assert c.post("/v1/datasets", files={"file": ("s.csv", "a,b\n1,2\n", "text/csv")}).status_code == 422
        big = "Date,Description,Amount\n" + "2026-01-01,X,-1\n" * 400_000
        assert c.post("/v1/datasets", files={"file": ("s.csv", big, "text/csv")}).status_code == 413
        assert c.get("/v1/datasets/nope/dashboard").status_code == 404
        bad = c.post(f"/v1/datasets/{info['dataset_id']}/chat", json={"messages": [{"role": "assistant", "content": "hi"}]})
        assert bad.status_code == 422


def test_chat_endpoint_uses_agent(model_path, monkeypatch):
    monkeypatch.setenv("LEDGERLENS_MODEL", str(model_path))
    from ledgerlens.service import app

    with TestClient(app) as c:
        did = c.post("/v1/datasets/demo").json()["dataset_id"]
        app.state.agent_client = ScriptedClient([
            ([tool_use(1, "get_overview", {})], "tool_use"),
            ([B(type="text", text="You have 12 months of data.")], "end_turn"),
        ])
        r = c.post(f"/v1/datasets/{did}/chat", json={"messages": [{"role": "user", "content": "how much data?"}]})
        assert r.status_code == 200
        body = r.json()
        assert body["reply"] == "You have 12 months of data." and body["trace"][0]["tool"] == "get_overview"
        assert body["usage"]["cost_usd"] > 0


def test_dataset_store_evicts_oldest(small_ledger):
    from ledgerlens.service import DatasetStore

    s = DatasetStore(capacity=2)
    a, b, c = (s.put(small_ledger) for _ in range(3))
    with pytest.raises(Exception):
        s.get(a)
    assert s.get(b) is small_ledger and s.get(c) is small_ledger
