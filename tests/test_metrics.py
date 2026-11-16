import joblib
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from ledgerlens import train
from ledgerlens.synth import Simulator, write_csv


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("metrics")
    write_csv(Simulator(23, 10, 15).run(), tmp / "s.csv")
    _, artifacts = train.run_seed(train.load(tmp / "s.csv"), seed=0, hide_frac=0.8, history_dropout=0.2,
                                  log=lambda *_: None)
    joblib.dump(artifacts, tmp / "e.joblib")
    return tmp / "e.joblib"


def samples(text: str) -> dict:
    out = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def value(snap: dict, name: str, **labels) -> float:
    return snap.get((name, tuple(sorted(labels.items()))), 0.0)


def test_metrics_count_requests_by_route_template(model_path, monkeypatch):
    monkeypatch.setenv("LEDGERLENS_MODEL", str(model_path))
    monkeypatch.delenv("LEDGERLENS_LLM", raising=False)
    monkeypatch.delenv("LEDGERLENS_REDIS_URL", raising=False)
    from ledgerlens.service import app

    with TestClient(app) as c:
        before = samples(c.get("/metrics").text)
        did = c.post("/v1/datasets/demo").json()["dataset_id"]
        for _ in range(3):
            assert c.get(f"/v1/datasets/{did}/dashboard").status_code == 200
        c.get("/v1/datasets/does-not-exist/dashboard")
        c.post("/v1/enrich", json={"transactions": [{"description": "SQ *BLUE BTL COF", "amount": -6.5,
                                                     "date": "2026-03-01"}]})
        assert c.get("/readyz").json() == {"ready": True, "state_ok": True}
        after = samples(c.get("/metrics").text)

    route = "/v1/datasets/{dataset_id}/dashboard"  # the template, never the dataset id
    d = lambda name, **kw: value(after, name, **kw) - value(before, name, **kw)  # noqa: E731
    assert d("ledgerlens_http_requests_total", method="GET", route=route, status="200") == 3
    assert d("ledgerlens_http_requests_total", method="GET", route=route, status="404") == 1
    assert d("ledgerlens_http_request_duration_seconds_count", method="GET", route=route) == 4
    assert d("ledgerlens_transactions_enriched_total", path="fast") == 1
    assert not any(did in str(k) for k in after)  # no raw ids in any label
    assert not any(dict(k[1]).get("route") == "/metrics" for k in after)  # scrapes don't count themselves
