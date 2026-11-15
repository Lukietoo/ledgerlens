from datetime import date

import pytest
from fastapi import HTTPException

from ledgerlens.guards import DailyBudget, RateLimiter


def test_rate_limiter_window():
    now = [0.0]
    rl = RateLimiter(limit=2, window=60, clock=lambda: now[0])
    rl.check("a")
    rl.check("a")
    with pytest.raises(HTTPException) as e:
        rl.check("a")
    assert e.value.status_code == 429 and "Retry-After" in e.value.headers
    rl.check("b")  # per-client
    now[0] = 61
    rl.check("a")  # window slid


def test_daily_budget_caps_and_resets():
    day = [date(2026, 11, 15)]
    b = DailyBudget(1.0, today=lambda: day[0])
    b.check()
    b.spend(0.6)
    b.check()
    b.spend(0.5)
    with pytest.raises(HTTPException) as e:
        b.check()
    assert e.value.status_code == 503
    day[0] = date(2026, 11, 16)
    b.check()
    assert b.spent_today == 0


def _fetch_module(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "fetch_model", Path(__file__).resolve().parents[1] / "scripts" / "fetch_model.py")
    fm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm)
    monkeypatch.setattr(fm, "ROOT", tmp_path)
    monkeypatch.setattr(fm, "TARGET", tmp_path / "models" / "enricher.joblib")
    monkeypatch.setattr(fm, "PINNED", tmp_path / "model.sha256")
    return fm


@pytest.mark.parametrize("tampered", [False, True])
def test_fetch_model_verifies_checksum(monkeypatch, tmp_path, tampered):
    import hashlib
    import shutil

    fm = _fetch_module(monkeypatch, tmp_path)
    good = tmp_path / "release.bin"
    good.write_bytes(b"model bytes")
    (tmp_path / "model.sha256").write_text(hashlib.sha256(b"model bytes").hexdigest() + "\n")
    served = tmp_path / "served.bin"
    served.write_bytes(b"evil bytes" if tampered else b"model bytes")
    monkeypatch.setattr(fm.urllib.request, "urlretrieve", lambda url, dst: shutil.copy(served, dst))
    monkeypatch.setenv("LEDGERLENS_MODEL_URL", "https://example.com/enricher.joblib")
    code = fm.main()
    assert code == (1 if tampered else 0)
    assert fm.TARGET.exists() is (not tampered)  # a mismatched pickle is never put where it'd be loaded
    assert list((tmp_path / "models").glob("tmp*")) == []  # temp file cleaned up either way


def test_service_guards(monkeypatch, tmp_path):
    import joblib
    from fastapi.testclient import TestClient

    from ledgerlens import train
    from ledgerlens.synth import Simulator, write_csv

    data = tmp_path / "s.csv"
    write_csv(Simulator(19, 10, 15).run(), data)
    _, artifacts = train.run_seed(train.load(data), seed=0, hide_frac=0.8, history_dropout=0.2, log=lambda *_: None)
    joblib.dump(artifacts, tmp_path / "e.joblib")
    monkeypatch.setenv("LEDGERLENS_MODEL", str(tmp_path / "e.joblib"))
    monkeypatch.setenv("LEDGERLENS_ALLOW_UPLOADS", "0")
    monkeypatch.setenv("LEDGERLENS_CHAT_PER_MIN", "2")
    monkeypatch.setenv("LEDGERLENS_DAILY_LLM_BUDGET_USD", "0")
    from ledgerlens.service import app

    with TestClient(app) as c:
        h = c.get("/healthz").json()
        assert h["uploads_enabled"] is False and h["llm_daily_budget_usd"] == 0
        up = c.post("/v1/datasets", files={"file": ("s.csv", "Date,Description,Amount\n2026-01-01,X,-1\n", "text/csv")})
        assert up.status_code == 403
        did = c.post("/v1/datasets/demo").json()["dataset_id"]
        msg = {"messages": [{"role": "user", "content": "hi"}]}
        assert c.post(f"/v1/datasets/{did}/chat", json=msg).status_code == 503  # budget 0
        assert c.post(f"/v1/datasets/{did}/chat", json=msg).status_code == 503
        assert c.post(f"/v1/datasets/{did}/chat", json=msg).status_code == 429  # 3rd call in a minute
