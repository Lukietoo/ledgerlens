import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ledgerlens import corrections_eval, train
from ledgerlens.finance import Ledger
from ledgerlens.synth import Simulator, write_csv


def ledger() -> Ledger:
    rows = []
    for m in range(1, 4):
        rows += [
            (f"2026-0{m}-03", "PMUSA NEW YORK NY", -6.0, "Pmusa", "GROCERIES", "PMUSA|out", 0.35, False),
            (f"2026-0{m}-04", "PMUSA NEW YORK NY", -7.0, "Pmusa", "GROCERIES", "PMUSA|out", 0.33, False),
            (f"2026-0{m}-05", "NETFLIX.COM", -15.49, "Netflix", "SUBSCRIPTIONS", "NETFLIX|out", 0.99, True),
            (f"2026-0{m}-06", "SQ *IVY OLIVE", -20.0, "Ivy Olive", "GROCERIES", "IVYOLIVE|out", 0.70, False),
            (f"2026-0{m}-15", "ACME PAYROLL", 3000.0, "Acme", "INCOME", "ACME|in", 0.99, True),
        ]
    df = pd.DataFrame(rows, columns=["date", "description", "amount", "merchant", "category", "merchant_key",
                                     "confidence", "merchant_in_directory"])
    df["is_recurring"] = False
    df["source"] = "model"
    return Ledger(df)


def test_correction_propagates_to_every_row_of_the_merchant():
    L = ledger()
    assert L.apply_correction("PMUSA|out", merchant="ParkMobile", category="TRANSPORT") == 6
    fixed = L.df[L.df["merchant_key"] == "PMUSA|out"]
    assert set(fixed["merchant"]) == {"ParkMobile"} and set(fixed["category"]) == {"TRANSPORT"}
    assert set(fixed["source"]) == {"user"} and set(fixed["confidence"]) == {1.0}
    assert L.corrections["PMUSA|out"] == {"merchant": "ParkMobile", "category": "TRANSPORT", "rows": 6}
    assert set(L.df.loc[L.df["merchant_key"] != "PMUSA|out", "source"]) == {"model"}  # nothing else touched


def test_category_correction_updates_spending():
    L = ledger()
    before = L.overview()["total_spending"]
    L.apply_correction("IVYOLIVE|out", category="TRANSFER")  # e.g. "that's my roommate, not a store"
    assert L.overview()["total_spending"] == pytest.approx(before - 60.0)
    L.apply_correction("IVYOLIVE|out", merchant="Ivy (roommate)")  # name-only fix keeps the category
    assert L.corrections["IVYOLIVE|out"]["category"] == "TRANSFER"


def test_correction_validation():
    L = ledger()
    with pytest.raises(KeyError):
        L.apply_correction("NOPE|out", category="DINING")
    with pytest.raises(ValueError):
        L.apply_correction("PMUSA|out", category="PIZZA")
    with pytest.raises(ValueError):
        L.apply_correction("PMUSA|out")
    with pytest.raises(ValueError):
        Ledger(ledger().df.drop(columns=["merchant_key"])).apply_correction("PMUSA|out", category="DINING")


def test_review_queue_ranks_by_impact_and_drops_reviewed():
    L = ledger()
    q = L.review_queue()
    assert [x["merchant_key"] for x in q[:2]] == ["PMUSA|out", "IVYOLIVE|out"]  # 6 unsure rows beat 3
    assert all(x["merchant_key"] not in ("NETFLIX|out", "ACME|in") for x in q)  # confident + known: skip
    L.apply_correction("PMUSA|out", category="TRANSPORT")
    assert "PMUSA|out" not in [x["merchant_key"] for x in L.review_queue()]


def _with_truth() -> pd.DataFrame:
    d = ledger().df.copy()
    d["true_merchant"] = d["merchant"].replace({"Pmusa": "ParkMobile"})
    d["true_category"] = d["category"].replace({"GROCERIES": "TRANSPORT"}).where(d["merchant"] == "Pmusa", d["category"])
    d["user_id"] = "u"
    return d


@pytest.mark.parametrize("strategy", ["notice", "queue"])
def test_simulation_strategies(strategy):
    r = corrections_eval.run_user(_with_truth(), strategy, 3)
    cat0, name0, n = r["curve"][0]
    assert (cat0, name0, n) == (9, 9, 15)  # Pmusa's 6 rows are wrong on both
    assert r["curve"][3][0] == 15 and r["curve"][3][1] == 15  # one fix repairs all 6
    assert r["fixes"] == ["PMUSA|out"] and r["rows_fixed"] == 6
    assert r["shared"] == {"PMUSA|out": ("ParkMobile", "TRANSPORT")}


def test_community_fixes_apply_before_any_effort():
    r = corrections_eval.run_user(_with_truth(), "queue", 0, shared={"PMUSA|out": ("ParkMobile", "TRANSPORT")})
    assert r["curve"][0][0] == 15  # fixed with zero actions by this user


def test_correction_endpoints(tmp_path, monkeypatch):
    data = tmp_path / "s.csv"
    write_csv(Simulator(23, 10, 15).run(), data)
    _, artifacts = train.run_seed(train.load(data), seed=0, hide_frac=0.8, history_dropout=0.2, log=lambda *_: None)
    joblib.dump(artifacts, tmp_path / "e.joblib")
    monkeypatch.setenv("LEDGERLENS_MODEL", str(tmp_path / "e.joblib"))
    from ledgerlens.service import app

    with TestClient(app) as c:
        did = c.post("/v1/datasets/demo").json()["dataset_id"]
        items = c.get(f"/v1/datasets/{did}/review", params={"limit": 3}).json()["items"]
        assert items and {"merchant_key", "merchant", "category", "rows", "example"} <= set(items[0])
        key, rows = items[0]["merchant_key"], items[0]["rows"]
        spend_before = c.get(f"/v1/datasets/{did}/dashboard").json()["overview"]["total_spending"]
        r = c.post(f"/v1/datasets/{did}/corrections", json={"merchant_key": key, "merchant": "Fixed Name",
                                                             "category": "TRANSFER"})
        assert r.status_code == 200 and r.json()["updated"] == rows
        feed = c.get(f"/v1/datasets/{did}/transactions", params={"q": "fixed name", "limit": 500}).json()
        assert feed["total"] == rows and {x["source"] for x in feed["rows"]} == {"user"}
        assert all("merchant_key" in x for x in feed["rows"])
        # transfers aren't spending, so the dashboard must reflect the fix immediately
        assert c.get(f"/v1/datasets/{did}/dashboard").json()["overview"]["total_spending"] <= spend_before
        assert key not in [x["merchant_key"] for x in c.get(f"/v1/datasets/{did}/review").json()["items"]]
        codes = [c.post(f"/v1/datasets/{did}/corrections", json=j).status_code for j in (
            {"merchant_key": "nope", "category": "DINING"}, {"merchant_key": key},
            {"merchant_key": key, "category": "PIZZA"}, {"merchant_key": key, "merchant": ""})]
        assert codes == [404, 422, 422, 422]
        assert c.get("/v1/datasets/nope/review").status_code == 404
