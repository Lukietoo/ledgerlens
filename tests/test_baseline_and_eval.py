import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from ledgerlens import evaluate
from ledgerlens.baseline import categorize, detect_recurring
from ledgerlens.synth import Simulator, write_csv
from ledgerlens.taxonomy import UNKNOWN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import label  # noqa: E402


@pytest.mark.parametrize(
    "raw, cat",
    [
        ("UBER *EATS HELP.UBER.COM CA", "DINING"),
        ("UBER *TRIP HELP.UBER.COM CA", "TRANSPORT"),
        ("GUSTO PAYROLL PPD ID: 1234567890", "INCOME"),
        ("ZELLE TO JANE DOE", "TRANSFER"),
        ("NETFLIX.COM 866-579-7172 CA", "SUBSCRIPTIONS"),
        ("SQ *MISSION LOTUS OAKLAND CA", UNKNOWN),
    ],
)
def test_categorize(raw, cat):
    assert categorize(raw) == cat


def test_detect_recurring():
    d0 = date(2026, 1, 3)
    rows = [("u1", "NETFLIX.COM 866-579-7172 CA", d0 + timedelta(days=30 * i), -15.49) for i in range(6)]
    rows += [("u1", "SAFEWAY #12 OAKLAND CA", d0 + timedelta(days=9 * i), -20.0 - 17 * i) for i in range(6)]
    flags = detect_recurring(rows)
    assert flags[:6] == [True] * 6
    assert flags[6:] == [False] * 6


def test_evaluate_end_to_end(tmp_path):
    data = tmp_path / "s.csv"
    write_csv(Simulator(5, 10, 20).run(), data)
    res = evaluate.eval_synthetic(data, "test")
    json.dumps(res)  # serializable
    assert 0 < res["category"]["accuracy"] < 1
    assert 0 <= res["merchant_exact_match"] <= 1
    assert res["recurring"]["f1"] > 0.5
    assert "category accuracy" in evaluate.to_markdown("x", res)


def test_labeler_resumes_and_feeds_eval(tmp_path):
    export = tmp_path / "bank.csv"
    export.write_text(
        "Transaction Date,Description,Debit,Credit\n"
        "01/02/2026,NETFLIX.COM 866-579-7172 CA,15.49,\n"
        "01/03/2026,SQ *MISSION LOTUS OAKLAND CA,22.10,\n"
        "01/03/2026,SQ *MISSION LOTUS OAKLAND CA,22.10,\n"  # dup description
        "01/15/2026,ACME CORP PAYROLL PPD ID: 99,,2100.00\n"
    )
    out = tmp_path / "real" / "labeled.csv"
    answers = iter(["", "2", "q"])  # accept, pick DINING, quit before payroll
    label.run([export], out, inp=lambda _: next(answers), echo=lambda *a: None)
    rows = label.load_done(out)
    assert [r["category"] for r in rows] == ["SUBSCRIPTIONS", "DINING"]
    assert rows[0]["amount"] == "-15.49"

    answers = iter([""])  # resume: only payroll left
    label.run([export], out, inp=lambda _: next(answers), echo=lambda *a: None)
    rows = label.load_done(out)
    assert [r["category"] for r in rows] == ["SUBSCRIPTIONS", "DINING", "INCOME"]
    assert rows[2]["amount"] == "2100.00"

    res = evaluate.eval_real(out)
    assert res["category"]["n"] == 3
    assert res["category"]["accuracy"] == pytest.approx(2 / 3)
