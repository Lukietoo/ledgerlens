"""The independent benchmark's split must never leak a merchant into training."""

import numpy as np
import pandas as pd

from ledgerlens import external_eval as ext


def fake(monkeypatch) -> pd.DataFrame:
    names = {"PLURALSIGHT": "online", "STARBUCKS": "coffee", "SAFEWAY": "chains", "SHELL OIL": "gas",
             "TARGET": "retail", "NETFLIX": "streaming", "JBG SMITH": "propertyMgmt"}
    monkeypatch.setattr(ext, "generator_merchants", lambda: names)
    rows = []
    for name, cat in [("PLURALSIGHT", "Education"), ("STARBUCKS", "Restaurants"), ("SAFEWAY", "Groceries"),
                      ("SHELL OIL", "Transportation"), ("TARGET", "Shopping"), ("NETFLIX", "Subscription"),
                      ("JBG SMITH", "Rent")]:
        for wrap in ("{} 1234 OAKLAND CA", "PAYPAL INST XFER {} WEB ID: X1", "WITHDRAWAL FROM {}", "{}"):
            rows.append({"description": "[debit] " + wrap.format(name), "category": cat})
    rows.append({"description": "[credit] ORIG CO NAME:ACME ENTRY DESCR:PAYROLL", "category": "Income"})
    path_df = pd.DataFrame(rows)
    m = path_df["description"].str.extract(r"^\[(debit|credit)\]\s*(.*)$")
    return path_df.assign(direction=m[0], raw=m[1], truth=path_df["category"].map(ext.THEIRS))


def test_every_wrapper_of_a_merchant_lands_in_one_group(monkeypatch):
    g = ext.merchant_groups(fake(monkeypatch))
    assert g.groupby("merchant").size().max() == 4  # 4 wrappers each, never split apart
    assert g.loc[g["raw"].str.contains("PLURALSIGHT"), "merchant"].nunique() == 1
    assert not g.loc[g["raw"].str.startswith("ORIG CO"), "matched"].any()  # template text: fallback key


def test_no_merchant_on_two_sides_and_test_never_written(monkeypatch):
    g = ext.merchant_groups(fake(monkeypatch))
    g["split"] = ext.assign_splits(g)
    assert (g.groupby("merchant")["split"].nunique() == 1).all()
    assert (ext.assign_splits(g) == g["split"]).all()  # deterministic
    tr = ext.training_rows(g)
    assert set(tr["description"]) == set(g.loc[g["split"] != "test", "raw"])
    assert not set(tr["merchant"]) & set(g.loc[g["split"] == "test", "merchant"])


def test_labels_are_refined_and_amounts_carry_no_signal(monkeypatch):
    g = ext.merchant_groups(fake(monkeypatch))
    g["split"] = "train"
    tr = ext.training_rows(g)
    assert set(tr.loc[tr["merchant"] == "STARBUCKS", "category"]) == {"COFFEE"}   # their Restaurants/coffee
    assert set(tr.loc[tr["merchant"] == "SHELL OIL", "category"]) == {"GAS"}
    assert set(tr.loc[tr["merchant"] == "JBG SMITH", "category"]) == {"HOUSING"}  # Rent and Mortgage
    debit = tr[tr["amount"] < 0]
    assert debit["amount"].nunique() == len(debit)  # no constant stand-in to key on
    by_cat = debit.groupby("category")["amount"].median().abs()
    assert by_cat.max() / by_cat.min() < 10  # same distribution for every label
    assert np.isfinite(tr["amount"]).all()
