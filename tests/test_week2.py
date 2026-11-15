import joblib
import numpy as np
import pandas as pd
import pytest

from ledgerlens import train
from ledgerlens.baseline import is_brand_match
from ledgerlens.features import add_clean, history_features, struct_amount_features
from ledgerlens.pipeline import Enricher
from ledgerlens.resolver import Resolver, load_directory
from ledgerlens.synth import Simulator, write_csv
from ledgerlens.taxonomy import CATEGORIES


@pytest.fixture(scope="module")
def resolver():
    return Resolver()


def test_directory_holds_out_aliases():
    entries = load_directory()
    texts = {e.text for e in entries if e.merchant == "Whole Foods Market"}
    assert texts == {"WHOLE FOODS MARKET", "WHOLEFDS"}  # canonical + primary only; WFM is held out


@pytest.mark.parametrize("query, merchant", [
    ("BLUE BTL COF", "Blue Bottle Coffee"),  # unseen alias
    ("SWEETGRN", "Sweetgreen"),
    ("TRADER JOES", "Trader Joe's"),
    ("ZELLE", "Zelle"),
])
def test_resolver_matches_aliases(resolver, query, merchant):
    m = resolver.match([query])
    assert m.merchant[0] == merchant
    assert m.sim[0] > 0.6


def test_resolver_ignores_payment_type_words(resolver):
    # used to match "ADP PAYROLL" on the word PAYROLL
    assert resolver.match(["SOFI TECHNOLOGI PAYROLL"]).merchant[0] != "ADP"


def test_resolver_exclude_and_abstain(resolver):
    m = resolver.match(["SWEETGRN"], exclude={"Sweetgreen"})
    assert m.merchant[0] != "Sweetgreen"
    assert resolver.resolve(["COPPER TIGER SUSHI"], threshold=0.6) == ["Copper Tiger Sushi"]
    scores = resolver.match(["SWEETGRN"]).cat_scores
    assert scores.shape == (1, len(CATEGORIES)) and np.isclose(scores.sum(), 1.0)


def test_brand_match_ignores_locations():
    assert is_brand_match("NETFLIX.COM 866-579-7172 CA")
    assert not is_brand_match("WINGSTOP #123 BERKELEY CA")  # BERKELEY rule fired on the city
    assert not is_brand_match("SQ *GOLDEN FOX CAFE OAKLAND CA")  # generic word


def test_features():
    df = add_clean(pd.DataFrame({
        "user_id": ["u"] * 4,
        "description": ["NETFLIX.COM 866-579-7172 CA"] * 3 + ["ACME PAYROLL PPD ID: 123"],
        "amount": [-15.49, -15.49, -15.49, 2000.0],
        "date": ["2026-01-05", "2026-02-05", "03/05/2026", "2026-01-15"],  # mixed formats
    }))
    s = struct_amount_features(df)
    assert list(s["entry"]) == ["none"] * 3 + ["pay"]
    assert list(s["is_online"])[:3] == [1, 1, 1]
    assert list(s["rule_cat"])[:3] == ["SUBSCRIPTIONS"] * 3
    h = history_features(df)
    assert list(h["hist_n"]) == [3, 3, 3, 1]
    assert h["hist_amount_cv"].iloc[0] == 0
    assert 28 <= h["hist_median_gap"].iloc[0] <= 31


def test_train_and_enrich_end_to_end(tmp_path):
    data = tmp_path / "s.csv"
    write_csv(Simulator(11, 12, 15).run(), data)
    d = train.load(data)
    report, artifacts = train.run_seed(d, seed=0, hide_frac=0.8, history_dropout=0.2, log=lambda *_: None)
    names = [r["model"] for r in report["category"]]
    assert names[0].startswith("Rules") and names[-1] == train.NO_HISTORY_ROW
    full = next(r for r in report["category"] if r["model"].startswith("+ merchant resolver"))
    assert full["cold"]["accuracy"] > 0.5 and full["warm"]["accuracy"] > 0.5
    assert 0.3 <= report["merchant"]["threshold"] <= 0.95

    joblib.dump(artifacts, tmp_path / "e.joblib")
    out = Enricher.load(tmp_path / "e.joblib").enrich(pd.DataFrame({
        "description": ["NETFLIX.COM 866-579-7172 CA", "ZELLE TO JANE DOE", "TST* MISSION LOTUS OAKLAND CA"],
        "amount": [-15.49, -40.0, -22.0],
        "date": ["2026-03-01", "2026-03-02", "2026-03-03"],
    }))
    assert list(out["merchant"][:2]) == ["Netflix", "Zelle"]
    assert out["merchant"].iloc[2] == "Mission Lotus"
    assert set(out["category"]) <= set(CATEGORIES)
    assert out["confidence"].between(0, 1).all()

    agg = train.aggregate(d, [report, report], type("A", (), {"hide_frac": 0.8, "history_dropout": 0.2}))
    assert "± 0.0%" in train.to_markdown(agg)
