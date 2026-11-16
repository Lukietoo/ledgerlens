import random
import statistics
import time

import joblib
import pandas as pd
import pytest

from ledgerlens import fastpath, train
from ledgerlens.pipeline import Enricher
from ledgerlens.synth import Simulator, write_csv

KEYS = ["merchant", "merchant_in_directory", "merchant_key", "category", "confidence", "is_recurring"]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fast")
    data = tmp / "s.csv"
    sim = Simulator(29, 14, 15)
    write_csv(sim.run(), data, sim.user_styles)
    _, artifacts = train.run_seed(train.load(data), seed=0, hide_frac=0.8, history_dropout=0.2, log=lambda *_: None)
    joblib.dump(artifacts, tmp / "e.joblib")
    df = pd.read_csv(data)[["user_id", "description", "amount", "date"]]
    return Enricher.load(tmp / "e.joblib"), df


def _same(enricher, df):
    slow = enricher.enrich(df).to_dict("records")
    fast = fastpath.enrich_records(enricher, df.to_dict("records"))
    assert len(slow) == len(fast)
    for s, f in zip(slow, fast):
        for k in KEYS:
            assert s[k] == f[k], (k, s["description"], s[k], f[k])
        assert (s["processor"] if isinstance(s["processor"], str) else None) == f["processor"]


def test_identical_on_single_transactions(setup):
    e, df = setup
    for i in range(300):
        _same(e, df.sample(1, random_state=i))


def test_identical_on_small_batches_with_history(setup):
    e, df = setup
    rng = random.Random(0)
    users = list(df["user_id"].unique())
    for i in range(60):  # same-user batches: repeated merchants exercise the history features
        u = df[df["user_id"] == rng.choice(users)]
        _same(e, u.sample(min(len(u), rng.randint(2, 16)), random_state=i))


def test_identical_on_awkward_inputs(setup):
    e, _ = setup
    _same(e, pd.DataFrame({
        "user_id": ["me"] * 6,
        "description": ["NETFLIX.COM 866-579-7172 CA", "NETFLIX.COM 866-579-7172 CA", "ZELLE TO JANE DOE",
                        "Tst* Golden Fox Oakland Ca", "ACME CORP PAYROLL PPD ID: 99", "#12345"],
        "amount": [-15.49, -15.49, -40.0, -22.005, 2100.0, -1.0],
        "date": ["2026-01-05", "02/05/2026", "2026-01-07", "2026-01-08T00:00:00", "2026-01-15", "2026-01-16"],
    }))


def test_dispatch(setup, monkeypatch):
    e, df = setup
    calls = []
    real = fastpath.enrich_records
    monkeypatch.setattr(fastpath, "enrich_records", lambda en, recs: calls.append(len(recs)) or real(en, recs))
    small = df.head(3).to_dict("records")
    big = df.head(fastpath.FAST_MAX_ROWS + 1).to_dict("records")
    assert len(e.enrich_records(small)) == 3 and calls == [3]
    assert len(e.enrich_records(big)) == len(big) and calls == [3]  # big batch: pandas path
    e.enrich_records(small, exclude={"Netflix"})  # evaluation-only cold start: pandas path
    assert calls == [3]
    assert all(r["processor"] is None or isinstance(r["processor"], str) for r in e.enrich_records(big))


def test_performance_budget(setup):
    # relative budget so it holds on any CI machine: fast path must stay at least 2x faster
    e, df = setup
    rows = [df.iloc[[i]] for i in range(0, 60)]
    for r in rows[:10]:
        e.enrich(r), fastpath.enrich_records(e, r.to_dict("records"))

    def median_ms(fn):
        ts = []
        for r in rows:
            t = time.perf_counter()
            fn(r)
            ts.append(time.perf_counter() - t)
        return statistics.median(ts) * 1000

    slow, fast = median_ms(e.enrich), median_ms(lambda r: fastpath.enrich_records(e, r.to_dict("records")))
    assert fast * 2 <= slow, f"fast path {fast:.2f} ms vs pandas {slow:.2f} ms"
