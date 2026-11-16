"""Latency/throughput benchmark for the inference paths (in-process, no HTTP).

    python -m ledgerlens.bench

Single-transaction and small-batch latency for the pandas path vs the fast path, and
large-batch throughput (which always uses the pandas path). Real test-split requests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ledgerlens.fastpath import enrich_records
from ledgerlens.pipeline import Enricher


def _timeit(fn, inputs, warmup: int = 20) -> np.ndarray:
    for x in inputs[:warmup]:
        fn(x)
    out = []
    for x in inputs:
        t = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t) * 1000)
    return np.array(out)


def summary(ms: np.ndarray) -> dict:
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "n": int(len(ms))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/transactions.csv"))
    ap.add_argument("--model", type=Path, default=Path("models/enricher.joblib"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    e = Enricher.load(args.model)
    test = pd.read_csv(args.data).query("split == 'test'")[["user_id", "description", "amount", "date"]]
    rng = np.random.default_rng(0)
    singles = [test.iloc[[i]] for i in rng.integers(0, len(test), args.n)]
    users = test["user_id"].unique()
    batches16 = []
    for _ in range(args.n // 5):
        u = test[test["user_id"] == rng.choice(users)]
        batches16.append(u.sample(min(16, len(u)), random_state=int(rng.integers(1 << 30))))
    big = [test.sample(100, random_state=i) for i in range(40)]

    report = {
        "single": {"pandas": summary(_timeit(e.enrich, singles)),
                   "fast": summary(_timeit(lambda d: enrich_records(e, d.to_dict("records")), singles))},
        "batch16": {"pandas": summary(_timeit(e.enrich, batches16, 5)),
                    "fast": summary(_timeit(lambda d: enrich_records(e, d.to_dict("records")), batches16, 5))},
    }
    ms100 = _timeit(e.enrich, big, 5)
    report["batch100_pandas"] = {**summary(ms100), "txns_per_s_one_core": float(100 / (np.median(ms100) / 1000))}
    for k in ("single", "batch16"):
        report[k]["speedup_p50"] = report[k]["pandas"]["p50_ms"] / report[k]["fast"]["p50_ms"]

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "week6_perf.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.results / "week6_perf.md").write_text(md)
    print(md)


def to_markdown(r: dict) -> str:
    L = ["## Inference latency (in-process, one core, M1 laptop)", "",
         "| request | path | p50 | p95 | p99 |", "|---|---|---|---|---|"]
    for k, label in (("single", "1 transaction"), ("batch16", "16 transactions, one user")):
        for path in ("pandas", "fast"):
            x = r[k][path]
            L.append(f"| {label} | {path} | {x['p50_ms']:.2f} ms | {x['p95_ms']:.2f} ms | {x['p99_ms']:.2f} ms |")
    b = r["batch100_pandas"]
    L += ["", f"Speedup at p50: **{r['single']['speedup_p50']:.1f}×** for one transaction, "
          f"**{r['batch16']['speedup_p50']:.1f}×** for 16. Batches of 100 use the pandas path: "
          f"{b['p50_ms']:.1f} ms p50, about {b['txns_per_s_one_core']:,.0f} transactions/s on one core.", "",
          "The fast path is checked to give identical outputs to the pandas path (tests/test_fastpath.py)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
