"""Cost per 1M transactions: model-only vs LLM-only vs hybrid.

    python -m ledgerlens.cost

Inputs (all produced by earlier steps):
  results/loadtest/light_batch100_stats.csv   measured throughput (4 workers, batches of 100)
  results/week3_routing.json                  share of traffic below the LLM threshold
  results/week3_llm_<model>.json              measured LLM tokens/txn, if llm_eval has been run;
                                              otherwise a pessimistic estimate (flagged in the output)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ledgerlens.llm import PRICES, SYSTEM_PROMPT
from ledgerlens.llm_eval import EST_OUTPUT_TOKENS_PER_TXN
from ledgerlens.pipeline import DEFAULT_LLM_THRESHOLD


def measured_throughput(stats_csv: Path, batch: int) -> float:
    with stats_csv.open() as f:
        agg = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    return float(agg["Requests/s"]) * batch


def tokens_per_txn(results: Path, model: str, batch_size: int = 25) -> tuple[float, float, bool]:
    p = results / f"week3_llm_{model}.json"
    if p.exists():
        t = json.loads(p.read_text())["tokens_per_txn"]
        return t["input"], t["output"], True
    return len(SYSTEM_PROMPT) / 4 / batch_size + 30, EST_OUTPUT_TOKENS_PER_TXN, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--workers", type=int, default=4, help="uvicorn workers in the load test (≈ vCPUs used)")
    ap.add_argument("--vcpu-hour-usd", type=float, default=0.04,
                    help="ASSUMPTION: on-demand cloud price per vCPU-hour")
    args = ap.parse_args()

    tps = measured_throughput(args.results / "loadtest" / "light_batch100_stats.csv", 100)
    vcpu_hours_per_m = 1e6 / tps / 3600 * args.workers
    model_cost = vcpu_hours_per_m * args.vcpu_hour_usd

    routing = json.loads((args.results / "week3_routing.json").read_text())
    routed = {reg: next(c["routed_to_llm"] for c in routing[reg]["curve"] if c["threshold"] == DEFAULT_LLM_THRESHOLD)
              for reg in ("cold", "warm")}
    ceiling = {reg: next(c["hybrid_upper_bound"] for c in routing[reg]["curve"]
                         if c["threshold"] == DEFAULT_LLM_THRESHOLD) for reg in ("cold", "warm")}

    rows = []
    for model, (pin, pout) in PRICES.items():
        tin, tout, measured = tokens_per_txn(args.results, model)
        per_m = 1e6 * (tin / 1e6 * pin + tout / 1e6 * pout)
        rows.append({"model": model, "tokens_in": tin, "tokens_out": tout, "measured": measured,
                     "llm_only_per_1m": per_m,
                     "hybrid_cold_per_1m": model_cost + routed["cold"] * per_m,
                     "hybrid_warm_per_1m": model_cost + routed["warm"] * per_m})

    report = {"throughput_txn_per_s": tps, "workers": args.workers, "vcpu_hour_usd_assumption": args.vcpu_hour_usd,
              "model_only_per_1m": model_cost, "routed_share": routed, "hybrid_ceiling": ceiling,
              "threshold": DEFAULT_LLM_THRESHOLD, "llm": rows}
    (args.results / "week3_cost.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.results / "week3_cost.md").write_text(md)
    print(md)


def to_markdown(r: dict) -> str:
    L = ["## Cost per 1M transactions", "",
         f"Model-only: measured **{r['throughput_txn_per_s']:,.0f} txns/s** on {r['workers']} workers "
         f"(M1 laptop, batches of 100) → **${r['model_only_per_1m']:.4f} per 1M** "
         f"(assumes ${r['vcpu_hour_usd_assumption']}/vCPU-hour).", "",
         f"Hybrid routes rows below confidence {r['threshold']} to the LLM: "
         f"{r['routed_share']['cold']:.1%} of traffic (cold directory) / {r['routed_share']['warm']:.1%} (warm).", "",
         "| LLM | tokens/txn in / out | LLM-only per 1M | hybrid per 1M (cold) | hybrid per 1M (warm) "
         "| LLM-only ÷ model-only |",
         "|---|---|---|---|---|---|"]
    for x in r["llm"]:
        flag = "" if x["measured"] else " (est.)"
        L.append(f"| {x['model']} | {x['tokens_in']:.0f} / {x['tokens_out']:.0f}{flag} | ${x['llm_only_per_1m']:,.0f} | "
                 f"${x['hybrid_cold_per_1m']:,.0f} | ${x['hybrid_warm_per_1m']:,.0f} | "
                 f"{x['llm_only_per_1m'] / r['model_only_per_1m']:,.0f}× |")
    L += ["", "(est.) = tokens estimated before any API call (≈4 chars/token, pessimistic output); "
          "run `python -m ledgerlens.llm_eval --yes --model <m>` to replace with measured usage.",
          "Standard API rates; the Batch API would halve LLM costs for non-real-time backfills.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
