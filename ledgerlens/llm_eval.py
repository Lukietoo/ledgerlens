"""Measure the LLM fallback: LLM-only vs model-only vs hybrid, on a test-split sample.

    python -m ledgerlens.llm_eval                 # dry run: prints the plan + estimated cost, calls nothing
    python -m ledgerlens.llm_eval --yes           # runs it (needs Anthropic credentials; costs money)
    python -m ledgerlens.llm_eval --yes --model claude-haiku-5-5

Writes results/week3_llm_<model>.{json,md}: accuracy for each strategy, plus measured
tokens per transaction, which cost.py uses instead of its estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ledgerlens.llm import DEFAULT_MODEL, PRICES, SYSTEM_PROMPT, LLMClassifier, Txn
from ledgerlens.normalize import canon
from ledgerlens.pipeline import DEFAULT_LLM_THRESHOLD, Enricher, apply_llm_fallback

# Rough pre-run estimate (≈4 chars/token): replaced by measured usage after a run.
EST_OUTPUT_TOKENS_PER_TXN = 60  # JSON row + low-effort thinking share; deliberately pessimistic


def estimate_cost(n_txns: int, model: str, batch_size: int) -> float:
    pin, pout = PRICES[model]
    n_req = -(-n_txns // batch_size)
    tin = n_req * len(SYSTEM_PROMPT) / 4 + n_txns * 30
    tout = n_txns * EST_OUTPUT_TOKENS_PER_TXN
    return tin / 1e6 * pin + tout / 1e6 * pout


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM fallback eval (costs money with --yes)")
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/transactions.csv"))
    ap.add_argument("--model-path", type=Path, default=Path("models/enricher.joblib"))
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES))
    ap.add_argument("--n", type=int, default=400, help="test transactions to sample")
    ap.add_argument("--threshold", type=float, default=DEFAULT_LLM_THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--regime", choices=["cold", "warm"], default="cold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yes", action="store_true", help="actually call the API")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    test = df[df.split == "test"]
    # unique descriptions, so the LLM can't score points on repeats
    sample = test.drop_duplicates("description").sample(args.n, random_state=args.seed).reset_index(drop=True)
    exclude = set(test.merchant) if args.regime == "cold" else set()
    model_out = Enricher.load(args.model_path).enrich(
        sample[["user_id", "description", "amount", "date"]], exclude=exclude)
    routed = int((model_out["confidence"] < args.threshold).sum())

    est = estimate_cost(args.n, args.model, args.batch_size) + estimate_cost(routed, args.model, args.batch_size)
    print(f"plan: {args.n} txns LLM-only + {routed} routed txns (hybrid, threshold {args.threshold}) "
          f"on {args.model}; estimated cost ≈ ${est:.2f}")
    if not args.yes:
        print("dry run: pass --yes to call the API")
        return

    import anthropic

    client = anthropic.Anthropic()
    truth_cat, truth_merch = sample["category"].to_numpy(), sample["merchant"].to_numpy()

    def score(cats, merchants) -> dict:
        cats, merchants = np.asarray(cats), list(merchants)
        return {"accuracy": float((cats == truth_cat).mean()),
                "merchant_exact": float(np.mean([canon(a) == canon(b) for a, b in zip(merchants, truth_merch)]))}

    # 1) LLM only: no model hint
    llm_only = LLMClassifier(client, model=args.model, batch_size=args.batch_size)
    preds = llm_only.classify([Txn(d, float(a)) for d, a in zip(sample.description, sample.amount)])
    llm_cats = [p.category if p else "UNKNOWN" for p in preds]
    llm_merch = [p.merchant if p else "" for p in preds]

    # 2) hybrid: model everywhere, LLM below the threshold (with the model's guess as a hint)
    hybrid_llm = LLMClassifier(client, model=args.model, batch_size=args.batch_size)
    hybrid = apply_llm_fallback(model_out, hybrid_llm, args.threshold)

    def usage(u) -> dict:
        return {"requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                "refusals": u.refusals, "seconds": round(u.seconds, 2), "cost_usd": round(u.cost_usd(), 4),
                "by_model": u.by_model}

    n_failed = sum(p is None for p in preds)
    report = {
        "model": args.model, "regime": args.regime, "n": args.n, "threshold": args.threshold,
        "routed": routed, "batch_size": args.batch_size,
        "model_only": score(model_out["category"], model_out["merchant"]),
        "llm_only": {**score(llm_cats, llm_merch), "unparsed_or_refused": n_failed, "usage": usage(llm_only.usage)},
        "hybrid": {**score(hybrid["category"], hybrid["merchant"]), "usage": usage(hybrid_llm.usage)},
        "tokens_per_txn": {"input": llm_only.usage.input_tokens / args.n,
                           "output": llm_only.usage.output_tokens / args.n},
    }
    args.results.mkdir(parents=True, exist_ok=True)
    stem = f"week3_llm_{args.model}"
    (args.results / f"{stem}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.results / f"{stem}.md").write_text(md)
    print(md)


def to_markdown(r: dict) -> str:
    def row(name, x, cost="–"):
        return f"| {name} | {x['accuracy']:.1%} | {x['merchant_exact']:.1%} | {cost} |"

    return "\n".join([
        f"## LLM fallback eval: {r['model']}, {r['regime']} directory, {r['n']} unique test txns", "",
        "| strategy | category acc | merchant exact | API cost for this sample |", "|---|---|---|---|",
        row("model only", r["model_only"], "$0"),
        row("LLM only", r["llm_only"], f"${r['llm_only']['usage']['cost_usd']:.2f}"),
        row(f"hybrid (LLM on {r['routed']} txns below {r['threshold']})", r["hybrid"],
            f"${r['hybrid']['usage']['cost_usd']:.2f}"),
        "",
        f"Measured tokens per transaction (LLM-only): {r['tokens_per_txn']['input']:.0f} in / "
        f"{r['tokens_per_txn']['output']:.0f} out. Refusals: {r['llm_only']['usage']['refusals']} requests.",
        "",
    ])


if __name__ == "__main__":
    main()
