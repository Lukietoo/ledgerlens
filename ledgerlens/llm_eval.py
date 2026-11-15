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
from ledgerlens.pipeline import DEFAULT_LLM_THRESHOLD, Enricher, RoutingPolicy, apply_llm_fallback, routed_mask

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
    ap.add_argument("--skip-llm-only", action="store_true",
                    help="reuse the LLM-only numbers from a previous run's results file instead of paying again")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    stem = f"week3_llm_{args.model}"
    previous = args.results / f"{stem}.json"
    if args.skip_llm_only and not previous.exists():
        raise SystemExit(f"--skip-llm-only needs a previous run at {previous}")

    df = pd.read_csv(args.data)
    test = df[df.split == "test"]
    # unique descriptions, so the LLM can't score points on repeats
    sample = test.drop_duplicates("description").sample(args.n, random_state=args.seed).reset_index(drop=True)
    exclude = set(test.merchant) if args.regime == "cold" else set()
    model_out = Enricher.load(args.model_path).enrich(
        sample[["user_id", "description", "amount", "date"]], exclude=exclude)

    policies = {
        "hybrid: confidence only (week 3)": RoutingPolicy(args.threshold, unknown_merchants=False,
                                                          category_for="routed"),
        "hybrid: + unknown merchants, LLM names only": RoutingPolicy(args.threshold),
        "hybrid: + unknown merchants, LLM names + categories": RoutingPolicy(args.threshold, category_for="routed"),
    }
    superset = RoutingPolicy(args.threshold, category_for="routed")  # routes the union of all policies
    keys_needed = model_out.loc[routed_mask(model_out, superset)[0], "merchant_key"].nunique()

    est = estimate_cost(keys_needed, args.model, args.batch_size)
    if not args.skip_llm_only:
        est += estimate_cost(args.n, args.model, args.batch_size)
    print(f"plan: {'' if args.skip_llm_only else f'{args.n} txns LLM-only + '}{keys_needed} merchant keys "
          f"for {len(policies)} hybrid policies (shared cache) on {args.model}; estimated cost ≈ ${est:.2f}")
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

    def usage(u) -> dict:
        return {"requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                "refusals": u.refusals, "seconds": round(u.seconds, 2), "cost_usd": round(u.cost_usd(), 4),
                "by_model": u.by_model}

    # 1) LLM only: no model hint, one call per row
    if args.skip_llm_only:
        old = json.loads(previous.read_text())
        llm_only_report, tokens = {**old["llm_only"], "reused_from_previous_run": True}, old["tokens_per_txn"]
    else:
        llm_only = LLMClassifier(client, model=args.model, batch_size=args.batch_size)
        preds = llm_only.classify([Txn(d, float(a)) for d, a in zip(sample.description, sample.amount)])
        llm_only_report = {**score([p.category if p else "UNKNOWN" for p in preds],
                                   [p.merchant if p else "" for p in preds]),
                           "unparsed_or_refused": sum(p is None for p in preds), "usage": usage(llm_only.usage)}
        tokens = {"input": llm_only.usage.input_tokens / args.n, "output": llm_only.usage.output_tokens / args.n}

    # 2) hybrids: one shared, cached classifier. The superset pays for every key once;
    #    each policy then reads its subset from the cache (no extra calls).
    shared = LLMClassifier(client, model=args.model, batch_size=args.batch_size)
    apply_llm_fallback(model_out, shared, superset)
    hybrids = {}
    for name, policy in policies.items():
        routed, _ = routed_mask(model_out, policy)
        out = apply_llm_fallback(model_out, shared, policy)
        hybrids[name] = {**score(out["category"], out["merchant"]), "rows_routed": int(routed.sum()),
                         "llm_calls": int(model_out.loc[routed, "merchant_key"].nunique())}

    report = {
        "model": args.model, "regime": args.regime, "n": args.n, "threshold": args.threshold,
        "batch_size": args.batch_size,
        "model_only": score(model_out["category"], model_out["merchant"]),
        "llm_only": llm_only_report,
        "hybrids": hybrids,
        "hybrid_usage": usage(shared.usage),
        "tokens_per_txn": tokens,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"{stem}.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (args.results / f"{stem}.md").write_text(md)
    print(md)


def to_markdown(r: dict) -> str:
    def row(name, x, calls):
        return f"| {name} | {x['accuracy']:.1%} | {x['merchant_exact']:.1%} | {calls} |"

    lo = r["llm_only"]
    return "\n".join([
        f"## LLM fallback eval: {r['model']}, {r['regime']} directory, {r['n']} unique test txns", "",
        "| strategy | category acc | merchant exact | LLM calls (txns or merchant keys) |", "|---|---|---|---|",
        row("model only", r["model_only"], "0"),
        row("LLM only", lo, f"{r['n']}" + (" (reused from previous run)" if lo.get("reused_from_previous_run") else "")),
        *[row(name, h, f"{h['llm_calls']} keys for {h['rows_routed']} rows") for name, h in r["hybrids"].items()],
        "",
        f"Hybrid API usage (all policies share one cache): {r['hybrid_usage']['requests']} requests, "
        f"${r['hybrid_usage']['cost_usd']:.4f}, {r['hybrid_usage']['refusals']} refusals.",
        f"Measured tokens per transaction (LLM-only): {r['tokens_per_txn']['input']:.0f} in / "
        f"{r['tokens_per_txn']['output']:.0f} out. LLM-only refusals: {lo['usage']['refusals']} requests.",
        "",
    ])


if __name__ == "__main__":
    main()
