"""The full system (model + Claude fallback) on the independent benchmark's held-out merchants.

    python -m ledgerlens.external_hybrid          # dry run: plan + estimated cost, calls nothing
    python -m ledgerlens.external_hybrid --yes    # runs it (Anthropic credentials; costs money)

Uses the production routing policy unchanged (pipeline.RoutingPolicy defaults): rows the model
is unsure about, or whose merchant isn't in the directory, go to the LLM, one call per merchant
key. Writes results/external_hybrid.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ledgerlens.external_eval import OURS, assign_splits, fetch, load, merchant_groups, metrics, score
from ledgerlens.llm import PRICES, LLMClassifier
from ledgerlens.llm_eval import estimate_cost
from ledgerlens.pipeline import Enricher, RoutingPolicy, apply_llm_fallback, routed_mask


def sample(n: int, seed: int = 0) -> pd.DataFrame:
    df = load(fetch())
    df = df[df["truth"].notna() & df["direction"].notna()].reset_index(drop=True)
    g = merchant_groups(df)
    held = g[assign_splits(g) == "test"]
    per = max(1, n // held["truth"].nunique())  # same count per bucket, so no class dominates
    return (held.groupby("truth", group_keys=False).apply(lambda x: x.sample(min(len(x), per), random_state=seed))
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid eval on the independent benchmark (costs money with --yes)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", default="claude-haiku-5-5", choices=sorted(PRICES))  # what production uses
    ap.add_argument("--enricher", type=Path, default=Path("models/enricher.joblib"))
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--yes", action="store_true", help="actually call the API")
    ap.add_argument("--out", type=Path, default=Path("results/external_hybrid"))
    args = ap.parse_args()

    df = sample(args.n)
    enricher = Enricher.load(args.enricher)
    x = pd.DataFrame({"description": df["raw"], "amount": df["direction"].map({"debit": -42.0, "credit": 1200.0}),
                      "date": "2026-03-15", "user_id": [f"u{i}" for i in range(len(df))]})
    enriched = enricher.enrich(x)
    policy = RoutingPolicy()
    routed, _ = routed_mask(enriched, policy)
    keys = enriched.loc[routed, "merchant_key"].nunique()
    est = estimate_cost(keys, args.model, args.batch_size)
    print(f"{len(df):,} held-out rows; {int(routed.sum()):,} routed ({routed.mean():.0%}) = {keys:,} unique merchant "
          f"keys -> {-(-keys // args.batch_size)} requests on {args.model}; estimated cost ≈ ${est:.2f}")
    if not args.yes:
        print("dry run: pass --yes to call the API")
        return

    import anthropic

    clf = LLMClassifier(anthropic.Anthropic(), model=args.model, batch_size=args.batch_size)
    hybrid = apply_llm_fallback(enriched, clf, policy)
    model_only = score(df, enricher)
    both = model_only.assign(hybrid=hybrid["category"].map(OURS).fillna("(other)").values)
    r = {
        "rows": len(df), "routed_share": round(float(routed.mean()), 4), "llm_model": args.model,
        "enricher": str(args.enricher), "model_only": metrics(both), "hybrid": metrics(both, "hybrid"),
        "llm_requests": clf.usage.requests, "cost_usd": round(clf.usage.cost_usd(), 4),
        "cost_per_1k_txns_usd": round(clf.usage.cost_usd() / len(df) * 1000, 4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(r, indent=2))
    md = (f"## Full system on the independent benchmark ({r['rows']:,} held-out-merchant rows)\n\n"
          f"| | accuracy | macro-F1 |\n|---|---|---|\n"
          f"| model only | {r['model_only']['accuracy']:.1%} | {r['model_only']['macro_f1']:.3f} |\n"
          f"| **model + {args.model} fallback** | **{r['hybrid']['accuracy']:.1%}** | **{r['hybrid']['macro_f1']:.3f}** |\n\n"
          f"{r['routed_share']:.0%} of rows routed to the LLM; {r['llm_requests']} requests; "
          f"measured cost ${r['cost_usd']:.3f} (${r['cost_per_1k_txns_usd']:.3f} per 1K transactions).\n")
    Path(f"{args.out}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
