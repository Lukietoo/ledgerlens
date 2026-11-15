"""How confident is the model when it's right? Decides what to send to the LLM.

    python -m ledgerlens.routing

Runs the production Enricher over the synthetic test split (cold and warm directory),
then sweeps a confidence threshold: rows below it go to the LLM fallback. For each
threshold we report the share of traffic routed and the model's accuracy on the rows
it keeps. No API calls; the LLM's accuracy on routed rows is measured by llm_eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ledgerlens.pipeline import CONFIDENCE_ONLY, Enricher, RoutingPolicy, routed_mask

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]


def coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> list[dict]:
    out = []
    for t in THRESHOLDS:
        keep = confidence >= t
        out.append({
            "threshold": t,
            "routed_to_llm": float(1 - keep.mean()),
            "model_acc_on_kept": float(correct[keep].mean()) if keep.any() else None,
            "model_acc_on_routed": float(correct[~keep].mean()) if (~keep).any() else None,
            # if the LLM were perfect on what it gets: the ceiling for the hybrid
            "hybrid_upper_bound": float(np.where(keep, correct, True).mean()),
        })
    return out


def calibration(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_conf": float(confidence[m].mean()), "accuracy": float(correct[m].mean())})
    return rows


def policy_traffic(out: pd.DataFrame, correct: np.ndarray, policy: RoutingPolicy) -> dict:
    """Share of rows a policy routes, and the share that actually costs an LLM call when
    answers are cached per merchant key (first sighting pays, repeats are free)."""
    routed, low = routed_mask(out, policy)
    keys = out.loc[routed, "merchant_key"]
    unknown_confident = routed & ~low
    return {
        "routed_rows": float(routed.mean()),
        "llm_calls_per_txn_cached": float(keys.nunique() / len(out)),
        "unique_keys_routed": int(keys.nunique()),
        # how good is the model's own category on rows routed only for a name?
        "model_acc_unknown_confident": float(correct[unknown_confident].mean()) if unknown_confident.any() else None,
        "share_unknown_confident": float(unknown_confident.mean()),
    }


POLICIES = {
    "confidence < 0.8 (week 3)": CONFIDENCE_ONLY,
    "confidence < 0.8 OR merchant not in directory": RoutingPolicy(),
}


def ece(cal: list[dict]) -> float:
    n = sum(r["n"] for r in cal)
    return sum(r["n"] / n * abs(r["accuracy"] - r["mean_conf"]) for r in cal)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/transactions.csv"))
    ap.add_argument("--model", type=Path, default=Path("models/enricher.joblib"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    test = df[df.split == "test"].reset_index(drop=True)
    enricher = Enricher.load(args.model)
    report = {}
    for regime, exclude in (("cold", set(test.merchant)), ("warm", set())):
        out = enricher.enrich(test[["user_id", "description", "amount", "date"]], exclude=exclude)
        correct = (out["category"].to_numpy() == test["category"].to_numpy())
        conf = out["confidence"].to_numpy()
        cal = calibration(conf, correct)
        report[regime] = {"accuracy": float(correct.mean()), "curve": coverage_curve(conf, correct),
                          "calibration": cal, "ece": ece(cal),
                          "policies": {name: policy_traffic(out, correct, p) for name, p in POLICIES.items()}}
        print(f"{regime}: accuracy {correct.mean():.3f}, ECE {ece(cal):.3f}")

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "week3_routing.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report, len(test))
    (args.results / "week3_routing.md").write_text(md)
    print("\n" + md)


def to_markdown(r: dict, n: int) -> str:
    L = [f"## Confidence routing: synthetic test split ({n:,} txns, unseen merchants)", ""]
    for regime in ("cold", "warm"):
        x = r[regime]
        L += [f"### {regime} directory: model accuracy {x['accuracy']:.1%}, ECE {x['ece']:.3f}", "",
              "| route to LLM below | traffic routed | model acc on kept | model acc on routed | hybrid ceiling |",
              "|---|---|---|---|---|"]
        for c in x["curve"]:
            f = lambda v: "–" if v is None else f"{v:.1%}"  # noqa: E731
            L.append(f"| {c['threshold']:.2f} | {c['routed_to_llm']:.1%} | {f(c['model_acc_on_kept'])} | "
                     f"{f(c['model_acc_on_routed'])} | {c['hybrid_upper_bound']:.1%} |")
        L += ["", "Calibration (confidence bin → accuracy):", "", "| bin | n | mean confidence | accuracy |",
              "|---|---|---|---|"]
        L += [f"| {c['bin']} | {c['n']:,} | {c['mean_conf']:.2f} | {c['accuracy']:.1%} |" for c in x["calibration"]]
        L += ["", "Routing policies (LLM calls assume answers are cached per merchant key):", "",
              "| policy | rows routed | LLM calls per txn (cached) | rows routed only for a name "
              "| model category acc on those |", "|---|---|---|---|---|"]
        for name, p in x["policies"].items():
            acc = "–" if p["model_acc_unknown_confident"] is None else f"{p['model_acc_unknown_confident']:.1%}"
            L.append(f"| {name} | {p['routed_rows']:.1%} | {p['llm_calls_per_txn_cached']:.2%} | "
                     f"{p['share_unknown_confident']:.1%} | {acc} |")
        L.append("")
    L.append("Hybrid ceiling = accuracy if the LLM got every routed row right. Measured hybrid numbers: week3_llm.md.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
