"""Evaluate the pipeline on synthetic and/or hand-labeled real transactions.

    python -m ledgerlens.evaluate                          # synthetic test split
    python -m ledgerlens.evaluate --real data/real/labeled.csv

Prints a markdown report and writes JSON to results/ so numbers can be tracked
across weeks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ledgerlens import baseline
from ledgerlens.normalize import canon, normalize
from ledgerlens.taxonomy import CATEGORIES, UNKNOWN


def category_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = list(CATEGORIES)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    confusions = Counter((t, pr) for t, pr in zip(y_true, y_pred) if t != pr)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "coverage": sum(p_ != UNKNOWN for p_ in y_pred) / max(1, len(y_pred)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(labels)
        },
        "top_confusions": [{"true": t, "pred": pr, "count": n} for (t, pr), n in confusions.most_common(10)],
    }


def merchant_exact_match(descs: list[str], merchants: list[str]) -> float:
    hits = sum(canon(normalize(d).clean) == canon(m) for d, m in zip(descs, merchants))
    return hits / max(1, len(descs))


def eval_synthetic(path: Path, split: str) -> dict:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    evald = [r for r in rows if r["split"] == split]
    descs = [r["description"] for r in evald]
    y_true = [r["category"] for r in evald]
    y_pred = [baseline.categorize(d) for d in descs]

    by_kind = {}
    for kind in sorted({r["merchant_kind"] for r in evald}):
        idx = [i for i, r in enumerate(evald) if r["merchant_kind"] == kind]
        by_kind[kind] = {
            "n": len(idx),
            "accuracy": accuracy_score([y_true[i] for i in idx], [y_pred[i] for i in idx]),
        }

    # Recurring detection needs full user histories, so it runs on every row
    # (it uses no learned parameters, so there's nothing to leak).
    flags = baseline.detect_recurring(
        (r["user_id"], r["description"], date.fromisoformat(r["date"]), float(r["amount"])) for r in rows
    )
    truth = [r["is_recurring"] == "1" for r in rows]
    rp, rr, rf, _ = precision_recall_fscore_support(truth, flags, average="binary", zero_division=0)

    return {
        "dataset": str(path),
        "split": split,
        "n_merchants": len({r["merchant"] for r in evald}),
        "category": category_metrics(y_true, y_pred),
        "category_by_merchant_kind": by_kind,
        "merchant_exact_match": merchant_exact_match(descs, [r["merchant"] for r in evald]),
        "recurring": {"n": len(rows), "precision": rp, "recall": rr, "f1": rf},
    }


def eval_real(path: Path) -> dict:
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("category") and r["category"] != "SKIP"]
    descs = [r["raw_description"] for r in rows]
    y_true = [r["category"] for r in rows]
    y_pred = [baseline.categorize(d) for d in descs]
    return {
        "dataset": str(path),
        "category": category_metrics(y_true, y_pred),
        "merchant_exact_match": merchant_exact_match(descs, [r["merchant"] for r in rows]),
    }


def to_markdown(name: str, res: dict) -> str:
    c = res["category"]
    lines = [
        f"### {name}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| transactions evaluated | {c['n']:,} |",
        f"| category accuracy | {c['accuracy']:.1%} |",
        f"| category macro-F1 | {c['macro_f1']:.3f} |",
        f"| rule coverage (non-UNKNOWN) | {c['coverage']:.1%} |",
        f"| merchant exact match (normalizer only) | {res['merchant_exact_match']:.1%} |",
    ]
    if "recurring" in res:
        r = res["recurring"]
        lines.append(f"| recurring P / R / F1 | {r['precision']:.3f} / {r['recall']:.3f} / {r['f1']:.3f} |")
    if "category_by_merchant_kind" in res:
        for k, v in res["category_by_merchant_kind"].items():
            lines.append(f"| accuracy on {k} merchants (n={v['n']:,}) | {v['accuracy']:.1%} |")
    lines += ["", "| category | P | R | F1 | support |", "|---|---|---|---|---|"]
    for cat, m in c["per_class"].items():
        if m["support"]:
            lines.append(f"| {cat} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']:,} |")
    lines += ["", "Top confusions (true → predicted):", ""]
    lines += [f"- {x['true']} → {x['pred']}: {x['count']:,}" for x in c["top_confusions"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate rules baseline")
    ap.add_argument("--synthetic", type=Path, default=Path("data/synthetic/transactions.csv"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--real", type=Path, help="hand-labeled CSV from scripts/label.py")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    if args.synthetic.exists():
        res = eval_synthetic(args.synthetic, args.split)
        (args.out_dir / f"baseline_synthetic_{args.split}.json").write_text(json.dumps(res, indent=2))
        reports.append(to_markdown(f"Rules baseline: synthetic ({args.split} split, unseen merchants)", res))
    if args.real:
        res = eval_real(args.real)
        (args.out_dir / "baseline_real.json").write_text(json.dumps(res, indent=2))
        reports.append(to_markdown("Rules baseline: real hand-labeled", res))
    if not reports:
        raise SystemExit("nothing to evaluate: run `python -m ledgerlens.synth` or pass --real")
    print("\n\n".join(reports))


if __name__ == "__main__":
    main()
