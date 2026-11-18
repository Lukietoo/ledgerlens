"""Independent benchmark: a third party's dataset, scored with zero tuning on it.

    python -m ledgerlens.external_eval                    # shipped model, all rows
    python -m ledgerlens.external_eval --heldout          # only merchants withheld from training
    python -m ledgerlens.external_eval --write-training data/synthetic/combined.csv
    python -m ledgerlens.train --data data/synthetic/combined.csv   # the shipped model's recipe

Every other number in this repo is on data from my own generator, so a model could look good
by learning that generator's habits. This scores the shipped model on a dataset someone else
built: DoDataThings/us-bank-transaction-categories-v2 (MIT, 68,000 rows, 500+ real merchant
names in 8 US statement styles; generator: github.com/wnstnb/foliome). Nothing here was tuned
on it: the mapping below was written from the two category lists before any score was seen.

The dataset has no amounts or dates, which the model normally uses. Each row gets one
label-independent stand-in amount for its direction (its [debit]/[credit] prefix) and is its
own user, so there is no history: every row is a first-seen merchant, the hardest setting.
A sensitivity run checks the result doesn't hinge on the stand-in value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ledgerlens.baseline import categorize
from ledgerlens.pipeline import Enricher

URL = ("https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2/resolve/"
       "3e8d052cb7b362fb1a04b6064d2331982553ea56/transactions-synthetic.csv")
SHA256 = "0424aed6e76f74a5b3b1ff61ccec43bc321622e6806da353b910b3b2c8108f6e"
PATH = Path("data/external/dodatathings/transactions-synthetic.csv")

# their label -> the bucket we score in. Ours is finer in places, so ours collapses into theirs.
THEIRS = {
    "Restaurants": "Restaurants", "Groceries": "Groceries", "Shopping": "Shopping",
    "Transportation": "Transportation", "Entertainment": "Entertainment", "Utilities": "Utilities",
    "Subscription": "Subscription", "Healthcare": "Healthcare", "Travel": "Travel",
    "Education": "Education", "Transfer": "Transfer", "Income": "Income", "Fees": "Fees",
    "Mortgage": "Housing", "Rent": "Housing",
    # no counterpart in our taxonomy: excluded and reported, not forced into a bucket
    "Insurance": None, "Personal Care": None,
}
OURS = {
    "DINING": "Restaurants", "COFFEE": "Restaurants", "GROCERIES": "Groceries", "SHOPPING": "Shopping",
    "TRANSPORT": "Transportation", "GAS": "Transportation", "ENTERTAINMENT": "Entertainment",
    "UTILITIES": "Utilities", "SUBSCRIPTIONS": "Subscription", "HEALTH": "Healthcare", "TRAVEL": "Travel",
    "EDUCATION": "Education", "TRANSFER": "Transfer", "INCOME": "Income", "FEES": "Fees",
    "HOUSING": "Housing",
    # no counterpart in theirs: always scored wrong
    "FITNESS": "(fitness)", "LOANS": "(loans)", "UNKNOWN": "(unknown)",
}
STAND_IN = {"debit": -42.0, "credit": 1200.0}  # fixed before scoring; see the sensitivity run


# their generator's merchant list (read as text, never executed), pinned to a commit
GEN_URL = ("https://raw.githubusercontent.com/wnstnb/foliome/a9a6d99e4cba4ac054e4ccbd7d3be7904ff053de/"
           "scripts/generate-training-data.js")
GEN_PATH = PATH.parent / "generate-training-data.js"
SPLIT_SALT = "ledgerlens-external-v1"
# their bucket -> our training label; sub-groups from their generator refine two buckets ours splits
TO_OURS = {"Restaurants": "DINING", "Groceries": "GROCERIES", "Shopping": "SHOPPING", "Transportation": "TRANSPORT",
           "Entertainment": "ENTERTAINMENT", "Utilities": "UTILITIES", "Subscription": "SUBSCRIPTIONS",
           "Healthcare": "HEALTH", "Travel": "TRAVEL", "Education": "EDUCATION", "Transfer": "TRANSFER",
           "Income": "INCOME", "Fees": "FEES", "Housing": "HOUSING"}
REFINE = {("Restaurants", "coffee"): "COFFEE", ("Transportation", "gas"): "GAS", ("Transportation", "ev"): "GAS"}


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def generator_merchants() -> dict[str, str]:
    """merchant name -> its sub-group in their generator (e.g. 'coffee', 'gas')."""
    if not GEN_PATH.exists():
        GEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GEN_URL, GEN_PATH)  # noqa: S310 (fixed https URL, pinned commit)
    text = GEN_PATH.read_text()
    start = text.index("const merchants = {")
    body = text[start:text.index("\n};", start)]
    out, sub = {}, None
    for line in body.splitlines():
        m = re.match(r"\s*(\w+): \[", line)
        if m:
            sub = m.group(1)
        for name in re.findall(r'"([^"]{2,60})"', line):
            out.setdefault(name, sub)
    return out


def merchant_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Which merchant each row is, so a split can keep every merchant on one side.

    Rows are matched to the longest merchant name from their generator that the description
    contains (88% of rows). The rest are template text (MORTGAGE PAYMENT, payroll lines,
    DR. PATEL PSYCHIATRY) and get a coarse fallback key: their first word.
    """
    names = generator_merchants()
    pats = sorted({_norm(n): n for n in names if len(_norm(n)) >= 3}.items(), key=lambda kv: -len(kv[0]))

    def match(raw: str) -> str | None:
        r = _norm(raw)
        return next((n for p, n in pats if p in r), None)

    m = df["raw"].map(match)
    first = df["raw"].str.upper().str.replace(r"[^A-Z ]", " ", regex=True).str.split().str[0].fillna("?")
    return df.assign(merchant=m.fillna("~" + first), matched=m.notna(), subgroup=m.map(names))


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Stable merchant-level split: 60% train, 15% val, 25% test, by a salted hash of the merchant."""
    def bucket(merchant: str) -> str:
        h = int(hashlib.sha256(f"{SPLIT_SALT}:{merchant}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "train" if h < 0.60 else "val" if h < 0.75 else "test"
    return df["merchant"].map(bucket)


def training_rows(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Their train/val rows in our synthetic schema (test rows are never written).

    Their data has no amounts. A constant stand-in here taught the model "amount == -42 means
    this source" (accuracy then moved 12 points with the stand-in at test time), so training
    rows get random amounts drawn independently of the label: the amount carries no signal.
    """
    d = df[df["split"] != "test"]
    rng = np.random.default_rng(seed)
    mag = np.exp(rng.normal(np.log(40.0), 1.2, len(d)))
    amount = np.where(d["direction"].to_numpy() == "credit", mag * 25, -mag).round(2)
    label = [REFINE.get((t, s), TO_OURS[t]) for t, s in zip(d["truth"], d["subgroup"])]
    return pd.DataFrame({
        "txn_id": [f"x{i:07d}" for i in range(len(d))], "user_id": [f"x{i}" for i in range(len(d))],
        "date": "2026-03-15", "description": d["raw"].values,
        "amount": amount, "merchant": d["merchant"].values,
        "category": label, "is_recurring": 0, "merchant_kind": "external", "split": d["split"].values,
        "card_style": "external", "ach_style": "external",
    })


def fetch(path: Path = PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)  # noqa: S310 (fixed https URL, pinned revision)
        tmp.rename(path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256:
        raise SystemExit(f"{path}: sha256 {got} != pinned {SHA256}")
    return path


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    m = df["description"].str.extract(r"^\[(debit|credit)\]\s*(.*)$")
    df["direction"], df["raw"] = m[0], m[1]
    df["truth"] = df["category"].map(THEIRS)
    return df


def score(df: pd.DataFrame, enricher: Enricher, scale: float = 1.0) -> pd.DataFrame:
    x = pd.DataFrame({
        "description": df["raw"],
        "amount": df["direction"].map(STAND_IN) * scale,
        "date": "2026-03-15",
        "user_id": [f"u{i}" for i in range(len(df))],  # one row per user: no history to lean on
    })
    out = enricher.enrich(x)
    return df.assign(pred=out["category"].map(OURS).fillna("(other)").values,
                     in_directory=out["merchant_in_directory"].astype(bool).values,
                     confidence=out["confidence"].values)


def metrics(d: pd.DataFrame, col: str = "pred") -> dict:
    # macro-F1 over THEIR buckets (evaluate.category_metrics assumes our label names)
    labels = sorted(d["truth"].unique())
    return {"rows": len(d), "accuracy": round(float(accuracy_score(d["truth"], d[col])), 4),
            "macro_f1": round(float(f1_score(d["truth"], d[col], labels=labels, average="macro",
                                             zero_division=0)), 4)}


def calibration(d: pd.DataFrame, bins: int = 10) -> dict:
    """Does a stated confidence of p mean right about p of the time? Expected calibration error
    (ECE) is the row-weighted gap between confidence and accuracy over equal-width bins."""
    ok = (d["truth"] == d["pred"]).to_numpy()
    conf = d["confidence"].to_numpy(dtype=float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        acc, mean_conf = float(ok[m].mean()), float(conf[m].mean())
        ece += m.mean() * abs(acc - mean_conf)
        rows.append({"bin": f"{b / bins:.1f}–{(b + 1) / bins:.1f}", "rows": int(m.sum()),
                     "mean_confidence": round(mean_conf, 3), "accuracy": round(acc, 3)})
    return {"ece": round(float(ece), 4), "bins": rows}


def by_confidence(d: pd.DataFrame) -> list[dict]:
    """If rows under a confidence cutoff went to a fallback, how good is what the model keeps?"""
    out = []
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        kept = d[d["confidence"] >= t]
        out.append({"threshold": t, "kept_share": round(len(kept) / len(d), 4),
                    "accuracy_kept": round(float((kept["truth"] == kept["pred"]).mean()), 4) if len(kept) else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=Path("models/enricher.joblib"))
    ap.add_argument("--out", type=Path, default=Path("results/external_dodatathings"))
    ap.add_argument("--heldout", action="store_true",
                    help="score only their test-split merchants (never in any training file)")
    ap.add_argument("--write-training", type=Path, metavar="CSV",
                    help="write our synthetic data + their train/val merchants to CSV (for ledgerlens.train) and exit")
    ap.add_argument("--synthetic", type=Path, default=Path("data/synthetic/transactions.csv"))
    args = ap.parse_args()

    df = load(fetch())
    excluded = Counter(df.loc[df["truth"].isna(), "category"])
    df = df[df["truth"].notna() & df["direction"].notna()].reset_index(drop=True)
    if args.write_training:
        g = merchant_groups(df)
        g["split"] = assign_splits(g)
        ext = training_rows(g)
        out = pd.concat([pd.read_csv(args.synthetic), ext], ignore_index=True)
        args.write_training.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.write_training, index=False)
        print(f"wrote {args.write_training}: {len(out) - len(ext):,} synthetic + {len(ext):,} external rows "
              f"({g.loc[g['split'] != 'test', 'merchant'].nunique()} merchants; their test merchants withheld)")
        return
    if args.heldout:
        g = merchant_groups(df)
        df = g[assign_splits(g) == "test"].reset_index(drop=True)
    enricher = Enricher.load(args.model)

    scored = score(df, enricher)
    scored["baseline"] = scored["raw"].map(categorize).map(OURS).fillna("(other)")
    cold = scored[~scored["in_directory"]]
    per_class = (scored.assign(ok=scored["truth"] == scored["pred"]).groupby("truth")["ok"].mean()
                 .sort_values().round(3).to_dict())
    confusions = (scored[scored["truth"] != scored["pred"]].groupby(["truth", "pred"]).size()
                  .sort_values(ascending=False).head(8))
    sens = {f"x{s}": metrics(score(df.sample(6000, random_state=0), enricher, s)) for s in (0.25, 4.0)}

    rep = {
        "slice": "held-out merchants (test split)" if args.heldout else "all rows",
        "model_path": str(args.model), "dataset": URL, "sha256": SHA256, "excluded_no_counterpart": dict(excluded),
        "model": metrics(scored), "rules_baseline": metrics(scored, "baseline"),
        "by_confidence": by_confidence(scored),
        "calibration": calibration(scored),
        "merchant_not_in_our_directory": metrics(cold) if len(cold) else None,
        "merchant_in_our_directory_share": round(float(scored["in_directory"].mean()), 4),
        "per_class_accuracy": per_class,
        "top_confusions": [{"truth": t, "pred": p, "rows": int(n)} for (t, p), n in confusions.items()],
        "amount_sensitivity_6k_sample": sens,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(rep, indent=2))
    Path(f"{args.out}.md").write_text(to_markdown(rep))
    print(to_markdown(rep))


def to_markdown(r: dict) -> str:
    c = r["merchant_not_in_our_directory"]
    L = [f"## Independent benchmark, {r['slice']} ({r['model_path']})", "",
         "[DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/"
         "us-bank-transaction-categories-v2) (MIT): built by someone else's generator from 500+ real merchant "
         "names in 8 US statement styles. With --heldout, only merchants withheld from every training file "
         "(merchant-level split) are scored. The category mapping was written before any score was seen. No amounts or dates in this data, and no history: every row is a "
         "first-seen merchant.", "",
         "| | rows | accuracy | macro-F1 |", "|---|---|---|---|",
         f"| rules baseline | {r['model']['rows']:,} | {r['rules_baseline']['accuracy']:.1%} | "
         f"{r['rules_baseline']['macro_f1']:.3f} |",
         f"| **Ledgerlens** | {r['model']['rows']:,} | **{r['model']['accuracy']:.1%}** | "
         f"**{r['model']['macro_f1']:.3f}** |"]
    if c:
        L.append(f"| Ledgerlens, merchant not in our directory | {c['rows']:,} | {c['accuracy']:.1%} | "
                 f"{c['macro_f1']:.3f} |")
    L += ["", f"{r['merchant_in_our_directory_share']:.0%} of rows matched a merchant in our directory. "
          f"Excluded (no matching category in our taxonomy): "
          + ", ".join(f"{k} ({v:,})" for k, v in r["excluded_no_counterpart"].items()) + ".", "",
          "Hardest classes: " + ", ".join(f"{k} {v:.0%}" for k, v in list(r["per_class_accuracy"].items())[:5])
          + ".", "", "Most common mistakes: "
          + "; ".join(f"{x['truth']} → {x['pred']} ({x['rows']:,})" for x in r["top_confusions"][:5]) + ".", "",
          "If low-confidence rows went to a fallback, what the model keeps: "
          + "; ".join(f"≥{x['threshold']:.1f}: {x['kept_share']:.0%} of rows at {x['accuracy_kept']:.1%}"
                      for x in r["by_confidence"][1:]) + ".", ""]
    if "calibration" in r:
        cal = r["calibration"]
        L += [f"Calibration: expected calibration error **{cal['ece']:.3f}** (0 = stated confidence matches "
              "accuracy exactly).", "", "| confidence | rows | mean confidence | accuracy |", "|---|---|---|---|"]
        L += [f"| {b['bin']} | {b['rows']:,} | {b['mean_confidence']:.2f} | {b['accuracy']:.1%} |"
              for b in cal["bins"]]
        L += [""]
    L += ["Stand-in amount ×0.25 / ×4 on a 6K sample: "
          + " / ".join(f"{v['accuracy']:.1%}" for v in r["amount_sensitivity_6k_sample"].values()) + ".", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
