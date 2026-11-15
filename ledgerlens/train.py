"""Week 2 experiment: train the stacked category model, run ablations, evaluate the resolver.

    python -m ledgerlens.train              # 3 seeds, reports mean ± std
    python -m ledgerlens.train --seeds 0    # quick single run

Writes results/week2.{md,json} and the trained pipeline (seed 0) to models/enricher.joblib.

Evaluation regimes on the test split (merchants never seen in training):
  cold: test merchants are also hidden from the resolver's directory -> pure generalization
  warm: full directory -> production setting where a merchant DB covers known chains
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ledgerlens import baseline
from ledgerlens.features import (AMOUNT, HISTORY, HISTORY_MISSING, RULES, STRUCT, add_clean,
                                 history_features, struct_amount_features)
from ledgerlens.model import (RESOLVER, TEXT, ResolverFeaturizer, as_categoricals, fit_lgbm,
                              fit_text_model, oof_stack_features, text_proba)
from ledgerlens.normalize import canon, normalize
from ledgerlens.resolver import Resolver
from ledgerlens.taxonomy import CATEGORIES

VARIANTS: list[tuple[str, list[str]]] = [
    ("LightGBM: text probs + format + amount", TEXT + STRUCT + AMOUNT),
    ("+ user history", TEXT + STRUCT + AMOUNT + HISTORY),
    ("+ rules prediction as a feature", TEXT + STRUCT + AMOUNT + HISTORY + RULES),
    ("+ merchant resolver (full model)", TEXT + STRUCT + AMOUNT + HISTORY + RULES + RESOLVER),
]
NO_HISTORY_ROW = "full model, first-seen txn (no user history)"


def cat_metrics(y_true, y_pred, kinds: pd.Series | None = None, clean_mask: np.ndarray | None = None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(CATEGORIES), average="macro", zero_division=0),
    }
    if clean_mask is not None:
        yt, yp = np.asarray(y_true)[clean_mask], np.asarray(y_pred)[clean_mask]
        out["acc_brand_clean"] = accuracy_score(yt, yp)
        out["macro_f1_brand_clean"] = f1_score(yt, yp, labels=list(CATEGORIES), average="macro", zero_division=0)
    if kinds is not None:
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        for k in ("chain", "local", "template"):
            m = (kinds == k).to_numpy()
            out[f"acc_{k}"] = accuracy_score(yt[m], yp[m]) if m.any() else None
    return out


def merchant_acc(pred: list[str], true: pd.Series, kinds: pd.Series) -> dict:
    hit = np.array([canon(p) == canon(t) for p, t in zip(pred, true)])
    out = {"overall": float(hit.mean())}
    for k in ("chain", "local", "template"):
        m = (kinds == k).to_numpy()
        out[k] = float(hit[m].mean())
    return out


@dataclass
class Data:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    resolver: Resolver
    rf: ResolverFeaturizer
    brand_named: set[str]
    clean: np.ndarray  # test rows not from a brand-named merchant

    @property
    def test_merchants(self) -> set[str]:
        return set(self.test["merchant"].unique())

    def M(self, pred) -> dict:
        return cat_metrics(self.test["category"].to_numpy(), pred,
                           self.test["merchant_kind"].reset_index(drop=True), self.clean)


def load(path: Path) -> Data:
    df = add_clean(pd.read_csv(path))
    feats = pd.concat([struct_amount_features(df), history_features(df)], axis=1)
    df = pd.concat([df, feats.drop(columns=[c for c in feats if c in df.columns])], axis=1)
    train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
    resolver = Resolver()
    # Test merchants the hand-written rules name by brand (NETFLIX, VENMO, ...) are not
    # truly "unseen" for any model that uses the rules. Report a brand-clean slice too.
    brand_named = set(test.loc[test["description"].map(baseline.is_brand_match), "merchant"])
    clean = ~test["merchant"].isin(brand_named).to_numpy()
    return Data(train, val, test, resolver, ResolverFeaturizer(resolver, df["clean"]), brand_named, clean)


def run_seed(d: Data, seed: int, hide_frac: float, history_dropout: float, log) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    train, val, test, rf = d.train, d.val, d.test, d.rf

    oof_text, oof_res = oof_stack_features(train, rf, hide_frac=hide_frac, seed=seed)
    text_model = fit_text_model(train["clean"], train["category"])
    val_merchants = val["merchant"].unique()
    val_hidden = set(val_merchants[rng.random(len(val_merchants)) < hide_frac])

    def X(part: pd.DataFrame, ptext: np.ndarray, res: pd.DataFrame) -> pd.DataFrame:
        text = pd.DataFrame(ptext, columns=TEXT, index=part.index)
        return pd.concat([part[STRUCT + AMOUNT + HISTORY + RULES], text, res[RESOLVER]], axis=1)

    def drop_history(Xp: pd.DataFrame, frac: float) -> pd.DataFrame:
        """History dropout: make some rows look like a first-seen merchant (no series yet)."""
        Xp = Xp.copy()
        m = rng.random(len(Xp)) < frac
        for col, v in HISTORY_MISSING.items():
            Xp.loc[m, col] = v
        return Xp

    X_tr = drop_history(X(train, oof_text, oof_res), history_dropout)
    X_va = drop_history(X(val, text_proba(text_model, val["clean"]), rf.frame(val["clean"], exclude=val_hidden)),
                        history_dropout)
    p_test = text_proba(text_model, test["clean"])
    X_cold = X(test, p_test, rf.frame(test["clean"], exclude=d.test_merchants))
    X_warm = X(test, p_test, rf.frame(test["clean"]))

    rows = [
        {"model": "Rules baseline (week 1)", "cold": d.M([baseline.categorize(x) for x in test.description]),
         "warm": None},
        {"model": "TF-IDF + logistic regression (text only)", "cold": d.M(np.array(CATEGORIES)[p_test.argmax(1)]),
         "warm": None},
    ]
    for name, cols in VARIANTS:
        clf = fit_lgbm(X_tr[cols], train["category"], X_va[cols], val["category"], seed=seed)
        pred_cold = clf.predict(as_categoricals(X_cold[cols]))
        warm = d.M(clf.predict(as_categoricals(X_warm[cols]))) if "res_sim" in cols else None
        rows.append({"model": name, "cold": d.M(pred_cold), "warm": warm})
        log(f"seed {seed} | {name}: cold {rows[-1]['cold']['accuracy']:.3f}"
            + (f", warm {warm['accuracy']:.3f}" if warm else ""))
    rows.append({"model": NO_HISTORY_ROW, "cold": d.M(clf.predict(as_categoricals(drop_history(X_cold, 1.0)[cols]))),
                 "warm": None})

    wrong = pd.DataFrame({"true": test["category"].to_numpy(), "pred": pred_cold,
                          "merchant": test["merchant"].to_numpy()})
    wrong = wrong[wrong.true != wrong.pred]
    confusions = (wrong.groupby(["true", "pred"])
                  .agg(n=("merchant", "size"), examples=("merchant", lambda s: ", ".join(s.value_counts().index[:3])))
                  .sort_values("n", ascending=False).head(10).reset_index().to_dict("records"))
    importance = pd.Series(clf.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False).head(15)

    # merchant resolution: tune abstain threshold on val, report on test
    def resolve(part, exclude, thr):
        return d.resolver.resolve(list(part["clean"]), thr, exclude, S=rf.S[rf.uniq.get_indexer(part["clean"])])

    grid = np.round(np.arange(0.30, 0.96, 0.025), 3)
    val_scores = [merchant_acc(resolve(val, val_hidden, t), val["merchant"], val["merchant_kind"])["overall"]
                  for t in grid]
    tau = float(grid[int(np.argmax(val_scores))])
    kinds = test["merchant_kind"]
    merch = {
        "threshold": tau,
        "normalizer_only (week 1)": merchant_acc([normalize(x).clean for x in test.description], test["merchant"], kinds),
        "resolver_cold": merchant_acc(resolve(test, d.test_merchants, tau), test["merchant"], kinds),
        "resolver_warm": merchant_acc(resolve(test, (), tau), test["merchant"], kinds),
    }
    report = {"seed": seed, "category": rows, "merchant": merch, "top_confusions_cold": confusions,
              "feature_importance_gain": (importance / importance.sum()).round(4).to_dict(),
              "lgbm_best_iteration": int(clf.best_iteration_ or clf.n_estimators)}
    artifacts = {"text_model": text_model, "lgbm": clf, "columns": cols, "tau": tau,
                 "resolver_entries": d.resolver.entries}
    return report, artifacts


def _agg(values: list) -> dict | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def aggregate(d: Data, reports: list[dict], args) -> dict:
    cat = []
    for i, row in enumerate(reports[0]["category"]):
        entry = {"model": row["model"]}
        for regime in ("cold", "warm"):
            if row[regime] is None:
                entry[regime] = None
                continue
            entry[regime] = {k: _agg([r["category"][i][regime][k] for r in reports]) for k in row[regime]}
        cat.append(entry)
    merch = {"threshold": _agg([r["merchant"]["threshold"] for r in reports])}
    for k in ("normalizer_only (week 1)", "resolver_cold", "resolver_warm"):
        merch[k] = {kk: _agg([r["merchant"][k][kk] for r in reports]) for kk in reports[0]["merchant"][k]}
    return {
        "seeds": [r["seed"] for r in reports],
        "settings": {"hide_frac": args.hide_frac, "history_dropout": args.history_dropout},
        "n": {"train": len(d.train), "val": len(d.val), "test": len(d.test),
              "test_merchants": len(d.test_merchants), "brand_named_test_merchants": sorted(d.brand_named),
              "brand_clean_test_txns": int(d.clean.sum())},
        "category": cat,
        "merchant": merch,
        "top_confusions_cold (seed 0)": reports[0]["top_confusions_cold"],
        "feature_importance_gain (seed 0)": reports[0]["feature_importance_gain"],
        "per_seed": reports,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + ablate the category model")
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/transactions.csv"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--history-dropout", type=float, default=0.2,
                    help="share of train/val rows whose user-history features are blanked")
    ap.add_argument("--hide-frac", type=float, default=0.8,
                    help="share of held-out merchants hidden from the directory when building stack features")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)  # noqa: E731

    d = load(args.data)
    log(f"features: {len(d.train):,} train / {len(d.val):,} val / {len(d.test):,} test rows; "
        f"{len(d.brand_named)} test merchants named by brand rules ({(~d.clean).mean():.1%} of test txns)")

    reports = []
    for seed in args.seeds:
        report, artifacts = run_seed(d, seed, args.hide_frac, args.history_dropout, log)
        reports.append(report)
        if seed == args.seeds[0]:
            args.models.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifacts, args.models / "enricher.joblib")

    agg = aggregate(d, reports, args)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "week2.json").write_text(json.dumps(agg, indent=2, default=float))
    md = to_markdown(agg)
    (args.results / "week2.md").write_text(md)
    print("\n" + md)
    log("done")


def to_markdown(r: dict) -> str:
    len_seeds = len(r["seeds"])

    def _pct(m: dict | None, std: bool = True) -> str:
        if m is None:
            return "–"
        return f"{m['mean']:.1%}" + (f" ± {m['std']:.1%}" if std and len_seeds > 1 else "")

    def _f1(m: dict) -> str:
        return f"{m['mean']:.3f}" + (f" ± {m['std']:.3f}" if len_seeds > 1 else "")

    n = r["n"]
    L = [f"## Category model: test split ({n['test']:,} txns, {n['test_merchants']} unseen merchants)", "",
         f"Mean ± std over {len_seeds} seeds ({', '.join(map(str, r['seeds']))}). "
         f"Settings: {r['settings']}.", "",
         "| model | accuracy | macro-F1 | brand-clean acc | brand-clean macro-F1 | chain | local | template "
         "| accuracy (warm directory) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for row in r["category"]:
        c, w = row["cold"], row["warm"]
        L.append(f"| {row['model']} | {_pct(c['accuracy'])} | {_f1(c['macro_f1'])} | {_pct(c['acc_brand_clean'])} | "
                 f"{_f1(c['macro_f1_brand_clean'])} | {_pct(c['acc_chain'], False)} | {_pct(c['acc_local'], False)} | "
                 f"{_pct(c['acc_template'], False)} | {_pct(w['accuracy']) if w else '–'} |")
    L += ["", "Cold (all columns except the last) = test merchants hidden from the merchant directory too. "
          "Warm = full directory (production setting). Chain/local/template columns are means.",
          f"Brand-clean = excludes the {len(n['brand_named_test_merchants'])} test merchants that the hand-written "
          f"rules name by brand ({', '.join(n['brand_named_test_merchants'])}), leaving "
          f"{n['brand_clean_test_txns']:,} txns. This is the leak-free generalization number.", "",
          "## Merchant resolution (exact canonical-name match)", "",
          f"Abstain threshold tuned on val: **{r['merchant']['threshold']['mean']:.3f}**", "",
          "| method | overall | chain | local | template |", "|---|---|---|---|---|"]
    for k in ("normalizer_only (week 1)", "resolver_cold", "resolver_warm"):
        m = r["merchant"][k]
        L.append(f"| {k} | {_pct(m['overall'], False)} | {_pct(m['chain'], False)} | {_pct(m['local'], False)} | "
                 f"{_pct(m['template'], False)} |")
    L += ["", "## Full model: top confusions (cold, seed 0)", "", "| true | predicted | txns | merchants |",
          "|---|---|---|---|"]
    L += [f"| {c['true']} | {c['pred']} | {c['n']:,} | {c['examples']} |" for c in r["top_confusions_cold (seed 0)"]]
    L += ["", "## Top features (LightGBM gain share, seed 0)", "", "| feature | share |", "|---|---|"]
    L += [f"| {k} | {v:.1%} |" for k, v in r["feature_importance_gain (seed 0)"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
