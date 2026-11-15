"""Does it work on a bank whose statement format it has never seen?

    python -m ledgerlens.format_eval                 # all 7 card + 4 ACH formats (~6 min)
    python -m ledgerlens.format_eval --formats card:wf

Leave-one-format-out. For each statement format (e.g. Wells Fargo's "PURCHASE AUTHORIZED
ON 03/14 ... S382... CARD 4821"):

  1. every user whose bank uses that format is removed from training;
  2. the model is scored on those users' transactions, twice:
       "model never saw it": the normalizer still has its hand-written rule for the format
       "nobody saw it":      that rule is switched off too (a new bank shows up tomorrow)
  3. "nobody saw it + learned": rule still off, but each statement's boilerplate is learned
     from the statement itself (boilerplate.py: no labels, no format knowledge);
  4. control: the same model on users whose format it *did* train on.

Scored only on rows the format actually shapes (card purchases for a card format;
payroll/rent/bills for an ACH format), split into merchants seen in training and the
merchant-held-out test merchants (unseen format AND unseen merchant: the strictest case).
Merchant resolution runs in the cold regime (test merchants hidden from the directory).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ledgerlens import baseline
from ledgerlens.boilerplate import learn_boilerplate, strip_boilerplate
from ledgerlens.features import HISTORY_MISSING, add_clean, history_features, struct_amount_features
from ledgerlens.model import (RESOLVER, TEXT, ResolverFeaturizer, as_categoricals, fit_lgbm,
                              fit_text_model, oof_stack_features, text_proba)
from ledgerlens.normalize import FORMAT_RULES, canon, disabled_rules
from ledgerlens.resolver import Resolver
from ledgerlens.synth import EMPLOYERS, PAYROLL_PROCESSORS, load_chain_merchants
from ledgerlens.train import VARIANTS, load

FULL = VARIANTS[-1][1]  # the full model's feature set
CARD_FORMATS = ["plain", "fixed", "pos", "wf", "boa", "title", "debit"]
ACH_FORMATS = ["chase", "boa", "wf", "plain"]
RAW_COLS = ["txn_id", "user_id", "date", "description", "amount", "merchant", "category",
            "merchant_kind", "split", "card_style", "ach_style"]


ACH_CHAINS = {m.name for m in load_chain_merchants() if m.rail == "ach"}
PAYROLL = {e for e, _ in EMPLOYERS} | {p for p, _ in PAYROLL_PROCESSORS}


def format_rows(df: pd.DataFrame, kind: str) -> pd.Series:
    """Rows whose descriptor is rendered by the user's card (or ACH) format.

    Payroll is matched on category AND employer: several employers are also stores
    (Starbucks, Target, Amazon...), and a coffee purchase is card-format, not payroll.
    """
    payroll = df["merchant"].isin(PAYROLL) & (df["category"] == "INCOME")
    ach = df["merchant"].isin(ACH_CHAINS) | payroll
    if kind == "ach":
        return ach
    return ~ach & df["merchant_kind"].isin(["chain", "local"])


def featurize(raw: pd.DataFrame, resolver: Resolver, text_model, exclude: set[str],
              rules_off: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame, ResolverFeaturizer]:
    """Same features the model trained on, recomputed from raw descriptors (optionally with
    some normalizer rules switched off)."""
    with disabled_rules(rules_off):
        d = add_clean(raw.reset_index(drop=True))
    feats = pd.concat([struct_amount_features(d), history_features(d)], axis=1)
    d = pd.concat([d, feats.drop(columns=[c for c in feats if c in d.columns])], axis=1)
    rf = ResolverFeaturizer(resolver, d["clean"])
    X = pd.concat([d.drop(columns=[c for c in TEXT + RESOLVER if c in d.columns]),
                   pd.DataFrame(text_proba(text_model, d["clean"]), columns=TEXT),
                   rf.frame(d["clean"], exclude=exclude)[RESOLVER]], axis=1)
    return d, X, rf


def score(raw: pd.DataFrame, clf, resolver, text_model, tau, exclude, rules_off=(), boilerplate=None) -> dict:
    if raw.empty:
        return {"n": 0}
    if boilerplate is not None:  # per-user words learned from that user's whole statement
        raw = raw.assign(description=[strip_boilerplate(d, boilerplate.get(u, set()))
                                      for d, u in zip(raw["description"], raw["user_id"])])
    d, X, rf = featurize(raw, resolver, text_model, exclude, rules_off)
    pred = clf.predict(as_categoricals(X[FULL]))
    merchants = resolver.resolve(list(d["clean"]), tau, exclude, S=rf.S[rf.uniq.get_indexer(d["clean"])])
    return {
        "n": int(len(d)),
        "category_acc": float(accuracy_score(d["category"], pred)),
        "merchant_exact": float(np.mean([canon(p) == canon(t) for p, t in zip(merchants, d["merchant"])])),
        "rules_baseline_acc": float(accuracy_score(d["category"], [baseline.categorize(x) for x in d["description"]])),
    }


def fit_without(data, users: set[str], seed: int, hide_frac=0.8, history_dropout=0.2):
    """Full stacked model trained with the held-out users removed (as in train.run_seed)."""
    rng = np.random.default_rng(seed)
    tr = data.train[~data.train["user_id"].isin(users)]
    va = data.val[~data.val["user_id"].isin(users)]
    oof_text, oof_res = oof_stack_features(tr, data.rf, hide_frac=hide_frac, seed=seed)
    text_model = fit_text_model(tr["clean"], tr["category"])
    val_m = va["merchant"].unique()
    val_hidden = set(val_m[rng.random(len(val_m)) < hide_frac])

    def X(part, ptext, res):
        Xp = pd.concat([part.drop(columns=[c for c in TEXT + RESOLVER if c in part.columns]),
                        pd.DataFrame(ptext, columns=TEXT, index=part.index), res[RESOLVER]], axis=1)
        m = rng.random(len(Xp)) < history_dropout
        for col, v in HISTORY_MISSING.items():
            Xp.loc[m, col] = v
        return Xp

    X_tr = X(tr, oof_text, oof_res)
    X_va = X(va, text_proba(text_model, va["clean"]), data.rf.frame(va["clean"], exclude=val_hidden))
    clf = fit_lgbm(X_tr[FULL], tr["category"], X_va[FULL], va["category"], seed=seed)
    return clf, text_model


def run(data_path: Path, model_path: Path, formats: list[str], seed: int, log) -> list[dict]:
    data = load(data_path)
    raw_all = pd.read_csv(data_path)
    if raw_all["card_style"].isna().all():
        raise SystemExit("this data has no card_style/ach_style columns; regenerate it: python -m ledgerlens.synth")
    resolver, tau = data.resolver, float(joblib.load(model_path)["tau"])
    test_merchants = set(data.test["merchant"])
    rows = []
    for fmt in formats:
        kind, style = fmt.split(":")
        col = "card_style" if kind == "card" else "ach_style"
        users = set(raw_all.loc[raw_all[col] == style, "user_id"])
        clf, text_model = fit_without(data, users, seed)
        held, ctrl = raw_all[raw_all["user_id"].isin(users)], raw_all[~raw_all["user_id"].isin(users)]
        held, ctrl = held[format_rows(held, kind)][RAW_COLS], ctrl[format_rows(ctrl, kind)][RAW_COLS]
        rules = FORMAT_RULES.get(fmt, ())
        # learned from each held-out user's full statement: unlabeled data a real upload would include
        learned = {u: learn_boilerplate(g["description"]) for u, g in raw_all[raw_all["user_id"].isin(users)].groupby("user_id")}
        r = {"format": fmt, "held_out_users": len(users), "format_rules": list(rules)}
        for name, sub in (("unseen_merchants", lambda x: x[x["split"] == "test"]),
                          ("seen_merchants", lambda x: x[x["split"] == "train"])):
            r[name] = {
                "control": score(sub(ctrl), clf, resolver, text_model, tau, test_merchants),
                "model_never_saw": score(sub(held), clf, resolver, text_model, tau, test_merchants),
                "nobody_saw": (score(sub(held), clf, resolver, text_model, tau, test_merchants, rules)
                               if rules else None),
                "nobody_saw_learned": (score(sub(held), clf, resolver, text_model, tau, test_merchants, rules, learned)
                                       if rules else None),
            }
        u = r["unseen_merchants"]
        log(f"{fmt:<11} control {u['control']['category_acc']:.3f} | never-saw {u['model_never_saw']['category_acc']:.3f}"
            + (f" | nobody-saw {u['nobody_saw']['category_acc']:.3f} | +learned {u['nobody_saw_learned']['category_acc']:.3f}"
               f" | merchant never/nobody/+learned {u['model_never_saw']['merchant_exact']:.3f}/{u['nobody_saw']['merchant_exact']:.3f}"
               f"/{u['nobody_saw_learned']['merchant_exact']:.3f}" if u["nobody_saw"] else " | (no format rule)"))
        rows.append(r)
    return rows


def _w(rows, key, cond, metric):
    """Row-weighted average across formats (weights = rows scored)."""
    pairs = [(r[key][cond][metric], r[key][cond]["n"]) for r in rows if r[key][cond] and r[key][cond]["n"]]
    n = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / n if n else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind in ("card", "ach"):
        rs = [r for r in rows if r["format"].startswith(kind)]
        if not rs:
            continue
        with_rule = [r for r in rs if r["format_rules"]]
        out[kind] = {k: {
            "control": _w(rs, k, "control", "category_acc"),
            "model_never_saw": _w(rs, k, "model_never_saw", "category_acc"),
            "nobody_saw_formats_with_rules": _w(with_rule, k, "nobody_saw", "category_acc"),
            "nobody_saw_learned": _w(with_rule, k, "nobody_saw_learned", "category_acc"),
            "merchant_nobody_saw_learned": _w(with_rule, k, "nobody_saw_learned", "merchant_exact"),
            "same_formats_model_never_saw": _w(with_rule, k, "model_never_saw", "category_acc"),
            "merchant_control": _w(rs, k, "control", "merchant_exact"),
            "merchant_model_never_saw": _w(rs, k, "model_never_saw", "merchant_exact"),
            "merchant_nobody_saw": _w(with_rule, k, "nobody_saw", "merchant_exact"),
            "merchant_same_formats_model_never_saw": _w(with_rule, k, "model_never_saw", "merchant_exact"),
        } for k in ("unseen_merchants", "seen_merchants")}
    return out


def to_markdown(rows: list[dict], summary: dict, seed: int) -> str:
    p = lambda v: "–" if v is None else f"{v:.1%}"  # noqa: E731
    L = ["## Unseen bank formats (leave-one-format-out)", "",
         f"Seed {seed}. Each row retrains the full model with every user of that statement format removed. "
         "Scored on rows the format shapes, for merchants that are *also* held out of training "
         "(the strictest case). **Never saw** = the ML model never saw the format, but the normalizer's "
         "hand-written rule for it is still on; **nobody saw** = that rule is switched off too.", "",
         "| held-out format | users | rows | control (format seen) | never saw | nobody saw | nobody saw + learned "
         "| rules baseline | merchant exact: control / never / nobody / +learned |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        u = r["unseen_merchants"]
        c, m, n, nl = u["control"], u["model_never_saw"], u["nobody_saw"], u["nobody_saw_learned"]
        L.append(f"| {r['format']} | {r['held_out_users']} | {m['n']:,} | {p(c['category_acc'])} | "
                 f"{p(m['category_acc'])} | {p(n['category_acc']) if n else 'no rule'} | {p(nl['category_acc']) if nl else '–'} | "
                 f"{p(m['rules_baseline_acc'])} | {p(c['merchant_exact'])} / {p(m['merchant_exact'])} / "
                 f"{p(n['merchant_exact']) if n else '–'} / {p(nl['merchant_exact']) if nl else '–'} |")
    L += ["", "### Summary (row-weighted across formats)", "",
          "| | merchants | control | model never saw format | nobody saw format* | nobody saw + learned* "
          "| merchant exact: control / never / nobody* / +learned* |",
          "|---|---|---|---|---|---|---|"]
    for kind, s in summary.items():
        for k, label in (("unseen_merchants", "unseen"), ("seen_merchants", "seen")):
            x = s[k]
            L.append(f"| {kind} formats | {label} | {p(x['control'])} | {p(x['model_never_saw'])} | "
                     f"{p(x['nobody_saw_formats_with_rules'])} (vs {p(x['same_formats_model_never_saw'])}) | "
                     f"{p(x['nobody_saw_learned'])} | {p(x['merchant_control'])} / {p(x['merchant_model_never_saw'])} / "
                     f"{p(x['merchant_nobody_saw'])} (vs {p(x['merchant_same_formats_model_never_saw'])}) / "
                     f"{p(x['merchant_nobody_saw_learned'])} |")
    L += ["", "\\* Only formats that have a format-specific normalizer rule; \"vs\" is the never-saw number on those "
          "same formats, so the pair isolates what switching off the rule costs."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/synthetic/transactions.csv"))
    ap.add_argument("--model-path", type=Path, default=Path("models/enricher.joblib"))
    ap.add_argument("--formats", nargs="+",
                    default=[f"card:{s}" for s in CARD_FORMATS] + [f"ach:{s}" for s in ACH_FORMATS])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    t0 = time.time()
    rows = run(args.data, args.model_path, args.formats, args.seed,
               lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True))
    summary = summarize(rows)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "week5_formats.json").write_text(json.dumps({"seed": args.seed, "rows": rows, "summary": summary},
                                                                indent=2))
    md = to_markdown(rows, summary, args.seed)
    (args.results / "week5_formats.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
