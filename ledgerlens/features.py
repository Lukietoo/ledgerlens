"""Label-free tabular features for the category model.

Three groups, kept separate so the ablation table can add them one at a time:

* STRUCT:  what the raw descriptor's *format* says (processor, ACH entry type, online, ...)
* AMOUNT:  sign, magnitude, price-point patterns ($x.99 is a subscription tell)
* HISTORY: per-user series stats for the same merchant (how often, how regular, how
           stable the amount). No labels involved.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ledgerlens.baseline import categorize, detect_recurring
from ledgerlens.boilerplate import learn_boilerplate, strip_clean
from ledgerlens.normalize import canon, normalize

ENTRY_TYPES: dict[str, str] = {
    "PAYROLL": "pay", "DIR DEP": "pay", "DIRECT DEP": "pay", "SALARY": "pay",
    "RENT": "rent", "LOAN": "loan", "TUITION": "tuition",
    "AUTOPAY": "bill", "AUTO PAY": "bill", "BILL PAY": "bill", "WEB PMTS": "bill", "ONLINE PMT": "bill",
    "TRANSFER": "xfer", "XFER": "xfer",
}
_ENTRY_RE = re.compile("|".join(sorted(map(re.escape, ENTRY_TYPES), key=len, reverse=True)))
_ACH_RE = re.compile(r"\b(?:PPD|WEB|CCD)\b|DES:|CO ID:")
_ONLINE_RE = re.compile(r"\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b|\.COM\b|\bWWW\.|\bHELP\.")
_STORE_RE = re.compile(r"#\s*\d|\bSTORE \d")

STRUCT = ["processor", "entry", "is_ach", "is_online", "has_store_no", "recurring_auth", "n_tokens", "clean_len"]
AMOUNT = ["amount", "log_abs_amount", "is_credit", "cents", "ends_99", "is_round"]
# Per-user only. A cross-user popularity feature ("how many users share this merchant") was
# tried and removed: computed in-batch it's ~1 for a single user's upload -> train/serve skew.
HISTORY = ["hist_n", "hist_median_gap", "hist_amount_cv", "hist_dom_std", "hist_rule_recurring"]
# What history features look like for a merchant seen once (no series yet).
HISTORY_MISSING = {"hist_n": 1, "hist_median_gap": -1, "hist_amount_cv": 0.0, "hist_dom_std": -1,
                   "hist_rule_recurring": 0}
RULES = ["rule_cat"]
CATEGORICAL = ["processor", "entry", "res_top1_cat", "rule_cat"]


def add_clean(df: pd.DataFrame, boilerplate: str | dict | None = "auto") -> pd.DataFrame:
    """Adds ``clean`` / ``processor`` / ``key`` (normalizer output, used for CATEGORY features)
    and ``name_clean`` (the same, minus each statement's learned boilerplate, used only to NAME
    the merchant).

    The split is measured, not cosmetic: feeding the learner's output to the category model too
    improved names but cost 1-3.5 pts of category accuracy on formats it already knew (README,
    "Unseen bank formats"). ``boilerplate``: "auto" learns per user from their rows in ``df``; a
    dict {user_id: words} uses words learned elsewhere (a user's full statement); None = off.
    """
    df = df.copy()
    norm = {d: normalize(d) for d in df["description"].unique()}
    df["clean"] = df["description"].map(lambda d: norm[d].clean)
    df["processor"] = df["description"].map(lambda d: norm[d].processor or "NONE")
    df["key"] = df["clean"].map(canon)
    df["name_clean"] = df["clean"]
    if boilerplate is not None and "user_id" in df:
        learned = boilerplate if isinstance(boilerplate, dict) else {
            u: learn_boilerplate(g) for u, g in df.groupby("user_id")["description"]}
        # after normalizing: the rules anchor on the very words the learner removes
        df["name_clean"] = [strip_clean(c, learned.get(u, set()))[0] for c, u in zip(df["clean"], df["user_id"])]
    return df


def struct_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["description"].str.upper()
    amt = df["amount"].astype(float)
    cents = (amt.abs() * 100).round().astype(int) % 100

    def entry(s: str) -> str:
        m = _ENTRY_RE.search(s)
        return ENTRY_TYPES[m.group(0)] if m else "none"

    rule = {d: categorize(d) for d in df["description"].unique()}
    return pd.DataFrame({
        "rule_cat": df["description"].map(rule),
        "processor": df["processor"],
        "entry": raw.map(entry),
        "is_ach": raw.str.contains(_ACH_RE).astype(int),
        "is_online": raw.str.contains(_ONLINE_RE).astype(int),
        "has_store_no": raw.str.contains(_STORE_RE).astype(int),
        "recurring_auth": raw.str.contains("RECURRING").astype(int),
        "n_tokens": df["clean"].str.split().str.len(),
        "clean_len": df["clean"].str.len(),
        "amount": amt,
        "log_abs_amount": np.log1p(amt.abs()),
        "is_credit": (amt > 0).astype(int),
        "cents": cents,
        "ends_99": (cents == 99).astype(int),
        "is_round": (cents == 0).astype(int),
    }, index=df.index)


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "g": df["user_id"].astype(str) + "|" + df["key"],
        "date": pd.to_datetime(df["date"], format="mixed"),
        "abs_amt": df["amount"].astype(float).abs(),
    }, index=df.index).sort_values("date")
    gb = d.groupby("g")
    d["gap"] = gb["date"].diff().dt.days
    d["dom"] = d["date"].dt.day
    stats = gb.agg(n=("abs_amt", "size"), mean=("abs_amt", "mean"), std=("abs_amt", "std"),
                   dom_std=("dom", "std"))
    stats["median_gap"] = d.groupby("g")["gap"].median()
    stats["cv"] = (stats["std"] / stats["mean"].replace(0, np.nan)).fillna(0)
    s = stats.reindex(d["g"]).set_axis(d.index)

    rule = detect_recurring(
        (u, desc, dt.date(), a) for u, desc, dt, a in
        zip(df["user_id"], df["description"], pd.to_datetime(df["date"], format="mixed"), df["amount"].astype(float))
    )
    out = pd.DataFrame({
        "hist_n": s["n"],
        "hist_median_gap": s["median_gap"].fillna(-1),
        "hist_amount_cv": s["cv"],
        "hist_dom_std": s["dom_std"].fillna(-1),
    }).reindex(df.index)
    out["hist_rule_recurring"] = np.array(rule, dtype=int)
    return out
