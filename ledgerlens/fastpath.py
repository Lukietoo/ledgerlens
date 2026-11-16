"""Pandas-free inference for small requests (the common API case: one or a few transactions).

Profiling showed a single-transaction request spending almost all of its ~10 ms in pandas
overhead (frame construction, groupbys, dtype checks), not in the model. This module computes
exactly the same features with plain Python + NumPy and feeds LightGBM a NumPy matrix.

Contract: identical outputs to Enricher.enrich (the pandas path), checked by
tests/test_fastpath.py on thousands of real test-split requests. Every helper below mirrors one
pandas expression in features.py / model.py; the comments name it.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime

import numpy as np
import pandas as pd

from ledgerlens.baseline import categorize, detect_recurring
from ledgerlens.boilerplate import learn_boilerplate, strip_clean
from ledgerlens.features import _ACH_RE, _ENTRY_RE, _ONLINE_RE, _STORE_RE, CATEGORICAL, ENTRY_TYPES
from ledgerlens.model import CAT_INDEX, PROCESSORS
from ledgerlens.normalize import canon, display_name, normalize
from ledgerlens.taxonomy import CATEGORIES, UNKNOWN

FAST_MAX_ROWS = 16  # above this, the vectorized pandas path wins

_CATS = {  # same fixed category lists as model.as_categoricals -> same integer codes
    "res_top1_cat": list(CATEGORIES),
    "rule_cat": [*CATEGORIES, UNKNOWN],
    "entry": ["none", "pay", "rent", "loan", "tuition", "bill", "xfer"],
    "processor": PROCESSORS,
}
_CODE = {col: {v: float(i) for i, v in enumerate(vals)} for col, vals in _CATS.items()}


def _parse_date(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:  # "03/14/2026" etc.: defer to the same parser the pandas path uses
        return pd.to_datetime(s, format="mixed").to_pydatetime()


def _sample_std(xs: list[float]) -> float:  # pandas .std() is the sample std (ddof=1); NaN for n=1
    return statistics.stdev(xs) if len(xs) > 1 else math.nan


def _history(rows: list[dict]) -> list[dict]:
    """features.history_features for a handful of rows."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(f"{r['user_id']}|{r['key']}", []).append(i)
    out: list[dict] = [{} for _ in rows]
    for idx in groups.values():
        idx = sorted(idx, key=lambda i: rows[i]["dt"])
        amts = [abs(rows[i]["amount"]) for i in idx]
        gaps = [(rows[b]["dt"] - rows[a]["dt"]).days for a, b in zip(idx, idx[1:])]
        mean, std = statistics.fmean(amts), _sample_std(amts)
        cv = std / mean if mean != 0 and not math.isnan(std) else 0.0  # (std / mean.replace(0, nan)).fillna(0)
        dom_std = _sample_std([float(rows[i]["dt"].day) for i in idx])
        stats = {
            "hist_n": len(idx),
            "hist_median_gap": statistics.median(gaps) if gaps else -1,
            "hist_amount_cv": cv,
            "hist_dom_std": -1 if math.isnan(dom_std) else dom_std,
        }
        for i in idx:
            out[i] = stats
    flags = detect_recurring((r["user_id"], r["description"], r["dt"].date(), r["amount"]) for r in rows)
    for o, f in zip(out, flags):
        o["hist_rule_recurring"] = int(f)
    return out


def enrich_records(enricher, records: list[dict]) -> list[dict]:
    """records: [{description, amount, date[, user_id]}] -> same fields Enricher.enrich returns."""
    rows = []
    for rec in records:
        desc = rec["description"]
        n = normalize(desc)
        amount = float(rec["amount"])
        rows.append({"description": desc, "amount": amount, "user_id": str(rec.get("user_id") or "me"),
                     "dt": _parse_date(str(rec["date"])), "clean": n.clean,
                     "processor": n.processor or "NONE", "key": canon(n.clean)})
    # name_clean: per-user boilerplate learned from this batch (needs >= 20 distinct lines, so
    # usually nothing here; kept for exact parity with add_clean)
    by_user: dict[str, list[str]] = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r["description"])
    learned = {u: learn_boilerplate(d) for u, d in by_user.items()}
    for r in rows:
        r["name_clean"] = strip_clean(r["clean"], learned[r["user_id"]])[0]

    hist = _history(rows)

    uniq = list(dict.fromkeys(r["clean"] for r in rows))
    proba = enricher.text_model.predict_proba(uniq)
    text = np.zeros((len(uniq), len(CATEGORIES)))
    text[:, [CAT_INDEX[c] for c in enricher.text_model.classes_]] = proba
    text_of = {t: text[i] for i, t in enumerate(uniq)}
    res = enricher.resolver.match(uniq)
    res_of = {t: i for i, t in enumerate(uniq)}

    names_uniq = list(dict.fromkeys(r["name_clean"] for r in rows))
    names = res if names_uniq == uniq else enricher.resolver.match(names_uniq)
    name_of = {t: i for i, t in enumerate(names_uniq)}

    feats = []
    for r, h in zip(rows, hist):
        raw = r["description"].upper()
        m = _ENTRY_RE.search(raw)
        cents = int(np.round(abs(r["amount"]) * 100)) % 100  # (amt.abs() * 100).round().astype(int) % 100
        j = res_of[r["clean"]]
        f = {
            "rule_cat": categorize(r["description"]), "processor": r["processor"],
            "entry": ENTRY_TYPES[m.group(0)] if m else "none",
            "is_ach": int(bool(_ACH_RE.search(raw))), "is_online": int(bool(_ONLINE_RE.search(raw))),
            "has_store_no": int(bool(_STORE_RE.search(raw))), "recurring_auth": int("RECURRING" in raw),
            "n_tokens": len(r["clean"].split()), "clean_len": len(r["clean"]),
            "amount": r["amount"], "log_abs_amount": math.log1p(abs(r["amount"])),
            "is_credit": int(r["amount"] > 0), "cents": cents, "ends_99": int(cents == 99), "is_round": int(cents == 0),
            **h,
            "res_sim": res.sim[j], "res_margin": res.margin[j], "res_top1_cat": res.category[j],
        }
        for k, c in enumerate(CATEGORIES):
            f[f"p_{c}"] = text_of[r["clean"]][k]
            f[f"r_{c}"] = res.cat_scores[j, k]
        feats.append(f)

    X = np.empty((len(feats), len(enricher.columns)))
    for i, f in enumerate(feats):
        for k, col in enumerate(enricher.columns):
            v = f[col]
            X[i, k] = _CODE[col].get(v, np.nan) if col in CATEGORICAL else float(v)
    # LGBMClassifier.predict_proba(DataFrame) == booster.predict(codes) with the same fixed categories
    P = enricher.lgbm.booster_.predict(X)

    out = []
    for r, p, h in zip(rows, P, hist):
        k = name_of[r["name_clean"]]
        known = bool(names.sim[k] >= enricher.tau)
        out.append({
            "description": r["description"], "amount": r["amount"],
            "merchant": str(names.merchant[k]) if known else display_name(r["name_clean"]),
            "merchant_in_directory": known,
            "merchant_key": canon(r["name_clean"]) + ("|in" if r["amount"] > 0 else "|out"),
            "category": str(enricher.lgbm.classes_[int(np.argmax(p))]),
            "confidence": round(float(np.max(p)), 4),
            "is_recurring": bool(h["hist_rule_recurring"]),
            "processor": None if r["processor"] == "NONE" else r["processor"],
        })
    return out


def is_eligible(n_rows: int, exclude) -> bool:
    return 0 < n_rows <= FAST_MAX_ROWS and not exclude


__all__ = ["FAST_MAX_ROWS", "enrich_records", "is_eligible", "date"]
