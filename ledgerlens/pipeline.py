"""Inference: load the trained artifacts and enrich a batch of transactions.

    from ledgerlens.pipeline import Enricher
    Enricher.load().enrich(df)   # df: description, amount, date[, user_id]

Returns one row per input with merchant, category, confidence, recurring flag and processor.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ledgerlens.features import AMOUNT, HISTORY, RULES, STRUCT, add_clean, history_features, struct_amount_features
from ledgerlens.llm import LLMClassifier, Txn
from ledgerlens.model import RESOLVER, TEXT, ResolverFeaturizer, as_categoricals, text_proba
from ledgerlens.normalize import canon, display_name
from ledgerlens.resolver import Resolver

DEFAULT_PATH = Path("models/enricher.joblib")
DEFAULT_LLM_THRESHOLD = 0.8  # see results/week3_routing.md: ~14-22% of traffic routed


@dataclass(frozen=True)
class RoutingPolicy:
    """Which rows go to the LLM, and which of its answers we keep.

    threshold:        rows with model confidence below this are routed
    unknown_merchants: also route rows whose merchant isn't in the directory
                       (the model often gets their *category* right but can only
                       echo the cleaned descriptor as a *name*)
    category_for:     "low_confidence" -> take the LLM's category only on low-confidence rows
                      "routed"         -> take it on every routed row
    """

    threshold: float = DEFAULT_LLM_THRESHOLD
    unknown_merchants: bool = True
    category_for: str = "low_confidence"

    def __post_init__(self):
        assert self.category_for in ("low_confidence", "routed"), self.category_for


CONFIDENCE_ONLY = RoutingPolicy(unknown_merchants=False, category_for="routed")  # week-3 behavior
# Default RoutingPolicy() is "names only": measured best on Haiku 5.5 (91.2% category / 82.2% merchant vs
# 91.2% / 52.8% confidence-only, 89.2% / 82.2% when also taking the LLM's category on confident rows).


def routed_mask(enriched: pd.DataFrame, policy: RoutingPolicy) -> tuple[np.ndarray, np.ndarray]:
    low = enriched["confidence"].to_numpy() < policy.threshold
    unknown = ~enriched["merchant_in_directory"].to_numpy(dtype=bool)
    return (low | unknown) if policy.unknown_merchants else low, low


def apply_llm_fallback(enriched: pd.DataFrame, classifier: LLMClassifier,
                       policy: RoutingPolicy | float = RoutingPolicy()) -> pd.DataFrame:
    """Send routed rows to the LLM (one call per *merchant key*, cached across calls).

    Adds ``source`` ("model" | "llm"). The LLM's merchant name is only used when the
    resolver didn't find the merchant in the directory (the directory is authoritative).
    A float ``policy`` means the week-3 confidence-only policy at that threshold.
    """
    if not isinstance(policy, RoutingPolicy):
        policy = RoutingPolicy(threshold=float(policy), unknown_merchants=False, category_for="routed")
    out = enriched.copy()
    out["source"] = "model"
    routed, low = routed_mask(out, policy)
    idx = np.flatnonzero(routed)
    if len(idx) == 0:
        return out
    # one representative row per merchant key; the classifier's cache dedupes across calls too
    reps: dict[str, int] = {}
    for i in idx:
        reps.setdefault(out.at[i, "merchant_key"], i)
    answers = classifier.classify_keyed({
        k: Txn(out.at[i, "description"], float(out.at[i, "amount"]), hint=out.at[i, "category"])
        for k, i in reps.items()
    })
    for i in idx:
        pred = answers.get(out.at[i, "merchant_key"])
        if pred is None:
            continue  # refusal / skipped: keep the model's answer
        if low[i] or policy.category_for == "routed":
            out.at[i, "category"] = pred.category
        if not out.at[i, "merchant_in_directory"] and pred.merchant:
            out.at[i, "merchant"] = pred.merchant
        out.at[i, "source"] = "llm"
    return out


class Enricher:
    def __init__(self, text_model, lgbm, columns: list[str], tau: float, resolver: Resolver):
        self.text_model, self.lgbm, self.columns, self.tau, self.resolver = text_model, lgbm, columns, tau, resolver

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> Enricher:
        a = joblib.load(path)
        return cls(a["text_model"], a["lgbm"], a["columns"], a["tau"], Resolver(a["resolver_entries"]))

    def enrich(self, df: pd.DataFrame, exclude: Collection[str] = ()) -> pd.DataFrame:
        """``exclude`` hides merchants from the directory (evaluation only: simulates cold start)."""
        df = df.reset_index(drop=True).copy()
        if "user_id" not in df:
            df["user_id"] = "me"
        df["amount"] = df["amount"].astype(float)
        df = add_clean(df)
        feats = pd.concat([struct_amount_features(df), history_features(df)], axis=1)
        res = ResolverFeaturizer(self.resolver, df["clean"]).frame(df["clean"], exclude=exclude)
        X = pd.concat([feats[STRUCT + AMOUNT + HISTORY + RULES],
                       pd.DataFrame(text_proba(self.text_model, df["clean"]), columns=TEXT),
                       res[RESOLVER]], axis=1)
        proba = self.lgbm.predict_proba(as_categoricals(X[self.columns]))
        # naming uses the boilerplate-stripped text; category above uses the text it trained on
        names = ResolverFeaturizer(self.resolver, df["name_clean"]).frame(df["name_clean"], exclude=exclude)
        known = names["res_sim"].to_numpy() >= self.tau
        return pd.DataFrame({
            "description": df["description"],
            "amount": df["amount"],
            "merchant": np.where(known, names["res_top1_merchant"], df["name_clean"].map(display_name)),
            "merchant_in_directory": known,
            # cache key for LLM answers: same cleaned merchant + same direction of money.
            # Sign matters: "UC BERKELEY" outgoing is tuition, incoming is payroll.
            "merchant_key": df["name_clean"].map(canon) + np.where(df["amount"].to_numpy() > 0, "|in", "|out"),
            "category": self.lgbm.classes_[proba.argmax(1)],
            "confidence": proba.max(1).round(4),
            "is_recurring": feats["hist_rule_recurring"].astype(bool),
            "processor": df["processor"].replace("NONE", None),
        })
