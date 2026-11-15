"""Inference: load the trained artifacts and enrich a batch of transactions.

    from ledgerlens.pipeline import Enricher
    Enricher.load().enrich(df)   # df: description, amount, date[, user_id]

Returns one row per input with merchant, category, confidence, recurring flag and processor.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ledgerlens.features import AMOUNT, HISTORY, RULES, STRUCT, add_clean, history_features, struct_amount_features
from ledgerlens.model import RESOLVER, TEXT, ResolverFeaturizer, as_categoricals, text_proba
from ledgerlens.normalize import display_name
from ledgerlens.resolver import Resolver

DEFAULT_PATH = Path("models/enricher.joblib")


class Enricher:
    def __init__(self, text_model, lgbm, columns: list[str], tau: float, resolver: Resolver):
        self.text_model, self.lgbm, self.columns, self.tau, self.resolver = text_model, lgbm, columns, tau, resolver

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> Enricher:
        a = joblib.load(path)
        return cls(a["text_model"], a["lgbm"], a["columns"], a["tau"], Resolver(a["resolver_entries"]))

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True).copy()
        if "user_id" not in df:
            df["user_id"] = "me"
        df["amount"] = df["amount"].astype(float)
        df = add_clean(df)
        feats = pd.concat([struct_amount_features(df), history_features(df)], axis=1)
        res = ResolverFeaturizer(self.resolver, df["clean"]).frame(df["clean"])
        X = pd.concat([feats[STRUCT + AMOUNT + HISTORY + RULES],
                       pd.DataFrame(text_proba(self.text_model, df["clean"]), columns=TEXT),
                       res[RESOLVER]], axis=1)
        proba = self.lgbm.predict_proba(as_categoricals(X[self.columns]))
        known = res["res_sim"].to_numpy() >= self.tau
        return pd.DataFrame({
            "description": df["description"],
            "amount": df["amount"],
            "merchant": np.where(known, res["res_top1_merchant"], df["clean"].map(display_name)),
            "merchant_in_directory": known,
            "category": self.lgbm.classes_[proba.argmax(1)],
            "confidence": proba.max(1).round(4),
            "is_recurring": feats["hist_rule_recurring"].astype(bool),
            "processor": df["processor"].replace("NONE", None),
        })
