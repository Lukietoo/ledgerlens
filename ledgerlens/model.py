"""Category model: TF-IDF text model stacked under LightGBM.

    text (cleaned descriptor) --char/word TF-IDF--> logistic regression --> 18 class probs ─┐
    raw descriptor format / amount / user history --------------------------------------------┼─> LightGBM
    merchant resolver (similarity, matched merchant's directory category, top-k votes) -------┘

Stacking is done out-of-fold with GroupKFold over *merchants*. In each fold the text model
never saw the held-out merchants, and the resolver directory has a random 80% of them hidden,
so LightGBM learns how far to trust each signal both when a merchant is known and when it's new.
(At 50% hidden the model over-trusted resolver matches on unseen merchants: -3 pts cold.)
"""

from __future__ import annotations

import warnings
from collections.abc import Collection

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from ledgerlens.features import CATEGORICAL
from ledgerlens.normalize import PROCESSOR_PREFIXES
from ledgerlens.resolver import Resolver
from ledgerlens.taxonomy import CATEGORIES, UNKNOWN

PROCESSORS = ["NONE", *sorted(set(PROCESSOR_PREFIXES.values()))]
TEXT = [f"p_{c}" for c in CATEGORIES]
RESOLVER = ["res_sim", "res_margin", "res_top1_cat"] + [f"r_{c}" for c in CATEGORIES]
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


# ---------------------------------------------------------------- text model

def fit_text_model(texts: pd.Series, y: pd.Series, C: float = 10.0) -> Pipeline:
    """Fit on unique (text, label) pairs, weighted by log frequency: 150K rows collapse to ~5K."""
    d = pd.DataFrame({"t": texts.to_numpy(), "y": y.to_numpy()}).value_counts().reset_index(name="n")
    pipe = Pipeline([
        ("tfidf", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                     token_pattern=r"[A-Z0-9&']+")),
        ])),
        ("lr", LogisticRegression(C=C, max_iter=3000)),
    ])
    pipe.fit(d["t"], d["y"], lr__sample_weight=1.0 + np.log(d["n"].to_numpy()))
    return pipe


def text_proba(model: Pipeline, texts: pd.Series) -> np.ndarray:
    """(n, 18) probabilities aligned to CATEGORIES, computed once per unique string."""
    uniq = texts.unique()
    p = model.predict_proba(uniq)
    out = np.zeros((len(uniq), len(CATEGORIES)))
    out[:, [CAT_INDEX[c] for c in model.classes_]] = p
    pos = {t: i for i, t in enumerate(uniq)}
    return out[[pos[t] for t in texts]]


# ---------------------------------------------------------------- resolver features

class ResolverFeaturizer:
    """Caches the (unique string x merchant) similarity matrix so per-fold masking is cheap."""

    def __init__(self, resolver: Resolver, texts: pd.Series):
        self.resolver = resolver
        self.uniq = pd.Index(texts.unique())
        self.S = resolver.merchant_similarity(list(self.uniq))

    def frame(self, texts: pd.Series, exclude: Collection[str] = ()) -> pd.DataFrame:
        u = texts.unique()
        m = self.resolver.match(list(u), exclude, S=self.S[self.uniq.get_indexer(u)])
        f = pd.DataFrame({"res_sim": m.sim, "res_margin": m.margin, "res_top1_cat": m.category,
                          "res_top1_merchant": m.merchant}, index=u)
        for i, c in enumerate(CATEGORIES):
            f[f"r_{c}"] = m.cat_scores[:, i]
        return f.reindex(texts.to_numpy()).set_axis(texts.index)


# ---------------------------------------------------------------- tabular model

def as_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    fixed = {"res_top1_cat": list(CATEGORIES), "rule_cat": [*CATEGORIES, UNKNOWN],
             "entry": ["none", "pay", "rent", "loan", "tuition", "bill", "xfer"]}
    for col in CATEGORICAL:
        if col in X:
            cats = fixed.get(col, PROCESSORS)
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def fit_lgbm(X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
             seed: int = 0) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=1500, learning_rate=0.08, num_leaves=31,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, verbose=-1,
    )
    with warnings.catch_warnings():
        # eval_set is "deprecated" in 4.7, but eval_X/eval_y don't label-encode string targets yet
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        clf.fit(as_categoricals(X), y, eval_set=[(as_categoricals(X_val), y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
    return clf


def oof_stack_features(train: pd.DataFrame, rf: ResolverFeaturizer, n_folds: int = 5,
                       hide_frac: float = 0.8, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold text probabilities and resolver features for the training rows."""
    rng = np.random.default_rng(seed)
    oof = np.zeros((len(train), len(CATEGORIES)))
    res_parts = []
    for tr_idx, ho_idx in GroupKFold(n_splits=n_folds).split(train, groups=train["merchant"]):
        tr, ho = train.iloc[tr_idx], train.iloc[ho_idx]
        oof[ho_idx] = text_proba(fit_text_model(tr["clean"], tr["category"]), ho["clean"])
        ho_merchants = ho["merchant"].unique()
        hidden = set(ho_merchants[rng.random(len(ho_merchants)) < hide_frac])
        res_parts.append(rf.frame(ho["clean"], exclude=hidden))
    return oof, pd.concat(res_parts).loc[train.index]
