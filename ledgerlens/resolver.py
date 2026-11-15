"""Merchant resolver: noisy cleaned descriptor -> canonical merchant from a directory.

The directory stands in for a production merchant database. Each chain has
exactly two entries, its canonical name and its *primary* descriptor. Every other
descriptor alias in merchants.csv (``WFM``, ``BLUE BTL COF``, ``VZWRLSS*APOCC VISB``)
is deliberately left out, so the resolver has to fuzzy-match aliases it has never seen.

Matching uses character n-gram TF-IDF with cosine similarity, aggregated to merchant
level (max over that merchant's entries). Below a confidence threshold the resolver
abstains, and the merchant name falls back to the title-cased descriptor. That's the
right behavior for the long tail of local businesses no directory will ever have.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from importlib import resources

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ledgerlens.normalize import display_name, normalize
from ledgerlens.synth import load_chain_merchants
from ledgerlens.taxonomy import CATEGORIES

CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


# ACH entry / payment-type words describe the *kind* of payment, not who it's with.
# Without this, "SOFI TECHNOLOGI PAYROLL" matches "ADP PAYROLL" on the word PAYROLL.
_NON_IDENTITY = frozenset("PAYROLL PAYMENT PMT PMTS DIR DEP DIRECT AUTOPAY AUTO PAY BILL WEB ONLINE "
                          "TRANSFER XFER DES PPD SALARY".split())


def _key(s: str) -> str:
    toks = re.sub(r"[^A-Z0-9]+", " ", s.upper()).split()
    kept = [t for t in toks if t not in _NON_IDENTITY]
    return " ".join(kept or toks)


@dataclass(frozen=True)
class Entry:
    merchant: str
    category: str
    text: str


def load_directory() -> list[Entry]:
    entries: list[Entry] = []
    for m in load_chain_merchants():
        for t in dict.fromkeys([m.name, m.descriptors[0]]):  # canonical + primary alias only
            entries.append(Entry(m.name, m.category, normalize(t).clean))
    text = resources.files("ledgerlens.resources").joinpath("entities.csv").read_text()
    for row in csv.DictReader(text.splitlines()):
        for t in row["descriptors"].split("|"):
            entries.append(Entry(row["merchant"], row["category"], normalize(t).clean))
    return entries


@dataclass
class Matches:
    """Merchant-level match results for a batch of query strings."""

    merchant: np.ndarray  # top-1 merchant name (object array)
    category: np.ndarray  # top-1 merchant's directory category
    sim: np.ndarray  # top-1 cosine similarity
    margin: np.ndarray  # top-1 minus top-2 (distinct merchants)
    cat_scores: np.ndarray  # (n, n_categories) similarity-weighted votes from top-k merchants


class Resolver:
    def __init__(self, entries: Sequence[Entry] | None = None, top_k: int = 5):
        entries = sorted(entries if entries is not None else load_directory(), key=lambda e: e.merchant)
        self.entries = entries
        self.top_k = top_k
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.D = self.vec.fit_transform([_key(e.text) for e in entries])
        names = [e.merchant for e in entries]
        self.merchants = np.array(list(dict.fromkeys(names)), dtype=object)
        self.starts = np.array([names.index(m) for m in self.merchants])
        cat_of = {e.merchant: e.category for e in entries}
        self.merchant_cat = np.array([cat_of[m] for m in self.merchants], dtype=object)
        self.merchant_cat_idx = np.array([CAT_INDEX[c] for c in self.merchant_cat])
        self._mid = {m: i for i, m in enumerate(self.merchants)}

    def merchant_similarity(self, texts: Sequence[str]) -> np.ndarray:
        """(n_texts, n_merchants) cosine similarity, max over each merchant's entries."""
        Q = self.vec.transform([_key(t) for t in texts])
        S = (Q @ self.D.T).toarray()
        return np.maximum.reduceat(S, self.starts, axis=1)

    def match(self, texts: Sequence[str], exclude: Collection[str] = (),
              S: np.ndarray | None = None) -> Matches:
        """Top-k matching. ``exclude`` hides merchants (simulates them missing from the directory)."""
        S = (self.merchant_similarity(texts) if S is None else S).copy()
        if exclude:
            cols = [self._mid[m] for m in exclude if m in self._mid]
            S[:, cols] = -1.0
        k = min(self.top_k, S.shape[1])
        top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-top_s, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        top_s = np.clip(np.take_along_axis(top_s, order, axis=1), 0.0, None)

        cat_scores = np.zeros((len(S), len(CATEGORIES)))
        rows = np.repeat(np.arange(len(S)), k)
        np.add.at(cat_scores, (rows, self.merchant_cat_idx[top].ravel()), top_s.ravel())
        cat_scores /= np.maximum(cat_scores.sum(axis=1, keepdims=True), 1e-9)
        return Matches(
            merchant=self.merchants[top[:, 0]],
            category=self.merchant_cat[top[:, 0]],
            sim=top_s[:, 0],
            margin=top_s[:, 0] - (top_s[:, 1] if k > 1 else 0.0),
            cat_scores=cat_scores,
        )

    def resolve(self, texts: Sequence[str], threshold: float, exclude: Collection[str] = (),
                S: np.ndarray | None = None) -> list[str]:
        """Final merchant names: directory match if confident, else the cleaned descriptor."""
        m = self.match(texts, exclude, S)
        return [str(name) if s >= threshold else display_name(t)
                for name, s, t in zip(m.merchant, m.sim, texts)]
