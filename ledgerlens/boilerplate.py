"""Learn a bank's statement boilerplate from the statement itself (no labels, no format rules).

A statement comes from one bank, so the bank's wrapper words ("PURCHASE AUTHORIZED ON",
"CHECKCARD", "POS DEBIT", "CARD", the home city) appear across many *different*
merchants. A real merchant word ("STARBUCKS") repeats too, but only within its own handful
of store variants, however often you shop there. So count over the statement's DISTINCT
lines (ignoring digits): a word in >= ``min_share`` of them is boilerplate. (Counting raw
lines doesn't work: 30 identical coffee-shop lines make any word look repetitive.)

Found by format_eval.py: with the hand-written rule for an unseen bank format switched
off, merchant-name accuracy fell to 0% because the wrapper stayed in the cleaned string.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

_WORD = re.compile(r"[A-Z][A-Z&'*-]*")


def _words(desc: str) -> list[str]:
    return _WORD.findall(desc.upper())


def learn_boilerplate(descriptions: Iterable[str], min_share: float = 0.25, min_distinct: int = 20) -> set[str]:
    distinct = {tuple(_words(d)) for d in descriptions}
    if len(distinct) < min_distinct:  # too little evidence to tell a wrapper from a favorite store
        return set()
    counts = Counter(w for ws in distinct for w in set(ws))
    # single letters are never boilerplate: Wells Fargo's "S8584..." reference would otherwise
    # teach "S" and turn "TRADER JOE S" into "TRADER JOE"
    return {w for w, c in counts.items() if len(w) >= 2 and c / len(distinct) >= min_share}


_SPLIT = re.compile(r"[\s:]+")  # "DES:PAYMENT" and "INDN:BRIAN" are two words each


def strip_boilerplate(desc: str, words: set[str]) -> str:
    if not words:
        return desc
    kept = [tok for tok in _SPLIT.split(desc) if tok and not (set(_words(tok)) and set(_words(tok)) <= words)]
    return " ".join(kept) if kept else desc
