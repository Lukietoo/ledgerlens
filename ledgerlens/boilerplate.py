"""Learn a bank's statement boilerplate from the statement itself (no labels, no format rules).

A statement comes from one bank, so the bank's wrapper words ("PURCHASE AUTHORIZED ON",
"CHECKCARD", "POS DEBIT", "CARD", the home city) appear across many *different*
merchants. A real merchant word ("STARBUCKS") repeats too, but only within its own handful
of store variants, however often you shop there. So count over the statement's DISTINCT
lines (ignoring digits): a word in >= ``min_share`` of them is boilerplate. (Counting raw
lines doesn't work: 30 identical coffee-shop lines make any word look repetitive.)

Found by format_eval.py: with the hand-written rule for an unseen bank format switched
off, merchant-name accuracy fell to 29% because the wrapper stayed in the cleaned string.
Used in training and serving alike (features.add_clean), so the model never sees a
different kind of string at inference than it trained on.

It runs on the normalizer's OUTPUT (strip_clean), not the raw text: the hand-written rules
anchor on exactly the words this removes (a trailing state code is how a city gets stripped;
"PURCHASE AUTHORIZED ON" is how the date in front of "SQ *" gets stripped). Stripping first
made 15% of rows gain words. After normalizing, it can only remove leftovers.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from ledgerlens.normalize import PROCESSOR_PREFIXES, finish

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


# Payment-processor prefixes ("SQ *", "TST*") repeat across many merchants for a heavy Square
# user, but they are signal (Toast = restaurant), not bank boilerplate. General payments
# knowledge, not knowledge of any one bank's format.
_PROTECTED = frozenset(PROCESSOR_PREFIXES)


def _is_boilerplate(tok: str, words: set[str]) -> bool:
    ws = set(_words(tok))
    if not ws or "*" in tok or ws & _PROTECTED:
        return False
    return ws <= words


def strip_boilerplate(desc: str, words: set[str]) -> str:
    if not words:
        return desc
    kept = [tok for tok in _SPLIT.split(desc) if tok and not _is_boilerplate(tok, words)]
    return " ".join(kept) if kept else desc


def strip_clean(clean: str, words: set[str]) -> tuple[str, str | None]:
    """Remove learned boilerplate from a normalizer output. Returns (clean, processor).

    If stripping exposes a processor prefix at the front ("PURCHASE AUTHORIZED ON SQ BLUE
    BTL COF" on a bank nobody wrote a rule for), it is dropped and reported, as the
    normalizer would have done had it been at the front of the raw text.
    """
    if not words:
        return clean, None
    toks = clean.split()
    kept = [t for t in toks if not (t in words and t not in _PROTECTED)]
    processor = None
    if kept and kept != toks and kept[0] in _PROTECTED and toks[0] != kept[0] and len(kept) > 1:
        processor, kept = PROCESSOR_PREFIXES[kept[0]], kept[1:]
    out = finish(" ".join(kept))
    return (out, processor) if out else (clean, None)
