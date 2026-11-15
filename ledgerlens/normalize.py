"""Deterministic cleanup of raw bank transaction descriptors.

Turns e.g. ``PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BOTTLE COF OAKLAND CA S38207461 CARD 4821``
into ``clean="BLUE BOTTLE COF"``, ``processor="SQUARE"``.

This is stage 1 of the pipeline. It strips bank wrapper text, ACH trailers, payment
processor prefixes, reference numbers, phone numbers, store numbers and trailing
locations. It does NOT try to map the result onto a canonical merchant; that's the
resolver's job (week 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV "
    "NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
)

# Small gazetteer of large US cities, as they appear in the 13-char city field.
# Deliberately NOT the same list the synthetic generator draws from, so the
# normalizer can't "cheat" on synthetic data.
CITY_GAZETTEER = frozenset(
    c.upper()
    for c in """
    NEW YORK|LOS ANGELES|CHICAGO|HOUSTON|PHOENIX|PHILADELPHIA|SAN ANTONIO|SAN DIEGO|DALLAS
    |SAN JOSE|AUSTIN|JACKSONVILLE|FORT WORTH|COLUMBUS|CHARLOTTE|SAN FRANCISCO|INDIANAPOLIS
    |SEATTLE|DENVER|WASHINGTON|BOSTON|NASHVILLE|DETROIT|PORTLAND|LAS VEGAS|MEMPHIS|LOUISVILLE
    |BALTIMORE|MILWAUKEE|ALBUQUERQUE|TUCSON|FRESNO|SACRAMENTO|ATLANTA|MIAMI|OAKLAND|MINNEAPOLIS
    |TULSA|CLEVELAND|NEW ORLEANS|TAMPA|BERKELEY|BROOKLYN|PALO ALTO|MOUNTAIN VIEW|SUNNYVALE
    |SANTA CLARA|CUPERTINO|IRVINE|PASADENA|PITTSBURGH|ST LOUIS|SALT LAKE CTY|ORLANDO|RALEIGH
    |CAMBRIDGE|SOMERVILLE|ARLINGTON|EMERYVILLE|SAN MATEO|REDWOOD CITY|MENLO PARK|ALAMEDA
    |DALY CITY|FREMONT|HAYWARD|RICHMOND|WALNUT CREEK|SANTA MONICA|LONG BEACH|ANAHEIM
    """.replace("\n", "").split("|")
    if c.strip()
)
_MAX_CITY_TOKENS = max(len(c.split()) for c in CITY_GAZETTEER)

# Third-party processors / aggregators that prefix the real merchant name.
PROCESSOR_PREFIXES: dict[str, str] = {
    "SQ": "SQUARE",
    "SQU": "SQUARE",
    "TST": "TOAST",
    "SP": "SHOPIFY",
    "PAYPAL": "PAYPAL",
    "PP": "PAYPAL",
    "CLV": "CLOVER",
    "IC": "INSTACART",
    "DD": "DOORDASH",
    "GOOGLE": "GOOGLE",
    "EB": "EVENTBRITE",
    "BKG": "BOOKING",
    "HLU": "HULU",
    "TM": "TICKETMASTER",
    "PY": "PAYMENTUS",
}
_PROCESSOR_RE = re.compile(
    r"^(?P<proc>" + "|".join(sorted(PROCESSOR_PREFIXES, key=len, reverse=True)) + r")\s?\*\s?(?P<rest>.*)$"
)

_BANK_WRAPPERS = [
    re.compile(p)
    for p in (
        r"^(RECURRING )?PURCHASE AUTHORIZED ON \d{2}/\d{2}\s+",
        r"^RECURRING PAYMENT AUTHORIZED ON \d{2}/\d{2}\s+",
        r"^POS (DEBIT|PURCHASE|WITHDRAWAL)\s+",
        r"^CHECKCARD \d{4}\s+",
        r"^DEBIT CARD PURCHASE\s*-?\s*",
        r"^(VISA|MC|DBT) (DEBIT|PURCHASE|CRD)\s+",
        r"^ACH (DEBIT|CREDIT|PMT)\s+",
    )
]

_TRAILERS = [
    re.compile(p)
    for p in (
        r"\s+S\d{9,}\s+CARD\s+\d{4}$",  # Wells Fargo style
        r"\s+CARD\s+\d{4}$",
        r"\s+\d{15,}$",  # BofA 23-digit ref
        r"\s+(PPD|WEB|CCD|TEL)(\s+ID:?\s*\S+)?$",
        r"\s+DES:.*$",  # BofA ACH:  CO DES:ENTRY ID:... INDN:... CO ID:... PPD
        r"\s+(PPD|WEB|CCD|TEL)\s+ID:?\s*\S+",
        r"\s+REF\s*#\s*\S+",
    )
]

_PHONE = re.compile(r"\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b")
_HOST_PREFIX = re.compile(r"\b(?:HELP|WWW|SUPPORT|PAY)\.\S*")  # often truncated: "HELP.LEMONADE"
_URL_PATH = re.compile(r"(?<=\w)/\S*")
_DUP_WORD = re.compile(r"\b(\w+)(?: \1\b)+")
_DOMAIN_SUFFIX = re.compile(r"\.(COM|NET|ORG|IO|AI|SO|CO)\b")
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b")
_STORE_NO = re.compile(r"(#\s*\d+|\b(STORE|STR|STO|UNIT|LOC)\s*#?\s*\d+\b)")
_STAR_CODE = re.compile(r"\*\s*(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,}\b")
_LONG_NUM = re.compile(r"\b\d{3,}\b")
_MIXED_REF = re.compile(r"\b(?=[A-Z]*\d)(?=\d*[A-Z])[A-Z0-9]{8,}\b")
_PUNCT = re.compile(r"[^A-Z0-9&' ]+")
_SPACES = re.compile(r"\s+")
_FILLER_TAIL = re.compile(r"(\s+(PAYMENT|PMT|PMTS|PURCHASE|DEBIT|CREDIT|RECURRING|AUTOPAY|AUTO PAY|ONLINE|WEB))+$")
_P2P = re.compile(r"^(ZELLE|VENMO|CASH APP)\b")


@dataclass(frozen=True)
class Normalized:
    raw: str
    clean: str
    processor: str | None


def _strip_location(s: str) -> str:
    tokens = s.split()
    if len(tokens) >= 2 and tokens[-1] in US_STATES:
        tokens = tokens[:-1]
        # greedily drop the longest trailing run of tokens that is a known city
        for n in range(min(_MAX_CITY_TOKENS, len(tokens) - 1), 0, -1):
            if " ".join(tokens[-n:]) in CITY_GAZETTEER:
                tokens = tokens[:-n]
                break
    return " ".join(tokens)


def normalize(raw: str) -> Normalized:
    s = _SPACES.sub(" ", raw.upper()).strip()

    for pat in _BANK_WRAPPERS:
        s = pat.sub("", s)
    for pat in _TRAILERS:
        s = pat.sub("", s)

    # P2P: the counterparty is a person; the merchant is the rail.
    p2p = _P2P.match(s)
    if p2p:
        return Normalized(raw=raw, clean=p2p.group(1), processor=None)

    processor = None
    m = _PROCESSOR_RE.match(s)
    # only strip when the word after "*" is a real name ("GOOGLE *FI" keeps GOOGLE)
    if m and len(re.sub(r"[^A-Z]", "", (m.group("rest").split() or [""])[0])) >= 3:
        processor = PROCESSOR_PREFIXES[m.group("proc")]
        s = m.group("rest")

    s = _PHONE.sub(" ", s)
    s = _HOST_PREFIX.sub(" ", s)
    s = _URL_PATH.sub(" ", s)
    s = _DOMAIN_SUFFIX.sub(" ", s)
    s = re.sub(r"\bWWW\.", "", s)
    s = _DATE.sub(" ", s)
    s = _STAR_CODE.sub(" ", s)
    s = _STORE_NO.sub(" ", s)
    s = _MIXED_REF.sub(" ", s)
    s = _LONG_NUM.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    s = _strip_location(s)
    s = _FILLER_TAIL.sub("", s).strip()
    s = _DUP_WORD.sub(r"\1", s)  # "DOORDASH DOORDASH" (name + domain) -> "DOORDASH"

    if not s:  # never return empty; fall back to the first raw token
        s = _PUNCT.sub("", raw.upper().split()[0]) if raw.strip() else ""
    return Normalized(raw=raw, clean=s, processor=processor)


def display_name(descriptor: str) -> str:
    """'MARIO S GRILL' -> "Mario's Grill" (descriptors drop apostrophes)."""
    out: list[str] = []
    for w in descriptor.title().split():
        if w == "S" and out:
            out[-1] += "'s"
        else:
            out.append(w)
    return " ".join(out)


def canon(s: str) -> str:
    """Comparison key: uppercase alphanumerics only. "Trader Joe's" == "TRADER JOE S"."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())
