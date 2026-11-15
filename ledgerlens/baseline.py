"""Rules-only baselines: what a quick hand-written system gets you.

These are intentionally the kind of thing an engineer would write in an afternoon:
a first-match keyword list for categories and a periodicity heuristic for
recurring charges. Every ML model in this repo has to beat these numbers.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import date

from ledgerlens.normalize import canon, normalize
from ledgerlens.taxonomy import UNKNOWN

# First match wins, so order matters (income/transfer patterns before generic words).
_RULES: list[tuple[str, str]] = [
    ("FEES", r"\b(SERVICE|MAINTENANCE|OVERDRAFT|INSUFFICIENT FUNDS|ATM|TRANSACTION|WIRE TRANSFER|LATE) FEE|\bFEE-"),
    ("INCOME", r"PAYROLL|DIR DEP|DIRECT DEP|SALARY|TAX REF|INTEREST (PAYMENT|EARNED|PAID)"),
    ("TRANSFER", r"VENMO|ZELLE|CASH APP|INST XFER|\bXFER\b|TRANSFER|ATM (CASH )?WITHDRAW|EPAYMENT|E-PAYMENT|"
                 r"CRD AUTOPAY|AUTOPAY PAYMENT|MOBILE PMT|MONEYLINK|MONEYLINE|ROBINHOOD|COINBASE|SECURITIES"),
    ("LOANS", r"\bLOAN|NAVIENT|NELNET|MOHELA|AFFIRM|KLARNA|AFTERPAY|FINANCIAL|AUTO FINANCE"),
    ("HOUSING", r"\bRENT\b|MORTGAGE|APPFOLIO|LEASING|APARTMENTS|RESIDENTIAL|RENTCAFE"),
    ("UTILITIES", r"PG&E|PGANDE|EDISON|ENERGY|ELECTRIC|\bWATER\b|COMCAST|XFINITY|AT&T|\bATT\b|VERIZON|"
                  r"T-MOBILE|TMOBILE|WIRELESS|WRLS|SPECTRUM|INTERNET|MINT MOBILE|SOCALGAS"),
    ("SUBSCRIPTIONS", r"NETFLIX|SPOTIFY|HULU|DISNEY|HBO|YOUTUBE|APPLE\.COM/BILL|PRIME VIDEO|AMAZON PRIME|ADOBE|"
                      r"MICROSOFT|DROPBOX|PATREON|SUBSCR|MEMBERSHIP|OPENAI|AUDIBLE"),
    ("DINING", r"UBER \*?EATS|DOORDASH|GRUBHUB"),
    ("TRANSPORT", r"\bUBER\b|\bLYFT\b|CLIPPER|\bBART\b|\bMTA\b|PARKING|PARKMOBILE|FASTRAK|\bTOLL|TRANSIT|\bLIME\b"),
    ("GAS", r"\bSHELL\b|CHEVRON|EXXON|\bMOBIL\b|\bARCO\b|VALERO|\b76\b|\bBP\b|GAS STATION|\bFUEL|SPEEDWAY|CIRCLE K"),
    ("TRAVEL", r"AIRLINE|AIRWAYS|AIR LINES|DELTA AIR|UNITED AIR|SOUTHWES|AIRBNB|HOTEL|MARRIOTT|HILTON|HYATT|"
               r"\bINN\b|EXPEDIA|BOOKING\.COM|HERTZ|RENT-A-CAR|AMTRAK"),
    ("COFFEE", r"COFFEE|\bCAFE\b|STARBUCKS|DUNKIN|PEETS|ESPRESSO|ROASTER|\bBOBA\b|\bTEA\b"),
    ("GROCERIES", r"GROCERY|\bMARKET\b|SUPERMARKET|SAFEWAY|KROGER|WHOLEFDS|WHOLE FOODS|TRADER JOE|COSTCO|"
                  r"\bALDI\b|\bFOODS\b|INSTACART|PRODUCE"),
    ("DINING", r"RESTAURANT|\bGRILL\b|KITCHEN|PIZZA|\bTACO|BURGER|SUSHI|\bTHAI\b|\bPHO\b|\bBBQ\b|DINER|BISTRO|"
               r"NOODLE|RAMEN|CHIPOTLE|MCDONALD|SUBWAY|WENDY|CHICK-FIL-A|\bTST\*"),
    ("HEALTH", r"PHARMACY|\bCVS\b|WALGREENS|MEDICAL|CLINIC|DENTAL|HEALTH|KAISER|DOCTOR|OPTOMETR"),
    ("FITNESS", r"FITNESS|\bGYM\b|\bYOGA\b|PILATES|CROSSFIT|CLIMBING|PELOTON|CLASSPASS|EQUINOX"),
    ("ENTERTAINMENT", r"CINEMA|THEATRE|THEATER|\bAMC\b|REGAL|TICKETMASTER|STUBHUB|STEAM|EVENTBRITE|\bBOWL"),
    ("EDUCATION", r"UNIVERSITY|COLLEGE|TUITION|BERKELEY|COURSERA|UDEMY|CHEGG|TEXTBOOK|SCHOOL"),
    ("SHOPPING", r"AMAZON|\bAMZN\b|TARGET|WAL-MART|WALMART|BEST ?BUY|HOME DEPOT|IKEA|\bNIKE\b|ETSY|EBAY|"
                 r"\bSTORE\b|\bSHOP\b|BOUTIQUE|OUTLET"),
]
_COMPILED = [(cat, re.compile(p)) for cat, p in _RULES]


def categorize(raw: str) -> str:
    s = raw.upper()
    for cat, pat in _COMPILED:
        if pat.search(s):
            return cat
    return UNKNOWN


# ---------------------------------------------------------------- recurring


def detect_recurring(
    txns: Iterable[tuple[str, str, date, float]],
    min_count: int = 3,
    periods: tuple[tuple[int, int], ...] = ((13, 16), (26, 35)),
    max_amount_cv: float = 0.25,
) -> list[bool]:
    """Flag txns that belong to a biweekly or monthly series.

    Input rows are (user_id, raw_description, date, amount). Groups by
    (user, normalized descriptor), then calls a group recurring if it has
    >= min_count charges, a median gap inside one of ``periods`` (days) and a low
    coefficient of variation in amount.
    """
    rows = list(txns)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (uid, desc, _, _) in enumerate(rows):
        groups[f"{uid}|{canon(normalize(desc).clean)}"].append(i)

    flags = [False] * len(rows)
    for idxs in groups.values():
        if len(idxs) < min_count:
            continue
        idxs.sort(key=lambda i: rows[i][2])
        dates = [rows[i][2] for i in idxs]
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        med = statistics.median(gaps)
        if not any(lo <= med <= hi for lo, hi in periods):
            continue
        amts = [abs(rows[i][3]) for i in idxs]
        mean = statistics.fmean(amts)
        cv = statistics.pstdev(amts) / mean if mean else 0.0
        if cv <= max_amount_cv:
            for i in idxs:
                flags[i] = True
    return flags
