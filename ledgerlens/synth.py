"""Synthetic bank transaction generator.

Simulates a population of users with 12 months of account history each, then renders
every transaction the way a real bank statement shows it. The noise model (see
README "Noise model") is the point of this module:

* card rail: optional processor prefix (SQ *, TST*, PAYPAL *), stable per-user store
  numbers, ISO 8583-style 22-char merchant name field + 13-char city field, online
  merchants show a phone number or domain instead of a city, Amazon-style ``*`` ref codes,
  and one of several bank-specific wrappers (Wells Fargo "PURCHASE AUTHORIZED ON",
  BofA "CHECKCARD", POS DEBIT, fixed-width, Title Case...)
* ACH rail: 16-char NACHA company name + entry description, rendered in one of
  several bank styles (Chase "PPD ID:", BofA "DES: ... INDN:", ...)
* P2P, card payments, brokerage, ATM, payroll and fees use their own templates.

Every row carries ground-truth labels: canonical merchant, category, is_recurring,
plus a merchant-level train/val/test split so evaluation measures generalization to
merchants the model has never seen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib import resources
from pathlib import Path

from ledgerlens.normalize import display_name
from ledgerlens.taxonomy import CATEGORIES

START = date(2025, 11, 1)
MONTHS = 12

# ---------------------------------------------------------------- merchants


@dataclass(frozen=True)
class Merchant:
    name: str
    category: str
    descriptors: tuple[str, ...]
    amount_min: float
    amount_max: float
    recurring: bool
    online: bool
    rail: str  # "card" | "ach"
    kind: str = "chain"  # chain | local


def load_chain_merchants() -> list[Merchant]:
    text = resources.files("ledgerlens.resources").joinpath("merchants.csv").read_text()
    out = []
    for row in csv.DictReader(text.splitlines()):
        assert row["category"] in CATEGORIES, row
        descs = tuple(d for d in row["descriptors"].split("|") if d.strip()) or (row["merchant"].upper(),)
        out.append(
            Merchant(
                name=row["merchant"],
                category=row["category"],
                descriptors=descs,
                amount_min=float(row["amount_min"]),
                amount_max=float(row["amount_max"]),
                recurring=row["recurring"] == "1",
                online=row["online"] == "1",
                rail=row["rail"],
            )
        )
    return out


_ADJ = (
    "GOLDEN BLUE RED LUCKY HAPPY SUNRISE MISSION OAK PACIFIC BAY NORTH SOUTH EAST WEST "
    "SILVER JADE COPPER URBAN RUSTIC LITTLE BIG OLD NEW SUNSET HIDDEN WILD FIRST CORNER "
    "MAIN STREET VILLAGE HARBOR CEDAR MAPLE IVY CRESCENT ROYAL"
).split()
_NOUN = (
    "DRAGON LOTUS BEAR FOX LEAF MOON HARBOR GARDEN BAMBOO ROSE PINE STONE RIVER OWL BRIDGE "
    "LANTERN TIGER CRANE ORCHARD FIG OLIVE SAGE BASIL PEARL CANYON MESA HILL PIER"
).split()
_OWNER = (
    "MARIO S|LUCY S|CHEN S|NGUYEN S|ROSA S|TONY S|AMIR S|PATEL S|KIM S|GARCIA S|SOFIA S|"
    "JOE S|ANNA S|RAJ S|ELENA S|OMAR S|MEI S|LEO S"
).split("|")
_LOCAL_KINDS: dict[str, tuple[list[str], float, float]] = {
    # category: (type words, amount_min, amount_max)
    "DINING": (
        "RESTAURANT KITCHEN GRILL TAQUERIA PIZZERIA SUSHI THAI PHO BBQ BISTRO DINER RAMEN TACOS "
        "BURGERS NOODLE HOUSE DUMPLING EATERY CANTINA".split(),
        9,
        85,
    ),
    "COFFEE": ("CAFE COFFEE ROASTERS ESPRESSO TEA HOUSE BOBA BAKERY".split(), 3, 16),
    "GROCERIES": ("MARKET GROCERY FOODS SUPERMARKET PRODUCE MERCADO DELI".split(), 6, 90),
    "SHOPPING": ("BOUTIQUE GOODS SUPPLY HARDWARE BOOKS OUTFITTERS THRIFT VINTAGE GIFTS".split(), 10, 150),
    "HEALTH": ("PHARMACY DENTAL CLINIC OPTOMETRY CHIROPRACTIC MEDICAL".split(), 15, 250),
    "FITNESS": ("YOGA FITNESS CLIMBING BOXING PILATES CROSSFIT".split(), 15, 120),
    "ENTERTAINMENT": ("THEATER CINEMA ARCADE KARAOKE COMEDY CLUB BOWL".split(), 10, 80),
}


def make_local_merchants(rng: random.Random, per_category: int) -> list[Merchant]:
    """Independent local businesses. ~30% have no category hint in their name at all,
    which is realistic and caps what any string-only model can do."""
    out: list[Merchant] = []
    seen: set[str] = set()
    for cat, (kinds, lo, hi) in _LOCAL_KINDS.items():
        made = 0
        while made < per_category:
            r = rng.random()
            head = rng.choice(_OWNER) if r < 0.25 else f"{rng.choice(_ADJ)} {rng.choice(_NOUN)}"
            name = head if rng.random() < 0.3 else f"{head} {rng.choice(kinds)}"
            if name in seen:
                continue
            seen.add(name)
            made += 1
            out.append(
                Merchant(
                    name=display_name(name),
                    category=cat,
                    descriptors=(name,),
                    amount_min=lo,
                    amount_max=hi,
                    recurring=False,
                    online=False,
                    rail="card",
                    kind="local",
                )
            )
    return out


# ---------------------------------------------------------------- places / people

# Includes plenty of cities NOT in the normalizer's gazetteer (on purpose).
CITIES = [
    ("BERKELEY", "CA"), ("OAKLAND", "CA"), ("SAN FRANCISCO", "CA"), ("EMERYVILLE", "CA"),
    ("ALBANY", "CA"), ("EL CERRITO", "CA"), ("SAN LEANDRO", "CA"), ("SOUTH SAN FRANCISCO", "CA"),
    ("SAN JOSE", "CA"), ("PALO ALTO", "CA"), ("MOUNTAIN VIEW", "CA"), ("LOS ANGELES", "CA"),
    ("SANTA MONICA", "CA"), ("CULVER CITY", "CA"), ("SAN DIEGO", "CA"), ("SACRAMENTO", "CA"),
    ("DAVIS", "CA"), ("WALNUT CREEK", "CA"), ("PLEASANTON", "CA"), ("DUBLIN", "CA"),
    ("NEW YORK", "NY"), ("BROOKLYN", "NY"), ("LONG ISLAND CITY", "NY"), ("HOBOKEN", "NJ"),
    ("JERSEY CITY", "NJ"), ("BOSTON", "MA"), ("CAMBRIDGE", "MA"), ("SOMERVILLE", "MA"),
    ("SEATTLE", "WA"), ("BELLEVUE", "WA"), ("REDMOND", "WA"), ("PORTLAND", "OR"),
    ("AUSTIN", "TX"), ("ROUND ROCK", "TX"), ("DALLAS", "TX"), ("PLANO", "TX"), ("HOUSTON", "TX"),
    ("CHICAGO", "IL"), ("EVANSTON", "IL"), ("DENVER", "CO"), ("BOULDER", "CO"),
    ("ATLANTA", "GA"), ("DECATUR", "GA"), ("MIAMI", "FL"), ("ORLANDO", "FL"), ("TAMPA", "FL"),
    ("PHOENIX", "AZ"), ("TEMPE", "AZ"), ("SCOTTSDALE", "AZ"), ("SALT LAKE CITY", "UT"),
    ("PROVO", "UT"), ("LAS VEGAS", "NV"), ("HENDERSON", "NV"), ("NASHVILLE", "TN"),
    ("CHARLOTTE", "NC"), ("DURHAM", "NC"), ("RALEIGH", "NC"), ("WASHINGTON", "DC"),
    ("ARLINGTON", "VA"), ("BETHESDA", "MD"), ("PHILADELPHIA", "PA"), ("PITTSBURGH", "PA"),
    ("ANN ARBOR", "MI"), ("DETROIT", "MI"), ("MINNEAPOLIS", "MN"), ("MADISON", "WI"),
    ("COLUMBUS", "OH"), ("CLEVELAND", "OH"), ("INDIANAPOLIS", "IN"), ("KANSAS CITY", "MO"),
]
HOME_CITIES = CITIES[:30]

FIRST = "JAMES MARIA DAVID SARAH MICHAEL PRIYA KEVIN EMILY JASON OLIVIA DANIEL GRACE ANDREW CHLOE BRIAN NINA ERIC JESSICA TYLER HANNAH".split()
LAST = "NGUYEN SMITH PATEL GARCIA KIM CHEN JOHNSON LEE WILLIAMS BROWN MARTINEZ TRAN DAVIS LOPEZ WANG SINGH WILSON ANDERSON".split()

EMPLOYERS = [
    # (canonical, ACH company name as printed)
    ("SoFi Technologies", "SOFI TECHNOLOGI"),
    ("UC Berkeley", "UC BERKELEY"),
    ("Google", "GOOGLE LLC"),
    ("Stripe", "STRIPE INC"),
    ("Target", "TARGET CORP"),
    ("Starbucks", "STARBUCKS CORP"),
    ("Kaiser Permanente", "KAISER FOUNDATN"),
    ("Chipotle", "CHIPOTLE MEXICAN"),
    ("Amazon", "AMAZON.COM SVCS"),
    ("DoorDash", "DOORDASH INC"),
    ("Accenture", "ACCENTURE LLP"),
    ("Deloitte", "DELOITTE LLP"),
    ("Wells Fargo", "WELLS FARGO PAYR"),
    ("Meta", "META PLATFORMS"),
]
PAYROLL_PROCESSORS = [("Gusto", "GUSTO"), ("ADP", "ADP PAYROLL"), ("Paychex", "PAYCHEX INC")]

# ---------------------------------------------------------------- rendering helpers


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _alnum(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def _phone(rng: random.Random) -> str:
    a = rng.choice(["800", "855", "866", "877", "888", "408", "415", "650", "206"])
    fmt = rng.choice(["{a}-{b}-{c}", "{a}{b}{c}", "{a}.{b}.{c}"])
    return fmt.format(a=a, b=_digits(rng, 3), c=_digits(rng, 4))


@dataclass
class StoreVariant:
    name_field: str  # already truncated to 22 chars
    city_field: str  # already truncated to 13 chars
    state: str


def make_card_variant(m: Merchant, rng: random.Random, city: str, state: str) -> StoreVariant:
    base = rng.choice(m.descriptors)
    if rng.random() < 0.15:  # some banks/acquirers drop apostrophes and spaces
        base = base.replace(" S ", "S ").replace("'", "")
    prefix = ""
    if m.kind == "local":
        r = rng.random()
        if r < 0.32:
            prefix = rng.choice(["SQ *", "SQ *", "SQU*"])
        elif r < 0.52 and m.category in ("DINING", "COFFEE"):
            prefix = "TST* "
        elif r < 0.58:
            prefix = "CLV*"
    elif m.online and rng.random() < 0.08 and not base.endswith("*"):
        prefix = "PAYPAL *"
    name = prefix + base
    if m.kind == "chain" and not m.online and rng.random() < 0.65:
        num = _digits(rng, rng.choice([3, 4, 5]))
        name += rng.choice([f" #{num}", f" {num}", f" #{num}", f" STORE {num}"])
    if base.endswith("*"):
        name += _alnum(rng, rng.choice([6, 9]))
    name = name[:22].rstrip()

    if m.online:
        r = rng.random()
        if r < 0.45:
            city_field = _phone(rng)
        elif r < 0.75:
            dom = "".join(ch for ch in m.name.upper() if ch.isalnum())
            city_field = rng.choice([f"{dom}.COM", f"WWW.{dom}.COM", f"HELP.{dom}.COM"])
        else:
            city_field = city
        state = rng.choice([state, "CA", "WA", "NY", "DE"])
    else:
        city_field = city
    return StoreVariant(name_field=name, city_field=city_field[:13].rstrip(), state=state)


CardStyle = Callable[[StoreVariant, date, bool, random.Random], str]


def _card_plain(v: StoreVariant, d: date, rec: bool, rng: random.Random) -> str:
    return f"{v.name_field} {v.city_field} {v.state}"


def _card_fixed(v: StoreVariant, d: date, rec: bool, rng: random.Random) -> str:
    return f"{v.name_field:<23}{v.city_field:<13} {v.state}"


def _card_pos(v: StoreVariant, d: date, rec: bool, rng: random.Random) -> str:
    return f"POS {rng.choice(['DEBIT', 'PURCHASE'])} {v.name_field} {v.city_field} {v.state}"


def _card_wf(v: StoreVariant, d: date, rec: bool, rng: random.Random) -> str:
    head = "RECURRING PAYMENT AUTHORIZED ON" if rec else "PURCHASE AUTHORIZED ON"
    return f"{head} {d:%m/%d} {v.name_field} {v.city_field} {v.state} S{_digits(rng, 15)} CARD {_digits(rng, 4)}"


def _card_boa(v: StoreVariant, d: date, rec: bool, rng: random.Random) -> str:
    return f"CHECKCARD {d:%m%d} {v.name_field} {v.city_field} {v.state} {_digits(rng, 23)}"


def _card_title(v: StoreVariant, d: date, rec: bool, rng: random.Random) -> str:
    return f"{v.name_field} {v.city_field} {v.state}".title()


def _card_debit(v: StoreVariant, d: date, rec: bool, rng: random.Random) -> str:
    return f"DEBIT CARD PURCHASE - {v.name_field} {v.city_field} {v.state}"


CARD_STYLES: dict[str, CardStyle] = {
    "plain": _card_plain,
    "fixed": _card_fixed,
    "pos": _card_pos,
    "wf": _card_wf,
    "boa": _card_boa,
    "title": _card_title,
    "debit": _card_debit,
}

ACH_ENTRY = {
    "INCOME": ["PAYROLL", "DIR DEP", "DIRECT DEP", "SALARY", "PAYROLL"],
    "HOUSING": ["RENT", "WEB PMTS", "PAYMENT", "RENT PMT"],
    "LOANS": ["LOAN PMT", "AUTOPAY", "PAYMENT", "AUTO PAY"],
    "UTILITIES": ["AUTOPAY", "BILL PAY", "WEB PMTS", "ONLINE PMT", "PAYMENT"],
    "EDUCATION": ["TUITION", "PAYMENT", "WEB PMTS"],
    "TRANSFER": ["TRANSFER", "PAYMENT", "WEB PMTS"],
}


@dataclass(frozen=True)
class AchOriginator:
    """What stays fixed across every ACH entry from one company to one account."""

    company: str  # 16-char NACHA company name
    entry: str  # 10-char entry description
    sec: str  # PPD / WEB
    company_id: str


def make_ach_originator(company: str, category: str, rng: random.Random) -> AchOriginator:
    return AchOriginator(
        company=company[:16].rstrip(),  # NACHA company name is 16 chars
        entry=rng.choice(ACH_ENTRY.get(category, ["PAYMENT"]))[:10],
        sec="PPD" if category == "INCOME" else rng.choice(["PPD", "WEB"]),
        company_id=_digits(rng, 10),
    )


def render_ach(style: str, o: AchOriginator, d: date, holder: str, rng: random.Random) -> str:
    co, entry, sec = o.company, o.entry, o.sec
    if style == "chase":
        return f"{co} {entry} {sec} ID: {o.company_id}"
    if style == "boa":
        return f"{co} DES:{entry} ID:{_alnum(rng, 9)} INDN:{holder} CO ID:{o.company_id} {sec}"
    if style == "wf":
        return f"{co} {entry} {d:%m%d%y} {_alnum(rng, 10)} {holder}"
    return f"{co} {entry}"


ACH_STYLES = ["chase", "boa", "wf", "plain"]

# ---------------------------------------------------------------- simulation


@dataclass
class Txn:
    user_id: str
    date: date
    description: str
    amount: float
    merchant: str
    category: str
    is_recurring: bool
    merchant_kind: str


@dataclass
class User:
    uid: str
    home: tuple[str, str]
    card_style: str
    ach_style: str
    holder: str
    variants: dict[tuple[str, str], list[StoreVariant]] = field(default_factory=dict)
    ach: dict[str, AchOriginator] = field(default_factory=dict)


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    if hi <= lo:
        return round(lo, 2)
    return round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 2)


def _sub_price(rng: random.Random, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    p = rng.uniform(lo, hi)
    return round(max(lo, math.floor(p) + 0.99) if hi - lo > 1 else p, 2)


def _month_start(i: int) -> date:
    y, m = divmod(START.month - 1 + i, 12)
    return date(START.year + y, m + 1, 1)


def _days_in_month(d: date) -> int:
    nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return (nxt - d.replace(day=1)).days


class Simulator:
    def __init__(self, seed: int, n_users: int, locals_per_category: int):
        self.rng = random.Random(seed)
        self.chains = load_chain_merchants()
        self.locals = make_local_merchants(self.rng, locals_per_category)
        self.by_cat: dict[str, list[Merchant]] = {}
        for m in self.chains:
            self.by_cat.setdefault(m.category, []).append(m)
        self.locals_by_cat: dict[str, list[Merchant]] = {}
        for m in self.locals:
            self.locals_by_cat.setdefault(m.category, []).append(m)
        self.n_users = n_users

    # -- descriptor for a card txn, stable per (user, merchant, city)
    def card_desc(self, u: User, m: Merchant, d: date, rec: bool, city: tuple[str, str]) -> str:
        key = (m.name, city[0])
        if key not in u.variants:
            n = 1 if (m.online or m.kind == "local" or self.rng.random() < 0.5) else self.rng.randint(2, 3)
            u.variants[key] = [make_card_variant(m, self.rng, *city) for _ in range(n)]
        v = self.rng.choices(u.variants[key], weights=[3, 1, 1][: len(u.variants[key])])[0]
        return CARD_STYLES[u.card_style](v, d, rec, self.rng)

    def emit(self, out: list[Txn], u: User, m: Merchant, d: date, amount: float, rec: bool,
             city: tuple[str, str] | None = None) -> None:
        if m.rail == "ach":
            if m.name not in u.ach:
                u.ach[m.name] = make_ach_originator(self.rng.choice(m.descriptors), m.category, self.rng)
            desc = render_ach(u.ach_style, u.ach[m.name], d, u.holder, self.rng)
        else:
            desc = self.card_desc(u, m, d, rec, city or u.home)
        out.append(Txn(u.uid, d, desc, amount, m.name, m.category, rec, m.kind))

    def _pick(self, rng: random.Random, pool: list[Merchant], k: int) -> list[Merchant]:
        return rng.sample(pool, min(k, len(pool)))

    def simulate_user(self, idx: int) -> list[Txn]:
        rng = self.rng
        u = User(
            uid=f"u{idx:04d}",
            home=rng.choice(HOME_CITIES),
            card_style=rng.choice(list(CARD_STYLES)),
            ach_style=rng.choice(ACH_STYLES),
            holder=f"{rng.choice(FIRST)} {rng.choice(LAST)}"[:22],
        )
        out: list[Txn] = []
        cat = self.by_cat
        loc = self.locals_by_cat

        # ---- recurring commitments: (merchant, day_of_month, amount, amount_jitter)
        recurring: list[tuple[Merchant, int, float, float]] = []
        for m in self._pick(rng, cat["SUBSCRIPTIONS"], rng.randint(2, 7)):
            recurring.append((m, rng.randint(1, 28), _sub_price(rng, m.amount_min, m.amount_max), 0.0))
        for m in self._pick(rng, cat["UTILITIES"], rng.randint(1, 3)):
            recurring.append((m, rng.randint(1, 28), _log_uniform(rng, m.amount_min, m.amount_max), 0.2))
        if rng.random() < 0.8:
            rent = rng.choice([m for m in cat["HOUSING"] if m.amount_min >= 1000])
            recurring.append((rent, 1, round(rng.uniform(rent.amount_min, rent.amount_max) / 25) * 25, 0.0))
        if rng.random() < 0.35:
            ins = next(m for m in cat["HOUSING"] if m.name == "Lemonade Insurance")
            recurring.append((ins, rng.randint(1, 28), _log_uniform(rng, ins.amount_min, ins.amount_max), 0.0))
        loans = [m for m in cat["LOANS"] if m.rail == "ach"]
        for m in self._pick(rng, loans, rng.choice([0, 0, 1, 1, 2])):
            recurring.append((m, rng.randint(1, 28), round(rng.uniform(m.amount_min, m.amount_max), 2), 0.0))
        if rng.random() < 0.5:
            m = rng.choice([g for g in cat["FITNESS"] if g.recurring])
            recurring.append((m, rng.randint(1, 28), _sub_price(rng, m.amount_min, m.amount_max), 0.0))
        if rng.random() < 0.3:
            m = rng.choice([e for e in cat["EDUCATION"] if e.recurring])
            recurring.append((m, rng.randint(1, 28), _sub_price(rng, m.amount_min, m.amount_max), 0.0))
        if rng.random() < 0.12:
            m = rng.choice([h for h in cat["HEALTH"] if h.recurring])
            recurring.append((m, rng.randint(1, 28), _sub_price(rng, m.amount_min, m.amount_max), 0.0))

        # ---- favorites for discretionary spend
        favs = {
            "GROCERIES": self._pick(rng, cat["GROCERIES"], 2) + self._pick(rng, loc["GROCERIES"], 1),
            "DINING": self._pick(rng, cat["DINING"], 6) + self._pick(rng, loc["DINING"], 5),
            "COFFEE": self._pick(rng, cat["COFFEE"], 2) + self._pick(rng, loc["COFFEE"], 2),
            "TRANSPORT": self._pick(rng, cat["TRANSPORT"], 3),
            "GAS": self._pick(rng, cat["GAS"], 2) if rng.random() < 0.55 else [],
            "SHOPPING": self._pick(rng, cat["SHOPPING"], 8) + self._pick(rng, loc["SHOPPING"], 2),
            "ENTERTAINMENT": self._pick(rng, cat["ENTERTAINMENT"], 3) + self._pick(rng, loc["ENTERTAINMENT"], 1),
            "HEALTH": self._pick(rng, [h for h in cat["HEALTH"] if not h.recurring], 2) + self._pick(rng, loc["HEALTH"], 1),
            "FITNESS": self._pick(rng, loc["FITNESS"], 1) if rng.random() < 0.3 else [],
            "EDUCATION": self._pick(rng, [e for e in cat["EDUCATION"] if not e.recurring], 2) if rng.random() < 0.4 else [],
            "LOANS": self._pick(rng, [m for m in cat["LOANS"] if m.rail == "card"], 1) if rng.random() < 0.3 else [],
        }
        monthly_counts = {
            "GROCERIES": (3, 8), "DINING": (4, 18), "COFFEE": (2, 18), "TRANSPORT": (1, 9),
            "GAS": (2, 4), "SHOPPING": (2, 8), "ENTERTAINMENT": (0, 3), "HEALTH": (0, 2),
            "FITNESS": (1, 4), "EDUCATION": (0, 1), "LOANS": (0, 2),
        }

        employer, employer_co = rng.choice(EMPLOYERS)
        if rng.random() < 0.25:
            employer, employer_co = rng.choice(PAYROLL_PROCESSORS)
        pay = round(rng.uniform(900, 4200), 2)
        biweekly = rng.random() < 0.6
        first_pay = START + timedelta(days=rng.randint(0, 13))
        p2p_friends = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(rng.randint(2, 6))]

        end = _month_start(MONTHS)

        # ---- payroll
        if biweekly:
            d = first_pay
            while d < end:
                out.append(self._income(u, employer, employer_co, d, pay * rng.uniform(0.97, 1.03)))
                d += timedelta(days=14)
        else:
            for i in range(MONTHS):
                for day in (1, 15):
                    d = _month_start(i).replace(day=day)
                    out.append(self._income(u, employer, employer_co, d, pay * rng.uniform(0.97, 1.03)))

        for i in range(MONTHS):
            ms = _month_start(i)
            dim = _days_in_month(ms)

            def rand_day() -> date:
                return ms + timedelta(days=rng.randrange(dim))

            for m, dom, amt, jitter in recurring:
                d = ms.replace(day=min(dom, dim))
                if rng.random() < 0.3:
                    d = min(end - timedelta(days=1), d + timedelta(days=rng.randint(0, 3)))
                a = amt if jitter == 0 else round(amt * rng.uniform(1 - jitter, 1 + jitter), 2)
                self.emit(out, u, m, d, -a, True)

            for c, pool in favs.items():
                if not pool:
                    continue
                lo, hi = monthly_counts[c]
                weights = [1.0 / (r + 1) for r in range(len(pool))]  # zipf-ish loyalty
                for _ in range(rng.randint(lo, hi)):
                    m = rng.choices(pool, weights=weights)[0]
                    self.emit(out, u, m, rand_day(), -_log_uniform(rng, m.amount_min, m.amount_max), False)

            # occasional trip: flight + lodging + spend in another city
            if rng.random() < 0.25:
                dest = rng.choice([c for c in CITIES if c != u.home])
                d0 = rand_day()
                for c in ("TRAVEL",):
                    flight = rng.choice([m for m in cat[c] if "Air" in m.name or "Jet" in m.name])
                    self.emit(out, u, flight, d0 - timedelta(days=rng.randint(7, 40)) if d0 - timedelta(days=40) >= START else d0,
                              -_log_uniform(rng, flight.amount_min, flight.amount_max), False)
                lodging = rng.choice([m for m in cat["TRAVEL"] if m.name in (
                    "Airbnb", "Marriott", "Hilton", "Hyatt", "Vrbo", "Booking.com", "Expedia", "Hotels.com")])
                self.emit(out, u, lodging, d0, -_log_uniform(rng, lodging.amount_min, lodging.amount_max), False, dest)
                trip_pool = self._pick(rng, cat["DINING"], 3) + self._pick(rng, loc["DINING"], 2) + self._pick(rng, cat["COFFEE"], 1)
                for _ in range(rng.randint(2, 6)):
                    m = rng.choice(trip_pool)
                    self.emit(out, u, m, min(end - timedelta(days=1), d0 + timedelta(days=rng.randint(0, 4))),
                              -_log_uniform(rng, m.amount_min, m.amount_max), False, dest)
                if rng.random() < 0.4:
                    car = rng.choice([m for m in cat["TRAVEL"] if "Rent" in m.name or m.name == "Hertz"])
                    self.emit(out, u, car, d0, -_log_uniform(rng, car.amount_min, car.amount_max), False, dest)

            self._month_misc(out, u, rng, ms, dim, p2p_friends, rand_day)

        out.sort(key=lambda t: t.date)
        return out

    def _income(self, u: User, employer: str, co: str, d: date, amount: float) -> Txn:
        rng = self.rng
        if "__payroll__" not in u.ach:
            u.ach["__payroll__"] = make_ach_originator(co, "INCOME", rng)
        if u.ach_style == "plain" and len(u.uid) % 2:  # some banks print "DIRECT DEP <co>"
            desc = f"DIRECT DEP {co}"
        else:
            desc = render_ach(u.ach_style, u.ach["__payroll__"], d, u.holder, rng)
        return Txn(u.uid, d, desc, round(amount, 2), employer, "INCOME", True, "template")

    def _month_misc(self, out, u, rng, ms, dim, friends, rand_day) -> None:
        def add(desc: str, amt: float, merchant: str, category: str, rec: bool = False, d: date | None = None):
            out.append(Txn(u.uid, d or rand_day(), desc, round(amt, 2), merchant, category, rec, "template"))

        # P2P
        for _ in range(rng.randint(0, 6)):
            who = rng.choice(friends)
            amt = _log_uniform(rng, 5, 300) * rng.choice([-1, -1, 1])
            rail = rng.choice(["Venmo", "Zelle", "Zelle", "Cash App", "PayPal"])
            if rail == "Venmo":
                desc = rng.choice([f"VENMO PAYMENT {_digits(rng, 10)} WEB ID: 3264681992",
                                   f"VENMO CASHOUT {_digits(rng, 10)} PPD ID: 5264681992",
                                   f"VENMO *{who.split()[0]} {_phone(rng)} NY"])
            elif rail == "Zelle":
                verb = "TO" if amt < 0 else "FROM"
                desc = rng.choice([f"ZELLE PAYMENT {verb} {who} {_alnum(rng, 11)}",
                                   f"ZELLE {verb} {who} ON {ms:%m}/{rng.randint(1, 28):02d} REF # PP0{_alnum(rng, 7)}",
                                   f"Zelle payment {verb.lower()} {who.title()} Conf# {_alnum(rng, 9).lower()}"])
            elif rail == "Cash App":
                desc = rng.choice([f"CASH APP*{who.split()[0]} {_digits(rng, 10)} CA", f"SQ *CASH APP {who.split()[0]} CA"])
            else:
                desc = f"PAYPAL INST XFER {_digits(rng, 13)} WEB ID: PAYPALSI77"
            add(desc, amt, rail, "TRANSFER")

        # credit card autopay
        if rng.random() < 0.7:
            card, desc = rng.choice([
                ("Chase Credit Card", f"CHASE CREDIT CRD AUTOPAY PPD ID: {_digits(rng, 10)}"),
                ("American Express", f"AMEX EPAYMENT ACH PMT {_alnum(rng, 10)}"),
                ("Discover", f"DISCOVER E-PAYMENT {_digits(rng, 4)} WEB ID: {_digits(rng, 10)}"),
                ("Citi", f"CITI AUTOPAY PAYMENT {_digits(rng, 6)} WEB ID: CITICARDAP"),
                ("Capital One", f"CAPITAL ONE MOBILE PMT {_alnum(rng, 8)}"),
            ])
            add(desc, -_log_uniform(rng, 50, 2500), card, "TRANSFER", True, ms + timedelta(days=min(dim - 1, 20)))

        # brokerage / savings
        if rng.random() < 0.35:
            br, desc = rng.choice([
                ("Robinhood", f"ROBINHOOD DEBITS {_digits(rng, 10)} WEB ID: {_digits(rng, 10)}"),
                ("Coinbase", f"COINBASE.COM {_alnum(rng, 8)} {_phone(rng)} CA"),
                ("Charles Schwab", f"SCHWAB BROKERAGE MONEYLINK {_alnum(rng, 8)}"),
                ("Fidelity", f"FID BKG SVC LLC MONEYLINE {_alnum(rng, 8)}"),
                ("SoFi Invest", f"SOFI SECURITIES {_alnum(rng, 8)} WEB ID: {_digits(rng, 10)}"),
            ])
            add(desc, -_log_uniform(rng, 25, 1000), br, "TRANSFER")

        # ATM
        if rng.random() < 0.3:
            street = f"{rng.randint(10, 9999)} {rng.choice(_NOUN)} {rng.choice(['ST', 'AVE', 'BLVD'])}"
            add(rng.choice([f"ATM WITHDRAWAL {_digits(rng, 6)} {street} {u.home[0][:13]} {u.home[1]}",
                            f"NON-CHASE ATM WITHDRAW {_digits(rng, 6)} {street}",
                            f"ATM CASH WITHDRAWAL {u.home[0][:13]} {u.home[1]}"]),
                -rng.choice([20, 40, 60, 80, 100, 200]), "ATM", "TRANSFER")

        # bank fees (rare)
        if rng.random() < 0.08:
            add(rng.choice(["MONTHLY SERVICE FEE", "MONTHLY MAINTENANCE FEE",
                            f"OVERDRAFT FEE FOR A ${_log_uniform(rng, 5, 90):.2f} ITEM", "INSUFFICIENT FUNDS FEE",
                            "NON-CHASE ATM FEE-WITH", "FOREIGN TRANSACTION FEE", "INTL TRANSACTION FEE",
                            "ATM FEE", "WIRE TRANSFER FEE", "LATE PAYMENT FEE"]),
                -rng.choice([2.5, 3.0, 3.5, 5.0, 12.0, 15.0, 25.0, 34.0, 35.0]), "Bank Fee", "FEES")

        # tax refund / interest
        if ms.month == 3 and rng.random() < 0.5:
            add(f"IRS TREAS 310 TAX REF {_digits(rng, 15)}", _log_uniform(rng, 100, 2500), "IRS", "INCOME")
        if rng.random() < 0.6:
            d = ms + timedelta(days=dim - 1)
            add(rng.choice(["INTEREST PAYMENT", "INTEREST EARNED", "INTEREST PAID"]), _log_uniform(rng, 0.05, 30),
                "Interest", "INCOME", True, d)

    def run(self) -> list[Txn]:
        txns: list[Txn] = []
        for i in range(self.n_users):
            txns.extend(self.simulate_user(i))
        return txns


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def assign_splits(merchant_category: dict[str, str]) -> dict[str, str]:
    """Merchant-level, category-stratified 70/15/15 split.

    Every txn of a merchant lands in the same split, so test numbers measure
    generalization to merchants never seen in training. Within each category,
    merchants are ordered by a stable hash; categories with >= 3 merchants are
    guaranteed at least one val and one test merchant.
    """
    by_cat: dict[str, list[str]] = {}
    for m, c in merchant_category.items():
        by_cat.setdefault(c, []).append(m)
    out: dict[str, str] = {}
    for ms in by_cat.values():
        ms.sort(key=_h)
        n = len(ms)
        n_test = max(1, round(n * 0.15)) if n >= 3 else 0
        n_val = max(1, round(n * 0.15)) if n >= 3 else 0
        for i, m in enumerate(ms):
            out[m] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


FIELDS = ["txn_id", "user_id", "date", "description", "amount", "merchant", "category",
          "is_recurring", "merchant_kind", "split"]


def write_csv(txns: list[Txn], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first_cat: dict[str, str] = {}
    for t in txns:
        first_cat.setdefault(t.merchant, t.category)
    splits = assign_splits(first_cat)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for i, t in enumerate(txns):
            w.writerow([f"t{i:07d}", t.user_id, t.date.isoformat(), t.description, f"{t.amount:.2f}",
                        t.merchant, t.category, int(t.is_recurring), t.merchant_kind, splits[t.merchant]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--locals-per-category", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/transactions.csv"))
    args = ap.parse_args()

    txns = Simulator(args.seed, args.users, args.locals_per_category).run()
    write_csv(txns, args.out)
    merchants = {t.merchant for t in txns}
    print(f"wrote {len(txns):,} txns, {len(merchants):,} merchants, {args.users} users -> {args.out}")


if __name__ == "__main__":
    main()
