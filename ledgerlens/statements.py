"""Parse bank CSV exports (Chase, BofA, SoFi, Amex, Apple Card...) into a common shape.

Column names vary by bank, so they are matched case-insensitively against known
aliases. Amounts are signed (negative = money out). Banks that split debits and
credits into two columns are merged into one signed amount.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

DESC_COLS = ["description", "transaction description", "original description", "payee", "name",
             "merchant", "details", "memo", "transaction"]
AMOUNT_COLS = ["amount", "transaction amount", "amount (usd)"]
DEBIT_COLS = ["debit", "withdrawals", "withdrawal"]
CREDIT_COLS = ["credit", "deposits", "deposit"]
DATE_COLS = ["date", "transaction date", "posting date", "posted date", "trans. date"]


class StatementError(ValueError):
    pass


def _find(header: list[str], candidates: list[str]) -> str | None:
    low = {h.strip().lower(): h for h in header}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def money(s: str) -> float | None:
    s = (s or "").replace("$", "").replace(",", "").strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    try:
        v = float(s.strip("()"))
    except ValueError:
        return None
    return -v if neg else v


def parse_csv(text: str, source: str = "upload") -> list[dict]:
    """Rows of {raw_description, amount (str, signed), date, source_file}."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    header = reader.fieldnames or []
    dcol = _find(header, DESC_COLS)
    if not dcol:
        raise StatementError(f"{source}: couldn't find a description column in {header}")
    acol, debit, credit = _find(header, AMOUNT_COLS), _find(header, DEBIT_COLS), _find(header, CREDIT_COLS)
    datecol = _find(header, DATE_COLS)
    rows = []
    for r in reader:
        desc = (r.get(dcol) or "").strip()
        if not desc:
            continue
        if acol:
            amt = money(r.get(acol, ""))
        else:
            d = money(r.get(debit, "")) if debit else None
            c = money(r.get(credit, "")) if credit else None
            amt = (c or 0.0) - abs(d or 0.0) if (d is not None or c is not None) else None
        rows.append({"raw_description": desc, "amount": "" if amt is None else f"{amt:.2f}",
                     "date": (r.get(datecol) or "").strip() if datecol else "", "source_file": source})
    return rows


def read_export(path: Path) -> list[dict]:
    return parse_csv(path.read_text(encoding="utf-8-sig"), source=path.name)
