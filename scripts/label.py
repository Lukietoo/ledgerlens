"""Hand-label real bank transactions for the held-out eval set.

    python scripts/label.py ~/Downloads/chase_export.csv ~/Downloads/sofi.csv

Reads one or more bank CSV exports (column names are auto-detected), dedupes by
description, pre-fills a suggested category (rules baseline) and merchant
(normalizer), and lets you accept or fix each one. Progress is appended to
data/real/labeled.csv after every row, so you can quit and resume anytime.

data/real/ is gitignored: your bank data never leaves your machine.

Keys:  Enter = accept suggestion   1-18 = pick category   m = edit merchant
       s = skip (not a real txn)   u = undo last   q = save & quit
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from ledgerlens.baseline import categorize
from ledgerlens.normalize import display_name, normalize
from ledgerlens.statements import StatementError, read_export
from ledgerlens.taxonomy import CATEGORIES, DESCRIPTIONS, UNKNOWN

OUT_FIELDS = ["raw_description", "amount", "date", "merchant", "category", "source_file"]

def load_done(out: Path) -> list[dict]:
    if not out.exists():
        return []
    with out.open(newline="") as f:
        return list(csv.DictReader(f))


def save(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)


def menu() -> str:
    cols = [f"{i + 1:>2} {c:<13}" for i, c in enumerate(CATEGORIES)]
    return "\n".join("  ".join(cols[i:i + 6]) for i in range(0, len(cols), 6))


def suggest_merchant(raw: str) -> str:
    clean = normalize(raw).clean
    return display_name(clean) if clean else raw


def run(inputs: list[Path], out: Path, inp=input, echo=print) -> int:
    done = load_done(out)
    seen = {r["raw_description"] for r in done}
    todo: list[dict] = []
    for p in inputs:
        try:
            parsed = read_export(p)
        except StatementError as e:
            raise SystemExit(str(e)) from e
        for r in parsed:
            if r["raw_description"] not in seen:
                seen.add(r["raw_description"])
                todo.append(r)

    echo(f"{len(done)} already labeled, {len(todo)} unique descriptions to go -> {out}\n")
    i = 0
    while i < len(todo):
        r = todo[i]
        cat = categorize(r["raw_description"])
        merchant = suggest_merchant(r["raw_description"])
        echo(f"[{len(done) + 1}]  {r['raw_description']}    ({r['amount']})")
        echo(f"   suggest: {merchant!r}  /  {cat}" + ("" if cat != UNKNOWN else "  <- pick a number"))
        while True:
            ans = inp("   > ").strip()
            if ans == "q":
                save(out, done)
                echo(f"saved {len(done)} rows")
                return len(done)
            if ans == "?":
                echo(menu())
                for c in CATEGORIES:
                    echo(f"     {c:<13} {DESCRIPTIONS[c]}")
                continue
            if ans == "u":
                if done:
                    prev = done.pop()
                    if i > 0 and todo[i - 1]["raw_description"] == prev["raw_description"]:
                        i -= 1  # re-ask it; rows from earlier sessions are just removed
                    echo(f"   undid: {prev['raw_description']}")
                break
            if ans == "m":
                merchant = inp("   merchant: ").strip() or merchant
                continue
            if ans == "s":
                done.append({**r, "merchant": "", "category": "SKIP"})
                i += 1
                break
            if ans.isdigit() and 1 <= int(ans) <= len(CATEGORIES):
                cat = CATEGORIES[int(ans) - 1]
            elif ans != "" or cat == UNKNOWN:
                echo(menu())
                continue
            done.append({**r, "merchant": merchant, "category": cat})
            i += 1
            break
        save(out, done)
        if ans != "u":
            echo("")
    echo(f"all done: {len(done)} rows in {out}")
    return len(done)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-label real bank transactions", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+", type=Path, help="bank CSV export(s)")
    ap.add_argument("--out", type=Path, default=Path("data/real/labeled.csv"))
    args = ap.parse_args()
    print(menu(), "\n", file=sys.stderr)
    run(args.exports, args.out)


if __name__ == "__main__":
    main()
