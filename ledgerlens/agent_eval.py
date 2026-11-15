"""Eval for the finance agent: ~50 templated questions about the demo user, auto-graded.

    python -m ledgerlens.agent_eval                          # dry run: writes the question set, estimates cost
    python -m ledgerlens.agent_eval --yes --model claude-haiku-5-5

Ground truth is computed with plain pandas from the enriched ledger the agent sees,
deliberately NOT by calling finance.py, so a bug in a tool can't grade itself correct.
(Exceptions: "subscriptions" and the cash-flow forecast are *defined* by those tools, so
their truth comes from them; those questions test whether the agent picks and reads the
right tool, not the tool's arithmetic.)

Grading is automatic and strict about the value, lenient about phrasing:
  number: some figure in the reply within max($0.50, 0.5%) of the truth
  count:  the exact integer appears in the reply
  name:   the merchant name (or its cleaned form) appears in the reply
  month:  the month name appears in the reply
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ledgerlens.agent import DEFAULT_AGENT_MODEL, FinanceAgent
from ledgerlens.finance import BILL_CATEGORIES, Ledger
from ledgerlens.llm import PRICES
from ledgerlens.normalize import canon

# Pre-run estimate per question (system + tools ≈ 1.6K tokens, ~2.5 model calls, tool results):
EST_TOKENS_PER_QUESTION = (6000, 600)


@dataclass
class Question:
    id: str
    kind: str  # number | count | name | month
    question: str
    answer: float | str
    note: str = ""


def _mname(period: str) -> str:
    y, m = period.split("-")
    return f"{calendar.month_name[int(m)]} {y}"


def build_questions(ledger: Ledger) -> list[Question]:
    d = ledger.df
    spend = d[d["is_spend"]]
    months = sorted(d["month"].unique())
    full = ledger._full_months()
    qs: list[Question] = []

    def add(kind, q, a, note=""):
        qs.append(Question(f"q{len(qs) + 1:02d}", kind, q, a if isinstance(a, str) else round(float(a), 2), note))

    # 1) category x month spending
    by_cm = -spend.groupby(["category", "month"])["amount"].sum()
    cats = (-spend.groupby("category")["amount"].sum()).sort_values(ascending=False).index[:8].tolist()
    picks = [(cats[i % len(cats)], months[(3 * i + 1) % len(months)]) for i in range(16)]
    for cat, m in dict.fromkeys(picks):
        if by_cm.get((cat, m), 0) > 0:
            add("number", f"How much did I spend on {cat.lower()} in {_mname(m)}?", by_cm[(cat, m)])
    # 2) total spending per month
    for m in months[1:-1:3]:
        add("number", f"What was my total spending in {_mname(m)}, not counting transfers?",
            -spend.loc[spend["month"] == m, "amount"].sum())
    # 3) merchant totals + visit counts
    top_m = (-spend.groupby("merchant")["amount"].sum()).sort_values(ascending=False)
    for merchant in top_m.index[2:8]:
        add("number", f"How much have I spent at {merchant} in total?", top_m[merchant])
    freq = spend.groupby("merchant").size().sort_values(ascending=False)
    for merchant in freq.index[:4]:
        add("count", f"How many transactions do I have at {merchant}?", int(freq[merchant]))
    # 4) average monthly category spend over full months
    for cat in cats[:4]:
        per = (-spend[spend["category"] == cat].groupby("month")["amount"].sum()).reindex(months, fill_value=0)
        add("number", f"On average, how much do I spend on {cat.lower()} per month?", per.loc[full].mean(),
            "average over full months")
    # 5) income
    inc = d[d["is_income"]].groupby("month")["amount"].sum()
    for m in months[2::4]:
        add("number", f"How much income did I receive in {_mname(m)}?", inc.get(m, 0.0))
    # 6) biggest purchase per month (by merchant name). Rent and loan payments aren't "purchases",
    #    so bills are excluded, and the question says so.
    purchases = spend[~spend["category"].isin(BILL_CATEGORIES)]
    for m in months[::3]:
        row = purchases[purchases["month"] == m].sort_values("amount").iloc[0]
        add("name", f"What was my single biggest purchase in {_mname(m)}, not counting rent, loans or utility "
            "bills? Name the merchant.", row["merchant"], f"${-row['amount']:.2f}")
    # 7) month with highest category spend
    for cat in ["DINING", "SHOPPING", "GROCERIES"]:
        per = -spend[spend["category"] == cat].groupby("month")["amount"].sum()
        if len(per):
            add("month", f"Which month did I spend the most on {cat.lower()}?",
                calendar.month_name[int(per.idxmax().split("-")[1])])
    # 8) tool-defined concepts: recurring + forecast
    rec = ledger.recurring()
    if rec["subscriptions"]:
        add("number", "How much do my subscriptions cost per month in total?", rec["subscriptions_monthly_total"])
        add("name", "What's my most expensive subscription?", rec["subscriptions"][0]["merchant"])
        add("count", "How many subscriptions do I have?", len(rec["subscriptions"]))
    rent = [b for b in rec["bills"] if b["category"] == "HOUSING" and b["monthly_cost"] > 500]
    if rent:
        add("number", "How much is my rent each month?", rent[0]["monthly_cost"])
    fc = ledger.cashflow_forecast(1)
    add("number", "Based on my recent months, how much should I expect to have left over next month?",
        fc["expected_monthly_net"])
    add("number", "What's my typical monthly income lately?", fc["expected_monthly_income"])
    return qs


_NUM = re.compile(r"-?\$?\s?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace("$", "").replace(",", "").replace(" ", "")))
        except ValueError:
            pass
    return out


def grade(q: Question, reply: str) -> bool:
    if q.kind == "number":
        tol = max(0.5, abs(float(q.answer)) * 0.005)
        return any(abs(abs(n) - abs(float(q.answer))) <= tol for n in numbers_in(reply))
    if q.kind == "count":
        return int(q.answer) in [int(n) for n in numbers_in(reply) if float(n).is_integer()]
    if q.kind == "name":
        return canon(str(q.answer)) in canon(reply)
    if q.kind == "month":
        return str(q.answer).lower() in reply.lower()
    raise ValueError(q.kind)


def demo_ledger(model_path: Path) -> Ledger:
    from ledgerlens.pipeline import Enricher
    from ledgerlens.service import _demo_frame

    raw = _demo_frame()
    out = Enricher.load(model_path).enrich(raw[["description", "amount", "date"]])
    out["source"] = "model"
    return Ledger(pd.concat([out, raw[["date"]]], axis=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Finance agent eval (costs money with --yes)")
    ap.add_argument("--model", default=DEFAULT_AGENT_MODEL, choices=sorted(PRICES))
    ap.add_argument("--model-path", type=Path, default=Path("models/enricher.joblib"))
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--yes", action="store_true", help="actually call the API")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    ledger = demo_ledger(args.model_path)
    qs = build_questions(ledger)[: args.limit]
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "week4_agent_questions.json").write_text(json.dumps([asdict(q) for q in qs], indent=2))
    pin, pout = PRICES[args.model]
    est = len(qs) * (EST_TOKENS_PER_QUESTION[0] / 1e6 * pin + EST_TOKENS_PER_QUESTION[1] / 1e6 * pout)
    kinds = pd.Series([q.kind for q in qs]).value_counts().to_dict()
    print(f"plan: {len(qs)} questions {kinds} on {args.model}; estimated cost ≈ ${est:.2f}")
    if not args.yes:
        print("dry run: pass --yes to call the API (question set written to results/week4_agent_questions.json)")
        return

    import anthropic

    agent = FinanceAgent(anthropic.Anthropic(), ledger, model=args.model)
    rows = []
    consecutive_errors = 0
    for q in qs:
        try:
            r = agent.run([{"role": "user", "content": q.question}])
            consecutive_errors = 0
        except anthropic.APIError as e:
            # keep answers already paid for; but a broken setup (bad schema, bad key) fails fast
            consecutive_errors += 1
            print(f"{q.id} ! API error: {type(e).__name__}: {str(e)[:200]}")
            if consecutive_errors >= 3:
                raise SystemExit("3 API errors in a row; stopping. Fix the error above and re-run.") from e
            rows.append({**asdict(q), "reply": "", "correct": False, "tools": [], "tool_errors": 0,
                         "stop_reason": f"api_error:{type(e).__name__}", "seconds": 0.0, "cost_usd": 0.0,
                         "input_tokens": 0, "output_tokens": 0})
            continue
        ok = grade(q, r.reply)
        rows.append({**asdict(q), "reply": r.reply, "correct": ok, "tools": [t.tool for t in r.trace],
                     "tool_errors": sum(t.is_error for t in r.trace), "stop_reason": r.stop_reason,
                     "seconds": r.seconds, "cost_usd": round(r.cost_usd(), 5),
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens})
        print(f"{q.id} {'✓' if ok else '✗'} {q.question[:70]}")

    df = pd.DataFrame(rows)
    report = {
        "model": args.model, "n": len(df), "accuracy": float(df["correct"].mean()),
        "by_kind": df.groupby("kind")["correct"].agg(["mean", "size"]).round(3).to_dict(orient="index"),
        "avg_tool_calls": float(df["tools"].map(len).mean()),
        "tool_errors": int(df["tool_errors"].sum()),
        "avg_seconds": float(df["seconds"].mean()), "p95_seconds": float(df["seconds"].quantile(0.95)),
        "total_cost_usd": round(float(df["cost_usd"].sum()), 4),
        "cost_per_question_usd": round(float(df["cost_usd"].mean()), 5),
        "stop_reasons": df["stop_reason"].value_counts().to_dict(),
        "rows": rows,
    }
    stem = f"week4_agent_{args.model}"
    (args.results / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    md = to_markdown(report)
    (args.results / f"{stem}.md").write_text(md)
    print("\n" + md)


def to_markdown(r: dict) -> str:
    L = [f"## Finance agent eval: {r['model']}, {r['n']} questions about the demo user", "",
         f"**Accuracy: {r['accuracy']:.1%}** · avg {r['avg_tool_calls']:.1f} tool calls · "
         f"avg {r['avg_seconds']:.1f}s (p95 {r['p95_seconds']:.1f}s) · ${r['cost_per_question_usd']:.4f}/question · "
         f"total ${r['total_cost_usd']:.2f} · tool errors {r['tool_errors']}", "",
         "| kind | accuracy | n |", "|---|---|---|"]
    L += [f"| {k} | {v['mean']:.1%} | {int(v['size'])} |" for k, v in r["by_kind"].items()]
    misses = [x for x in r["rows"] if not x["correct"]]
    if misses:
        L += ["", "### Misses", "", "| question | expected | reply |", "|---|---|---|"]
        for x in misses:
            reply = x["reply"].replace("\n", " ").replace("|", "\\|")[:220]
            L.append(f"| {x['question']} | {x['answer']} | {reply} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
