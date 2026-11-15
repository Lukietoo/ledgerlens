"""Deterministic personal-finance analytics over an enriched ledger.

These functions back both the dashboard and the agent's tools. The agent never
does arithmetic itself: every number it quotes comes from here.

Definitions (also given to the agent):
  spending  = money out, excluding TRANSFER (card payments, P2P, brokerage, ATM),
              since those move money rather than spend it
  income    = money in categorized INCOME
  recurring = a merchant charged on a regular monthly/biweekly cadence
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ledgerlens.normalize import canon

BILL_CATEGORIES = {"UTILITIES", "HOUSING", "LOANS"}
NOT_SPENDING = {"TRANSFER"}


def _money(x: float) -> float:
    return round(float(x), 2)


@dataclass
class Ledger:
    """One person's enriched transactions: date, description, amount, merchant, category, is_recurring."""

    df: pd.DataFrame

    def __post_init__(self):
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df["amount"] = df["amount"].astype(float)
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["is_spend"] = (df["amount"] < 0) & ~df["category"].isin(NOT_SPENDING)
        df["is_income"] = (df["amount"] > 0) & (df["category"] == "INCOME")
        self.df = df.sort_values("date").reset_index(drop=True)

    # ---------------------------------------------------------------- helpers
    def _range(self, start: str | None, end: str | None) -> pd.DataFrame:
        d = self.df
        if start:
            d = d[d["date"] >= pd.Timestamp(start)]
        if end:
            d = d[d["date"] <= pd.Timestamp(end)]
        return d

    @property
    def as_of(self) -> pd.Timestamp:
        return self.df["date"].max()

    def _full_months(self) -> list[str]:
        """Months with data from (roughly) start to end; drops a partial first/last month."""
        months = sorted(self.df["month"].unique())
        first, last = self.df["date"].min(), self.df["date"].max()
        if first.day > 3:
            months = months[1:]
        if last.day < (last + pd.offsets.MonthEnd(0)).day - 3:
            months = months[:-1]
        return months

    # ---------------------------------------------------------------- tools
    def overview(self) -> dict:
        d = self.df
        spend = d[d["is_spend"]]
        return {
            "first_date": d["date"].min().date().isoformat(),
            "last_date": d["date"].max().date().isoformat(),
            "transactions": int(len(d)),
            "total_spending": _money(-spend["amount"].sum()),
            "total_income": _money(d.loc[d["is_income"], "amount"].sum()),
            "months": sorted(d["month"].unique().tolist()),
            "top_categories": [
                {"category": c, "spending": _money(v)}
                for c, v in (-spend.groupby("category")["amount"].sum()).sort_values(ascending=False).head(5).items()
            ],
        }

    def spending_by_category(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        d = self._range(start_date, end_date)
        spend = d[d["is_spend"]]
        by = (-spend.groupby("category")["amount"].sum()).sort_values(ascending=False)
        return {"start_date": start_date, "end_date": end_date, "total": _money(by.sum()),
                "by_category": {c: _money(v) for c, v in by.items()}}

    def monthly_spending(self, category: str | None = None) -> dict:
        if category and category.upper() in {"INCOME", *NOT_SPENDING}:
            # Silently returning $0 per month here misled the agent into "income is miscategorized"
            # (agent eval q42). Fail loudly and point at the right tool instead.
            raise ValueError(
                f"{category.upper()} is not spending, so monthly_spending has nothing to report for it. "
                "For income use cashflow_forecast (expected_monthly_income) or "
                "search_transactions(direction='in', category='INCOME') for per-month amounts.")
        d = self.df[self.df["is_spend"]]
        if category:
            d = d[d["category"] == category.upper()]
        by = -d.groupby("month")["amount"].sum()
        by = by.reindex(sorted(self.df["month"].unique()), fill_value=0.0)
        full = self._full_months()
        return {"category": category, "by_month": {m: _money(v) for m, v in by.items()},
                "average_per_full_month": _money(by.loc[by.index.isin(full)].mean()) if full else None,
                "full_months": full}

    def search_transactions(self, query: str | None = None, category: str | None = None,
                            start_date: str | None = None, end_date: str | None = None,
                            min_amount: float | None = None, max_amount: float | None = None,
                            direction: str | None = None, sort: str = "date", limit: int = 25) -> dict:
        """Filter transactions. Amount filters apply to the absolute value; direction is "in" | "out"."""
        d = self._range(start_date, end_date)
        if category:
            d = d[d["category"] == category.upper()]
        if query:
            q = canon(query)
            d = d[d["merchant"].map(canon).str.contains(q, regex=False)
                  | d["description"].map(canon).str.contains(q, regex=False)]
        absamt = d["amount"].abs()
        if min_amount is not None:
            d = d[absamt >= min_amount]
            absamt = d["amount"].abs()
        if max_amount is not None:
            d = d[absamt <= max_amount]
        if direction == "out":
            d = d[d["amount"] < 0]
        elif direction == "in":
            d = d[d["amount"] > 0]
        if sort == "amount":
            d = d.reindex(d["amount"].abs().sort_values(ascending=False).index)
        else:
            d = d.sort_values("date", ascending=False)
        limit = max(1, min(int(limit), 100))
        return {
            "matched": int(len(d)),
            "total_amount": _money(d["amount"].sum()),
            "shown": [
                {"date": r.date.date().isoformat(), "merchant": r.merchant, "category": r.category,
                 "amount": _money(r.amount), "description": r.description}
                for r in d.head(limit).itertuples()
            ],
        }

    def recurring(self) -> dict:
        """Recurring outflows per merchant, split into subscriptions and bills."""
        # transfers excluded: a card autopay repeats monthly but pays for spending already counted
        d = self.df[self.df["is_spend"] & self.df["is_recurring"]]
        items = []
        for merchant, g in d.groupby("merchant"):
            if len(g) < 2:
                continue
            g = g.sort_values("date")
            gap = float(np.median(np.diff(g["date"].values).astype("timedelta64[D]").astype(int)))
            latest, first = -g["amount"].iloc[-1], -g["amount"].iloc[0]
            # a monthly charge costs what it charges; only normalize other cadences (weekly, biweekly...)
            per_month = latest if 26 <= gap <= 35 or gap <= 0 else latest * (30.4 / gap)
            category = g["category"].mode().iloc[0]
            items.append({
                "merchant": merchant, "category": category,
                "kind": "bill" if category in BILL_CATEGORIES else "subscription",
                "latest_charge": _money(latest), "monthly_cost": _money(per_month),
                "cadence_days": int(round(gap)), "charges": int(len(g)),
                "first_charge": g["date"].iloc[0].date().isoformat(),
                "last_charge": g["date"].iloc[-1].date().isoformat(),
                "next_expected": (g["date"].iloc[-1] + pd.Timedelta(days=round(gap))).date().isoformat(),
                "price_change": _money(latest - first),
            })
        items.sort(key=lambda x: -x["monthly_cost"])
        subs = [i for i in items if i["kind"] == "subscription"]
        bills = [i for i in items if i["kind"] == "bill"]
        return {"subscriptions": subs, "bills": bills,
                "subscriptions_monthly_total": _money(sum(i["monthly_cost"] for i in subs)),
                "bills_monthly_total": _money(sum(i["monthly_cost"] for i in bills))}

    def cashflow_forecast(self, months_ahead: int = 1) -> dict:
        """Projection from the last 3 full months: income - recurring bills/subscriptions - typical other spending."""
        months = self._full_months()[-3:]
        d = self.df[self.df["month"].isin(months)]
        n = max(1, len(months))
        income = d.loc[d["is_income"], "amount"].sum() / n
        rec = self.recurring()
        recurring_total = rec["subscriptions_monthly_total"] + rec["bills_monthly_total"]
        recurring_merchants = {i["merchant"] for i in rec["subscriptions"] + rec["bills"]}
        other = d[d["is_spend"] & ~d["merchant"].isin(recurring_merchants)]
        discretionary = -other["amount"].sum() / n
        net = income - recurring_total - discretionary
        months_ahead = max(1, min(int(months_ahead), 12))
        return {
            "based_on_months": months, "as_of": self.as_of.date().isoformat(),
            "expected_monthly_income": _money(income), "recurring_monthly": _money(recurring_total),
            "typical_other_spending_monthly": _money(discretionary),
            "expected_monthly_net": _money(net),
            "months_ahead": months_ahead, "expected_net_over_period": _money(net * months_ahead),
        }

    # ---------------------------------------------------------------- dashboard
    def dashboard(self) -> dict:
        spend = self.df[self.df["is_spend"]]
        pivot = (-spend.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0))
        months = sorted(self.df["month"].unique())
        pivot = pivot.reindex(months, fill_value=0)
        top_merchants = (-spend.groupby("merchant")["amount"].sum()).sort_values(ascending=False).head(8)
        return {
            "overview": self.overview(),
            "monthly_by_category": [{"month": m, **{c: _money(v) for c, v in row.items() if v}}
                                    for m, row in pivot.iterrows()],
            "recurring": self.recurring(),
            "forecast": self.cashflow_forecast(),
            "top_merchants": [{"merchant": m, "spending": _money(v)} for m, v in top_merchants.items()],
            "enrichment": {
                "llm_routed": int((self.df.get("source", pd.Series(dtype=str)) == "llm").sum()),
                "merchants_in_directory": float(self.df.get("merchant_in_directory", pd.Series([0])).mean()),
                "mean_confidence": float(self.df.get("confidence", pd.Series([np.nan])).mean()),
            },
        }
