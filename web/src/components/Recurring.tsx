import type { Dashboard, RecurringItem } from "@/lib/api";
import { usd } from "@/lib/api";

export function Recurring({ data }: { data: Dashboard }) {
  const r = data.recurring;
  const f = data.forecast;
  return (
    <div className="space-y-8">
      <section aria-labelledby="subs-title">
        <div className="flex items-baseline justify-between">
          <h2 id="subs-title" className="text-lg font-semibold">Subscriptions</h2>
          <p className="num text-sm text-ink-soft">{usd(r.subscriptions_monthly_total)} a month</p>
        </div>
        <List items={r.subscriptions} empty="No subscriptions found." />
      </section>

      <section aria-labelledby="bills-title">
        <div className="flex items-baseline justify-between">
          <h2 id="bills-title" className="text-lg font-semibold">Bills</h2>
          <p className="num text-sm text-ink-soft">{usd(r.bills_monthly_total)} a month</p>
        </div>
        <List items={r.bills} empty="No recurring bills found." />
      </section>

      <section aria-labelledby="fc-title" className="rounded-lg bg-green-soft/60 px-4 py-3">
        <h2 id="fc-title" className="text-lg font-semibold">Next month, if it looks like the last three</h2>
        <dl className="num mt-2 grid grid-cols-[1fr_auto] gap-y-1 text-sm">
          <dt className="text-ink-soft">Income</dt><dd className="text-right">{usd(f.expected_monthly_income)}</dd>
          <dt className="text-ink-soft">Subscriptions and bills</dt><dd className="text-right">−{usd(f.recurring_monthly)}</dd>
          <dt className="text-ink-soft">Everything else, typically</dt><dd className="text-right">−{usd(f.typical_other_spending_monthly)}</dd>
          <dt className="border-t border-rule pt-1 font-semibold">Left over</dt>
          <dd className={`border-t border-rule pt-1 text-right font-semibold ${f.expected_monthly_net < 0 ? "text-red" : "text-green"}`}>
            {usd(f.expected_monthly_net)}
          </dd>
        </dl>
      </section>
    </div>
  );
}

function List({ items, empty }: { items: RecurringItem[]; empty: string }) {
  if (!items.length) return <p className="mt-2 text-sm text-ink-soft">{empty}</p>;
  return (
    <ul className="mt-2 divide-y divide-rule/70">
      {items.map((s) => (
        <li key={s.merchant} className="flex items-baseline justify-between gap-3 py-2 text-sm">
          <div className="min-w-0">
            <p className="truncate font-medium">{s.merchant}</p>
            <p className="text-[12px] text-ink-faint">
              next {new Date(s.next_expected + "T00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}
              {s.price_change > 0.5 && <span className="ml-2 text-red">up {usd(s.price_change)}</span>}
            </p>
          </div>
          <p className="num shrink-0">{usd(s.monthly_cost)}</p>
        </li>
      ))}
    </ul>
  );
}
