// Typed client for the FastAPI service (proxied at /api by next.config.ts).

export type DatasetInfo = {
  dataset_id: string;
  transactions: number;
  first_date: string;
  last_date: string;
  llm_fallback: boolean;
};

export type RecurringItem = {
  merchant: string;
  category: string;
  kind: "subscription" | "bill";
  latest_charge: number;
  monthly_cost: number;
  cadence_days: number;
  charges: number;
  last_charge: string;
  next_expected: string;
  price_change: number;
};

export type Dashboard = {
  overview: {
    first_date: string;
    last_date: string;
    transactions: number;
    total_spending: number;
    total_income: number;
    months: string[];
    top_categories: { category: string; spending: number }[];
  };
  monthly_by_category: ({ month: string } & Record<string, number | string>)[];
  recurring: {
    subscriptions: RecurringItem[];
    bills: RecurringItem[];
    subscriptions_monthly_total: number;
    bills_monthly_total: number;
  };
  forecast: {
    based_on_months: string[];
    expected_monthly_income: number;
    recurring_monthly: number;
    typical_other_spending_monthly: number;
    expected_monthly_net: number;
  };
  top_merchants: { merchant: string; spending: number }[];
  enrichment: { llm_routed: number; merchants_in_directory: number; mean_confidence: number };
};

export type Txn = {
  date: string;
  description: string;
  merchant: string;
  category: string;
  amount: number;
  confidence: number;
  source: "model" | "llm" | "user";
  merchant_in_directory: boolean;
  is_recurring: boolean;
  merchant_key: string;
};

export type ReviewItem = {
  merchant_key: string;
  merchant: string;
  category: string;
  rows: number;
  mean_confidence: number;
  example: string;
  total: number;
};

// mirrors ledgerlens/taxonomy.py
export const CATEGORIES = ["GROCERIES", "DINING", "COFFEE", "TRANSPORT", "GAS", "TRAVEL", "SHOPPING",
  "SUBSCRIPTIONS", "UTILITIES", "ENTERTAINMENT", "HEALTH", "FITNESS", "EDUCATION", "HOUSING", "LOANS",
  "TRANSFER", "INCOME", "FEES"] as const;

export type TraceStep = { tool: string; input: Record<string, unknown>; output_preview: string; is_error: boolean; ms: number };

export type ChatReply = {
  reply: string;
  stop_reason: string;
  model: string;
  trace: TraceStep[];
  seconds: number;
  usage: { input_tokens: number; output_tokens: number; cost_usd: number };
};

export type ChatMessage = { role: "user" | "assistant"; content: string };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<{ ok: boolean; uploads_enabled?: boolean }>("/healthz"),
  demo: () => call<DatasetInfo>("/v1/datasets/demo", { method: "POST" }),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return call<DatasetInfo>("/v1/datasets", { method: "POST", body });
  },
  dashboard: (id: string) => call<Dashboard>(`/v1/datasets/${id}/dashboard`),
  transactions: (id: string, params: { q?: string; category?: string; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && qs.set(k, String(v)));
    return call<{ total: number; rows: Txn[] }>(`/v1/datasets/${id}/transactions?${qs}`);
  },
  review: (id: string, limit = 5) => call<{ items: ReviewItem[] }>(`/v1/datasets/${id}/review?limit=${limit}`),
  correct: (id: string, body: { merchant_key: string; merchant?: string; category?: string }) =>
    call<{ updated: number }>(`/v1/datasets/${id}/corrections`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  chat: (id: string, messages: ChatMessage[]) =>
    call<ChatReply>(`/v1/datasets/${id}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages }),
    }),
};

export const usd = (n: number, cents = true) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: cents ? 2 : 0, minimumFractionDigits: cents ? 2 : 0 });

export const monthLabel = (ym: string) => {
  const [y, m] = ym.split("-").map(Number);
  return new Date(y, m - 1, 1).toLocaleString("en-US", { month: "short" }) + (m === 1 ? ` ’${String(y).slice(2)}` : "");
};

export const titleCase = (s: string) => s.charAt(0) + s.slice(1).toLowerCase();
