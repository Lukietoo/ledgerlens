"""LLM fallback: classify the transactions the model isn't confident about.

Batches several transactions per request and uses structured outputs (a JSON schema
with the category enum), so every response parses. Server-side refusal fallbacks
(``fallbacks: "default"``) are on by default; if the whole chain still refuses, those
rows keep the model's prediction.

The client is injected, so tests and offline runs use a fake. Nothing here runs
unless you construct an ``LLMClassifier`` with a real ``anthropic.Anthropic()`` client.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ledgerlens.taxonomy import CATEGORIES, DESCRIPTIONS

DEFAULT_MODEL = "claude-opus-5-5"

# $ per 1M tokens (input, output), standard API rates. Cached 2026-11-03.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM_PROMPT = (
    "You categorize raw bank transaction descriptors for a personal finance app.\n\n"
    "For each transaction, return:\n"
    "- merchant: the business or counterparty a person would recognize "
    "(\"Blue Bottle Coffee\", not \"SQ *BLUE BTL COF\"). Strip processor prefixes, store numbers, "
    "locations and reference codes. For person-to-person transfers, use the payment rail (\"Zelle\").\n"
    "- category: exactly one of the categories below.\n\n"
    "Amounts are signed: negative is money out, positive is money in. A hint from the in-house "
    "model is included when available; it is right more often than not, but these rows were sent "
    "to you because it was unsure.\n\n"
    "Categories:\n" + "\n".join(f"- {c}: {DESCRIPTIONS[c]}" for c in CATEGORIES)
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "merchant": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                },
                "required": ["id", "merchant", "category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


@dataclass
class Txn:
    description: str
    amount: float
    hint: str | None = None  # the in-house model's guess


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refusals: int = 0
    seconds: float = 0.0
    by_model: dict[str, list[int]] = field(default_factory=dict)  # model -> [in, out]

    def add(self, response, seconds: float) -> None:
        self.requests += 1
        self.seconds += seconds
        u = response.usage
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        tally = self.by_model.setdefault(response.model, [0, 0])
        tally[0] += u.input_tokens
        tally[1] += u.output_tokens

    def cost_usd(self) -> float:
        total = 0.0
        for model, (tin, tout) in self.by_model.items():
            pin, pout = PRICES.get(model, PRICES[DEFAULT_MODEL])
            total += tin / 1e6 * pin + tout / 1e6 * pout
        return total


@dataclass
class Prediction:
    merchant: str
    category: str


class LLMClassifier:
    def __init__(self, client, model: str = DEFAULT_MODEL, effort: str = "low", batch_size: int = 25,
                 max_tokens: int = 8000, refusal_fallbacks: bool = True):
        self.client = client
        self.model = model
        self.effort = effort
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.refusal_fallbacks = refusal_fallbacks
        self.usage = Usage()
        # merchant key -> answer. Unbounded and in-process: fine for one service instance;
        # production would back this with Redis/Postgres and write confirmed names into the directory.
        self.cache: dict[str, Prediction] = {}
        self.cache_hits = 0

    @staticmethod
    def render(batch: list[Txn]) -> str:
        lines = []
        for i, t in enumerate(batch):
            hint = f" | model hint: {t.hint}" if t.hint else ""
            lines.append(f"{i}. {t.description!r} | amount: {t.amount:+.2f}{hint}")
        return "Transactions:\n" + "\n".join(lines)

    def _request(self, batch: list[Txn]):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self.render(batch)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )
        if self.refusal_fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return self.client.beta.messages.create(**kwargs)

    def classify_keyed(self, items: dict[str, Txn]) -> dict[str, Prediction]:
        """Classify one representative txn per key, answering repeats from the cache.
        Keys the LLM refused or skipped are absent from the result (and not cached)."""
        hits = {k: self.cache[k] for k in items if k in self.cache}
        self.cache_hits += len(hits)
        todo = [k for k in items if k not in self.cache]
        for k, pred in zip(todo, self.classify([items[k] for k in todo])):
            if pred is not None:
                self.cache[k] = pred
                hits[k] = pred
        return hits

    def classify(self, txns: list[Txn]) -> list[Prediction | None]:
        """One prediction per input; None where the LLM refused or skipped a row."""
        out: list[Prediction | None] = [None] * len(txns)
        for start in range(0, len(txns), self.batch_size):
            batch = txns[start:start + self.batch_size]
            t0 = time.perf_counter()
            response = self._request(batch)
            self.usage.add(response, time.perf_counter() - t0)
            if response.stop_reason == "refusal":
                self.usage.refusals += 1
                continue
            text = next((b.text for b in response.content if b.type == "text"), None)
            if text is None:
                continue
            for r in json.loads(text)["results"]:
                i = r["id"]
                if 0 <= i < len(batch) and r["category"] in CATEGORIES:
                    out[start + i] = Prediction(merchant=r["merchant"].strip(), category=r["category"])
        return out
