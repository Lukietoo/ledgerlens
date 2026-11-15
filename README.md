# Ledgerlens

Turns messy bank transaction strings into clean, structured data:

```
PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BOTTLE COF OAKLAND CA S38207461 CARD 4821
  -> merchant: Blue Bottle Coffee   category: COFFEE   processor: Square   recurring: no
```

Every fintech (SoFi, Plaid, Ramp, Copilot...) needs this layer before it can
build budgeting, underwriting, or fraud features on top. Ledgerlens is a
classic-ML pipeline (rules → normalizer → merchant resolver → category model →
recurring detector), with an LLM used only as a fallback for low-confidence cases.

**Status:** Week 1 of 5 is done: synthetic data, normalizer, rules baseline, eval harness, and labeling tool.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m ledgerlens.synth        # ~215K labeled txns -> data/synthetic/
.venv/bin/python -m ledgerlens.evaluate     # baseline report -> results/
.venv/bin/pytest -q
```

## Layout

| path | what |
|---|---|
| `ledgerlens/taxonomy.py` | 18 categories (roughly Plaid's primary categories) |
| `ledgerlens/resources/merchants.csv` | 266 chain merchants with real-world descriptor variants, price ranges, rail (card/ACH) |
| `ledgerlens/normalize.py` | deterministic descriptor cleanup (stage 1) |
| `ledgerlens/synth.py` | synthetic user/transaction simulator + noise model |
| `ledgerlens/baseline.py` | rules-only category + recurring baselines |
| `ledgerlens/evaluate.py` | metrics on synthetic test split and/or real labeled set |
| `scripts/label.py` | CLI to hand-label your own bank exports |

## Synthetic data

`synth.py` simulates 300 users × 12 months. Each user has a home city, one bank
(which fixes the statement format), an employer, 2–7 subscriptions,
utilities, rent, maybe loans/gym, and "favorite" stores with Zipf-like loyalty.
Some months include trips with spend in another city. About 215K transactions
across about 1,000 merchants (266 chains + ≈760 generated local businesses and templates).

### Noise model

What real descriptors look like, and what the generator reproduces:

| effect | example | why it happens |
|---|---|---|
| 22-char name / 13-char city fields | `LULULEMON ATHLETICA #5 LONG ISLAND C NY` | ISO 8583 card-acceptor field widths |
| processor prefix | `SQ *`, `TST* `, `CLV*`, `PAYPAL *` | Square / Toast / Clover / PayPal are the merchant of record |
| store numbers | `COSTCO WHSE #8873`, `STARBUCKS STORE 05412` | per-location merchant IDs (stable per user) |
| online merchants | `NETFLIX.COM 866-579-7172 CA`, `HELP.UBER.COM` | phone/URL replaces city |
| marketplace ref codes | `AMZN MKTP US*2K4AB1CD2` | per-order references |
| bank wrappers | `PURCHASE AUTHORIZED ON 03/14 … S382… CARD 4821`, `CHECKCARD 0413 … <23 digits>`, `POS DEBIT`, fixed-width, Title Case | each bank formats differently |
| ACH | `ZILLOW RENT DES:RENT ID:… INDN:… CO ID:… WEB`, `… PPD ID: 1234567890` | 16-char NACHA company name + entry description |
| abbreviations | `WHOLEFDS`, `BLUE BTL COF`, `SOUTHWES`, `VZWRLSS*APOCC VISB` | merchants pick their own short names |
| P2P / payroll / fees | `ZELLE TO DANIEL LEE ON 07/08 REF # PP05…`, `GUSTO PAYROLL PPD ID:` | separate templates |
| no category hint | `SQ *JADE PEARL PALO ALTO CA` (it's a restaurant) | ~30% of local merchant names |

### Splits

The split is by **merchant** and stratified by category (70/15/15). No test
merchant ever appears in training, so test numbers measure generalization to
*new* merchants, not memorization. Recurring detection uses full user histories
and has no learned parameters yet, so it's evaluated on all rows.

## Week 1 results: rules baseline (synthetic test split, unseen merchants)

| metric | value |
|---|---|
| category accuracy | **56.8%** |
| category macro-F1 | **0.601** |
| rule coverage (anything matched) | 61.3% |
| merchant exact match (normalizer only) | **44.2%** |
| recurring precision / recall / F1 | 0.973 / 0.890 / **0.930** |
| accuracy: chain / local merchants | 52.9% / 48.4% |

Full per-class table: [`results/baseline_synthetic_test.md`](results/baseline_synthetic_test.md).

Takeaways (what the ML in weeks 2–3 has to fix):
- **Rules are precise but don't cover enough.** Most errors are `UNKNOWN`: 39% of transactions match no rule at all.
  Precision is ~1.0 for most classes, but recall collapses on anything the rules author didn't anticipate.
- **GAS and HOUSING get 0 recall** on unseen merchants: brand-name rules don't transfer.
- **The normalizer gets 44% exact-match merchants.** Abbreviations (`WHOLEFDS`, `BLUE BTL COF`)
  need a learned resolver, not more regex.
- Caveat: template categories (INCOME / TRANSFER / LOANS) score 100% because the
  same person wrote the generator templates and the rules. **This is why the real hand-labeled set is
  the headline eval.**

## Labeling real data (do this; it's the number founders will ask about)

1. Export CSVs from your bank(s): SoFi, Chase, BofA, Amex and Apple Card all offer CSV export.
2. Run `.venv/bin/python scripts/label.py ~/Downloads/*.csv`. Columns are auto-detected and descriptions are
   deduped. Each row gets a suggested merchant and category: press Enter to accept, `1-18` to change the category,
   `m` to edit the merchant, `s` to skip, `u` to undo, `?` for help, `q` to save and quit. You can resume anytime.
3. Aim for **1–2K unique descriptions**. Then run `.venv/bin/python -m ledgerlens.evaluate --real data/real/labeled.csv`.

`data/real/` is gitignored, so your bank data stays on your machine.

## Roadmap

- [x] **Week 1**: synthetic generator, normalizer, rules baseline, eval harness, labeling tool
- [ ] **Week 2**: merchant resolver (char n-gram TF-IDF + kNN), LightGBM category model, feature ablations
- [ ] **Week 3**: FastAPI serving, locust load test, LLM fallback, cost per 1M transactions
- [ ] **Week 4**: Next.js dashboard + Claude tool-calling finance agent with eval set
- [ ] **Week 5**: deploy, writeup, demo video
