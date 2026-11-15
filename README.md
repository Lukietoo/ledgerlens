# Ledgerlens

Turns messy bank transaction strings into clean, structured data:

```
PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BOTTLE COF OAKLAND CA S38207461 CARD 4821
  -> merchant: Blue Bottle Coffee   category: COFFEE   processor: Square   recurring: no
```

Every fintech (SoFi, Plaid, Ramp, Copilot...) needs this layer before it can
build budgeting, underwriting, or fraud features on top. Ledgerlens is a
classic-ML pipeline (normalizer → merchant resolver → stacked category model →
recurring detector), with an LLM planned only as a fallback for low-confidence cases.

**Status:** Weeks 1–3 of 5 are done (the measured LLM-fallback accuracy is pending a paid API run).

| on merchants never seen in training | rules baseline | Ledgerlens |
|---|---|---|
| category accuracy | 53.5% | **81.5%** |
| category macro-F1 | 0.583 | **0.682** |
| accuracy, leak-free slice ([why](#leakage-check)) | 42.8% | **76.0%** |
| merchant name exactly right (known chains) | 49.2% | **94.1%** |
| throughput (4 workers, M1 laptop) | – | **11.5K txns/s**, p95 38 ms per 100-txn batch |
| serving cost per 1M txns (model only) | – | **≈ $0.004** vs ≈ $1,400 for an LLM-only pipeline |

Synthetic data, mean of 3 seeds. Real hand-labeled numbers: [in progress](#labeling-real-data).

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m ledgerlens.synth        # ~216K labeled txns -> data/synthetic/
.venv/bin/python -m ledgerlens.evaluate     # week-1 rules baseline -> results/
.venv/bin/python -m ledgerlens.train        # week-2 model + ablations (~3 min) -> results/, models/
.venv/bin/pytest -q
```

```python
from ledgerlens.pipeline import Enricher
Enricher.load().enrich(df)   # df: description, amount, date[, user_id]
# -> merchant, merchant_in_directory, category, confidence, is_recurring, processor
```

```bash
.venv/bin/uvicorn ledgerlens.service:app --workers 4          # API on :8000, model only
LEDGERLENS_LLM=1 .venv/bin/uvicorn ledgerlens.service:app     # + LLM fallback (needs Anthropic credentials)

curl -s localhost:8000/v1/enrich -H 'content-type: application/json' -d '{"transactions": [
  {"description": "PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BTL COF OAKLAND CA S38207461 CARD 4821",
   "amount": -6.50, "date": "2026-03-14"}]}'
```

## Layout

| path | what |
|---|---|
| `ledgerlens/taxonomy.py` | 18 categories (roughly Plaid's primary categories) |
| `ledgerlens/resources/merchants.csv` | 266 chain merchants with real-world descriptor variants, price ranges, rail (card/ACH) |
| `ledgerlens/resources/entities.csv` | P2P rails, card issuers, brokerages, payroll processors |
| `ledgerlens/normalize.py` | deterministic descriptor cleanup (stage 1) |
| `ledgerlens/synth.py` | synthetic user/transaction simulator + noise model |
| `ledgerlens/baseline.py` | rules-only category + recurring baselines |
| `ledgerlens/resolver.py` | char n-gram TF-IDF merchant resolver with abstention |
| `ledgerlens/features.py` | format / amount / user-history features |
| `ledgerlens/model.py` | text model + out-of-fold stacking + LightGBM |
| `ledgerlens/train.py` | multi-seed training, ablations, resolver eval |
| `ledgerlens/pipeline.py` | `Enricher`: inference over a batch of transactions; LLM fallback routing |
| `ledgerlens/service.py` | FastAPI app: `POST /v1/enrich`, `GET /healthz` |
| `ledgerlens/llm.py` | Claude fallback classifier (batched, structured outputs, refusal fallbacks) |
| `ledgerlens/routing.py` | confidence calibration + threshold sweep |
| `ledgerlens/llm_eval.py` | LLM-only vs model-only vs hybrid eval (dry-run unless `--yes`) |
| `ledgerlens/cost.py` | cost per 1M transactions from measured throughput and tokens |
| `loadtest/locustfile.py` | locust load test against the API |
| `ledgerlens/evaluate.py` | metrics on synthetic test split and/or real labeled set |
| `scripts/label.py` | CLI to hand-label your own bank exports |

## Synthetic data

`synth.py` simulates 300 users × 12 months. Each user has a home city, one bank
(which fixes the statement format), an employer, 2–7 subscriptions,
utilities, rent, maybe loans/gym, and "favorite" stores with Zipf-like loyalty.
Some months include trips with spend in another city. About 216K transactions
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
| ACH | `ZILLOW RENT DES:RENT ID:… INDN:… CO ID:… WEB`, `… PPD ID: 1234567890` | 16-char NACHA company name + entry description, stable per payer |
| abbreviations | `WHOLEFDS`, `BLUE BTL COF`, `SOUTHWES`, `VZWRLSS*APOCC VISB` | merchants pick their own short names |
| P2P / payroll / fees | `ZELLE TO DANIEL LEE ON 07/08 REF # PP05…`, `GUSTO PAYROLL PPD ID:` | separate templates |
| no category hint | `SQ *JADE PEARL PALO ALTO CA` (it's a restaurant) | ~30% of local merchant names |
| price points | $15.49, $11.99, $20.00 | subscriptions end in .99 / .49 / .00 (see lesson 3 below) |

### Splits

The split is by **merchant** and stratified by category (70/15/15). No test
merchant ever appears in training, so test numbers measure generalization to
*new* merchants, not memorization.

## Week 1: rules baseline

Synthetic test split, unseen merchants ([full report](results/baseline_synthetic_test.md)):

| metric | value |
|---|---|
| category accuracy | 53.5% |
| category macro-F1 | 0.583 |
| rule coverage (anything matched) | 60.8% |
| merchant exact match (normalizer only) | 45.5% |
| recurring precision / recall / F1 | 0.977 / 0.888 / **0.930** |

Rules are precise but don't cover enough. 39% of transactions match no rule, and GAS and HOUSING get
0 recall on unseen merchants because brand-name rules don't transfer.

## Week 2: merchant resolver + stacked category model

```
cleaned descriptor ──char/word TF-IDF──> logistic regression ──> 18 class probs ─┐
raw format (processor, ACH entry type, online) + amount + per-user history ──────┤
rules-baseline prediction ───────────────────────────────────────────────────────┼─> LightGBM
merchant resolver (similarity, matched merchant's category, top-5 votes) ────────┘
```

**Merchant resolver.** The directory stands in for a production merchant DB and holds each chain's
canonical name plus *one* descriptor. Every other alias is held out, so `BLUE BTL COF`, `SWEETGRN` and
`VZWRLSS` have to be fuzzy-matched (char 2–4-gram TF-IDF, cosine). Below a confidence threshold
(tuned on val) the resolver abstains and returns the cleaned name, which is the right call for
the long tail of local businesses.

**Stacking without leakage.** Text-model probabilities and resolver features for training rows
are produced out-of-fold (GroupKFold over *merchants*). In each fold, 80% of the held-out
merchants are also hidden from the directory, so LightGBM learns how much to trust a resolver match.

### Results ([full report](results/week2.md), mean ± std over 3 seeds)

| model (test split, unseen merchants) | accuracy | macro-F1 | leak-free acc | local merchants |
|---|---|---|---|---|
| Rules baseline (week 1) | 53.5% | 0.583 | 42.8% | 44.2% |
| TF-IDF + logistic regression | 45.1% | 0.362 | 50.4% | 71.3% |
| LightGBM: text + format + amount | 52.3% ± 0.9 | 0.387 | 58.8% | 80.7% |
| + user history | 60.9% ± 1.3 | 0.456 | 67.3% | 87.9% |
| + rules prediction as a feature | 80.8% ± 0.0 | 0.673 | 75.1% | 89.9% |
| **+ merchant resolver (full model)** | **81.5% ± 0.1** | **0.682** | **76.0%** | 89.8% |
| full model, first-seen txn (no history) | 73.4% ± 0.6 | 0.637 | 66.4% | 79.1% |

With the full merchant directory available (production setting), the full model reaches **86.7% ± 3.0**.
Merchant resolution goes from **45.5% → 82.8%** exact match overall (**94.1%** on known chains).
Inference takes about 25 ms for a small batch on a laptop (see the Week 3 load test).

### What I learned (the parts worth talking about)

1. <a id="leakage-check"></a>**My own rules leaked into the "unseen" test set.** Feeding the
   rules' prediction to LightGBM was the single biggest jump (+20 pts). But I wrote those rules
   while looking at the merchant list, and keywords like `NETFLIX`, `AIRBNB` and `TICKETMASTER` name 9
   test merchants (23% of test txns). `baseline.is_brand_match` flags rule hits on brand words
   (not generic ones like `COFFEE`, and not city names: the `BERKELEY` tuition rule also fires on
   *purchases in Berkeley*, a real baseline bug). The **leak-free column** excludes those
   merchants. It's the honest generalization number, and the model still gains +33 pts there.
2. **Train/serve skew.** A "how many users share this merchant" feature helped offline but is
   ~1 for a single user's upload at inference. It was removed. **History dropout** (blanking
   per-user history on 20% of training rows) makes first-seen transactions work: 73.4% with no history at all.
3. **The model learned a bug in my generator.** A smoke test classified a $15.49 Netflix charge as
   DINING. The generator priced every subscription at $x.99, so the model learned ".99 ⟹
   subscription". I fixed the generator to use realistic price endings. Synthetic data needs its own QA.
4. **Generic payment words fool string matching.** `SOFI TECHNOLOGI PAYROLL` resolved to "ADP" on
   the word PAYROLL. The resolver now ignores payment-type words (PAYROLL, AUTOPAY, PMT...).
5. **Seed variance is real.** Single runs moved ±1–2 pts. Cold-start results are stable (±0.1);
   warm-directory results are not (±3.0). That's the next thing to fix.

Selection caveat: the 80% hide rate was chosen after comparing two settings (50% vs 80%) on the
test split. At 50%, the resolver *lowered* cold accuracy by 3 pts.

Remaining errors are mostly genuinely ambiguous: renters insurance billed monthly looks like a
subscription, Dave & Buster's is food *and* entertainment, and a Tesla Supercharger
charge has no obvious hint.

## Week 3: serving, load test, LLM fallback, cost

```
POST /v1/enrich ──> Enricher (normalizer → resolver → LightGBM) ──> confidence ≥ 0.8 ──> response
                                                                 └─> confidence < 0.8 ──> Claude (batched,
                                                                     structured output, category enum) ──> response
```

### Load test ([locust](loadtest/locustfile.py), 4 uvicorn workers, M1 laptop, locust on the same machine)

| workload | concurrency | throughput | p50 | p95 | p99 | errors |
|---|---|---|---|---|---|---|
| 1 txn / request | 4 users | 213 req/s | 13 ms | 17 ms | 26 ms | 0 |
| 100 txns / request | 4 users | **11,500 txns/s** | 26 ms | 38 ms | 73 ms | 0 |
| 1 txn / request | 32 users (saturated) | 279 req/s | 110 ms | 170 ms | 200 ms | 0 |
| 100 txns / request | 32 users (saturated) | 9,800 txns/s | 260 ms | 740 ms | 1.3 s | 0 |

Each request has about 9 ms of fixed overhead (mostly pandas). Batching amortizes it, and that's the lever for real-time
single-transaction latency later.

### When to call the LLM ([routing report](results/week3_routing.md))

The production path reproduces the training numbers (cold 81.6%, warm 85.4%), so there's no train/serve skew.
Sweeping a confidence threshold:

| route to LLM below 0.8 | traffic routed | model acc on rows it keeps | hybrid ceiling (perfect LLM) |
|---|---|---|---|
| cold directory | 21.5% | 87.7% | 90.3% |
| warm directory | 13.9% | 92.0% | 93.1% |

**Limit of confidence routing:** the model is somewhat overconfident (ECE ≈ 0.07). The 0.9–1.0 bin holds
most of the traffic at 89% accuracy (cold), so a *confident* mistake (renters insurance → SUBSCRIPTIONS)
never reaches the LLM. Better calibration, or routing on "merchant not in directory", is the next lever.

### LLM fallback design ([llm.py](ledgerlens/llm.py))

- 25 transactions per request, with `output_config.format` constrained to a JSON schema whose category field is the
  18-value enum, so every response parses and every category is valid.
- The model's guess goes along as a hint ("these rows were sent to you because it was unsure").
- `effort: "low"`: this is classification, not reasoning-heavy work.
- Server-side refusal fallbacks (`fallbacks: "default"`) are on. If the chain still refuses, the rows keep the model's answer.
- A merchant found in the directory is never overwritten by the LLM's name.
- Defaults to `claude-opus-5-5`; any model in `PRICES` can be passed (`LEDGERLENS_LLM_MODEL`, `--model`).

### Cost per 1M transactions ([cost report](results/week3_cost.md))

| | per 1M txns |
|---|---|
| Ledgerlens model only (measured throughput, assumes $0.04/vCPU-hour) | **≈ $0.004** |
| hybrid, Opus 5.5 fallback on 21.5% of traffic | ≈ $298 (est.) |
| hybrid, Haiku 5.5 fallback | ≈ $7 (est.) |
| LLM only, Opus 5.5 | ≈ $1,381 (est.) |
| LLM only, Haiku 5.5 | ≈ $35 (est.) |

(est.) = token counts estimated before any API call (pessimistic output). The measured version needs:

```bash
.venv/bin/python -m ledgerlens.llm_eval                    # dry run: plan + cost estimate, no API calls
.venv/bin/python -m ledgerlens.llm_eval --yes              # ~$0.73 on Opus 5.5 for 400 txns; ~$0.02 on Haiku 5.5
.venv/bin/python -m ledgerlens.cost                        # re-prices with measured tokens
```

That run also answers the question that matters: **does the hybrid beat both model-only and LLM-only on accuracy?**

## Labeling real data

This is the number founders will ask about.

1. Export CSVs from your bank(s): SoFi, Chase, BofA, Amex and Apple Card all offer CSV export.
2. Run `.venv/bin/python scripts/label.py ~/Downloads/*.csv`. Columns are auto-detected and descriptions are
   deduped. Each row gets a suggested merchant and category: press Enter to accept, `1-18` to change the category,
   `m` to edit the merchant, `s` to skip, `u` to undo, `?` for help, `q` to save and quit. You can resume anytime.
3. Aim for **1–2K unique descriptions**. Then run `.venv/bin/python -m ledgerlens.evaluate --real data/real/labeled.csv`,
   which reports the rules baseline and the trained model side by side. Rows are unique descriptions, so this is the
   first-seen regime (expect it to look like the 73% row, not the 81% one).

`data/real/` is gitignored, so your bank data stays on your machine.

## Roadmap

- [x] **Week 1**: synthetic generator, normalizer, rules baseline, eval harness, labeling tool
- [x] **Week 2**: merchant resolver, stacked TF-IDF + LightGBM category model, ablations, leakage audit
- [x] **Week 3**: FastAPI serving, locust load test, confidence routing, LLM fallback, cost per 1M transactions
  - [ ] measured LLM-only / hybrid accuracy (`llm_eval --yes`, needs API credentials, ~$0.02–0.73)
- [ ] **Week 4**: Next.js dashboard + Claude tool-calling finance agent with eval set
- [ ] **Week 5**: deploy, writeup, demo video
