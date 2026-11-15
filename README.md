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

**Status:** all 5 weeks built. The writeup is [docs/WRITEUP.md](docs/WRITEUP.md); the demo script is [docs/DEMO.md](docs/DEMO.md).

| on merchants never seen in training | rules baseline | Ledgerlens |
|---|---|---|
| category accuracy | 53.5% | **81.2%** |
| category macro-F1 | 0.583 | **0.672** |
| accuracy, leak-free slice ([why](#leakage-check)) | 42.8% | **75.8%** |
| merchant name exactly right (known chains) | 50.1% | **94.1%** |
| throughput (4 workers, M1 laptop) | – | **11.5K txns/s**, p95 38 ms per 100-txn batch |
| bank statement formats never seen in training | – | **78.6%** category (vs 78.2% on seen formats) |
| hybrid with Claude Haiku 5.5 fallback (400 unseen txns) | – | **91.2% category / 82.2% merchant**, matching Haiku-only (91.5% / 83.8%) |
| serving cost per 1M txns | – | ≈ $0.004 model only · **≈ $0.59–1.15 hybrid** (cached) · $26 Haiku-only |

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
| `ledgerlens/statements.py` | bank CSV parser (column aliases, split debit/credit columns) |
| `ledgerlens/finance.py` | deterministic analytics: spending, recurring, forecast (the agent's tools) |
| `ledgerlens/agent.py` | Claude finance agent: tool loop, trace, guardrails |
| `ledgerlens/guards.py` | public-deploy guards: per-client rate limits, daily LLM spend cap |
| `scripts/fetch_model.py` | download + SHA-256-verify the model for deploys (or train) |
| `ledgerlens/format_eval.py` | leave-one-bank-format-out evaluation (unseen statement formats) |
| `ledgerlens/boilerplate.py` | learns a bank's wrapper words from one statement, no labels |
| `ledgerlens/agent_eval.py` | 42-question agent eval with independent ground truth (97.6% on Haiku 5.5) |
| `web/` | Next.js dashboard: decode hero, spending chart, transaction feed, recurring, chat |
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
| merchant exact match (normalizer only) | 46.8% |
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
| TF-IDF + logistic regression | 46.0% | 0.366 | 51.7% | 71.2% |
| LightGBM: text + format + amount | 51.8% ± 0.4 | 0.396 | 58.2% | 80.4% |
| + user history | 61.3% ± 0.7 | 0.458 | 65.7% | 88.4% |
| **+ rules prediction as a feature** | **81.2% ± 0.3** | **0.672** | **75.8%** | 90.9% |
| + merchant resolver (full model) | 80.5% ± 0.9 | 0.659 | 74.9% | 89.4% |
| full model, first-seen txn (no history) | 73.1% ± 0.2 | 0.630 | 65.9% | 79.4% |

With the full merchant directory available (production setting), the full model reaches **86.7% ± 3.2**.
Merchant resolution goes from **46.8% → 83.5%** exact match overall (**94.1%** on known chains).
**In the cold regime the resolver is neutral** (80.5% ± 0.9 vs 81.2% ± 0.3 without it, within noise): it can't match
a merchant the directory doesn't have. It earns its place in the warm regime and in merchant names.
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
   per-user history on 20% of training rows) makes first-seen transactions work: 73.1% with no history at all.
3. **The model learned a bug in my generator.** A smoke test classified a $15.49 Netflix charge as
   DINING. The generator priced every subscription at $x.99, so the model learned ".99 ⟹
   subscription". I fixed the generator to use realistic price endings. Synthetic data needs its own QA.
4. **Generic payment words fool string matching.** `SOFI TECHNOLOGI PAYROLL` resolved to "ADP" on
   the word PAYROLL. The resolver now ignores payment-type words (PAYROLL, AUTOPAY, PMT...).
5. **Seed variance is real.** Single runs moved ±1–2 pts, enough to flip which of two rows looks best. Warm-directory
   results vary ±3.2 between seeds. That's the next thing to fix.

Selection caveat: the 80% hide rate was chosen after comparing two settings (50% vs 80%) on the
test split. At 50% the resolver lowered cold accuracy by 3 pts; at 80% it's neutral. An earlier version of this
README said the full model was best in the cold regime. That stopped being true after the Week 4 city fix,
so the claim was dropped.

Remaining errors are mostly genuinely ambiguous: renters insurance billed monthly looks like a
subscription, Dave & Buster's is food *and* entertainment, and a Tesla Supercharger
charge has no obvious hint.

## Week 3: serving, load test, LLM fallback, cost

```
POST /v1/enrich ──> Enricher (normalizer → resolver → LightGBM) ──> confident + merchant in directory ──> response
                                                                 └─> confidence < 0.8 OR merchant unknown
                                                                     ──> per-merchant cache ──hit──> response
                                                                                            └─miss─> Claude (batched,
                                                                     structured output, category enum) ──> cache ──> response
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

The production path reproduces the training numbers (cold 80.6%, warm 85.1%), so there's no train/serve skew.
Sweeping a confidence threshold:

| route to LLM below 0.8 | traffic routed | model acc on rows it keeps | hybrid ceiling (perfect LLM) |
|---|---|---|---|
| cold directory | 22.8% | 88.7% | 91.2% |
| warm directory | 15.7% | 92.7% | 93.9% |

**Limit of confidence routing:** the model is somewhat overconfident (ECE ≈ 0.07). The 0.9–1.0 bin holds
most of the traffic at 89% accuracy (cold), so a *confident* mistake (renters insurance → SUBSCRIPTIONS)
never reaches the LLM. Better calibration, or routing on "merchant not in directory", is the next lever.

### LLM fallback design ([llm.py](ledgerlens/llm.py))

- 25 transactions per request, with `output_config.format` constrained to a JSON schema whose category field is the
  18-value enum, so every response parses and every category is valid.
- The model's guess goes along as a hint ("these rows were sent to you because it was unsure").
- `effort: "low"`: this is classification, not reasoning-heavy work.
- Server-side refusal fallbacks (`fallbacks: "default"`) are on. If the chain still refuses, the rows keep the model's answer.
- A merchant found in the directory is never overwritten by the LLM's name. On confident rows the LLM supplies the
  name only, and the model's category stands (measured: taking the LLM's category there costs 2 pts).
- One LLM lookup per merchant key, cached across requests (in-process today; Redis/Postgres in production,
  where confirmed names would also be written back into the directory).
- Defaults to `claude-opus-5-5`; any model in `PRICES` can be passed (`LEDGERLENS_LLM_MODEL`, `--model`).

### Measured: model vs LLM vs hybrid ([report](results/week3_llm_claude-haiku-5-5.md))

400 unique test descriptions (cold directory, unseen merchants), Claude Haiku 5.5, 25 txns per request:

| strategy | category acc (95% CI) | merchant exact (95% CI) | LLM calls on this sample |
|---|---|---|---|
| Ledgerlens model only | 76.5% [72.1–80.4] | 40.5% [35.8–45.4] | 0 |
| Haiku only | 91.5% [88.4–93.9] | 83.8% [79.8–87.0] | 400 txns |
| hybrid v1: LLM on low-confidence rows | 91.2% [88.1–93.6] | 52.8% [47.9–57.6] | 66 merchant keys |
| **hybrid v2: + merchants not in directory, LLM names only** (shipped) | **91.2%** [88.1–93.6] | **82.2%** [78.2–85.7] | 190 merchant keys |
| hybrid v2 variant: also take the LLM's category on those rows | 89.2% [85.8–91.9] | 82.2% [78.2–85.7] | 190 merchant keys |

0 refusals and 0 unparseable responses. Both runs together cost $0.02.
Unique descriptions over-weight hard, rare rows, which is why model-only reads 76.5% here vs ~81% on all traffic.
**This eval ran on the model from before the Week 4 city fix** (re-running it costs ~$0.02 and needs an API key).
Hybrid v1 first measured 92.0% with one LLM answer per *description*. It now uses one answer per *merchant key*
(cached), which reads 91.2%; that difference is within noise.

**What this says, plainly:**
- **An off-the-shelf small LLM beats my model by 15 points** on merchants it has never seen. Part of that is world
  knowledge: it already knows what "SWEETGRN" is, and the synthetic chains are real brands.
- **Hybrid v2 matches Haiku-only on both category and merchant name** (within the CIs). It sends the LLM only
  low-confidence rows and merchants missing from the directory, and only *once per merchant* thanks to the cache.
- **v1 left merchant names on the table** (52.8%). Unseen chains the model categorized *confidently* never
  reached the LLM, so they kept a cleaned descriptor ("Sweetgrn") as their name. Routing on "not in directory"
  fixed it: **+29 pts merchant accuracy at the same category accuracy**.
- **Don't take the LLM's category when the model is confident.** It *lowered* accuracy (91.2% → 89.2%). On confident
  rows the model's category is right ~87% of the time (cold) and ~98% (warm), so the LLM is there for names.

### Cost per 1M transactions ([cost report](results/week3_cost.md))

LLM answers are cached per merchant key (cleaned merchant + money in/out), so a merchant costs one call the
first time it's seen and nothing after. Over a year of test traffic that's **4.4 LLM calls per 100 txns
(cold) / 2.3 (warm)** for hybrid v2, even though it *routes* 96% / 35% of rows.

| | per 1M txns |
|---|---|
| Ledgerlens model only (measured throughput, assumes $0.04/vCPU-hour) | **≈ $0.004** |
| **hybrid v2 + cache, Haiku 5.5** (measured tokens: 103 in / 31 out per txn) | **≈ $0.59 warm / $1.15 cold** |
| hybrid v1 without cache, Haiku 5.5 | ≈ $4.08 / $5.92 |
| Haiku 5.5 only | ≈ $26 |
| hybrid v2 + cache / LLM-only, Opus 5.5 | ≈ $31–61 / $1,381 (est.) |

(est.) = token counts estimated before any API call. Cache rates come from 300 synthetic users; real traffic from
more users repeats merchants more, so they're conservative. Re-run with a different model:

```bash
.venv/bin/python -m ledgerlens.llm_eval                                      # dry run: plan + cost estimate
.venv/bin/python -m ledgerlens.llm_eval --yes --model claude-haiku-5-5       # ~$0.02 for 400 txns
.venv/bin/python -m ledgerlens.cost                                          # re-prices with measured tokens
```

## Week 4: dashboard + finance agent

```bash
.venv/bin/uvicorn ledgerlens.service:app --workers 1     # API on :8000 (chat needs ANTHROPIC_API_KEY)
cd web && npm install && npm run dev                      # dashboard on :3000, proxies /api -> :8000
```

**Dashboard** ([web/](web/src/app/page.tsx), Next.js 15 + Tailwind + Recharts). Upload a bank CSV or load a
demo account (one synthetic user, 12 months, deterministic so the demo video is reproducible). It shows:
- a hero that decodes a real line from the statement (`SQ *BLUE BTL COF` → Blue Bottle Coffee),
- spending by month, stacked by category,
- the transaction feed: the bank's raw text under each clean merchant, an amber "Claude" tag where the LLM
  named it, and an "unsure" tag below the confidence threshold,
- subscriptions, bills, and next month's forecast,
- a chat panel.

Uploaded statements are held **in memory only** (bounded store, oldest evicted), never written to disk or logged.
Because of that, the dataset endpoints need a single worker; scaling out would need a shared store (Redis).

**Agent** ([agent.py](ledgerlens/agent.py)). Claude with six tools over deterministic pandas analytics
([finance.py](ledgerlens/finance.py)): overview, spending by category, monthly spending, transaction search,
recurring charges, and a cash-flow forecast.
- **The model never does arithmetic.** Every number in an answer comes from a tool, and the chat panel shows
  which calculations ran.
- Strict tool schemas. Refusal fallbacks on.
- Transaction text is treated as data: a Venmo memo saying "ignore previous instructions" is not an instruction.
- "Spending" excludes transfers (card payments, P2P, brokerage) to avoid double counting.
- A hand-written tool loop rather than the SDK's beta tool runner, because the UI needs a per-call trace and the
  tests inject a scripted fake client.

**Agent eval** ([agent_eval.py](ledgerlens/agent_eval.py), 42 questions). Spend by category and month,
merchant totals and visit counts, monthly averages, income, the biggest purchase, peak months, subscriptions,
rent and the forecast. Ground truth is computed with plain pandas, **independently of the agent's tools**, so a
buggy tool can't grade itself correct. Grading: the exact figure (within max($0.50, 0.5%)), count, merchant
or month must appear in the reply.

```bash
.venv/bin/python -m ledgerlens.agent_eval                                 # dry run: writes questions, estimates cost
.venv/bin/python -m ledgerlens.agent_eval --yes --model claude-haiku-5-5  # ~$0.04; Opus 5.5 ~$1.50
```

**Results on Claude Haiku 5.5 ([report](results/week4_agent_claude-haiku-5-5.md)): 41/42 correct (97.6%).** Avg 1.2 tool calls, 3.2 s (p95 6.3 s), **$0.0008 per question**; 0 tool errors, 0 refusals; the whole run cost $0.03.

| question type | accuracy | n |
|---|---|---|
| dollar amounts | 96.6% | 29 |
| counts | 100% | 5 |
| merchant names | 100% | 5 |
| months | 100% | 3 |

**The one miss was a tool-design bug, not the model.** Asked about typical monthly income, the agent called
`monthly_spending(category="INCOME")`. Income isn't spending, so the tool returned $0 for every month with no
warning, and the agent concluded the income was miscategorized. The tool now rejects non-spending categories
with an error that names the right tool (`cashflow_forecast`, or `search_transactions` with `direction='in'`),
and a regression test covers it. The fix is unit-tested but not yet re-run against the API.

The first attempt at this eval never reached the model: the API rejected the tool schemas (an optional enum
declared as `type: ["string", "null"]`). The fake-client tests couldn't catch that, so a schema-shape test
now does.

### Bugs this week's end-to-end check caught

- **A Week 1 normalizer bug.** 9 of 71 gazetteer cities (New York, Dallas, Minneapolis...) sat at a line break
  in the source string and kept their indentation, so they never matched. In the dashboard, a pilates studio,
  a gas station and a café in New York all resolved to "New York Times". Fixed, regression-tested, and every
  offline number in this README re-run.
- **Card autopay counted as a subscription.** A monthly transfer looked recurring; it's now excluded, since it
  pays for spending already counted.

## Week 5: deploy

One [Render blueprint](render.yaml) deploys the API and the dashboard; [CI](.github/workflows/ci.yml) runs the
tests and the web build on every push. CI needs no API key: the tests use a fake LLM client.

**Public-demo safety.** On a public server the chat endpoint spends real money and uploads would be strangers'
bank data, so the blueprint turns uploads off, limits chat to 6 requests/minute per client, and caps LLM spend at
$2/day ([guards.py](ledgerlens/guards.py)). Past the cap, enrichment falls back to model-only and chat returns a
clear 503. Keys live in Render's dashboard, never in the repo.

**The model artifact.** Training needs about 600 MB of RAM, more than a free build box has, so the trained model
ships as a GitHub release asset. It's a pickle, and loading a pickle runs code, so
[fetch_model.py](scripts/fetch_model.py) downloads it over HTTPS and refuses to install it unless its SHA-256
matches [model.sha256](model.sha256), which is committed with the code. With no URL set, it trains instead.

Steps:

1. Push the repo to GitHub, then publish the model and pin its checksum:
   `scripts/release_model.sh model-v1` (needs `gh auth login`; commit the updated `model.sha256`).
2. In Render: **New → Blueprint**, pick the repo. When prompted, paste `ANTHROPIC_API_KEY` and the
   `LEDGERLENS_MODEL_URL` that the release script printed.
3. Open the `ledgerlens-web` URL and click **Try it on a demo account**.

Free Render services sleep when idle; the first request after a nap takes about 30–60 s.

## Unseen bank formats

Synthetic data's obvious weakness is "it only works on the formats you generated." So the test holds out entire
*banks*: for each of the 7 card-statement formats and 4 ACH formats, [format_eval.py](ledgerlens/format_eval.py)
retrains the full model with every user of that format removed, then scores those users' transactions
([full report](results/week5_formats.md)).

The catch: the normalizer has a hand-written rule for each format (strip `PURCHASE AUTHORIZED ON …`), so a held-out
format isn't really unseen. Each format is therefore scored twice: with its rule on (**the model never saw it**) and
with its rule switched off (**nobody saw it**, as if a new bank showed up tomorrow).

| card formats, row-weighted | category (unseen merchants) | merchant name (known merchants) |
|---|---|---|
| control: users whose format the model trained on | 78.2% | 82.5% |
| **model never saw the format** | **78.6%** | **82.5%** |
| nobody saw it: format rule switched off | 75.7% | 28.9% |
| **nobody saw it + boilerplate learned from the statement** | **77.1%** | **80.9%** |

- **The ML model generalizes to new statement formats:** no measurable loss.
- **The hand-written cleanup was the fragile part.** Without its rule, the bank's wrapper text stays in the merchant
  name, and merchant accuracy collapses to 29%.
- **So I replaced the need for the rule.** [boilerplate.py](ledgerlens/boilerplate.py) learns a bank's wrapper words
  from the statement itself, with no labels and no format knowledge. A bank's boilerplate appears across many
  *different* merchants (`PURCHASE`, `CHECKCARD`, `POS DEBIT`, the home city, the account holder's name), while a real
  merchant word repeats only within its own few store variants, however often you shop there. That recovers merchant
  accuracy from 29% to 81%, against 82% with hand-written rules.

Caveats: single seed; the control is a different set of users than the held-out ones, so per-format gaps of a few
points are population noise (`card:plain` even scores higher unseen). ACH formats are near 99% on category; the
two ACH formats without standard markers (`wf`, `plain`) lose 6–8 points on small slices (~200 rows each).
The boilerplate learner is measured here but not yet switched on in the production pipeline.

This test also caught two bugs: the evaluation briefly counted Starbucks *purchases* as payroll rows (Starbucks is
also an employer in the data), and the normalizer's URL rule ran before its date rule (`11/01` → `11`). Both are fixed
and tested; the date fix changes 0 of 98,098 existing outputs.

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
- [x] **Week 3**: FastAPI serving, locust load test, confidence routing, LLM fallback (measured on Haiku 5.5), cost per 1M transactions
  - [x] route "merchant not in directory" rows to the LLM too, with a per-merchant cache (merchant accuracy 52.8% → 82.2%)
- [x] **Week 4**: Next.js dashboard + Claude tool-calling finance agent with eval set
- [x] **Week 5**: Render blueprint + CI, public-demo guards, verified model download, [writeup](docs/WRITEUP.md), [demo script](docs/DEMO.md)
- [x] Unseen-bank-format evaluation + label-free boilerplate learner ([results](results/week5_formats.md))
- [ ] Switch the boilerplate learner on in the production pipeline (measure it doesn't hurt known formats first)
- [ ] Hand-labeled real-data eval (`scripts/label.py`) and re-run of the LLM fallback eval on the fixed model
