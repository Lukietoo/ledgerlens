# Ledgerlens: decoding bank statements with a small model, a cheap LLM, and honest evals

Every transaction in a bank feed arrives as something like

```
PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BTL COF OAKLAND CA S382074619283746 CARD 4821
```

Before a fintech can build budgeting, underwriting or fraud features, it has to turn that into
*Blue Bottle Coffee · Coffee · paid through Square · not recurring*. Ledgerlens is that layer,
built in five weeks: a classic-ML enrichment pipeline, a cheap LLM fallback for what it can't
handle, an API, a dashboard, and a Claude agent that answers questions about your money.

This writeup is mostly about measurement: which numbers are trustworthy, and the bugs the
measurements caught.

## The data problem

Real bank data is private, so I built a generator that reproduces how descriptors get mangled:
- 22-character merchant fields (an ISO 8583 artifact) and 13-character city fields,
- Square/Toast/Clover processor prefixes,
- store numbers that stay stable for each user,
- each bank's own wrapper format,
- NACHA ACH layouts,
- phone numbers or URLs in place of cities for online merchants.

300 simulated users × 12 months gives about 216K labeled transactions across about 1,000 merchants.

The split is **by merchant**, stratified by category. Every number below is on merchants the
model never saw in training. A random split would have reported much higher numbers that
measure memorization.

## The pipeline

1. **Normalizer:** deterministic cleanup of bank wrappers, reference numbers, processor prefixes and
   locations.
2. **Merchant resolver:** character n-gram TF-IDF against a merchant directory that deliberately
   holds out all but one alias per chain, so `BLUE BTL COF` and `SWEETGRN` must be fuzzy-matched. It
   abstains below a confidence threshold rather than guessing.
3. **Category model:** TF-IDF + logistic regression probabilities, descriptor-format features,
   amount features, per-user history, the rules baseline's guess, and resolver features, stacked
   **out-of-fold by merchant** into LightGBM.
4. **LLM fallback:** low-confidence rows and merchants missing from the directory go to Claude
   Haiku, batched, with a JSON-schema-constrained output, and **cached per merchant**.

## Results (unseen merchants, mean of 3 seeds)

| | rules baseline | Ledgerlens |
|---|---|---|
| category accuracy | 53.5% | **84.4%** |
| macro-F1 | 0.583 | **0.699** |
| accuracy, leak-free slice | 42.8% | **81.6%** |
| independent benchmark (a third party's data, merchants never seen) | 33.9% | **75.3%** |
| same benchmark, model + Claude Haiku 5.5 fallback (1,988-row sample) | – | **90.3%** (73.8% model-only on that sample) |
| bank formats never seen in training | – | **78.6%** (vs 78.2% seen) |
| merchant exact match (overall / known chains) | 46.8% / 50.1% | **87.5% / 94.3%** |

With the hybrid LLM fallback (Claude Haiku 5.5, 400 unique hard transactions):

| strategy | category | merchant | cost per 1M txns |
|---|---|---|---|
| model only | 76.5% | 40.5% | ≈ $0.004 |
| Haiku on everything | 91.5% | 83.8% | ≈ $26 |
| **hybrid + per-merchant cache** | **91.2%** | **82.2%** | **≈ $0.59–1.15** |

The hybrid matches the LLM-only pipeline within the confidence intervals at about 1/25th of the LLM
cost. The model answers the majority of traffic in about 13 ms, and the cache means each new
merchant costs one LLM call ever. The API serves **11,500 transactions/s at 38 ms p95** (batches
of 100, 4 workers, an M1 laptop, locust).

The finance agent (Claude with six deterministic analytics tools) scored **41/42 (97.6%)** on an
auto-graded eval at **$0.0008 per question**.

## What the measurements caught

- **My own data was flattering me.** On a third party's dataset (68K rows, someone else's generator),
  the model trained only on my data scored 40%, no better than hand-written rules. It had learned
  my generator's vocabulary, not banking's. Training on both sources, with whole merchants held out,
  took it to 75% on merchants neither source had shown it, and to 84% on my own test set (from 80.5%).
  The retrain also found a shortcut: a constant stand-in amount let the model tell the two sources
  apart, until the training amounts were randomized.

**My own rules leaked into the test set.** Feeding the rules baseline's guess to LightGBM was the
biggest single jump (+20 points). But I wrote those rules while looking at the merchant list,
and keywords like `NETFLIX` and `AIRBNB` named 9 test merchants (23% of test traffic). I added a
check that flags rule matches on brand words inside the merchant name, and report a leak-free
slice. It's the number I'd quote, and the model still gains 33 points there.

**The model learned a bug in my generator.** A smoke test called a $15.49 Netflix charge DINING.
The generator priced every subscription at $x.99, so the model had learned ".99 means
subscription". Synthetic data needs its own QA.

**Train/serve skew.** A "how many users share this merchant" feature looked great offline and was
always about 1 for a single user's upload. I removed it. History dropout (blanking user history
on 20% of training rows) keeps first-time transactions at 73% instead of collapsing.

**An off-the-shelf small LLM beat my model by 15 points.** That's partly world knowledge: Haiku
already knows what "SWEETGRN" is, and the synthetic chains are real brands. The model's real job
turned out to be the cheap, fast path for the easy majority, and *knowing when it's unsure*.

**The first hybrid left merchant names on the table** (52.8%). Unseen chains that the model
categorized *confidently* never reached the LLM. Routing on "merchant not in directory" fixed it
(+29 points). Taking the LLM's *category* on those confident rows made accuracy worse (−2 points),
so the LLM is there for names.

**A whitespace bug from week 1 surfaced in week 4.** Nine cities in the normalizer's list,
including New York and Dallas, carried indentation from a multi-line string and never matched.
The dashboard showed a pilates studio, a gas station and a café all resolving to "New York Times".
I fixed it, added a regression test, and re-ran every number. One earlier claim, that the full
model beat its ablation in the cold-start setting, stopped being true, and I removed it.

**The fragile part wasn't the model.** To answer "it only works on formats you generated", I held out
entire bank statement formats from training. The model didn't notice (78.6% vs 78.2% in-format), but switching off
the normalizer's hand-written rule for the held-out format dropped merchant-name accuracy to 29%: the bank's wrapper
text stayed in the name. A label-free learner that finds each statement's boilerplate (words that appear across many
*different* merchants) brought it back to 86%. Turning it on in production took two more tries. Run on the raw text,
it removed the words the hand-written rules anchor on, so a dataset-wide check found 15% of rows gaining words. Fed
to the category model, it cost 1–3.5 points of category accuracy. Shipped: it runs on the normalizer's output and
only names merchants, with category metrics unchanged (0 of 49 moved) and merchant names up from 83.5% to 87.5%.

**Users can fix what's left, and a fix goes a long way.** Correcting one transaction fixes every transaction from
that merchant, about 6 on average, and a review list puts the most uncertain, highest-traffic merchants first. In
simulation, 3 fixes per user take merchants the model never saw from 81% to 94%. If half the users make one review
action each, the other half go from 80% to 91% with no effort at all. That last number needs care: the synthetic
population shares merchants densely, and real community fixes would need several users to agree first.

**The agent's one miss was a tool bug.** Asked for monthly income, it called a spending-only tool,
got $0 for every month with no warning, and decided the income was miscategorized. The tool now
rejects that call with an error that names the right tool.

## Engineering choices

- **The agent never does arithmetic.** Every number comes from a deterministic pandas function, and
  the dashboard shows which calculations ran.
- **Statement text is untrusted input.** A Venmo memo can say "ignore previous instructions", so
  the agent's prompt treats tool output as data.
- **Statements stay in memory only**: a bounded store, size caps, never written to disk or logged.
  The public demo disables uploads entirely.
- **A public deploy spends real money**, so chat and uploads are rate-limited per client and LLM
  spend has a daily cap.
- **The model artifact is a pickle.** Deploys download it from a GitHub release and verify a SHA-256
  pinned in the repo before loading it.
- **Paid evals are dry-run by default.** They print a cost estimate and need `--yes`. CI uses a
  fake LLM client: no key, no network, no cost.
- **The API holds no state, so it scales out.** Statements, corrections, the LLM answer cache,
  rate limits and the spend cap live in Redis, so any worker or pod can serve any request.
  Statements are stored as JSON, not pickle, so a compromised Redis can't run code in the API.
  Each worker keeps a decoded copy and checks a version number on every read, so a fix made on
  one replica shows up on all of them right away.
- **Corrections are lock-free.** The first version locked a statement, rewrote the whole JSON
  blob and released the lock. The load test showed writers queuing past the lock timeout, and a
  killed worker left its lock held. Now a statement is written once, and corrections are stored
  beside it as hash fields. Corrections to different merchants commute and each field is
  last-writer-wins, so a fix is one O(1) atomic `MULTI`, and readers rebuild only when the
  version moves.
- **Failure modes are chosen on purpose.** If Redis is down, rate limiting fails open (keep
  serving) and the spend cap fails closed (never spend money it can't count). Readiness does
  not depend on Redis: every replica shares it, so gating on it would pull them all out of the
  load balancer at once and turn "saved statements unavailable" into a full outage.
- **Small requests skip pandas.** A pure-Python/NumPy path gives identical outputs for up to 16
  transactions, 7.8× faster at p50 for one transaction.
- **It runs on Kubernetes, and CI proves it.** Every push builds the images, boots a 3-node kind
  cluster, and kills API pods under load. The test asserts failed requests stay under 1%, the
  Deployment heals, saved corrections survive, and Prometheus finds every replica through
  pod discovery.

## Limits

- **Every accuracy number is on synthetic data.** The independent benchmark is someone else's
  synthetic data, which rules out "it learned my generator", but not "real statements are messier
  than both generators". Public real descriptor data doesn't exist; `scripts/label.py` is built for
  hand-labeling real exports.
- The LLM-fallback numbers were measured on the model from before the city fix. Re-running costs
  about $0.02.
- Warm-directory accuracy varies ±3 points between seeds.
- Confidence routing can't catch confident mistakes. Calibration is the next lever.

## Stack

Python 3.13, scikit-learn, LightGBM, pandas, FastAPI, Redis, Prometheus, Grafana, Docker,
Kubernetes (kind in CI), locust, the Anthropic Python SDK (Claude Haiku 5.5 for the fallback and
the agent), Next.js 15, Tailwind, Recharts.
117 tests, CI on every push (including a Kubernetes chaos test on a kind cluster), deployable to Render
from one blueprint file or to Kubernetes with `kubectl apply -k deploy/`.
