# Model card: Ledgerlens transaction enricher (model-v3)

**Artifact:** [`enricher.joblib`, release model-v3](https://github.com/Lukietoo/ledgerlens/releases/tag/model-v3),
pinned by SHA-256 in [`model.sha256`](../model.sha256). The container build and `scripts/fetch_model.py` refuse any
other file, because a joblib pickle runs code when loaded.

## What it does

Takes one raw bank-statement line (description, signed amount, date, and optionally the same user's earlier
transactions) and returns a merchant name, whether that merchant is in the directory, one of 18 spending
categories, a confidence, a recurring flag, and the payment processor.

Real outputs of the shipped model, without the LLM:

```
"PURCHASE AUTHORIZED ON 03/14 SQ *BLUE BOTTLE COF OAKLAND CA S38207461 CARD 4821"
    -> Blue Bottle Coffee (in directory) · COFFEE · confidence 0.95
"SQ *BLUE BTL COF OAKLAND CA"
    -> "Blue Btl Cof" (not in directory) · COFFEE · confidence 0.97  -> routed to Claude for the name
```

The second alias is deliberately left out of the directory so evaluation measures unseen spellings. The category
is still right; the name isn't in the directory, so production asks Claude for it.

## Architecture

- **Normalizer:** rules strip processor prefixes, store numbers, phone numbers, locations and bank wrappers.
- **Merchant resolver:** character n-gram TF-IDF nearest neighbour against a directory of about 1,000 merchants,
  with a similarity score and an abstain threshold.
- **Category model:** a TF-IDF + logistic-regression text model stacked under LightGBM, with format, amount,
  user-history, rules-prediction and resolver features. Stacking is out-of-fold, grouped by merchant, with 80% of
  each fold's merchants hidden from the directory, so the model learns how far to trust a match.
- **In production:** rows the model is unsure about, or whose merchant isn't in the directory, go to Claude Haiku
  5.5, one call per merchant, cached ([routing](../ledgerlens/routing.py)).

## Training data

| source | rows | what it is |
|---|---|---|
| my generator ([synth.py](../ledgerlens/synth.py)) | 192K train / 47K val | 300 simulated users × 12 months, about 1,000 merchants, with a documented noise model for 8+ bank formats |
| [DoDataThings v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) (MIT), train/val merchants only | about 46K | a third party's generator: 500+ real merchant names in 8 US statement styles. Pinned by revision and SHA-256 |

**No real customer data was used.** Real bank descriptors with labels aren't public. Both sources are synthetic,
built by different people with different generators, which is the closest available stand-in for real-world
variety. Third-party rows have no amounts, so training gives them random amounts that carry no label signal (a
constant stand-in let the model tell the two sources apart).

## Evaluation

All splits are by **merchant**: no test merchant appears in any training file, and tests enforce this.

| setting | accuracy | macro-F1 |
|---|---|---|
| my test split (22,906 txns, 153 unseen merchants), 3 seeds | 84.4% ± 0.2 | 0.699 |
| same, leak-free slice (excludes merchants the rules name by brand) | 81.6% | – |
| same, full merchant directory (production setting) | 87.5% ± 0.6 | – |
| same, a user's first transaction at a merchant (no history) | 81.5% ± 0.2 | 0.683 |
| **third-party benchmark, withheld merchants (14,206 rows)** | **75.3%** | **0.735** |
| third-party, merchants not in the directory (10,277 rows) | 71.3% | 0.685 |
| **third-party, model + Claude Haiku 5.5 fallback (1,988-row sample)** | **90.3%** | **0.910** |
| rules baseline: my test split / third-party | 53.5% / 33.9% | 0.583 / 0.421 |

Merchant name exact match: 87.5% overall, 94.3% on known chains (rules-only normalizer: 46.8%).

**Calibration** (third-party, withheld merchants): expected calibration error 0.053. Rows at 0.9+ confidence (61% of
rows) are right 94.8% of the time against a mean confidence of 0.98, so the model is mildly overconfident at the
top. The routing threshold was set from measured accuracy-by-confidence, not from raw confidence.

## Known failure modes

From reading 60 random misses on the third-party benchmark ([full analysis](../results/external_errors.md)):

- **Brand names with no clue in the text (about 43% of misses):** `MITSUWA`, `CEREBRAL`, `WYZANT`. This is what the
  Claude fallback exists for.
- **Both labels defensible (about 22%):** airport parking (Travel or Transportation), `ESPN+` (Entertainment or
  Subscription). No model can score 100% against one labeling.
- **Generic words misread (about 18%):** `SQ *FARM STAND` read as Shopping, not Groceries.
- **Name collisions:** `O'REILLY MEDIA` read as O'Reilly Auto Parts.
- **A directory match outvoted:** `RITE AID` resolves correctly to a HEALTH merchant, but the model outputs
  ENTERTAINMENT. Trusting the directory at similarity ≥ 0.9 was measured at +0.5 points and not shipped.

## Intended use and limits

- **Intended:** categorizing personal transactions for budgeting, subscription tracking, and as the data layer for
  the finance agent. It's a portfolio project, not a production system.
- **Not for:** credit, lending, fraud or any decision about a person. Categories are spending labels, not risk
  signals.
- **Untested on real statements.** Every number above is on synthetic data from two independent generators. The
  labeling tool for a real-data eval is built ([scripts/label.py](../scripts/label.py)); real data stays on the
  owner's machine and is never committed.
- **US-centric:** formats, merchants and categories are American.
- **Privacy:** the fallback sends low-confidence descriptions to the Anthropic API. The public demo serves a
  synthetic user only; uploads are off.
