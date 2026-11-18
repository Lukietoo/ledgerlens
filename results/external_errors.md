## Why the model misses on the independent benchmark (error analysis)

Held-out merchants of the third-party benchmark, model-v3: 3,503 misses out of 14,206 rows (75.3%
accuracy). I read a random sample of 60 misses (`random_state=1`) and assigned each one cause.
With n = 60, each share below is about ±12 points (95%), so the ranking is reliable and the exact
percentages aren't.

| cause | misses | share | examples (descriptor: their label → ours) | what fixes it |
|---|---|---|---|---|
| **needs world knowledge**: a brand name with no clue in the text | 26 | 43% | `MITSUWA 7650`: Groceries → Shopping; `CEREBRAL.COM`: Healthcare → Shopping; `ROYAL CARIBBEAN LINE`: Travel → Restaurants; `WYZANT`: Education → Shopping | the LLM fallback (the hybrid scores 90.3% on this benchmark), or a bigger merchant directory |
| **both labels defensible**: the benchmark's label is one reasonable answer of two | 13 | 22% | `AIRPORT PARKING INC.`: Travel → Transportation; `ESPN+; SUBSCRIPTION`: Entertainment → Subscription; `DAVE & BUSTER'S`: Entertainment → Restaurants; `CASH APP CASHOUT`: Income → Transfer | nothing; this caps any model's measured accuracy below 100% |
| **generic words misread** | 11 | 18% | `SQ *FARM STAND`: Groceries → Shopping; `PYPL*DRIVING RANGE`: Entertainment → Restaurants | more varied training vocabulary |
| **name collision** | 4 | 7% | `O'REILLY MEDIA 3313`: Education → Transportation (read as O'Reilly Auto Parts) | the resolver's margin feature, or the LLM |
| **directory match ignored**: the resolver found the merchant, but the model outvoted its category | 3 | 5% | `RITE AID`: Healthcare → Entertainment (directory: HEALTH) | see below |
| **statement template misled it** | 2 | 3% | `Payment to: OVERSTOCK`: Shopping → Transfer | template stripping (the boilerplate learner) |
| **abbreviation missed** | 1 | 2% | `WIRE TRF OUT REF: …`: Transfer → Restaurants | a normalizer rule for `TRF` |

**What this says.**
- The largest bucket is the one the system already routes to Claude. Merchants outside the
  directory with low confidence are exactly what the fallback exists for, and the measured hybrid
  result (73.8% → 90.3%) is consistent with that.
- About a fifth of the "errors" are label disagreements, so a perfect model wouldn't score 100%
  here. Reading the misses is how you find that out; a confusion matrix alone can't.

**A fix I measured and didn't ship.** On rows where the resolver is very sure of the merchant
(similarity ≥ 0.9, 12% of rows), the directory's category is right 90.5% of the time and the
model's 87.0%. Overriding the model there lifts overall accuracy only from 75.3% to 75.8%. That's
real but small, and it would add a special case to both the pandas path and the fast path (which
are tested to give identical outputs), so it isn't worth it yet. The cause is known: the stacked model trains with 80% of
the directory hidden (so it learns to handle new merchants) and therefore under-trusts a match.

**Calibration.** Expected calibration error is 0.053. The 0.9–1.0 band (61% of rows) is right
94.8% of the time against a mean stated confidence of 0.98, so the model is mildly overconfident
at the top. That's why the routing threshold was picked from measured accuracy-by-confidence
([external_dodatathings.md](external_dodatathings.md)), not from the raw confidence value.
