## Independent benchmark, held-out merchants (test split) (models/enricher.joblib)

[DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) (MIT): built by someone else's generator from 500+ real merchant names in 8 US statement styles. With --heldout, only merchants withheld from every training file (merchant-level split) are scored. The category mapping was written before any score was seen. No amounts or dates in this data, and no history: every row is a first-seen merchant.

| | rows | accuracy | macro-F1 |
|---|---|---|---|
| rules baseline | 14,206 | 33.9% | 0.421 |
| **Ledgerlens** | 14,206 | **75.3%** | **0.735** |
| Ledgerlens, merchant not in our directory | 10,277 | 71.3% | 0.685 |

28% of rows matched a merchant in our directory. Excluded (no matching category in our taxonomy): Insurance (4,000), Personal Care (4,000).

Hardest classes: Shopping 44%, Travel 52%, Healthcare 54%, Education 57%, Entertainment 66%.

Most common mistakes: Groceries → Shopping (215); Healthcare → Shopping (192); Groceries → Restaurants (166); Education → Shopping (151); Healthcare → Entertainment (137).

If low-confidence rows went to a fallback, what the model keeps: ≥0.5: 81% of rows at 86.9%; ≥0.6: 77% of rows at 89.5%; ≥0.7: 73% of rows at 90.8%; ≥0.8: 69% of rows at 92.6%; ≥0.9: 61% of rows at 94.8%.

Calibration: expected calibration error **0.053** (0 = stated confidence matches accuracy exactly).

| confidence | rows | mean confidence | accuracy |
|---|---|---|---|
| 0.1–0.2 | 183 | 0.17 | 9.8% |
| 0.2–0.3 | 902 | 0.26 | 16.5% |
| 0.3–0.4 | 847 | 0.35 | 26.6% |
| 0.4–0.5 | 832 | 0.45 | 44.8% |
| 0.5–0.6 | 553 | 0.55 | 34.7% |
| 0.6–0.7 | 528 | 0.64 | 64.2% |
| 0.7–0.8 | 518 | 0.76 | 55.8% |
| 0.8–0.9 | 1,176 | 0.86 | 77.0% |
| 0.9–1.0 | 8,667 | 0.98 | 94.8% |

Stand-in amount ×0.25 / ×4 on a 6K sample: 75.6% / 75.3%.
