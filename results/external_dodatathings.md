## Independent benchmark, held-out merchants (model-v3: trained on our synthetic data + their train/val merchants)

[DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) (MIT): built by someone else's generator from 500+ real merchant names in 8 US statement styles. Scored on merchants withheld from every training file (merchant-level split); the category mapping was written before any score was seen. No amounts or dates in this data, and no history: every row is a first-seen merchant.

| | rows | accuracy | macro-F1 |
|---|---|---|---|
| rules baseline | 14,206 | 33.9% | 0.421 |
| **Ledgerlens** | 14,206 | **75.3%** | **0.735** |
| Ledgerlens, merchant not in our directory | 10,277 | 71.3% | 0.685 |

28% of rows matched a merchant in our directory. Excluded (no matching category in our taxonomy): Insurance (4,000), Personal Care (4,000).

Hardest classes: Shopping 44%, Travel 52%, Healthcare 54%, Education 57%, Entertainment 66%.

Most common mistakes: Groceries → Shopping (215); Healthcare → Shopping (192); Groceries → Restaurants (166); Education → Shopping (151); Healthcare → Entertainment (137).

If low-confidence rows went to a fallback, what the model keeps: ≥0.5: 81% of rows at 86.9%; ≥0.6: 77% of rows at 89.5%; ≥0.7: 73% of rows at 90.8%; ≥0.8: 69% of rows at 92.6%; ≥0.9: 61% of rows at 94.8%.

Stand-in amount ×0.25 / ×4 on a 6K sample: 75.6% / 75.3%.
