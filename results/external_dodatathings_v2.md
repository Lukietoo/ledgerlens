## Independent benchmark, held-out merchants (model-v2: trained on our synthetic data only)

[DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) (MIT): built by someone else's generator from 500+ real merchant names in 8 US statement styles. Scored on merchants withheld from every training file (merchant-level split); the category mapping was written before any score was seen. No amounts or dates in this data, and no history: every row is a first-seen merchant.

| | rows | accuracy | macro-F1 |
|---|---|---|---|
| rules baseline | 14,206 | 33.9% | 0.421 |
| **Ledgerlens** | 14,206 | **39.2%** | **0.384** |
| Ledgerlens, merchant not in our directory | 10,277 | 30.6% | 0.315 |

28% of rows matched a merchant in our directory. Excluded (no matching category in our taxonomy): Insurance (4,000), Personal Care (4,000).

Hardest classes: Fees 0%, Housing 2%, Travel 14%, Transportation 18%, Subscription 23%.

Most common mistakes: Housing → Travel (690); Fees → Groceries (431); Fees → Restaurants (430); Housing → Restaurants (409); Entertainment → Shopping (329).

If low-confidence rows went to a fallback, what the model keeps: ≥0.5: 64% of rows at 56.6%; ≥0.6: 54% of rows at 63.6%; ≥0.7: 46% of rows at 69.6%; ≥0.8: 39% of rows at 75.3%; ≥0.9: 27% of rows at 82.0%.

Stand-in amount ×0.25 / ×4 on a 6K sample: 40.2% / 38.9%.
