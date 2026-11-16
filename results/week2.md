## Category model: test split (22,906 txns, 153 unseen merchants)

Mean ± std over 3 seeds (0, 1, 2). Settings: {'hide_frac': 0.8, 'history_dropout': 0.2}.

| model | accuracy | macro-F1 | brand-clean acc | brand-clean macro-F1 | chain | local | template | accuracy (warm directory) |
|---|---|---|---|---|---|---|---|---|
| Rules baseline (week 1) | 53.5% ± 0.0% | 0.583 ± 0.000 | 42.8% ± 0.0% | 0.387 ± 0.000 | 50.5% | 44.2% | 100.0% | – |
| TF-IDF + logistic regression (text only) | 65.4% ± 0.0% | 0.552 ± 0.000 | 73.7% ± 0.0% | 0.535 ± 0.000 | 61.8% | 70.0% | 86.1% | – |
| LightGBM: text probs + format + amount | 67.4% ± 0.2% | 0.578 ± 0.007 | 74.4% ± 0.1% | 0.542 ± 0.009 | 62.6% | 77.9% | 85.8% | – |
| + user history | 77.3% ± 2.6% | 0.647 ± 0.032 | 81.2% ± 0.7% | 0.584 ± 0.013 | 72.7% | 90.1% | 88.7% | – |
| + rules prediction as a feature | 85.8% ± 0.2% | 0.716 ± 0.007 | 83.2% ± 0.2% | 0.589 ± 0.006 | 83.1% | 89.8% | 99.7% | – |
| + merchant resolver (full model) | 84.4% ± 0.2% | 0.699 ± 0.006 | 81.6% ± 0.4% | 0.574 ± 0.007 | 81.6% | 88.3% | 99.9% | 87.5% ± 0.6% |
| full model, first-seen txn (no user history) | 81.5% ± 0.2% | 0.683 ± 0.004 | 78.6% ± 0.4% | 0.558 ± 0.009 | 79.9% | 79.2% | 99.9% | – |

Cold (all columns except the last) = test merchants hidden from the merchant directory too. Warm = full directory (production setting). Chain/local/template columns are means.
Brand-clean = excludes the 9 test merchants that the hand-written rules name by brand (Afterpay, Airbnb, Coinbase, FasTrak, Mint Mobile, Netflix, Southern California Edison, Ticketmaster, Xbox Game Pass), leaving 17,552 txns. This is the leak-free generalization number.

## Merchant resolution (exact canonical-name match)

Abstain threshold tuned on val: **0.625**

| method | overall | chain | local | template |
|---|---|---|---|---|
| normalizer_only (week 1) | 46.8% | 50.1% | 49.8% | 11.6% |
| resolver_cold | 56.7% | 59.9% | 64.7% | 11.6% |
| resolver_warm | 87.5% | 94.3% | 64.7% | 79.2% |

## Full model: top confusions (cold, seed 0)

| true | predicted | txns | merchants |
|---|---|---|---|
| HOUSING | SUBSCRIPTIONS | 1,176 | Lemonade Insurance |
| ENTERTAINMENT | DINING | 470 | Fandango, Dave & Buster's, Ticketmaster |
| EDUCATION | SUBSCRIPTIONS | 300 | LeetCode |
| LOANS | TRANSFER | 274 | Afterpay |
| ENTERTAINMENT | SHOPPING | 218 | Dave & Buster's, Fandango, Maple Fox |
| SHOPPING | DINING | 140 | Sephora, Rustic Basil, Sunrise Bamboo Hardware |
| ENTERTAINMENT | GROCERIES | 117 | Ticketmaster, Bowlero, Maple Fox |
| COFFEE | DINING | 116 | Main Garden, Sofia's, Sunrise Tiger |
| GAS | TRANSPORT | 95 | Tesla Supercharger |
| GROCERIES | DINING | 86 | Wild Lotus, Village Lantern Mercado, Mission Garden Supermarket |

## Top features (LightGBM gain share, seed 0)

| feature | share |
|---|---|
| rule_cat | 27.0% |
| p_TRANSFER | 8.4% |
| p_DINING | 8.3% |
| amount | 7.1% |
| p_GROCERIES | 6.7% |
| log_abs_amount | 5.7% |
| p_SUBSCRIPTIONS | 5.7% |
| p_FEES | 4.6% |
| p_SHOPPING | 4.3% |
| p_TRANSPORT | 4.3% |
| p_HEALTH | 4.2% |
| p_EDUCATION | 3.7% |
| entry | 3.6% |
| hist_amount_cv | 3.2% |
| p_FITNESS | 3.1% |
