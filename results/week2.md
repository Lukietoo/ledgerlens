## Category model: test split (22,906 txns, 153 unseen merchants)

Mean ± std over 3 seeds (0, 1, 2). Settings: {'hide_frac': 0.8, 'history_dropout': 0.2}.

| model | accuracy | macro-F1 | brand-clean acc | brand-clean macro-F1 | chain | local | template | accuracy (warm directory) |
|---|---|---|---|---|---|---|---|---|
| Rules baseline (week 1) | 53.5% ± 0.0% | 0.583 ± 0.000 | 42.8% ± 0.0% | 0.387 ± 0.000 | 50.5% | 44.2% | 100.0% | – |
| TF-IDF + logistic regression (text only) | 46.0% ± 0.0% | 0.366 ± 0.000 | 51.7% ± 0.0% | 0.325 ± 0.000 | 40.6% | 71.2% | 37.2% | – |
| LightGBM: text probs + format + amount | 51.8% ± 0.4% | 0.396 ± 0.004 | 58.2% ± 0.2% | 0.376 ± 0.007 | 46.7% | 80.4% | 31.5% | – |
| + user history | 61.3% ± 0.7% | 0.458 ± 0.004 | 65.7% ± 0.5% | 0.396 ± 0.011 | 57.6% | 88.4% | 33.2% | – |
| + rules prediction as a feature | 81.2% ± 0.3% | 0.672 ± 0.011 | 75.8% ± 0.2% | 0.496 ± 0.002 | 76.5% | 90.9% | 99.6% | – |
| + merchant resolver (full model) | 80.5% ± 0.9% | 0.659 ± 0.017 | 74.9% ± 1.2% | 0.478 ± 0.017 | 76.0% | 89.4% | 99.4% | 86.7% ± 3.2% |
| full model, first-seen txn (no user history) | 73.1% ± 0.2% | 0.630 ± 0.003 | 65.9% ± 0.1% | 0.453 ± 0.005 | 68.5% | 79.4% | 99.7% | – |

Cold (all columns except the last) = test merchants hidden from the merchant directory too. Warm = full directory (production setting). Chain/local/template columns are means.
Brand-clean = excludes the 9 test merchants that the hand-written rules name by brand (Afterpay, Airbnb, Coinbase, FasTrak, Mint Mobile, Netflix, Southern California Edison, Ticketmaster, Xbox Game Pass), leaving 17,552 txns. This is the leak-free generalization number.

## Merchant resolution (exact canonical-name match)

Abstain threshold tuned on val: **0.608**

| method | overall | chain | local | template |
|---|---|---|---|---|
| normalizer_only (week 1) | 46.8% | 50.1% | 49.8% | 11.6% |
| resolver_cold | 44.2% | 48.0% | 44.4% | 11.6% |
| resolver_warm | 83.5% | 94.1% | 44.4% | 79.2% |

## Full model: top confusions (cold, seed 0)

| true | predicted | txns | merchants |
|---|---|---|---|
| HOUSING | SUBSCRIPTIONS | 1,176 | Lemonade Insurance |
| ENTERTAINMENT | SHOPPING | 422 | Fandango, Dave & Buster's, Maple Fox |
| GAS | SHOPPING | 300 | Marathon, Tesla Supercharger |
| EDUCATION | SUBSCRIPTIONS | 300 | LeetCode |
| ENTERTAINMENT | DINING | 289 | Dave & Buster's, Fandango, Bowlero |
| GROCERIES | SHOPPING | 261 | Stop & Shop, Mei's, Wild Lotus |
| GAS | COFFEE | 214 | Tesla Supercharger |
| GAS | GROCERIES | 212 | Tesla Supercharger, Marathon |
| GAS | DINING | 168 | Tesla Supercharger, Marathon |
| DINING | SHOPPING | 146 | Wingstop, Happy Sage, Sunrise Pearl |

## Top features (LightGBM gain share, seed 0)

| feature | share |
|---|---|
| rule_cat | 40.8% |
| amount | 9.4% |
| log_abs_amount | 7.2% |
| entry | 6.7% |
| hist_amount_cv | 6.2% |
| res_top1_cat | 5.0% |
| hist_n | 3.9% |
| p_FITNESS | 3.1% |
| cents | 3.0% |
| res_sim | 2.9% |
| p_SUBSCRIPTIONS | 2.5% |
| p_GAS | 2.5% |
| r_GROCERIES | 2.5% |
| p_TRAVEL | 2.3% |
| r_SHOPPING | 2.1% |
