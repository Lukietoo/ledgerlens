## Category model: test split (22,906 txns, 153 unseen merchants)

Mean ± std over 3 seeds (0, 1, 2). Settings: {'hide_frac': 0.8, 'history_dropout': 0.2}.

| model | accuracy | macro-F1 | brand-clean acc | brand-clean macro-F1 | chain | local | template | accuracy (warm directory) |
|---|---|---|---|---|---|---|---|---|
| Rules baseline (week 1) | 53.5% ± 0.0% | 0.583 ± 0.000 | 42.8% ± 0.0% | 0.387 ± 0.000 | 50.5% | 44.2% | 100.0% | – |
| TF-IDF + logistic regression (text only) | 45.1% ± 0.0% | 0.362 ± 0.000 | 50.4% ± 0.0% | 0.324 ± 0.000 | 39.2% | 71.3% | 37.2% | – |
| LightGBM: text probs + format + amount | 52.3% ± 0.9% | 0.387 ± 0.010 | 58.8% ± 1.1% | 0.379 ± 0.008 | 46.9% | 80.7% | 35.6% | – |
| + user history | 60.9% ± 1.3% | 0.456 ± 0.019 | 67.3% ± 1.2% | 0.420 ± 0.019 | 55.7% | 87.9% | 46.0% | – |
| + rules prediction as a feature | 80.8% ± 0.0% | 0.673 ± 0.001 | 75.1% ± 0.1% | 0.490 ± 0.002 | 76.2% | 89.9% | 99.9% | – |
| + merchant resolver (full model) | 81.5% ± 0.1% | 0.682 ± 0.002 | 76.0% ± 0.1% | 0.502 ± 0.002 | 77.2% | 89.8% | 99.3% | 86.7% ± 3.0% |
| full model, first-seen txn (no user history) | 73.4% ± 0.6% | 0.637 ± 0.005 | 66.4% ± 0.4% | 0.460 ± 0.005 | 68.9% | 79.1% | 99.6% | – |

Cold (all columns except the last) = test merchants hidden from the merchant directory too. Warm = full directory (production setting). Chain/local/template columns are means.
Brand-clean = excludes the 9 test merchants that the hand-written rules name by brand (Afterpay, Airbnb, Coinbase, FasTrak, Mint Mobile, Netflix, Southern California Edison, Ticketmaster, Xbox Game Pass), leaving 17,552 txns. This is the leak-free generalization number.

## Merchant resolution (exact canonical-name match)

Abstain threshold tuned on val: **0.608**

| method | overall | chain | local | template |
|---|---|---|---|---|
| normalizer_only (week 1) | 45.5% | 49.2% | 46.0% | 11.6% |
| resolver_cold | 42.9% | 47.1% | 40.8% | 11.6% |
| resolver_warm | 82.8% | 94.1% | 40.8% | 79.2% |

## Full model: top confusions (cold, seed 0)

| true | predicted | txns | merchants |
|---|---|---|---|
| HOUSING | SUBSCRIPTIONS | 1,176 | Lemonade Insurance |
| ENTERTAINMENT | SHOPPING | 412 | Fandango, Dave & Buster's, Ticketmaster |
| EDUCATION | SUBSCRIPTIONS | 300 | LeetCode |
| GAS | SHOPPING | 287 | Marathon, Tesla Supercharger |
| ENTERTAINMENT | DINING | 283 | Dave & Buster's, Fandango, Bowlero |
| GAS | DINING | 237 | Tesla Supercharger, Marathon |
| GROCERIES | SHOPPING | 234 | Stop & Shop, Mission Lotus, Mei's |
| GAS | GROCERIES | 212 | Tesla Supercharger |
| GAS | COFFEE | 158 | Tesla Supercharger |
| EDUCATION | SHOPPING | 134 | College Board |

## Top features (LightGBM gain share, seed 0)

| feature | share |
|---|---|
| rule_cat | 40.8% |
| amount | 9.7% |
| log_abs_amount | 7.0% |
| entry | 6.7% |
| hist_amount_cv | 6.4% |
| res_top1_cat | 5.0% |
| hist_n | 4.0% |
| cents | 3.0% |
| p_FITNESS | 2.9% |
| p_SUBSCRIPTIONS | 2.7% |
| r_GROCERIES | 2.6% |
| p_GAS | 2.5% |
| p_TRAVEL | 2.4% |
| p_DINING | 2.2% |
| hist_rule_recurring | 2.1% |
