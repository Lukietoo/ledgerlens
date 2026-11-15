### Rules baseline: synthetic (test split, unseen merchants)

| metric | value |
|---|---|
| transactions evaluated | 22,906 |
| category accuracy | 53.5% |
| category macro-F1 | 0.583 |
| rule coverage (non-UNKNOWN) | 60.8% |
| merchant exact match (normalizer only) | 45.5% |
| recurring P / R / F1 | 0.977 / 0.888 / 0.930 |
| accuracy on chain merchants (n=16,677) | 50.5% |
| accuracy on local merchants (n=4,293) | 44.2% |
| accuracy on template merchants (n=1,936) | 100.0% |

| category | P | R | F1 | support |
|---|---|---|---|---|
| GROCERIES | 1.00 | 0.47 | 0.64 | 1,542 |
| DINING | 0.71 | 0.26 | 0.38 | 3,748 |
| COFFEE | 1.00 | 0.74 | 0.85 | 3,617 |
| TRANSPORT | 1.00 | 1.00 | 1.00 | 2,225 |
| GAS | 0.00 | 0.00 | 0.00 | 894 |
| TRAVEL | 1.00 | 0.79 | 0.89 | 442 |
| SHOPPING | 0.11 | 0.15 | 0.13 | 763 |
| SUBSCRIPTIONS | 1.00 | 0.35 | 0.52 | 2,412 |
| UTILITIES | 1.00 | 0.58 | 0.74 | 924 |
| ENTERTAINMENT | 1.00 | 0.58 | 0.73 | 1,800 |
| HEALTH | 1.00 | 0.42 | 0.59 | 89 |
| FITNESS | 1.00 | 0.54 | 0.70 | 592 |
| EDUCATION | 0.32 | 0.34 | 0.33 | 453 |
| HOUSING | 0.00 | 0.00 | 0.00 | 1,176 |
| LOANS | 1.00 | 1.00 | 1.00 | 293 |
| TRANSFER | 1.00 | 1.00 | 1.00 | 1,369 |
| INCOME | 1.00 | 1.00 | 1.00 | 567 |

Top confusions (true → predicted):

- DINING → UNKNOWN: 2,291
- SUBSCRIPTIONS → UNKNOWN: 1,524
- HOUSING → UNKNOWN: 1,164
- GAS → UNKNOWN: 789
- COFFEE → UNKNOWN: 640
- SHOPPING → UNKNOWN: 617
- ENTERTAINMENT → UNKNOWN: 563
- GROCERIES → SHOPPING: 457
- UTILITIES → UNKNOWN: 384
- DINING → SHOPPING: 360
