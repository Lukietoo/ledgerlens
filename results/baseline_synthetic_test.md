### Rules baseline: synthetic (test split, unseen merchants)

| metric | value |
|---|---|
| transactions evaluated | 20,874 |
| category accuracy | 56.8% |
| category macro-F1 | 0.601 |
| rule coverage (non-UNKNOWN) | 61.3% |
| merchant exact match (normalizer only) | 44.2% |
| recurring P / R / F1 | 0.973 / 0.890 / 0.930 |
| accuracy on chain merchants (n=14,636) | 52.9% |
| accuracy on local merchants (n=4,118) | 48.4% |
| accuracy on template merchants (n=2,120) | 100.0% |

| category | P | R | F1 | support |
|---|---|---|---|---|
| GROCERIES | 1.00 | 0.66 | 0.79 | 1,130 |
| DINING | 0.73 | 0.25 | 0.38 | 3,375 |
| COFFEE | 1.00 | 0.79 | 0.88 | 3,605 |
| TRANSPORT | 1.00 | 1.00 | 1.00 | 2,146 |
| GAS | 0.00 | 0.00 | 0.00 | 714 |
| TRAVEL | 1.00 | 0.72 | 0.84 | 390 |
| SHOPPING | 0.16 | 0.11 | 0.13 | 693 |
| SUBSCRIPTIONS | 1.00 | 0.31 | 0.47 | 1,944 |
| UTILITIES | 1.00 | 0.68 | 0.81 | 828 |
| ENTERTAINMENT | 1.00 | 0.56 | 0.72 | 1,588 |
| HEALTH | 1.00 | 0.49 | 0.66 | 112 |
| FITNESS | 1.00 | 0.67 | 0.80 | 495 |
| EDUCATION | 0.35 | 0.33 | 0.34 | 359 |
| HOUSING | 0.00 | 0.00 | 0.00 | 1,128 |
| LOANS | 1.00 | 1.00 | 1.00 | 247 |
| TRANSFER | 1.00 | 1.00 | 1.00 | 1,387 |
| INCOME | 1.00 | 1.00 | 1.00 | 733 |

Top confusions (true → predicted):

- DINING → UNKNOWN: 2,025
- SUBSCRIPTIONS → UNKNOWN: 1,332
- HOUSING → UNKNOWN: 1,128
- GAS → UNKNOWN: 678
- ENTERTAINMENT → UNKNOWN: 626
- SHOPPING → UNKNOWN: 580
- COFFEE → UNKNOWN: 561
- GROCERIES → UNKNOWN: 375
- DINING → SHOPPING: 349
- UTILITIES → UNKNOWN: 264
