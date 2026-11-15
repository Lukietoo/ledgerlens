## Confidence routing: synthetic test split (22,906 txns, unseen merchants)

### cold directory: model accuracy 80.6%, ECE 0.083

| route to LLM below | traffic routed | model acc on kept | model acc on routed | hybrid ceiling |
|---|---|---|---|---|
| 0.00 | 0.0% | 80.6% | – | 80.6% |
| 0.50 | 6.9% | 83.4% | 42.3% | 84.5% |
| 0.60 | 13.6% | 84.8% | 53.8% | 86.9% |
| 0.70 | 18.2% | 87.1% | 51.0% | 89.5% |
| 0.80 | 22.8% | 88.7% | 53.1% | 91.2% |
| 0.85 | 25.7% | 89.1% | 55.9% | 91.9% |
| 0.90 | 30.2% | 90.0% | 58.9% | 93.0% |
| 0.95 | 43.2% | 93.4% | 63.7% | 96.2% |
| 0.98 | 66.0% | 99.0% | 71.0% | 99.7% |

Calibration (confidence bin → accuracy):

| bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.1–0.2 | 26 | 0.18 | 0.0% |
| 0.2–0.3 | 65 | 0.26 | 36.9% |
| 0.3–0.4 | 544 | 0.36 | 34.9% |
| 0.4–0.5 | 943 | 0.45 | 48.0% |
| 0.5–0.6 | 1,549 | 0.55 | 65.3% |
| 0.6–0.7 | 1,035 | 0.65 | 43.0% |
| 0.7–0.8 | 1,060 | 0.75 | 61.4% |
| 0.8–0.9 | 1,701 | 0.86 | 76.4% |
| 0.9–1.0 | 15,983 | 0.97 | 90.0% |

Routing policies (LLM calls assume answers are cached per merchant key):

| policy | rows routed | LLM calls per txn (cached) | rows routed only for a name | model category acc on those |
|---|---|---|---|---|
| confidence < 0.8 (week 3) | 22.8% | 1.19% | 0.0% | – |
| confidence < 0.8 OR merchant not in directory | 95.7% | 4.36% | 73.0% | 88.3% |

### warm directory: model accuracy 85.1%, ECE 0.061

| route to LLM below | traffic routed | model acc on kept | model acc on routed | hybrid ceiling |
|---|---|---|---|---|
| 0.00 | 0.0% | 85.1% | – | 85.1% |
| 0.50 | 4.0% | 87.5% | 26.3% | 88.0% |
| 0.60 | 6.9% | 89.0% | 33.0% | 89.7% |
| 0.70 | 11.3% | 91.9% | 31.7% | 92.8% |
| 0.80 | 15.7% | 92.7% | 44.3% | 93.9% |
| 0.85 | 17.4% | 93.2% | 46.6% | 94.4% |
| 0.90 | 21.4% | 94.3% | 51.4% | 95.5% |
| 0.95 | 32.8% | 97.5% | 59.8% | 98.3% |
| 0.98 | 51.3% | 99.8% | 71.2% | 99.9% |

Calibration (confidence bin → accuracy):

| bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.1–0.2 | 14 | 0.17 | 0.0% |
| 0.2–0.3 | 44 | 0.29 | 4.5% |
| 0.3–0.4 | 374 | 0.36 | 15.8% |
| 0.4–0.5 | 477 | 0.46 | 37.3% |
| 0.5–0.6 | 665 | 0.56 | 42.3% |
| 0.6–0.7 | 1,008 | 0.66 | 29.7% |
| 0.7–0.8 | 1,019 | 0.76 | 76.3% |
| 0.8–0.9 | 1,293 | 0.86 | 71.2% |
| 0.9–1.0 | 18,012 | 0.98 | 94.3% |

Routing policies (LLM calls assume answers are cached per merchant key):

| policy | rows routed | LLM calls per txn (cached) | rows routed only for a name | model category acc on those |
|---|---|---|---|---|
| confidence < 0.8 (week 3) | 15.7% | 1.00% | 0.0% | – |
| confidence < 0.8 OR merchant not in directory | 35.7% | 2.13% | 20.0% | 98.9% |

Hybrid ceiling = accuracy if the LLM got every routed row right. Measured hybrid numbers: week3_llm.md.
