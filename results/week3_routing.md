## Confidence routing: synthetic test split (22,906 txns, unseen merchants)

### cold directory: model accuracy 81.6%, ECE 0.076

| route to LLM below | traffic routed | model acc on kept | model acc on routed | hybrid ceiling |
|---|---|---|---|---|
| 0.00 | 0.0% | 81.6% | – | 81.6% |
| 0.50 | 5.6% | 83.7% | 45.8% | 84.6% |
| 0.60 | 9.5% | 85.0% | 49.4% | 86.4% |
| 0.70 | 14.8% | 86.3% | 54.2% | 88.4% |
| 0.80 | 21.5% | 87.7% | 59.4% | 90.3% |
| 0.85 | 25.4% | 88.3% | 61.9% | 91.3% |
| 0.90 | 29.4% | 89.2% | 63.4% | 92.3% |
| 0.95 | 37.8% | 90.0% | 67.8% | 93.8% |
| 0.98 | 61.1% | 95.8% | 72.5% | 98.4% |

Calibration (confidence bin → accuracy):

| bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.1–0.2 | 10 | 0.20 | 0.0% |
| 0.2–0.3 | 80 | 0.27 | 38.8% |
| 0.3–0.4 | 413 | 0.35 | 39.5% |
| 0.4–0.5 | 775 | 0.45 | 50.5% |
| 0.5–0.6 | 903 | 0.55 | 54.5% |
| 0.6–0.7 | 1,201 | 0.66 | 62.9% |
| 0.7–0.8 | 1,554 | 0.75 | 70.8% |
| 0.8–0.9 | 1,795 | 0.85 | 74.5% |
| 0.9–1.0 | 16,175 | 0.98 | 89.2% |

Routing policies (LLM calls assume answers are cached per merchant key):

| policy | rows routed | LLM calls per txn (cached) | rows routed only for a name | model category acc on those |
|---|---|---|---|---|
| confidence < 0.8 (week 3) | 21.5% | 1.19% | 0.0% | – |
| confidence < 0.8 OR merchant not in directory | 95.8% | 4.66% | 74.3% | 87.3% |

### warm directory: model accuracy 85.4%, ECE 0.070

| route to LLM below | traffic routed | model acc on kept | model acc on routed | hybrid ceiling |
|---|---|---|---|---|
| 0.00 | 0.0% | 85.4% | – | 85.4% |
| 0.50 | 3.2% | 87.3% | 26.9% | 87.7% |
| 0.60 | 5.1% | 88.4% | 30.8% | 89.0% |
| 0.70 | 8.0% | 89.8% | 35.4% | 90.6% |
| 0.80 | 13.9% | 92.0% | 44.2% | 93.1% |
| 0.85 | 16.3% | 92.8% | 47.5% | 94.0% |
| 0.90 | 20.1% | 93.2% | 54.5% | 94.5% |
| 0.95 | 27.6% | 94.9% | 60.4% | 96.3% |
| 0.98 | 45.0% | 99.7% | 68.0% | 99.8% |

Calibration (confidence bin → accuracy):

| bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.2–0.3 | 19 | 0.28 | 0.0% |
| 0.3–0.4 | 160 | 0.36 | 21.2% |
| 0.4–0.5 | 552 | 0.46 | 29.5% |
| 0.5–0.6 | 447 | 0.55 | 37.4% |
| 0.6–0.7 | 653 | 0.66 | 43.5% |
| 0.7–0.8 | 1,346 | 0.75 | 56.2% |
| 0.8–0.9 | 1,419 | 0.86 | 77.6% |
| 0.9–1.0 | 18,310 | 0.98 | 93.2% |

Routing policies (LLM calls assume answers are cached per merchant key):

| policy | rows routed | LLM calls per txn (cached) | rows routed only for a name | model category acc on those |
|---|---|---|---|---|
| confidence < 0.8 (week 3) | 13.9% | 1.05% | 0.0% | – |
| confidence < 0.8 OR merchant not in directory | 34.6% | 2.37% | 20.8% | 98.1% |

Hybrid ceiling = accuracy if the LLM got every routed row right. Measured hybrid numbers: week3_llm.md.
