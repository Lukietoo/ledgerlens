## Unseen bank formats (leave-one-format-out)

Seed 0. Each row retrains the full model with every user of that statement format removed. Scored on rows the format shapes, for merchants that are *also* held out of training (the strictest case). **Never saw** = the ML model never saw the format, but the normalizer's hand-written rule for it is still on; **nobody saw** = that rule is switched off too.

| held-out format | users | rows | control (format seen) | never saw | nobody saw | nobody saw + learned | rules baseline | merchant exact: control / never / nobody / +learned |
|---|---|---|---|---|---|---|---|---|
| card:plain | 33 | 2,216 | 78.1% | 82.1% | no rule | – | 51.0% | 47.1% / 60.6% / – / – |
| card:fixed | 39 | 2,401 | 78.0% | 76.3% | no rule | – | 47.4% | 47.3% / 58.5% / – / – |
| card:pos | 51 | 3,785 | 75.2% | 75.5% | 72.2% | 73.9% | 44.0% | 47.8% / 52.3% / 0.0% / 58.5% |
| card:wf | 50 | 3,201 | 79.5% | 79.1% | 74.8% | 78.3% | 50.1% | 49.3% / 44.5% / 0.0% / 38.3% |
| card:boa | 43 | 2,943 | 79.0% | 79.2% | 78.9% | 79.2% | 58.0% | 49.0% / 46.0% / 0.0% / 38.8% |
| card:title | 34 | 2,324 | 79.4% | 79.4% | no rule | – | 44.6% | 49.7% / 39.8% / – / – |
| card:debit | 50 | 3,608 | 78.1% | 79.7% | 77.4% | 77.8% | 53.6% | 50.0% / 42.2% / 0.0% / 43.6% |
| ach:chase | 75 | 291 | 98.0% | 100.0% | 100.0% | 100.0% | 54.6% | 0.0% / 0.0% / 0.0% / 0.0% |
| ach:boa | 85 | 180 | 99.9% | 100.0% | 100.0% | 100.0% | 33.3% | 0.0% / 0.0% / 0.0% / 0.0% |
| ach:wf | 71 | 238 | 98.2% | 89.9% | no rule | – | 74.8% | 0.0% / 0.0% / – / – |
| ach:plain | 69 | 186 | 99.7% | 93.5% | no rule | – | 61.3% | 0.0% / 0.0% / – / – |

### Summary (row-weighted across formats)

| | merchants | control | model never saw format | nobody saw format* | nobody saw + learned* | merchant exact: control / never / nobody* / +learned* |
|---|---|---|---|---|---|---|
| card formats | unseen | 78.2% | 78.6% | 75.7% (vs 78.2%) | 77.1% | 48.6% / 48.6% / 0.0% (vs 46.4%) / 45.5% |
| card formats | seen | 92.7% | 92.1% | 90.2% (vs 92.2%) | 91.7% | 82.5% / 82.5% / 28.9% (vs 81.2%) / 80.9% |
| ach formats | unseen | 99.0% | 96.0% | 100.0% (vs 100.0%) | 100.0% | 0.0% / 0.0% / 0.0% (vs 0.0%) / 0.0% |
| ach formats | seen | 99.6% | 98.9% | 99.4% (vs 99.2%) | 99.2% | 70.4% / 70.4% / 50.8% (vs 74.6%) / 73.8% |

\* Only formats that have a format-specific normalizer rule; "vs" is the never-saw number on those same formats, so the pair isolates what switching off the rule costs.
