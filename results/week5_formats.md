## Unseen bank formats (leave-one-format-out)

Seed 0. Each row retrains the full model with every user of that statement format removed. Scored on rows the format shapes, for merchants that are *also* held out of training (the strictest case). **Never saw** = the ML model never saw the format, but the normalizer's hand-written rule for it is still on; **nobody saw** = that rule is switched off and the boilerplate learner is off too (the old pipeline); **+ learned** = rule off, learner on (the current pipeline on a brand-new bank). Control, never-saw and +learned all run the current pipeline (learner on).

| held-out format | users | rows | control (format seen) | never saw | nobody saw | nobody saw + learned | rules baseline | merchant exact: control / never / nobody / +learned |
|---|---|---|---|---|---|---|---|---|
| card:plain | 33 | 2,216 | 78.1% | 82.1% | no rule | – | 51.0% | 62.5% / 62.9% / – / – |
| card:fixed | 39 | 2,401 | 78.0% | 76.3% | no rule | – | 47.4% | 61.7% / 68.9% / – / – |
| card:pos | 51 | 3,785 | 75.2% | 75.5% | 72.2% | 72.2% | 44.0% | 61.0% / 69.3% / 0.0% / 58.7% |
| card:wf | 50 | 3,201 | 79.5% | 79.1% | 74.8% | 74.8% | 50.1% | 62.4% / 63.2% / 0.0% / 37.5% |
| card:boa | 43 | 2,943 | 79.0% | 79.2% | 78.9% | 78.9% | 58.0% | 62.9% / 60.2% / 0.0% / 60.2% |
| card:title | 34 | 2,324 | 79.4% | 79.4% | no rule | – | 44.6% | 62.9% / 60.2% / – / – |
| card:debit | 50 | 3,608 | 78.1% | 79.7% | 77.4% | 77.4% | 53.6% | 64.4% / 53.9% / 0.0% / 53.9% |
| ach:chase | 75 | 291 | 98.0% | 100.0% | 100.0% | 100.0% | 54.6% | 0.0% / 0.0% / 0.0% / 0.0% |
| ach:boa | 85 | 180 | 99.9% | 100.0% | 100.0% | 100.0% | 33.3% | 0.0% / 0.0% / 0.0% / 0.0% |
| ach:wf | 71 | 238 | 98.2% | 89.9% | no rule | – | 74.8% | 0.0% / 0.0% / – / – |
| ach:plain | 69 | 186 | 99.7% | 93.5% | no rule | – | 61.3% | 0.0% / 0.0% / – / – |

### Summary (row-weighted across formats)

| | merchants | control | model never saw format | nobody saw format* | nobody saw + learned* | merchant exact: control / never / nobody* / +learned* |
|---|---|---|---|---|---|---|
| card formats | unseen | 78.2% | 78.6% | 75.7% (vs 78.2%) | 75.7% | 62.6% / 62.6% / 0.0% (vs 61.8%) / 52.7% |
| card formats | seen | 92.7% | 92.1% | 90.2% (vs 92.2%) | 90.2% | 89.6% / 89.6% / 28.9% (vs 89.1%) / 86.4% |
| ach formats | unseen | 99.0% | 96.0% | 100.0% (vs 100.0%) | 100.0% | 0.0% / 0.0% / 0.0% (vs 0.0%) / 0.0% |
| ach formats | seen | 99.6% | 98.9% | 99.4% (vs 99.2%) | 99.4% | 76.0% / 76.0% / 50.8% (vs 74.6%) / 73.8% |

\* Only formats that have a format-specific normalizer rule; "vs" is the never-saw number on those same formats, so the pair isolates what switching off the rule costs.
