## User corrections: 300 simulated users, 22,906 transactions at merchants the model never saw

Row-level accuracy over each user's own transactions after k actions (cold directory).

| actions per user | category: notice | category: review queue | names: notice | names: review queue |
|---|---|---|---|---|
| 0 | 80.6% | 80.6% | 47.9% | 47.9% |
| 1 | 85.6% | 86.4% | 63.6% | 63.1% |
| 2 | 90.2% | 90.7% | 74.4% | 71.7% |
| 3 | 93.9% | 93.9% | 82.6% | 78.3% |
| 5 | 97.7% | 98.2% | 91.2% | 85.9% |
| 10 | 99.8% | 99.9% | 98.8% | 93.4% |

notice: 2,501 fixes, **5.5 rows fixed per fix** (one fix applies to every row of that merchant).

queue: 2,023 fixes, **6.3 rows fixed per fix** (one fix applies to every row of that merchant).

### Community corrections (150 users contribute, 150 benefit with zero effort)

| review actions per contributor | shared fixes | others' category | others' names |
|---|---|---|---|
| 1 | 59 | 79.7% → 90.9% | 47.2% → 65.7% |
| 3 | 162 | 79.7% → 96.1% | 47.2% → 72.8% |
| 5 | 262 | 79.7% → 96.5% | 47.2% → 75.9% |
| 10 | 503 | 79.7% → 96.8% | 47.2% → 79.9% |

### Read these numbers with their scale

- Each user here has a **median of 10 distinct unseen merchants** (p90: 13), so 5 actions covers about half of them and 10 covers nearly all. The curves are steep partly because the tail is short. The scale-free numbers are **rows fixed per fix** and the first few actions.
- The community result reflects a **dense synthetic population**: the median unseen merchant is shared by 5 of 300 users, and 87% of these rows are at merchants with 10+ users. A real long tail of local shops is sparser, so expect less lift.
- Simulated users are always right. Production community corrections would need agreement from several users (and abuse protection) before applying to anyone else; per-user corrections don't.
