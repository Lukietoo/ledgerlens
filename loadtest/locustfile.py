"""Load test for POST /v1/enrich.

    LEDGERLENS_BATCH=1   locust -f loadtest/locustfile.py --headless -u 32 -r 8 -t 45s --host http://127.0.0.1:8000
    LEDGERLENS_BATCH=100 locust ...

Payloads are real rows from the synthetic test split (unseen merchants), so the
resolver and text model do realistic work. Model-only: no LLM calls.
"""

import csv
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

BATCH = int(os.environ.get("LEDGERLENS_BATCH", "1"))
DATA = Path(os.environ.get("LEDGERLENS_DATA", "data/synthetic/transactions.csv"))

with DATA.open() as f:
    POOL = [
        {"id": r["txn_id"], "description": r["description"], "amount": float(r["amount"]),
         "date": r["date"], "user_id": r["user_id"]}
        for r in csv.DictReader(f) if r["split"] == "test"
    ]


class EnrichUser(HttpUser):
    wait_time = between(0, 0.01)

    @task
    def enrich(self):
        txns = random.sample(POOL, BATCH)
        with self.client.post("/v1/enrich", json={"transactions": txns}, name=f"/v1/enrich [batch={BATCH}]",
                              catch_response=True) as r:
            if r.status_code != 200 or len(r.json()["results"]) != BATCH:
                r.failure(f"bad response {r.status_code}")
