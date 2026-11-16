"""Stateful load: each simulated user opens a statement, then browses and corrects it.

Exercises the shared-state path (Redis) on every request: whichever worker serves a request
must find the user's statement and see their latest corrections.
"""

import csv
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

DATA = Path(os.environ.get("LEDGERLENS_DATA", "data/synthetic/transactions.csv"))
with DATA.open() as f:
    POOL = [{"description": r["description"], "amount": float(r["amount"]), "date": r["date"]}
            for r in csv.DictReader(f) if r["split"] == "test"]


class StatementUser(HttpUser):
    wait_time = between(0.01, 0.05)

    def on_start(self):
        # statements are created before the timed run (run_scaling.py) and shared by many users, so
        # the numbers are steady-state traffic, and corrections on one statement contend for its lock
        pool = os.environ.get("LL_DATASETS")
        if pool:
            self.did = random.choice(pool.split(","))
        else:
            self.did = self.client.post("/v1/datasets/demo", name="POST /datasets/demo").json()["dataset_id"]
        items = self.client.get(f"/v1/datasets/{self.did}/review?limit=8", name="GET /review").json()["items"]
        self.keys = [x["merchant_key"] for x in items] or ["none"]

    @task(3)
    def dashboard(self):
        self.client.get(f"/v1/datasets/{self.did}/dashboard", name="GET /dashboard")

    @task(5)
    def feed(self):
        self.client.get(f"/v1/datasets/{self.did}/transactions?limit=25&offset={random.randint(0, 600)}",
                        name="GET /transactions")

    @task(2)
    def review(self):
        self.client.get(f"/v1/datasets/{self.did}/review?limit=4", name="GET /review")

    @task(1)
    def correct(self):
        with self.client.post(f"/v1/datasets/{self.did}/corrections", name="POST /corrections",
                              json={"merchant_key": random.choice(self.keys), "merchant": "Load Test Fix"},
                              catch_response=True) as r:
            if r.status_code in (200, 404):  # 404: key already merged into another correction; fine
                r.success()

    @task(5)
    def enrich_one(self):
        self.client.post("/v1/enrich", json={"transactions": [random.choice(POOL)]}, name="POST /enrich (1 txn)")
