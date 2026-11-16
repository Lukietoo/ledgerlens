"""In-cluster load generator for the Kubernetes chaos test (stdlib only; runs in the API image).

    python k8s_load.py http://api:8000 --seconds 75 --concurrency 16

Every request opens a fresh connection, so each one is load-balanced by the Service across
whatever API pods are Ready at that moment, which is the path real clients take.

Before the load: create a statement and save one correction. During: mixed reads/enriches.
After: read the statement back (from whichever pod answers) and check the correction survived
every pod that was killed. Prints one JSON line prefixed with RESULT.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request

TXNS = ["SQ *BLUE BTL COF 0412 OAKLAND CA", "PAYPAL *SPOTIFYUSA 4029357733", "TST* SWEETGREEN 0231",
        "UBER *TRIP HELP.UBER.COM", "AMZN Mktp US*2K4L19", "SHELL OIL 57444", "NETFLIX.COM 866-579-7172"]


def call(base: str, method: str, path: str, body: dict | None = None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"null")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--seconds", type=float, default=75)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()

    for _ in range(60):  # Service ready?
        try:
            call(a.base, "GET", "/readyz")
            break
        except Exception:
            time.sleep(1)
    did = call(a.base, "POST", "/v1/datasets/demo", timeout=60)[1]["dataset_id"]
    key = call(a.base, "GET", f"/v1/datasets/{did}/review?limit=1")[1]["items"][0]["merchant_key"]
    call(a.base, "POST", f"/v1/datasets/{did}/corrections", {"merchant_key": key, "merchant": "Chaos Survivor"})

    lock = threading.Lock()
    ok, failures, lat = 0, [], []
    t0 = time.time()
    stop = t0 + a.seconds

    def worker():
        nonlocal ok
        while time.time() < stop:
            r = random.random()
            try:
                s = time.perf_counter()
                if r < 0.4:
                    call(a.base, "POST", "/v1/enrich", {"transactions": [
                        {"description": random.choice(TXNS), "amount": -round(random.uniform(3, 90), 2),
                         "date": "2026-03-01"}]})
                elif r < 0.7:
                    call(a.base, "GET", f"/v1/datasets/{did}/dashboard")
                else:
                    call(a.base, "GET", f"/v1/datasets/{did}/transactions?limit=25&offset={random.randint(0, 300)}")
                with lock:
                    ok += 1
                    lat.append(time.perf_counter() - s)
            except (urllib.error.URLError, OSError, ValueError) as e:  # HTTPError is a URLError
                with lock:
                    failures.append({"t": round(time.time() - t0, 2), "error": str(e)[:120]})

    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    feed = call(a.base, "GET", f"/v1/datasets/{did}/transactions?q=chaos+survivor&limit=5")[1]
    lat.sort()
    q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))] * 1000, 1) if lat else None  # noqa: E731
    print("RESULT " + json.dumps({
        "ok": ok, "failed": len(failures), "seconds": a.seconds, "rps": round((ok + len(failures)) / a.seconds, 1),
        "p50_ms": q(0.5), "p99_ms": q(0.99), "failures": failures[:50],
        "correction_survived": feed["total"] >= 1 and all(r["source"] == "user" for r in feed["rows"]),
    }), flush=True)


if __name__ == "__main__":
    main()
