#!/usr/bin/env bash
# Chaos test against a running cluster (CI runs it on kind; see .github/workflows/k8s.yml).
#
# Starts an in-cluster load Job through the api Service, then, under load:
#   t=20s  delete one API pod normally (a deploy, a node drain): expect zero failed requests
#   t=45s  force-kill another (a crash, an OOM kill): expect only that pod's in-flight requests to fail
# and asserts the Deployment heals back to full strength, a correction saved before the chaos
# is still there (state is in Redis, not in any pod), and Prometheus sees every replica.
# Writes results/k8s_chaos.json and .md.
set -euo pipefail
NS=ledgerlens
OUT=${OUT:-results/k8s_chaos}
MAX_FAIL_RATE=${MAX_FAIL_RATE:-0.01}
log() { printf '\n== %s\n' "$*"; }


log "load job"
kubectl -n $NS delete job chaos-load --ignore-not-found --wait=true
kubectl -n $NS create configmap chaos-load --from-file=k8s_load.py=loadtest/k8s_load.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n $NS apply -f - <<'YAML'
apiVersion: batch/v1
kind: Job
metadata: {name: chaos-load}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      securityContext: {runAsNonRoot: true, runAsUser: 10001, seccompProfile: {type: RuntimeDefault}}
      containers:
        - name: load
          image: ledgerlens-api:dev
          imagePullPolicy: IfNotPresent
          command: ["python", "/load/k8s_load.py", "http://api:8000", "--seconds", "75", "--concurrency", "16"]
          securityContext: {allowPrivilegeEscalation: false, capabilities: {drop: [ALL]}}
          resources: {requests: {cpu: 250m, memory: 128Mi}}
          volumeMounts: [{name: load, mountPath: /load}]
      volumes: [{name: load, configMap: {name: chaos-load}}]
YAML
kubectl -n $NS wait --for=condition=ready pod -l job-name=chaos-load --timeout=120s
sleep 25   # setup (create statement + correction) then 20 s of steady load

log "graceful delete"
mapfile -t PODS < <(kubectl -n $NS get pods -l app.kubernetes.io/name=api -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
VICTIM1=${PODS[0]}
T1=$(date +%s)
kubectl -n $NS delete pod "$VICTIM1" --wait=false
sleep 25

log "force kill"
mapfile -t PODS < <(kubectl -n $NS get pods -l app.kubernetes.io/name=api -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}')
VICTIM2=""
for p in "${PODS[@]}"; do [ "$p" != "$VICTIM1" ] && VICTIM2=$p && break; done
kubectl -n $NS delete pod "$VICTIM2" --grace-period=0 --force
T2=$(date +%s)

log "heal"
kubectl -n $NS rollout status deploy/api --timeout=180s
HEALED=$(( $(date +%s) - T2 ))
READY=$(kubectl -n $NS get deploy api -o jsonpath='{.status.readyReplicas}')
WANT=$(kubectl -n $NS get deploy api -o jsonpath='{.spec.replicas}')

kubectl -n $NS wait --for=condition=complete job/chaos-load --timeout=180s
RESULT=$(kubectl -n $NS logs job/chaos-load | sed -n 's/^RESULT //p')
echo "$RESULT" | python3 -m json.tool | head -40

log "prometheus sees every replica"
kubectl -n $NS port-forward svc/prometheus 9090:9090 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
UP=0
for _ in $(seq 1 30); do
  sleep 2
  UP=$(curl -fsS 'http://127.0.0.1:9090/api/v1/query' --data-urlencode 'query=count(up{job="ledgerlens-api"}==1)' \
       | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(int(float(r[0]["value"][1])) if r else 0)' || echo 0)
  [ "$UP" -ge "$WANT" ] && break
done
SERVED=$(curl -fsS 'http://127.0.0.1:9090/api/v1/query' \
  --data-urlencode 'query=sum(ledgerlens_http_requests_total{job="ledgerlens-api"})' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(int(float(r[0]["value"][1])) if r else 0)')

mkdir -p "$(dirname "$OUT")"
RESULT="$RESULT" VICTIM1="$VICTIM1" VICTIM2="$VICTIM2" T1="$T1" T2="$T2" HEALED="$HEALED" READY="$READY" \
WANT="$WANT" UP="$UP" SERVED="$SERVED" OUT="$OUT" MAX_FAIL_RATE="$MAX_FAIL_RATE" python3 - <<'PY'
import json, os
r = json.loads(os.environ["RESULT"])
e = os.environ
total = r["ok"] + r["failed"]
rate = r["failed"] / max(1, total)
report = {**{k: r[k] for k in ("ok", "failed", "rps", "p50_ms", "p99_ms", "correction_survived", "failures")},
          "failure_rate": round(rate, 5), "graceful_victim": e["VICTIM1"], "killed_victim": e["VICTIM2"],
          "healed_s": int(e["HEALED"]), "ready_replicas": int(e["READY"] or 0), "replicas": int(e["WANT"]),
          "prometheus_targets_up": int(e["UP"]), "prometheus_requests_counted": int(e["SERVED"])}
json.dump(report, open(e["OUT"] + ".json", "w"), indent=2)
md = f"""## Kubernetes chaos test (kind, 3 API replicas behind a Service, Redis-backed state)

75 s of in-cluster load (16 concurrent clients, a new connection per request). Under load, one API
pod was deleted normally (like a deploy or node drain), then 25 s later another was force-killed
(like a crash).

| | |
|---|---|
| requests | {total:,} ({r['rps']} req/s) |
| failed | **{r['failed']}** ({rate:.2%}) |
| latency p50 / p99 | {r['p50_ms']} ms / {r['p99_ms']} ms |
| back to {report['replicas']}/{report['replicas']} ready after the kill | {report['healed_s']} s |
| correction saved before the chaos, still there after | {'yes' if r['correction_survived'] else 'NO'} |
| replicas scraped by Prometheus (pod discovery) | {report['prometheus_targets_up']} |
"""
open(e["OUT"] + ".md", "w").write(md)
print(md)
assert r["correction_survived"], "a correction was lost"
assert report["ready_replicas"] == report["replicas"], "deployment did not heal"
assert report["prometheus_targets_up"] >= report["replicas"], "prometheus is missing replicas"
assert rate <= float(e["MAX_FAIL_RATE"]), f"failure rate {rate:.2%} above {float(e['MAX_FAIL_RATE']):.2%}"
print("chaos test passed")
PY
