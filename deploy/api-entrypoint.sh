#!/bin/sh
# One uvicorn worker per container by default (Kubernetes scales with replicas). With
# WEB_CONCURRENCY > 1 (docker compose), workers share Prometheus samples through a scratch dir
# that must start empty.
set -eu
if [ "${WEB_CONCURRENCY:-1}" -gt 1 ]; then
  export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus}"
  rm -rf "$PROMETHEUS_MULTIPROC_DIR" && mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
fi
exec uvicorn ledgerlens.service:app --host 0.0.0.0 --port "${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-1}" --timeout-graceful-shutdown 20 "$@"
