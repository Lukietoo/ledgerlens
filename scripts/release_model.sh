#!/usr/bin/env bash
# Pin the current model's checksum and publish it as a GitHub release asset.
#
#   scripts/release_model.sh model-v1
#
# Writes model.sha256 (commit it), then uploads models/enricher.joblib with `gh`.
# Run it yourself: publishing a release is a deliberate, public step.
set -euo pipefail

tag="${1:?usage: scripts/release_model.sh <tag, e.g. model-v1>}"
cd "$(dirname "$0")/.."
model="models/enricher.joblib"
[ -f "$model" ] || { echo "no $model; run: python -m ledgerlens.train" >&2; exit 1; }

shasum -a 256 "$model" | awk '{print $1}' > model.sha256
echo "pinned $(cat model.sha256) in model.sha256 (commit this file)"

gh release create "$tag" "$model" \
  --title "Ledgerlens model $tag" \
  --notes "Trained enricher (stacked TF-IDF + LightGBM) on synthetic data. sha256: $(cat model.sha256)"

repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo
echo "Set this on the API service (Render dashboard or render.yaml):"
echo "  LEDGERLENS_MODEL_URL=https://github.com/${repo}/releases/download/${tag}/enricher.joblib"
