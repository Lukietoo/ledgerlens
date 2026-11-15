"""Put a verified model at models/enricher.joblib for a deployment build.

    LEDGERLENS_MODEL_URL=https://github.com/<you>/ledgerlens/releases/download/model-v1/enricher.joblib \
        python scripts/fetch_model.py

The artifact is a joblib pickle, and loading a pickle runs code, so the download must
match the SHA-256 pinned in model.sha256 (committed alongside the code that made it)
before it is moved into place. With no URL set, it trains locally instead (needs ~650 MB
of RAM, so it suits a bigger build box or a laptop).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "enricher.joblib"
PINNED = ROOT / "model.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    url = os.environ.get("LEDGERLENS_MODEL_URL", "").strip()
    if not url:
        print("LEDGERLENS_MODEL_URL not set: training locally (synthetic data + 1 seed)")
        subprocess.run([sys.executable, "-m", "ledgerlens.synth"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "ledgerlens.train", "--seeds", "0"], cwd=ROOT, check=True)
        return 0
    if not url.startswith("https://"):
        print("refusing a non-HTTPS model URL", file=sys.stderr)
        return 1
    expected = PINNED.read_text().split()[0].strip().lower()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=TARGET.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"downloading model from {url}")
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 (https enforced above)
        got = sha256(tmp_path)
        if got != expected:
            print(f"checksum mismatch: expected {expected}, got {got}; refusing to install", file=sys.stderr)
            return 1
        tmp_path.replace(TARGET)
        print(f"installed {TARGET.relative_to(ROOT)} (sha256 {got[:12]}…)")
        return 0
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
