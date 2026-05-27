#!/usr/bin/env bash
set -euo pipefail

cd /workspace/resplat

if python - <<'PY'
try:
    import pointops._C  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
then
  exit 0
fi

echo "[resplat] Building pointops from mounted source..."
cd /workspace/resplat/src/model/encoder/pointops
python setup.py install
