#!/usr/bin/env bash
# Vendor ONNX Runtime Web into web/vendor/ so the demo runs with no CDN.
set -euo pipefail
V="${1:-1.29.0}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/web/vendor"
mkdir -p "$DIR"
for f in ort.all.min.js ort-wasm-simd-threaded.jsep.wasm ort-wasm-simd-threaded.jsep.mjs; do
  echo "fetching $f"
  curl -sfL "https://cdn.jsdelivr.net/npm/onnxruntime-web@$V/dist/$f" -o "$DIR/$f"
done
echo "onnxruntime-web $V vendored into web/vendor/"
