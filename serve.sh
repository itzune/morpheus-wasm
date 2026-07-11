#!/usr/bin/env bash
# Serve morpheus-wasm locally for browser testing.
# wllama downloads fine over a plain server (it falls back to a full fetch
# when the server lacks HTTP Range support), so python's http.server works.
set -e
cd "$(dirname "$0")"
PORT="${1:-8765}"
echo "morpheus-wasm → http://127.0.0.1:${PORT}/"
echo "(open in Chrome/Edge for WebGPU; Firefox = WASM-SIMD fallback)"
python3 -m http.server "$PORT" --bind 127.0.0.1
