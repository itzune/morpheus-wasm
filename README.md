# morpheus-wasm

Browser-based Basque autocomplete running a 91M-parameter **Mamba-2** model
entirely client-side via [wllama](https://github.com/ngxson/wllama)
(WebAssembly llama.cpp). No backend — inference happens in the browser on
WebGPU (with a WASM-SIMD fallback).

This is the **Phase 1 proof of concept**: validate that wllama can load the
Morpheus GGUF and produce Basque completions as inline ghost text.

![status: working](https://img.shields.io/badge/status-working-brightgreen)
![model: 91M Mamba-2](https://img.shields.io/badge/model-91M%20Mamba--2-blue)
![backend: wllama 3.5.1](https://img.shields.io/badge/wllama-3.5.1-purple)

---

## Run locally

```bash
# 1. Get the patched model file (one-time, ~55 MB):
python3 test/patch-headcount.py
#   downloads the upstream GGUF from HF and writes
#   model/morpheus-v2-mamba.Q4_K_M.wllama.gguf

# 2. Serve + open:
./serve.sh
#   → http://127.0.0.1:8765/
```

Open in **Chrome or Edge** for WebGPU acceleration. Firefox will fall back to
single-threaded WASM SIMD (works, but slower). The model loads once and is
cached by the browser.

### What you'll see
- A loading panel with a download progress bar.
- Once ready, an editor pre-filled with `Kaixo, zer ` and a greyed ghost
  completion (e.g. `zaldiak?`). Press **Tab** to accept, or keep typing to
  get fresh suggestions. Try the example chips below the editor.

---

## Why a patched model?

The upstream GGUF (`itzune/morpheus-gguf`) **does not load** in any released
wllama (3.2.3–3.5.1). Root cause:

- The llama.cpp HuggingFace converter writes
  `mamba2.attention.head_count = 0` (commented "unused").
- wllama 3.5.1's bundled llama.cpp (dd4623a) computes the `ssm_in` tensor
  width as `2*d_inner + 2*n_group*d_state + n_head`, reading `n_head` from
  `head_count` (=0) → expects width **3200**, but the tensor is **3224**
  (= 3200 + `dt_rank`). Hard load failure.
- Newer llama.cpp (post Jul-2025) uses `ssm.time_step_rank` (=24) instead and
  loads fine — but wllama is pinned to the old build.

**Fix:** set `head_count = 24` (= `d_inner/head_dim` = `dt_rank`). A 4-byte
in-place patch; file size unchanged. This makes the dd4623a loader compute the
correct width (3224) and is harmless for newer loaders (which ignore
`head_count`). See [`test/patch-headcount.py`](test/patch-headcount.py).

---

## Deploy to GitHub Pages

The app loads the model from **HuggingFace** when served from a non-localhost
origin (GitHub Pages), so you must upload the patched GGUF to HF first:

```bash
# 1. Upload the patched model to the HF repo as a new file:
huggingface-cli upload itzune/morpheus-gguf \
  model/morpheus-v2-mamba.Q4_K_M.wllama.gguf \
  morpheus-v2-mamba.Q4_K_M.wllama.gguf

# 2. Push this repo to GitHub and enable Pages (root folder).
```

`app.js` auto-detects the origin: localhost → loads `./model/*.gguf` from the
local server; elsewhere → `loadModelFromHF('itzune/morpheus-gguf',
'morpheus-v2-mamba.Q4_K_M.wllama.gguf')`.

---

## Project layout

```
index.html       # page structure
style.css        # light theme + ghost-text selection styling
app.js           # wllama load + greedy completion + ghost text
serve.sh         # local dev server launcher
test/
  patch-headcount.py   # regenerates the loadable GGUF (download + 4-byte patch)
  range_server.py      # HTTP server w/ Range support (optional; plain server works too)
  load-test.mjs        # headless Playwright smoke test
  probe*.html / *.mjs  # wllama version / model-load probes
RESEARCH.md      # design notes, risk analysis, phase plan
```

## Notes & known limitations

- **Greedy, raw output.** Phase 1 does straight greedy decoding (`temp=0`,
  8 tokens). Output quality and the exact continuation can differ from the
  Python demo (which does token-level inference engineering) because of
  tokenizer divergence (llama.cpp's built-in SentencePiece vs. the original)
  and llama.cpp version differences. Phase 2 will add `n_probs`-based
  next-word extraction and retokenization handling.
- **No BOS token** — the model was trained with `add_bos_token=false`; the
  app verifies this and warns if a BOS token is unexpectedly required.
- **Context**: `n_ctx=2048` (the model was trained at 1M; 2K is ample for
  autocomplete).
