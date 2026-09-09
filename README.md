# morpheus-wasm

Browser-based Basque autocomplete running a 91M-parameter **Mamba-2** model
entirely client-side via [wllama](https://github.com/ngxson/wllama)
(WebAssembly llama.cpp). No backend — inference happens in the browser on
WebGPU (with a WASM-SIMD fallback).

Two modes:
- **Ghost text** (`index.html`) — inline greyed completions, Tab to accept
- **Predictive keyboard** (`keyboard.html`) — phone-style chat UI with word
  suggestion chips + on-screen virtual keyboard

![status: working](https://img.shields.io/badge/status-working-brightgreen)
![model: 91M Mamba-2](https://img.shields.io/badge/model-91M%20Mamba--2-blue)
![backend: wllama 3.6.1](https://img.shields.io/badge/wllama-3.6.1-purple)

---

## Run locally

```bash
# 1. Get the patched model file (one-time, ~55 MB):
python3 test/patch-gguf.py
#   downloads the upstream GGUF from HF and writes
#   model/morpheus-v2-mamba.Q4_K_M.ugm.gguf

# 2. Serve + open:
./serve.sh
#   → http://127.0.0.1:8765/
```

Open in **Chrome or Edge** for WebGPU acceleration. Firefox falls back to
single-threaded WASM SIMD (works, but slower). The model loads once and is
cached by the browser.

### What you'll see
- A loading panel with a download progress bar.
- Once ready, an editor pre-filled with `Kaixo, zer ` and a greyed ghost
  completion (e.g. ` moduz?`). Press **Tab** to accept, or keep typing to
  get fresh suggestions. Try the example chips below the editor.

---

## Why a patched model?

The upstream GGUF (`itzune/morpheus-gguf`) needs **two fixes** before it
works correctly in wllama. Both are applied in a single pass by
[`test/patch-gguf.py`](test/patch-gguf.py):

### 1. Head-count fix (load compatibility)

The llama.cpp HuggingFace converter writes
`mamba2.attention.head_count = 0` (commented "unused"). wllama 3.6.1's
bundled llama.cpp uses `head_count` to compute the `ssm_in` tensor width
(`2*d_inner + 2*n_group*d_state + n_head`), so 0 makes it reject the model.
Setting `head_count = 24` (= `d_inner/head_dim` = `dt_rank`) produces the
correct width and is harmless for newer llama.cpp builds (which use
`ssm.time_step_rank` instead).

### 2. UGM tokenizer fix (tokenization fidelity)

The model was trained with a SentencePiece **unigram** tokenizer, but the
GGUF carries `tokenizer.ggml.model = "llama"` → llama.cpp's SPM tokenizer,
which uses **BPE-style pair merging** instead of the unigram **Viterbi
algorithm**. The result: every word preceded by a space gets mis-tokenized
(`" zer"` → `[▁][▁][z][er]` instead of `[▁zer]`), shifting the argmax and
producing wrong completions ("kostatzen" instead of " moduz?").

**Fix:** change `tokenizer.ggml.model` to `"t5"`, which maps to
`LLAMA_VOCAB_TYPE_UGM` — the real unigram Viterbi algorithm. Also set
`add_space_prefix = true` and `remove_extra_whitespaces = true` (matching
the reference SentencePiece's `add_dummy_prefix` and
`remove_extra_whitespaces` normalizer specs). These are standard GGUF
metadata keys read by llama.cpp, so **no source patching or wllama fork
is needed**.

After the patch, llama.cpp's tokenization matches the reference
SentencePiece model **100%**, and string-prompt completions match
token-ID-prompt completions exactly — the same outputs as the Python demo.

---

## Inference strategies

Shared model loading + token helpers live in `morpheus.js`, imported by
both modes.

### Ghost text mode (`app.js`)

| Strategy | Description |
|---|---|
| **Digit-token repair** | Walks the greedy path; if a token contains digits (gazette pollution like "1964ko"), swaps it for the best non-digit alternative from top-k logprobs. If a swap changed the path, re-generates from the swap point so subsequent tokens are correctly conditioned. |
| **`filter_suggestion`** | Strips ▁ markers and U+FFFD chars, collapses whitespace and punctuation runs, strips trailing space-punct sequences, rejects pure-punctuation output. |
| **`ghost_suffix`** | Smart Compose overlap: if the user already typed part of the prediction, only the non-typed suffix is shown as ghost. Also deduplicates punctuation at the boundary. |
| **Byte-fallback garbage detection** | Non-Latin characters (code > U+00FF) indicate byte-fallback tokens from a tokenization trap. Triggers a retokenization fallback that tries shorter prefixes to land on a compatible tokenization path. |
| **Confidence threshold** | Ghost text is only shown when average token confidence (excluding EOS) ≥ 18%. |
| **Greedy decoding** | `temperature=0, penalty_repeat=1.1, n_probs=5, logprobs=5, max_tokens=3` — matching the demo's parameters. |

### Predictive keyboard mode (`keyboard.js`)

| Strategy | Description |
|---|---|
| **`keyboardCandidates()`** | Ported from `demo/server.py::_keyboard_candidates`. Two modes: **next-word prediction** (cursor after space — single call, returns top-k first-token words + greedy word) and **word completion** (cursor mid-word — tries progressively shorter prefixes in parallel, extracts greedy multi-token completions + top-k single-token alternatives, also detects next-word candidates when the model predicts a space token). |
| **Sticky merge** | Carries forward previous candidates that match the new prefix. Prevents predictions from vanishing when the user types the first letter of a predicted word and the tokenization path switches. Sticky survivors get a small probability boost. |
| **Android-style chip layout** | 3 chips with the highest-probability word in the CENTER (like Gboard). 2 chips: [2nd, 1st]. 1 chip: [1st]. |
| **Chip acceptance** | Different handling for word completions (replace partial word + trailing space), next-word suggestions (insert with leading space), and punctuation (attach to previous word, no space before). |
| **Virtual keyboard** | Full QWERTY layout with ñ, shift (one-shot + auto-shift after sentence-ending punctuation), symbol layout (123/ABC toggle), long-press for accented characters (é, ü, ñ, ç, …), backspace, space, enter (send message). |
| **Parallel fallback paths** | Word-completion fallback paths (progressively shorter prefixes + from-scratch) run concurrently via `Promise.allSettled`. Needs wllama 3.6.0+ (PR #270 fixed the concurrent-completion bug); `loadModel()` sets `n_parallel` + `kv_unified:false` so each slot owns its SSM state (required for a recurrent model). |

---

## Deploy to GitHub Pages

The app loads the model from **HuggingFace** when served from a non-localhost
origin (GitHub Pages), so you must upload the patched GGUF to HF first:

```bash
# 1. Upload the patched model to the HF repo:
huggingface-cli upload itzune/morpheus-gguf \
  model/step_0074000.Q4_K_M.ugm.gguf \
  morpheus-v2-mamba.Q4_K_M.ugm.gguf

# 2. Push this repo to GitHub and enable Pages (root folder).
```

`app.js` auto-detects the origin: localhost → loads `./model/*.gguf` from
the local server; elsewhere → `loadModelFromHF('itzune/morpheus-gguf',
'morpheus-v2-mamba.Q4_K_M.ugm.gguf')`.

---

## Project layout

```
morpheus.js     # shared module: model loading + token helpers + keyboardCandidates()
index.html      # ghost text mode page
keyboard.html   # predictive keyboard mode page
app.js          # ghost text mode: digit repair, ghost suffix, retokenization fallback
keyboard.js     # keyboard mode: sticky merge, chips, virtual keyboard
style.css       # ghost text mode (light theme)
keyboard.css    # keyboard mode (dark chat theme + virtual keyboard)
serve.sh        # local dev server launcher
test/
  patch-gguf.py       # regenerates the loadable GGUF (download + head_count + UGM)
  load-test.mjs       # headless Playwright smoke test (ghost mode)
  ux-test.mjs         # interactive ghost-text accept test
  keyboard-test.mjs   # predictive keyboard test (chips, typing, send)
RESEARCH.md      # design notes, risk analysis, phase plan
```

## Notes & known limitations

- **No BOS token** — the model was trained with `add_bos_token=false`; the
  app verifies this and warns if a BOS token is unexpectedly required.
- **Context**: `n_ctx=2048` (the model was trained at 1M; 2K is ample for
  autocomplete).
- **wllama API**: `createCompletion` accepts string prompts only (no token
  IDs). The UGM patch makes string-prompt tokenization match the reference
  SentencePiece, so this is no longer a quality barrier. The digit-repair
  re-generation uses string prompts (original + repaired prefix text) since
  wllama has no public `detokenize()` — this works because tokenization is
  now correct.
