# Morpheus-WASM: Research Notes

> **Goal**: Run the Morpheus v2 Basque autocomplete model entirely in the
> browser, deployed as a free GitHub Pages site, using
> [wllama](https://github.com/ngxson/wllama) (WebAssembly binding for
> llama.cpp).

---

## 1. Why This Project Exists

### The Problem

Morpheus v2 is a 91M-parameter Mamba-2 language model for Basque autocomplete,
trained on ~10B tokens. The trained model is exported to GGUF format (55MB
Q4_K_M / 66MB Q5_K_M) and published on HuggingFace at
[`itzune/morpheus-gguf`](https://huggingface.co/itzune/morpheus-gguf).

The existing demo (`morpheus-mamba/demo/`) requires a backend server running
`llama-server` (llama.cpp's HTTP server) plus a Python FastAPI layer that
implements keyboard-specific inference engineering (tokenization repair,
retokenization fallback, sticky merge, next-word candidate extraction). This
works for local Docker deployments and on the GPU server, but **cannot be
hosted for free on HuggingFace Spaces** — as of July 2026, both Docker and
Gradio Spaces require a paid PRO/Team/Enterprise plan. Only **Static Spaces**
(plain HTML/CSS/JS, no server-side execution) remain free, but we'll use
**GitHub Pages** instead for better control and no HF dependencies.

### The Opportunity

The model is tiny (55MB quantized). Modern browsers support WebAssembly SIMD
and WebGPU. The [wllama](https://github.com/ngxson/wllama) library provides a
full WebAssembly binding for llama.cpp that can:

- Load GGUF files directly from HuggingFace Hub
- Run inference entirely in the browser (no backend server)
- Use WebGPU for GPU acceleration when available
- Fall back to CPU (WASM SIMD) when WebGPU is unavailable

If we can port our inference logic to the browser, we get a **free, zero-cost,
zero-infrastructure** public demo that anyone can use by visiting a URL.

---

## 2. Deployment Platform: GitHub Pages

### Why Not HuggingFace Spaces

On ~July 8, 2026, HuggingFace moved Docker **and** Gradio Spaces behind a
paywall. Free accounts can no longer create CPU Basic Docker or Gradio Spaces.
Confirmed via API:

```
402 Payment Required:
"Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a Team or Enterprise plan for organization itzune.
Subscribe at https://huggingface.co/enterprise"
```

HF Static Spaces (plain HTML/CSS/JS) remain free, but have limitations:
- Spaces sleep after inactivity (cold starts)
- Limited control over HTTP headers (needed for multi-threaded WASM)
- Tied to HF infrastructure

### Why GitHub Pages

GitHub Pages is free, always-on, and gives full control over the repository:

| Property | GitHub Pages | HF Static Space |
|----------|-------------|-----------------|
| Cost | Free | Free |
| Always on | ✅ | ❌ (sleeps after inactivity) |
| Custom HTTP headers | ❌ (limitation) | ❌ |
| Custom domain | ✅ | ✅ (PRO only) |
| HTTPS | ✅ | ✅ |
| Build from repo | ✅ (Actions or Jekyll) | ✅ |
| Independence from HF | ✅ | ❌ |

**One limitation**: GitHub Pages does not support custom HTTP headers. This
means `Cross-Origin-Embedder-Policy` (COEP) and `Cross-Origin-Opener-Policy`
(COOP) cannot be set, which disables wllama's multi-threaded WASM mode.
However:

1. **WebGPU doesn't need COOP/COEP** — the primary acceleration path works fine.
2. **wllama falls back to single-threaded WASM automatically** when COOP/COEP
   are absent — slower but functional.
3. **A service worker can inject COOP/COEP headers** even on GitHub Pages
   (known technique used by many WASM projects). This is a Phase 2+ optimization.
4. The model is 91M params — single-threaded WASM SIMD is adequate for
   real-time autocomplete at this scale.

### GitHub Pages Setup

The repo (`itzune/morpheus-wasm`) will be published via GitHub Pages:

1. Push to `github.com/itzune/morpheus-wasm`
2. Settings → Pages → Source: `main` branch, `/` root
3. Site live at `https://itzune.github.io/morpheus-wasm/`

No build step needed for a static site (no Jekyll). GitHub Actions can be added
later for bundling/minification if needed.

### HF Spaces Paywall (for reference)

Forum discussions confirming the July 2026 change:
- [Docker SDK now marked as "Paid" when creating a new Space?](https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580)
- [New free accounts cannot create CPU Basic Gradio Spaces](https://discuss.huggingface.co/t/new-free-accounts-cannot-create-cpu-basic-gradio-spaces-only-zerogpu-available/177629)

Account status (confirmed):
- `xezpeleta` personal account: `isPro: False`
- `itzune` org: `canPay: False`

---

## 3. wllama: WebAssembly Binding for llama.cpp

### Overview

**wllama** ([github.com/ngxson/wllama](https://github.com/ngxson/wllama)) is a
TypeScript/WASM binding for llama.cpp, maintained by Xuan-Son Nguyen (a
llama.cpp maintainer). It runs LLM inference entirely in the browser.

### Key Facts

| Property | Value |
|----------|-------|
| npm package | `@wllama/wllama` |
| Current version | v3.5.0 (synced with llama.cpp v3.5.0, June 2026) |
| License | MIT |
| WebGPU support | ✅ (V3.1+, automatic, all layers offloaded by default) |
| CPU fallback | ✅ (WebAssembly SIMD, no GPU needed) |
| Max model size | 2GB (ArrayBuffer limitation; can split larger models) |
| HF Hub integration | ✅ (`loadModelFromHF()`) |
| OpenAI-compatible API | ✅ (`createChatCompletion()`, `createCompletion()`) |
| Low-level API | ✅ (tokenize, detokenize, sampling, logits) |
| Worker-based | ✅ (inference in Web Worker, doesn't block UI) |
| Multi-thread | ✅ (requires COOP/COEP headers) |

### Architecture

```
Browser
├── Main thread (UI)
│   ├── predictive-keyboard.html (our frontend)
│   └── wllama JS API calls
└── Web Worker
    ├── wllama.wasm (compiled llama.cpp)
    ├── WebGPU backend (if available)
    └── WASM SIMD backend (fallback)
```

### Mamba-2 Support

llama.cpp supports Mamba/Mamba-2 architectures (the `mamba2` GGUF architecture
type). wllama is synced to llama.cpp v3.5.0, which includes the critical
SSM_SCAN fix (commit `dc2187d48`, merged 2025-07-04) for Mamba-2 models with
`n_groups > 1`. Without this fix, greedy outputs are silently incorrect.

Evidence that Mamba models work in-browser:
- NVIDIA Nemotron-3-Nano (hybrid Mamba + Attention, 4B) has been demoed running
  locally in-browser on WebGPU.
- llama.cpp's README lists Mamba as a supported architecture.
- No known Mamba-specific issues in wllama's GitHub issues.

Our model (91M params, Mamba-2, Q4_K_M = 55MB) is well within browser
constraints — it's smaller than most models wllama is designed to run.

---

## 4. Model Details

### Morpheus v2 (the model we'll serve)

| Property | Value |
|----------|-------|
| Architecture | Mamba-2 |
| Parameters | 91M (94,137,792) |
| Vocabulary | 4K Unigram (SentencePiece) |
| Training | ~10B tokens, 76K steps, 14.9 hours |
| Best checkpoint | step 74,000 (held-out PPL 7.13) |
| HF repo (safetensors) | `itzune/morpheus` |
| HF repo (GGUF) | `itzune/morpheus-gguf` |

### Available GGUF Files

| File | Size | Bits/weight | Notes |
|------|------|-------------|-------|
| `morpheus-v2-mamba.Q4_K_M.gguf` | 55MB | 4.64 | Default, smallest |
| `morpheus-v2-mamba.Q5_K_M.gguf` | 66MB | 5.60 | Higher quality |

Both are well under the 2GB wllama limit. No splitting needed.

### Inference Semantics

- **No BOS token** — inference must match training (prompts are raw text,
  no special prefix).
- **Greedy decoding** for autocomplete (temperature=0, top_p=1, top_k=0).
- **llama.cpp SP tokenizer divergence**: The SentencePiece tokenizer as loaded
  by llama.cpp diverges slightly from the reference SP model for this vocab.
  In the server demo, we work around this by using token-ID prompts. In the
  browser, wllama handles tokenization internally — we need to verify this
  doesn't cause issues.

---

## 5. Current Demo Architecture (Server-Based)

The existing `morpheus-mamba/demo/` has a Python backend (`server.py`) that
implements significant inference engineering for agglutinative Basque:

### Inference Engineering Strategies (documented in paper §5.5)

1. **Retokenization fallback**: When a typed prefix (e.g. `Kaix`) tokenizes
   into subwords that can't produce the target word (`Kaixo`), we query
   alternate shorter prefixes and filter candidates by the typed string.

2. **Sticky merge (carry-forward)**: When typing the first letter of a new
   word, previous candidates vanish (no prefix match yet). We carry forward
   previous candidates matching the new prefix with a +0.1 probability boost,
   merged with fresh candidates. Pool stores 5, display shows 3.

3. **Top-k exceeding display-k**: Fetch 5 candidates from the server, display
   3 chips. The larger pool gives sticky merge more material to work with.

4. **Next-word candidate extraction**: When the model's continuation starts
   with whitespace, extract the next word as a candidate (the user hasn't
   typed anything yet for this word).

5. **Token-level digit repair**: When greedy generation produces digits or
   artifacts, swap the greedy token with top-k alternatives and regenerate.

6. **From-scratch fallback**: For short prefixes where retokenization can't
   reach a whole-word token, generate from scratch with the prefix as context.

7. **Completion logging with replay**: Log user acceptance events to
   `logs/completions.jsonl` for offline evaluation across checkpoints.

### What Must Be Ported to the Browser

For a faithful port, these strategies need JavaScript implementations using
wllama's API:

| Strategy | wllama API needed | Porting difficulty |
|----------|-------------------|--------------------|
| Greedy completion | `createCompletion()` | Easy |
| Token-level top-k | Low-level logits/sampling API | Medium |
| Retokenization fallback | `tokenize()` / `detokenize()` | Medium |
| Sticky merge | Pure JS (already in frontend) | Easy (already done) |
| Next-word extraction | String parsing on completion | Easy |
| Digit repair | Top-k token alternatives | Medium |
| Completion logging | `fetch()` to external service or localStorage | Easy |
| From-scratch fallback | `createCompletion()` with different prompts | Easy |

---

## 6. Project Plan

### Phase 1: Proof of Concept (GitHub Pages + simple completion)

**Goal**: Validate that wllama can load our Mamba-2 GGUF and produce correct
Basque completions in the browser.

**Deliverables**:
- Static HTML page with a text input
- Loads `morpheus-v2-mamba.Q4_K_M.gguf` from `itzune/morpheus-gguf` via
  `wllama.loadModelFromHF()`
- Calls `wllama.createCompletion()` with greedy params
- Displays completion as ghost text
- Deployed to GitHub Pages

**Success criteria**:
- Model loads without errors
- Typing "Kaixo, zer " produces a sensible Basque continuation
- Inference completes in < 1 second on a modern laptop
- Works on both WebGPU and WASM-SIMD fallback

### Phase 2: Predictive Keyboard (full port)

**Goal**: Port the smartphone-style predictive keyboard to the browser.

**Deliverables**:
- Virtual Basque keyboard UI (QWERTY + ñ, shift, backspace, symbols)
- Suggestion chip strip (top-3, Android-style center alignment)
- Tokenization repair / retokenization fallback in JS
- Sticky merge (already in JS, adapt to wllama)
- Next-word candidate extraction
- Auto-space on chip accept, punctuation attachment, one-shot auto-shift
- Real keyboard support (Tab to accept center suggestion)

**Success criteria**:
- Typing "Kai" shows "Kaixo" as top suggestion
- Typing "beti b" shows "bezala" via fallback
- Sticky merge prevents vanishing on first-letter typing
- Keyboard UX matches the server-based demo

### Phase 3: Polish & Deploy

**Deliverables**:
- Responsive mobile-first design
- Model download progress bar (55MB, cached by browser after first load)
- Fallback messaging if WebGPU unavailable
- Deployed to GitHub Pages at `https://itzune.github.io/morpheus-wasm/`
- README with screenshots and usage instructions

### Phase 4: Evaluation (optional)

**Deliverables**:
- Run the typing simulation (`scripts/simulate_typing.py` methodology)
  against the browser-based demo
- Compare CSR, word accuracy, acceptance rate with the server-based demo
- Document any quality differences between wllama (WASM/WebGPU) and
  native llama.cpp

---

## 7. Technical Decisions

### Static Space vs Docker Space vs GitHub Pages

**Decision**: GitHub Pages.

Docker Spaces require a paid plan (confirmed July 2026). HF Static Spaces are
free but sleep after inactivity and offer less control. GitHub Pages is free,
always-on, and independent from HuggingFace infrastructure. The only trade-off
is no custom HTTP headers (affects multi-threaded WASM, see COOP/COEP below).

### wllama vs transformers.js

**Decision**: wllama.

transformers.js uses ONNX Runtime and requires model conversion to ONNX format.
Mamba-2 is not a standard ONNX architecture and would require custom export.
wllama uses the same llama.cpp backend we already use for the server demo,
loads the same GGUF files we already published, and has the same Mamba-2
support (including the SSM_SCAN fix). No model conversion needed.

### Q4_K_M vs Q5_K_M

**Decision**: Q4_K_M (55MB) as default, with option to switch.

The 11MB difference is significant for browser download time. Q4_K_M quality
is sufficient for autocomplete (we use it as the default in the server demo
too). Q5_K_M can be offered as a "higher quality" toggle.

### WebGPU-first vs WASM-only

**Decision**: WebGPU-first with WASM fallback.

wllama automatically uses WebGPU when available (Chrome 113+, Edge 113+,
Safari 18+) and falls back to WASM SIMD otherwise. We should:
- Default to WebGPU (all layers offloaded, fast)
- Provide clear messaging if WebGPU is unavailable
- Allow manual `n_gpu_layers` adjustment for debugging

### COOP/COEP Headers

wllama's multi-threaded WASM requires `Cross-Origin-Embedder-Policy:
require-corp` and `Cross-Origin-Opener-Policy: same-origin` headers.
**GitHub Pages does not support custom HTTP headers**, so multi-threaded
WASM will not be available by default.

This is acceptable because:
1. **WebGPU** (the primary acceleration) does not require COOP/COEP.
2. wllama automatically falls back to **single-threaded WASM SIMD** when the
   headers are absent — slower but functional.
3. At 91M params, single-threaded inference is adequate for real-time
   autocomplete (a few tokens per keystroke).
4. A **service worker** can inject COOP/COEP headers even on GitHub Pages
   (known technique). This is a Phase 2+ optimization if single-threaded
   performance is insufficient.

The wllama demo (on HF Static Space) works, confirming that headerless hosting
is viable for wllama.

---

## 8. Risks and Unknowns

### Model Loading

**Risk**: wllama's llama.cpp build might not include Mamba-2 support.

**Mitigation**: wllama is synced to llama.cpp v3.5.0 (June 2026), which
includes Mamba-2. The SSM_SCAN fix (`dc2187d48`) was merged July 2025, over a
year ago. Very likely included. But we must test by actually loading the model.

### Tokenizer Divergence

**Risk**: The SentencePiece tokenizer as loaded by llama.cpp/wllama may
diverge from the reference SP model, as observed in the server demo.

**Mitigation**: In the server demo, we work around this using token-ID prompts.
In wllama, we use the text-level API (`createCompletion()`), which handles
tokenization internally. If we see garbled output, we may need to use
wllama's lower-level token APIs (similar to the server's token-ID approach).

### Inference Speed

**Risk**: WASM inference may be too slow for real-time autocomplete.

**Mitigation**: The model is 91M params — very small. wllama is designed for
models up to several GB. With WebGPU, inference should be fast. Even on WASM
SIMD (CPU), 91M params should produce tokens in < 500ms on a modern laptop.
The model generates at most a few tokens per keystroke (we cap at 5-10 tokens
for autocomplete).

### Cross-Origin Headers

**Risk**: GitHub Pages doesn't support COOP/COEP headers, disabling
multi-threaded WASM.

**Mitigation**: wllama falls back to single-threaded automatically. WebGPU
(the primary path) doesn't need these headers. A service worker can inject
them if needed (Phase 2+ optimization). At 91M params, single-threaded
performance is adequate.

### GGUF File Hosting

**Risk**: The GGUF file on HF Hub might have CORS restrictions when loaded
from a GitHub Pages site.

**Mitigation**: HF Hub serves files with permissive CORS (`Access-Control-Allow-Origin: *`).
wllama's `loadModelFromHF()` is designed for cross-origin loading. The wllama
demo (on HF) loads models from HF Hub without issues, and cross-origin loading
from GitHub Pages to HF Hub is a standard pattern. If CORS issues arise, the
GGUF can be hosted in the GitHub repo itself (55MB, under GitHub's 100MB file
limit) or via a CDN.

---

## 9. File Structure (Planned)

```
morpheus-wasm/
├── README.md              ← Project overview
├── RESEARCH.md            ← This file
├── index.html             ← Main entry point (GitHub Pages root)
├── app.js                 ← Application logic (wllama calls, UI)
├── style.css              ← Styling
├── keyboard.js            ← Virtual keyboard + chip logic (Phase 2)
├── inf-engine.js          ← Inference engineering port (Phase 2)
│                            (retokenization fallback, sticky merge, etc.)
├── vendor/
│   └── wllama/            ← wllama WASM + JS (from CDN or npm build)
└── .github/
    └── workflows/
        └── deploy.yml     ← GitHub Actions: deploy to Pages (optional,
                             can also use built-in Pages from main branch)
```

No YAML front matter needed for GitHub Pages (unlike HF Static Spaces). Just
push to `main` and enable Pages in repo settings.

---

## 10. References

- **wllama**: https://github.com/ngxson/wllama
- **wllama demo (Static Space on HF)**: https://huggingface.co/spaces/ngxson/wllama
- **wllama docs**: https://github.ngxson.com/wllama/docs/
- **wllama basic example**: https://github.ngxson.com/wllama/examples/basic/
- **wllama V3 guide (WebGPU)**: https://github.com/ngxson/wllama/blob/master/guides/intro-v3.md
- **HF Docker Spaces docs**: https://huggingface.co/docs/hub/spaces-sdks-docker
- **HF Spaces paywall discussion**: https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580
- **Llamas on the Web (WebGPU blog post)**: https://reeselevine.github.io/llamas-on-the-web/
- **llama.cpp Mamba-2 SSM_SCAN fix**: commit `dc2187d48` (2025-07-04)
- **Morpheus GGUF models**: https://huggingface.co/itzune/morpheus-gguf
- **Morpheus safetensors**: https://huggingface.co/itzune/morpheus
- **Morpheus paper**: `morpheus-on-device-basque-autocompletion.pdf` (in morpheus-mamba repo)

---

## 11. Status

- [x] Research completed (this document)
- [ ] Phase 1: Proof of concept (simple completion in browser)
- [ ] Phase 2: Predictive keyboard (full port)
- [ ] Phase 3: Polish & deploy to GitHub Pages
- [ ] Phase 4: Evaluation (optional)

**Next action**: Build the Phase 1 proof of concept — a minimal HTML page that
loads the model via wllama and produces a Basque completion.
