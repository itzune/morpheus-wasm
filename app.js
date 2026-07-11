// ──────────────────────────────────────────────────────────────────
// Morpheus-WASM — Phase 1 POC
//
// Loads the Morpheus v2 Mamba-2 Basque GGUF (91M params, Q4_K_M, 55MB)
// entirely in the browser via wllama (WebAssembly binding for llama.cpp),
// then produces greedy Basque completions shown as inline ghost text.
//
// No backend server. Inference runs on WebGPU when available, with an
// automatic fallback to single-threaded WASM SIMD.
// ──────────────────────────────────────────────────────────────────

import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.5.1/esm/index.js';

// ── Config ────────────────────────────────────────────────────────
const WLLAMA_VERSION = '3.5.1';
const WLLAMA_CDN = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}`;
// The wllama.wasm binary. wllama's Web Worker code is embedded inside the
// ESM bundle (built via Blob), so this is the only binary asset we point at.
const WLLAMA_WASM = `${WLLAMA_CDN}/src/wasm/wllama.wasm`;

const MODEL_REPO = 'itzune/morpheus-gguf';
// NOTE on the filename: the upstream llama.cpp HF converter writes
// `mamba2.attention.head_count = 0` (commented "unused"). wllama 3.5.1's
// bundled llama.cpp (dd4623a) uses head_count to compute the ssm_in tensor
// width, so 0 makes it reject the model ("wrong shape; expected 768,3200
// got 768,3224"). The patched file below sets head_count = 24
// (= d_inner/head_dim = dt_rank), which is the correct value and loads in
// wllama while staying compatible with newer llama.cpp (which uses dt_rank).
const MODEL_FILE = 'morpheus-v2-mamba.Q4_K_M.wllama.gguf';
// When served from localhost, load the model from the same origin (faster,
// no HF round-trip) — served by test/range_server.py with HTTP Range support.
// Otherwise (e.g. GitHub Pages) fetch from the HuggingFace Hub.
const MODEL_LOCAL_URL = './model/morpheus-v2-mamba.Q4_K_M.wllama.gguf';
const IS_LOCAL = ['localhost', '127.0.0.1', '0.0.0.0'].includes(location.hostname);
const MODEL_SIZE_MB = 55;

// Inference params. The model was trained with NO BOS token (the GGUF carries
// add_bos_token=false), and autocomplete uses greedy decoding (temperature=0).
const N_CTX = 2048;          // context window — ample for autocomplete prompts
const MAX_TOKENS = 8;        // short continuation for ghost text
const DEBOUNCE_MS = 180;     // keystroke debounce before querying the model

// ── DOM ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const editor        = $('editor');
const editorCard    = $('editorCard');
const loadingPanel  = $('loadingPanel');
const loadIcon      = $('loadIcon');
const loadTitle     = $('loadTitle');
const loadStatus    = $('loadStatus');
const progressWrap  = $('progressWrap');
const progressFill  = $('progressFill');
const progressPct   = $('progressPct');
const progressLabel = $('progressLabel');
const loadBadges    = $('loadBadges');
const badgeBackend  = $('badgeBackend');
const loadBanner    = $('loadBanner');
const mBackend      = $('mBackend');
const mLatency      = $('mLatency');
const mTps          = $('mTps');
const debugLog      = $('debugLog');

// ── State ─────────────────────────────────────────────────────────
let wllama = null;
let modelReady = false;
let currentGhost = '';          // ghost text currently displayed (as a selection)
let _showingGhost = false;      // guard against our own input events
let _prefixMatchPending = false; // typing matched ghost[0] — shrink, don't re-query
let debounceTimer = null;
let abortCtrl = null;           // abort in-flight completion when input changes
let lastQuery = '';             // last text we queried (for staleness check)

// ── Logging ───────────────────────────────────────────────────────
function log(...args) {
  console.log('[morpheus]', ...args);
  const line = args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
  debugLog.textContent += line + '\n';
  debugLog.scrollTop = debugLog.scrollHeight;
}

// ── UI helpers ────────────────────────────────────────────────────
function setLoad(title, status) {
  loadTitle.textContent = title;
  if (status !== undefined) loadStatus.innerHTML = status;
}

function setProgress(pct, label) {
  progressWrap.style.display = 'block';
  progressFill.style.width = pct + '%';
  progressPct.textContent = pct + '%';
  if (label) progressLabel.textContent = label;
}

function setBackendBadge(state, text) {
  badgeBackend.className = 'badge ' + state;
  badgeBackend.innerHTML = `<span class="dot"></span>${text}`;
}

function showBanner(type, html) {
  loadBanner.className = 'banner show ' + type;
  loadBanner.innerHTML = html;
}
function clearBanner() { loadBanner.className = 'banner'; loadBanner.innerHTML = ''; }

function failLoading(title, err) {
  log('ERROR:', err);
  loadIcon.className = 'icon-wrap error';
  loadIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
  setLoad(title, 'See the debug log below for details.');
  const msg = (err && err.message) ? err.message : String(err);
  showBanner('error',
    `<b>Failed to load the model.</b><br><code>${msg}</code>` +
    `<br><br>Possible causes: you are offline, the HuggingFace Hub is unreachable, ` +
    `or your browser does not support the required WebAssembly features.`);
  // Keep the editor hidden; show debug log so the user can expand it.
  document.querySelector('details.debug').open = true;
}

// ── Model loading ─────────────────────────────────────────────────
async function init() {
  log('wllama version target:', WLLAMA_VERSION);
  log('libllama version:', Wllama.getLibllamaVersion());

  // Construct the wllama instance. isSupportWebGPU() is an instance method,
  // so we detect the backend right after construction.
  try {
    wllama = new Wllama({ 'default': WLLAMA_WASM });
  } catch (e) {
    failLoading('Could not start wllama', e);
    return;
  }

  let supportsGPU = false;
  try { supportsGPU = wllama.isSupportWebGPU(); } catch (e) { supportsGPU = false; }
  log('WebGPU supported:', supportsGPU);
  if (supportsGPU) {
    setBackendBadge('ok', 'WebGPU available');
  } else {
    setBackendBadge('warn', 'WebGPU unavailable — using CPU (WASM SIMD)');
    showBanner('warn',
      `<b>WebGPU is not available in this browser.</b> Inference will run on the CPU ` +
      `via single-threaded WASM SIMD — slower but functional. For GPU acceleration, ` +
      `use a recent Chrome, Edge, or Safari 18+.`);
  }

  setLoad('Downloading model…',
    IS_LOCAL
      ? `Loading <code>${MODEL_FILE}</code> (${MODEL_SIZE_MB}&nbsp;MB) from the local server.`
      : `Fetching <code>${MODEL_FILE}</code> (${MODEL_SIZE_MB}&nbsp;MB) from
         <code>huggingface.co/${MODEL_REPO}</code>.<br>This happens once — your browser caches it.`);

  try {
    const loadOpts = {
      n_ctx: N_CTX,
      progressCallback: ({ loaded, total }) => {
        if (!total) return;
        const pct = Math.min(100, Math.round((loaded / total) * 100));
        setProgress(pct, 'Downloading model');
        if (pct >= 100) setLoad('Initializing model…', 'Model downloaded. Preparing the Mamba-2 recurrent state…');
      },
    };
    if (IS_LOCAL) {
      await wllama.loadModelFromUrl(MODEL_LOCAL_URL, loadOpts);
    } else {
      await wllama.loadModelFromHF({ repo: MODEL_REPO, file: MODEL_FILE }, loadOpts);
    }
  } catch (e) {
    failLoading('Model download failed', e);
    return;
  }

  // ── Verify model actually loaded ──
  // loadModelFromHF/Url can return without throwing even when the worker fails
  // to load the model (e.g. tensor-shape mismatch). The metadata then comes back
  // all zeros — so we check and fail loudly instead of showing a broken editor.
  let meta, addBos, eos;
  try {
    meta = wllama.getModelMetadata();
    addBos = wllama.mustAddBosToken();
    eos = wllama.getEOS();
  } catch (e) {
    failLoading('Model metadata unavailable', e);
    return;
  }
  if (!meta?.hparams?.nVocab || meta.hparams.nVocab <= 0) {
    failLoading('Model failed to load',
      new Error('The model did not load correctly (vocab size is 0). ' +
        'This usually means a tensor-shape mismatch between the GGUF and the ' +
        'wllama/llama.cpp build. See the debug log for the underlying error.'));
    return;
  }
  try {
    const arch = meta.meta?.['general.architecture'] || 'unknown';
    const name = meta.meta?.['general.name'] || 'unknown';
    log('model name:', name);
    log('architecture:', arch,
        '| vocab:', meta.hparams?.nVocab,
        '| layers:', meta.hparams?.nLayer,
        '| embd:', meta.hparams?.nEmbd,
        '| ctx_train:', meta.hparams?.nCtxTrain);
    log('add_bos_token:', addBos, '| EOS id:', eos, '| threads:', wllama.getNumThreads());
    if (arch !== 'mamba2' && arch !== 'mamba') {
      showBanner('warn',
        `<b>Unexpected architecture: <code>${arch}</code></b> (expected <code>mamba2</code>). ` +
        `The model may not run correctly.`);
    }
  } catch (e) {
    log('metadata query failed:', e);
  }

  // BOS sanity check: our model MUST NOT add a BOS token (training used raw text).
  if (addBos) {
    showBanner('warn',
      `<b>Unexpected: model wants a BOS token.</b> This model was trained without one. ` +
      `Completions may be slightly off. (add_bos_token = true)`);
  }

  // Switch UI from loading panel to editor.
  modelReady = true;
  loadingPanel.style.display = 'none';
  editorCard.classList.remove('hidden');
  const archStr = (meta && meta.meta?.['general.architecture']) || 'mamba2';
  const backendStr = supportsGPU ? 'WebGPU' : `WASM-SIMD · ${wllama.getNumThreads()} thread`;
  mBackend.textContent = `${archStr} · ${backendStr}`;
  log('ready. backend =', backendStr);
  editor.focus();

  // Wire up interaction.
  editor.addEventListener('input', onInput);
  editor.addEventListener('keydown', onKeyDown);
  editor.addEventListener('keyup', onKeyUp);
  editor.addEventListener('mousedown', onMouseDown);
  editor.addEventListener('mouseup', onMouseUp);
  document.querySelectorAll('#exampleChips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      editor.value = chip.dataset.text;
      editor.focus();
      const end = editor.value.length;
      editor.setSelectionRange(end, end);
      onInput();
    });
  });

  // Trigger an initial completion so the user immediately sees it working.
  editor.value = 'Kaixo, zer ';
  editor.setSelectionRange(editor.value.length, editor.value.length);
  onInput();
}

// ── Completion ────────────────────────────────────────────────────
async function complete(text) {
  // Abort any in-flight request — only the latest keystroke matters.
  if (abortCtrl) { try { abortCtrl.abort(); } catch (e) {} }
  abortCtrl = new AbortController();

  const t0 = performance.now();
  try {
    const response = await wllama.createCompletion({
      prompt: text,
      max_tokens: MAX_TOKENS,
      temperature: 0,   // greedy decoding
      top_p: 1.0,
      stream: false,
      abortSignal: abortCtrl.signal,
    });
    const elapsed = performance.now() - t0;
    const out = response.choices?.[0]?.text ?? '';
    const timings = response.timings || {};
    return { text: out, elapsed, timings };
  } catch (e) {
    if (e && (e.name === 'AbortError' || /abort/i.test(e.message || ''))) return null; // superseded
    log('completion error:', e);
    return null;
  }
}

// ── Ghost text (selection-based, like the sibling demo) ───────────
function showGhost(suggestion) {
  if (!suggestion) { currentGhost = ''; return; }
  const start = editor.selectionStart;
  const end = editor.selectionEnd;
  if (start !== end) return; // don't clobber an existing selection
  currentGhost = suggestion;
  _showingGhost = true;
  editor.value = editor.value.substring(0, start) + suggestion + editor.value.substring(end);
  editor.setSelectionRange(start, start + suggestion.length);
  editor.classList.add('ghost-active');
  _showingGhost = false;
}

function clearGhost() {
  if (!currentGhost) return;
  currentGhost = '';
  const start = editor.selectionStart;
  const end = editor.selectionEnd;
  if (start !== end) {
    editor.value = editor.value.substring(0, start) + editor.value.substring(end);
    editor.setSelectionRange(start, start);
  }
  editor.classList.remove('ghost-active');
}

function getPrefix() { return editor.value.substring(0, editor.selectionStart); }

// ── Input handling ────────────────────────────────────────────────
function onInput() {
  if (_showingGhost) return;

  // Prefix-match shrink: user typed the first char of the ghost → shrink it,
  // no need to re-query the model.
  if (_prefixMatchPending) {
    _prefixMatchPending = false;
    currentGhost = currentGhost.slice(1);
    if (currentGhost) { showGhost(currentGhost); return; }
    // Ghost fully consumed — fall through to re-query.
  }

  // Only autocomplete when the cursor is at the end of the text.
  if (editor.selectionStart !== editor.value.length) { clearGhost(); return; }

  const prefix = getPrefix();
  clearGhost();
  if (!prefix.trim()) {
    mLatency.textContent = '—'; mTps.textContent = '—';
    return;
  }

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => doComplete(prefix), DEBOUNCE_MS);
}

async function doComplete(prefix) {
  if (!modelReady) return;
  lastQuery = prefix;
  const result = await complete(prefix);
  if (!result) return;
  // Staleness guard: user may have typed more since we sent this query.
  if (lastQuery !== prefix) return;
  if (getPrefix() !== prefix) { clearGhost(); return; }

  // Update latency / throughput metrics.
  mLatency.textContent = result.elapsed.toFixed(0) + ' ms';
  const tps = result.timings?.predicted_per_second;
  if (tps) mTps.textContent = tps.toFixed(1);
  else if (result.timings?.predicted_n && result.timings?.predicted_ms) {
    mTps.textContent = (result.timings.predicted_n / (result.timings.predicted_ms / 1000)).toFixed(1);
  } else {
    mTps.textContent = (MAX_TOKENS / (result.elapsed / 1000)).toFixed(1);
  }

  let suggestion = result.text;
  // Trim trailing whitespace/newlines that would make the ghost look odd.
  suggestion = suggestion.replace(/\s+$/, '');
  if (suggestion) showGhost(suggestion);
}

// ── Keyboard interaction ──────────────────────────────────────────
function onKeyDown(e) {
  // Tab → accept the ghost (move cursor to end of the selection).
  if (e.key === 'Tab') {
    if (editor.selectionStart !== editor.selectionEnd) {
      e.preventDefault();
      const end = editor.selectionEnd;
      editor.setSelectionRange(end, end);
      editor.classList.remove('ghost-active');
      currentGhost = '';
    }
    return;
  }
  // Escape → dismiss ghost.
  if (e.key === 'Escape') {
    if (currentGhost) { e.preventDefault(); clearGhost(); }
    return;
  }
  // If a ghost is showing and the user types its first char, shrink it
  // instead of clearing + re-querying (smoother UX, fewer model calls).
  if (editor.selectionStart !== editor.selectionEnd && currentGhost) {
    if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.key.startsWith('Arrow') && e.key.length === 1) {
      if (e.key === currentGhost[0]) {
        _prefixMatchPending = true;
        const start = editor.selectionStart;
        editor.value = editor.value.substring(0, start) + editor.value.substring(editor.selectionEnd);
        editor.setSelectionRange(start, start);
        return; // let the browser type the char naturally, then onInput shrinks the ghost
      }
      clearGhost(); // divergent char — clear, let it type, re-query
    }
  }
}

function onKeyUp(e) {
  if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) {
    clearGhost();
  }
}

function onMouseDown() {
  // Clear ghost on click so the user can reposition the cursor freely.
  if (editor.selectionStart !== editor.selectionEnd) clearGhost();
}
function onMouseUp() {
  // After a click at end of text, refresh the suggestion.
  if (editor.selectionStart === editor.selectionEnd &&
      editor.selectionStart === editor.value.length) {
    onInput();
  }
}

// ── Go ────────────────────────────────────────────────────────────
init().catch(e => failLoading('Initialization failed', e));
