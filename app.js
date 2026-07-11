// ──────────────────────────────────────────────────────────────────
// Morpheus-WASM — Ghost-text autocomplete mode
//
// Loads a 91M-parameter Mamba-2 GGUF (Q4_K_M, 55 MB) entirely client-side
// via wllama (WebAssembly llama.cpp), then produces greedy Basque
// completions shown as inline ghost text.  No backend server.
//
// Shared model loading + token helpers live in morpheus.js.
// Inference strategies ported from the Python demo (demo/server.py):
//   - Digit-token repair via n_probs logprobs + re-generate from swap point
//   - filter_suggestion (strip artifacts, collapse punct, reject pure-punct)
//   - ghost_suffix (overlap computation, punct dedup)
//   - Byte-fallback garbage detection + retokenization fallback
//   - Confidence threshold (min 18 %)
// Tokenization fidelity is guaranteed by the UGM-patched GGUF (Viterbi
// algorithm matching the reference SentencePiece unigram model).
// ──────────────────────────────────────────────────────────────────

import {
  CONFIG, IS_LOCAL, MODEL_LOCAL, MODEL_SIZE_MB,
  loadModel, complete,
  tokenHasDigit, hasByteFallbackGarbage, isPurePunct,
  extractCurrentWord, extractFirstWord,
  filterSuggestion,
} from './morpheus.js';

// ── Mode-specific config ──────────────────────────────────────────
const MAX_TOKENS     = 3;       // short continuation for ghost text
const DEBOUNCE_MS    = 150;     // keystroke debounce
const MIN_CONFIDENCE = 0.18;    // don't show ghost below 18 % confidence

// ── DOM ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const editor       = $('editor');
const editorCard   = $('editorCard');
const loadingPanel = $('loadingPanel');
const loadIcon     = $('loadIcon');
const loadTitle    = $('loadTitle');
const loadStatus   = $('loadStatus');
const progressWrap = $('progressWrap');
const progressFill = $('progressFill');
const progressPct  = $('progressPct');
const progressLbl  = $('progressLabel');
const badgeBackend = $('badgeBackend');
const loadBanner   = $('loadBanner');
const mBackend     = $('mBackend');
const mLatency     = $('mLatency');
const mConfidence  = $('mConfidence');
const debugLog     = $('debugLog');

// ── State ─────────────────────────────────────────────────────────
let modelReady = false;
let currentGhost = '';
let _showingGhost = false;
let _prefixMatchPending = false;
let debounceTimer = null;
let abortCtrl = null;
let lastQuery = '';

// ══════════════════════════════════════════════════════════════════
//  Logging & UI helpers
// ══════════════════════════════════════════════════════════════════
function log(...args) {
  console.log('[morpheus]', ...args);
  const line = args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
  debugLog.textContent += line + '\n';
  debugLog.scrollTop = debugLog.scrollHeight;
}

function setLoad(title, status) {
  loadTitle.textContent = title;
  if (status !== undefined) loadStatus.innerHTML = status;
}

function setProgress(pct, label) {
  progressWrap.style.display = 'block';
  progressFill.style.width = pct + '%';
  progressPct.textContent = pct + '%';
  if (label) progressLbl.textContent = label;
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
  document.querySelector('details.debug').open = true;
}

// ══════════════════════════════════════════════════════════════════
//  Model loading
// ══════════════════════════════════════════════════════════════════
async function init() {
  try {
    const { supportsGPU, arch } = await loadModel({
      onLog: log,
      onProgress: (pct) => setProgress(pct, 'Downloading model'),
      onStatus: (title, html) => setLoad(title, html),
      onBackendBadge: setBackendBadge,
      onBanner: (type, html) => { if (type === 'warn') showBanner('warn', html); },
    });

    // ── Switch to editor ──
    modelReady = true;
    loadingPanel.style.display = 'none';
    editorCard.classList.remove('hidden');
    const backendStr = supportsGPU ? 'WebGPU' : `WASM-SIMD`;
    mBackend.textContent = `${arch} · ${backendStr}`;
    log('ready. backend =', backendStr);
    editor.focus();

    // Wire up interaction
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

    // Trigger an initial completion
    editor.value = 'Kaixo, zer ';
    editor.setSelectionRange(editor.value.length, editor.value.length);
    onInput();
  } catch (e) {
    failLoading('Initialization failed', e);
  }
}

// ══════════════════════════════════════════════════════════════════
//  Inference — digit-token repair (ghost-mode specific)
// ══════════════════════════════════════════════════════════════════

/**
 * Extract average confidence from logprobs.
 * Excludes EOS/stop tokens (logprob near 0.0 → prob near 1.0) which
 * are not real predictions and would inflate the average.
 */
function computeConfidence(logprobs) {
  const content = logprobs?.content;
  if (!content || !content.length) return 1.0;
  const real = content
    .filter(c => c.logprob < -0.01)
    .map(c => Math.exp(c.logprob));
  if (real.length === 0) return 0.0;
  return real.reduce((a, b) => a + b, 0) / real.length;
}

/**
 * Extract next-token candidates from logprobs position 0.
 * Returns [{text, prob}, ...] — digit and garbage tokens are skipped.
 */
function extractCandidates(logprobs, topK = 3) {
  const content = logprobs?.content;
  if (!content?.[0]?.top_logprobs) return [];
  const candidates = [];
  const seen = new Set();
  for (const tok of content[0].top_logprobs) {
    const text = tok.token || '';
    if (!text.trim()) continue;
    if (tokenHasDigit(text)) continue;
    if (hasByteFallbackGarbage(text)) continue;
    if (seen.has(text)) continue;
    seen.add(text);
    candidates.push({ text, prob: Math.exp(tok.logprob) });
    if (candidates.length >= topK) break;
  }
  return candidates;
}

/**
 * Generate completion with digit-token repair.
 *
 * Strategy (ported from server.py::_generate_with_repair):
 *   1. Generate max_tokens with top-k logprobs per position.
 *   2. Walk the greedy path. If a token contains digits, swap it for the
 *      best non-digit alternative from top_logprobs.
 *   3. If no swap was needed → return original content (zero extra cost).
 *   4. If a swap changed the path → re-generate from the swap point so
 *      subsequent tokens are correctly conditioned.
 *   5. If a digit token has NO non-digit alternative: if it's the first
 *      token, return empty; otherwise truncate at the good prefix.
 *
 * Returns { suggestion, confidence, candidates, latency }.
 */
async function generateWithRepair(prompt, maxTokens) {
  const t0 = performance.now();

  // ── First call ──
  const resp = await complete(prompt, { maxTokens, abortSignal: abortCtrl?.signal });

  const choice   = resp.choices?.[0] ?? {};
  const content  = choice.text ?? '';
  const logprobs = choice.logprobs;

  const candidates  = extractCandidates(logprobs);
  const confidence  = computeConfidence(logprobs);
  const latency     = performance.now() - t0;

  const lpContent = logprobs?.content;
  if (!lpContent?.length) {
    // No logprobs → return raw content (digit repair disabled)
    return { suggestion: content, confidence, candidates, latency };
  }

  // ── Walk the greedy path, looking for digit tokens to repair ──
  const repairedTexts = [];
  let firstSwap = -1;

  for (let i = 0; i < lpContent.length; i++) {
    const chosen = lpContent[i].token || '';
    const needsRepair = tokenHasDigit(chosen) || (i === 0 && !chosen.trim());

    if (needsRepair) {
      // Find best non-digit alternative (top_logprobs sorted by logprob desc)
      const alts = lpContent[i].top_logprobs || [];
      let foundAlt = false;

      for (const alt of alts) {
        const altToken = alt.token || '';
        if (!tokenHasDigit(altToken) && altToken.trim() && altToken !== chosen) {
          repairedTexts.push(altToken);
          if (firstSwap === -1) firstSwap = i;
          foundAlt = true;
          break;
        }
      }

      if (!foundAlt) {
        // Can't repair this position
        if (i === 0) {
          return { suggestion: '', confidence, candidates, latency };
        }
        // Truncate at good prefix
        return { suggestion: repairedTexts.join(''), confidence, candidates, latency };
      }
    } else {
      repairedTexts.push(chosen);
    }
  }

  if (firstSwap === -1) {
    // No digit tokens → return original content (zero extra cost)
    return { suggestion: content, confidence, candidates, latency };
  }

  // ── Re-generate from the first swap point ──
  const prefixText = repairedTexts.slice(0, firstSwap + 1).join('');
  const remaining  = maxTokens - (firstSwap + 1);

  if (remaining <= 0) {
    return { suggestion: prefixText, confidence, candidates, latency };
  }

  const regen = await complete(prompt + prefixText, { maxTokens: remaining });
  const regenContent = regen.choices?.[0]?.text ?? '';
  return {
    suggestion: prefixText + regenContent,
    confidence,
    candidates,
    latency: performance.now() - t0,
  };
}

// ══════════════════════════════════════════════════════════════════
//  Ghost suffix (overlap computation)
// ══════════════════════════════════════════════════════════════════

/**
 * Compute the ghost suffix to display, given the typed prefix and the
 * model's suggestion.
 *
 * If the user already typed part of the prediction, only show the
 * non-typed suffix. Also deduplicates punctuation at the boundary.
 */
function ghostSuffix(prefix, ctx, suggestion) {
  if (!suggestion) return '';

  // If the suggestion starts with the context tail, show only the suffix
  const ctxLower = ctx.toLowerCase();
  const sugLower = suggestion.toLowerCase();

  if (sugLower.startsWith(ctxLower)) {
    return suggestion.slice(ctx.length);
  }

  // Check overlap: typed text ends with part of the suggestion start
  for (let i = Math.min(prefix.length, suggestion.length); i >= 1; i--) {
    if (prefix.slice(-i).toLowerCase() === suggestion.slice(0, i).toLowerCase()) {
      return suggestion.slice(i);
    }
  }

  return suggestion;
}

// ══════════════════════════════════════════════════════════════════
//  Retokenization fallback (byte-fallback garbage rescue)
// ══════════════════════════════════════════════════════════════════

/**
 * Simplified retokenization fallback for byte-fallback garbage.
 *
 * When the suggestion contains non-Latin chars (byte-fallback garbage),
 * the typed prefix's tokenization may be incompatible. Try progressively
 * shorter prefixes to land on a compatible path.
 *
 * Returns a clean suggestion string, or '' if nothing usable found.
 */
async function keyboardFallback(text) {
  const [textBeforeWord, currentWord] = extractCurrentWord(text);
  if (!currentWord) return '';

  const maxFallback = Math.min(2, currentWord.length - 1);
  for (let fallback = 0; fallback <= maxFallback; fallback++) {
    const shorterLen = currentWord.length - fallback;
    if (shorterLen < 1) break;
    const shorterWord = currentWord.slice(0, shorterLen);
    const prefix = textBeforeWord + shorterWord;

    let resp;
    try {
      resp = await complete(prefix, { maxTokens: 5, abortSignal: abortCtrl?.signal });
    } catch { continue; }

    const content = resp.choices?.[0]?.text ?? '';
    if (!content || hasByteFallbackGarbage(content)) continue;

    // Greedy multi-token word completion
    const firstWord = extractFirstWord(content);
    if (firstWord) {
      const fullWord = shorterWord + firstWord;
      if (fullWord.startsWith(currentWord)
          && fullWord.length >= currentWord.length
          && !tokenHasDigit(fullWord)
          && !hasByteFallbackGarbage(fullWord)) {
        // Return only the untyped suffix as ghost
        return fullWord.slice(currentWord.length);
      }
    }

    // Top-k single-token alternatives at position 0
    const alts = resp.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs || [];
    for (const alt of alts) {
      const token = alt.token || '';
      if (!token.trim() || tokenHasDigit(token) || hasByteFallbackGarbage(token)) continue;
      // A whitespace-prefixed token means model thinks current word is complete
      if (/\s/.test(token[0])) {
        const nextWord = token.trim();
        if (nextWord && !hasByteFallbackGarbage(nextWord)) return ' ' + nextWord;
        continue;
      }
      const fullWord = shorterWord + token;
      if (fullWord.startsWith(currentWord) && fullWord.length >= currentWord.length) {
        return fullWord.slice(currentWord.length);
      }
    }
  }

  return '';
}

// ══════════════════════════════════════════════════════════════════
//  Completion orchestration (mirrors demo's autocomplete_greedy flow)
// ══════════════════════════════════════════════════════════════════

async function doComplete(prefix) {
  if (!modelReady) return;
  lastQuery = prefix;

  // smart_context: currently pass-through (full text)
  const ctx = prefix;

  // Generate with digit repair
  let result;
  try {
    result = await generateWithRepair(ctx, MAX_TOKENS);
  } catch (e) {
    if (e?.name === 'AbortError' || /abort/i.test(e?.message || '')) return;
    log('completion error:', e);
    return;
  }

  // Staleness guard
  if (lastQuery !== prefix) return;
  if (getPrefix() !== prefix) { clearGhost(); return; }

  // Update metrics
  mLatency.textContent = Math.round(result.latency) + ' ms';
  mConfidence.textContent = (result.confidence * 100).toFixed(1) + '%';

  let suggestion = result.suggestion;

  // filter_suggestion: strip artifacts, collapse punct, reject pure-punct
  suggestion = filterSuggestion(suggestion);

  // If suggestion is pure punct and user already ends with punct, drop it
  if (isPurePunct(suggestion) && ctx.trim()
      && CONFIG.PUNCT_CHARS.includes(ctx.trim().slice(-1))) {
    suggestion = '';
  }

  // Byte-fallback garbage fallback: retokenization rescue
  if (suggestion && hasByteFallbackGarbage(suggestion)) {
    log('byte-fallback garbage detected, trying retokenization fallback…');
    const rescued = await keyboardFallback(prefix);
    suggestion = rescued || '';
  }

  // ghost_suffix: overlap computation + punct dedup
  const ghost = ghostSuffix(prefix, ctx, suggestion);

  // Confidence threshold
  if (result.confidence < MIN_CONFIDENCE) {
    clearGhost();
    return;
  }

  if (ghost) showGhost(ghost);
  else clearGhost();
}

// ══════════════════════════════════════════════════════════════════
//  Ghost text (selection-based, like the sibling demo)
// ══════════════════════════════════════════════════════════════════

function showGhost(suggestion) {
  if (!suggestion) { currentGhost = ''; return; }
  const start = editor.selectionStart;
  const end = editor.selectionEnd;
  if (start !== end) return;
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

// ══════════════════════════════════════════════════════════════════
//  Input handling
// ══════════════════════════════════════════════════════════════════

function onInput() {
  if (_showingGhost) return;

  // Prefix-match shrink: user typed the first char of the ghost → shrink
  if (_prefixMatchPending) {
    _prefixMatchPending = false;
    currentGhost = currentGhost.slice(1);
    if (currentGhost) { showGhost(currentGhost); return; }
    // Ghost fully consumed — fall through to re-query
  }

  // Only autocomplete when cursor is at the end of the text
  if (editor.selectionStart !== editor.value.length) { clearGhost(); return; }

  const prefix = getPrefix();
  clearGhost();
  if (!prefix.trim()) {
    mLatency.textContent = '—';
    mConfidence.textContent = '—';
    return;
  }

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    // Abort any in-flight request — only the latest keystroke matters
    if (abortCtrl) { try { abortCtrl.abort(); } catch {} }
    abortCtrl = new AbortController();
    doComplete(prefix);
  }, DEBOUNCE_MS);
}

// ══════════════════════════════════════════════════════════════════
//  Keyboard interaction
// ══════════════════════════════════════════════════════════════════

function onKeyDown(e) {
  // Tab → accept ghost
  if (e.key === 'Tab') {
    if (editor.selectionStart !== editor.selectionEnd) {
      e.preventDefault();
      const end = editor.selectionEnd;
      editor.setSelectionRange(end, end);
      editor.classList.remove('ghost-active');
      currentGhost = '';
      // Fetch next prediction after accepting
      onInput();
    }
    return;
  }
  // Escape → dismiss ghost
  if (e.key === 'Escape') {
    if (currentGhost) { e.preventDefault(); clearGhost(); }
    return;
  }
  // If ghost is showing and user types its first char, shrink it
  if (editor.selectionStart !== editor.selectionEnd && currentGhost) {
    if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.key.startsWith('Arrow') && e.key.length === 1) {
      if (e.key === currentGhost[0]) {
        _prefixMatchPending = true;
        const start = editor.selectionStart;
        editor.value = editor.value.substring(0, start) + editor.value.substring(editor.selectionEnd);
        editor.setSelectionRange(start, start);
        return;
      }
      clearGhost();
    }
  }
}

function onKeyUp(e) {
  if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) clearGhost();
}

function onMouseDown() {
  if (editor.selectionStart !== editor.selectionEnd) clearGhost();
}
function onMouseUp() {
  if (editor.selectionStart === editor.selectionEnd
      && editor.selectionStart === editor.value.length) {
    onInput();
  }
}

// ── Go ────────────────────────────────────────────────────────────
init();
