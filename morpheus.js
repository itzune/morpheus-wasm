// ──────────────────────────────────────────────────────────────────
// morpheus.js — shared module: model loading + inference helpers
//
// Imported by both ghost-mode (app.js) and keyboard-mode (keyboard.js).
// Encapsulates wllama lifecycle, token-level helpers, and the
// keyboard-candidates algorithm ported from demo/server.py.
// ──────────────────────────────────────────────────────────────────

import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/index.js';

// ── Config ────────────────────────────────────────────────────────
export const CONFIG = {
  WLLAMA_VERSION: '3.6.1',
  WLLAMA_CDN:     'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1',
  MODEL_REPO:     'itzune/morpheus-gguf',
  MODEL_FILE:     'morpheus-v2-mamba.Q4_K_M.ugm.gguf',
  N_CTX:          2048,
  N_PARALLEL:     4,      // parallel inference slots (needs wllama 3.6.0+, PR #270)
  N_THREADS:      Math.max(1, navigator.hardwareConcurrency || 4),
  N_PROBS:        5,      // top-k logprobs per position
  PENALTY_REPEAT: 1.1,
  PUNCT_CHARS:    '.!,?;:()[]{}',
  // ── Experimental latency knobs (Tier 2 measurement) ──
  // cache_prompt: ask llama-server to reuse the slot's cached prompt prefix
  //   across calls. For transformers this is a big win; for Mamba the
  //   *append* case (ghost mode: each keystroke = prev prompt + 1 token)
  //   may hit, but the *truncate* case (keyboard fallback prefixes) is
  //   upstream-broken (llama.cpp #19264) → clears & recomputes (safe, no gain).
  //   Set to true and watch usage.prompt_tokens_details.cached_tokens.
  CACHE_PROMPT:      true,
  // timings_per_token: request per-token ms breakdown in the response.
  TIMINGS_PER_TOKEN: true,
};

export const IS_LOCAL      = ['localhost', '127.0.0.1', '0.0.0.0'].includes(location.hostname);
export const MODEL_LOCAL   = './model/' + CONFIG.MODEL_FILE;
export const MODEL_SIZE_MB = 55;
export const WLLAMA_WASM   = `${CONFIG.WLLAMA_CDN}/src/wasm/wllama.wasm`;

// ── State ─────────────────────────────────────────────────────────
let wllama = null;
let meta = null;

export function getWllama() { return wllama; }
export function getModelMeta() { return meta; }
export function isReady() { return wllama !== null && meta !== null; }

// ══════════════════════════════════════════════════════════════════
//  Token / text helpers (shared between modes)
// ══════════════════════════════════════════════════════════════════

/** Check if a token's text contains any digit character. */
export function tokenHasDigit(text) {
  return /[0-9]/.test(text);
}

/**
 * Check if text contains non-Latin characters (byte-fallback garbage).
 * Basque uses Latin script: legitimate non-ASCII chars (ñ, ç, ü, á, …)
 * fall within Latin-1 Supplement (U+0080–U+00FF). Anything above U+00FF
 * can only come from byte-fallback bytes forming unintended UTF-8.
 */
export function hasByteFallbackGarbage(text) {
  for (const c of text) if (c.codePointAt(0) > 0xFF) return true;
  return false;
}

/** Check if string is only punctuation, no word characters. */
export function isPurePunct(s) {
  const stripped = s.trim();
  return stripped.length > 0 && [...stripped].every(c => CONFIG.PUNCT_CHARS.includes(c));
}

/**
 * Split text into (text_before_word, current_word) at the cursor.
 * Cursor is at end of `text`. If text ends with whitespace,
 * current_word is empty (next-word prediction mode).
 */
export function extractCurrentWord(text) {
  if (!text || /\s$/.test(text)) return [text, ''];
  let i = text.length - 1;
  while (i >= 0 && !/\s/.test(text[i])) i--;
  return [text.slice(0, i + 1), text.slice(i + 1)];
}

/**
 * Extract the first word from generated content.
 * Returns '' if content starts with whitespace (model thinks prefix
 * is a complete word and is starting a new one).
 * Strips trailing punctuation.
 */
export function extractFirstWord(content) {
  if (!content || /\s/.test(content[0])) return '';
  let word = '';
  for (const c of content) {
    if (/\s/.test(c)) break;
    word += c;
  }
  return word.replace(new RegExp(`[${CONFIG.PUNCT_CHARS}]+$`), '');
}

/**
 * filter_suggestion — strip artifacts, collapse punct, reject pure-punct.
 * Ported from demo/server.py::_filter_suggestion.
 */
export function filterSuggestion(suggestion) {
  if (!suggestion) return suggestion;

  // Replace ▁ markers with spaces, strip replacement chars
  let text = suggestion.replace(/▁/g, ' ').replace(/\uFFFD/g, '');

  // Preserve leading whitespace, strip trailing
  text = text.replace(/\s+$/, '');

  // Collapse internal whitespace
  text = text.replace(/\s+/g, ' ');

  // Collapse runs of 2+ punctuation chars to single first char
  const pEsc = CONFIG.PUNCT_CHARS.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  text = text.replace(new RegExp(`([${pEsc}])[${pEsc}]+`, 'g'), '$1');

  // Strip trailing space-then-punct sequences: "hello . ," → "hello"
  while (new RegExp(` [${pEsc}]+$`).test(text)) {
    text = text.replace(new RegExp(` [${pEsc}]+$`), '');
  }

  // If the result has no word characters (only punct/whitespace), reject it.
  const stripped = text.trim();
  if (stripped) {
    const hasWord = /[a-zA-ZñÑçÇüÜáéíóúÁÉÍÓÚ]/.test(stripped);
    if (!hasWord && !/^[.?!]$/.test(stripped)) return '';
  }

  return text;
}

// ══════════════════════════════════════════════════════════════════
//  Model loading
// ══════════════════════════════════════════════════════════════════

/**
 * Load the Mamba-2 GGUF model via wllama.
 *
 * @param {object} callbacks — DOM-specific hooks:
 *   - onLog(...args)         — debug logging
 *   - onProgress(pct)        — download progress (0–100)
 *   - onStatus(title, html)  — loading stage update
 *   - onBackendBadge(s, txt) — WebGPU badge update
 *   - onBanner(type, html)   — warning/error banner
 * @returns {Promise<{wllama, meta, supportsGPU, addBos, arch}>}
 */
export async function loadModel(callbacks = {}) {
  const {
    onLog = () => {},
    onProgress = () => {},
    onStatus = () => {},
    onBackendBadge = () => {},
    onBanner = () => {},
  } = callbacks;

  onLog('wllama version:', CONFIG.WLLAMA_VERSION);
  onLog('libllama version:', Wllama.getLibllamaVersion());

  wllama = new Wllama({ 'default': WLLAMA_WASM });

  let supportsGPU = false;
  try { supportsGPU = wllama.isSupportWebGPU(); } catch { supportsGPU = false; }
  onLog('WebGPU supported:', supportsGPU);

  if (supportsGPU) {
    onBackendBadge('ok', 'WebGPU available');
  } else {
    onBackendBadge('warn', 'WebGPU unavailable — using CPU (WASM SIMD)');
    onBanner('warn',
      `<b>WebGPU is not available.</b> Inference will run on the CPU via ` +
      `single-threaded WASM SIMD — slower but functional. For GPU acceleration, ` +
      `use a recent Chrome, Edge, or Safari 18+.`);
  }

  onStatus('Downloading model…',
    IS_LOCAL
      ? `Loading <code>${CONFIG.MODEL_FILE}</code> (${MODEL_SIZE_MB}&nbsp;MB) from the local server.`
      : `Fetching <code>${CONFIG.MODEL_FILE}</code> (${MODEL_SIZE_MB}&nbsp;MB) from
         <code>huggingface.co/${CONFIG.MODEL_REPO}</code>.<br>This happens once — your browser caches it.`);

  const loadOpts = {
    n_ctx: CONFIG.N_CTX,
    n_threads: CONFIG.N_THREADS,
    n_parallel: CONFIG.N_PARALLEL,
    // Mamba is recurrent: each parallel slot must own its SSM state.
    // kv_unified:false gives every slot its own cache (n_ctx / n_parallel)
    // instead of sharing one — required for parallel recurrent inference.
    kv_unified: false,
    n_batch: 512,
    n_ubatch: 512,
    // Backend toggle: ?cpu=1 forces CPU (WASM SIMD) by offloading 0 layers,
    // for A/B comparison against WebGPU. Without the param, wllama auto-
    // offloads all layers when WebGPU is available.
    n_gpu_layers: new URLSearchParams(location.search).has('cpu') ? 0 : undefined,
    progressCallback: ({ loaded, total }) => {
      if (!total) return;
      const pct = Math.min(100, Math.round((loaded / total) * 100));
      onProgress(pct);
      if (pct >= 100) {
        onStatus('Initializing model…',
          'Model downloaded. Preparing the Mamba-2 recurrent state…');
      }
    },
  };

  if (IS_LOCAL) {
    await wllama.loadModelFromUrl(MODEL_LOCAL, loadOpts);
  } else {
    await wllama.loadModelFromHF({ repo: CONFIG.MODEL_REPO, file: CONFIG.MODEL_FILE }, loadOpts);
  }

  // ── Verify model actually loaded ──
  meta = wllama.getModelMetadata();
  const addBos = wllama.mustAddBosToken();

  if (!meta?.hparams?.nVocab || meta.hparams.nVocab <= 0) {
    throw new Error('The model did not load correctly (vocab size is 0). ' +
      'This usually means a tensor-shape mismatch.');
  }

  const arch = meta.meta?.['general.architecture'] || 'unknown';
  onLog('architecture:', arch,
    '| vocab:', meta.hparams?.nVocab,
    '| layers:', meta.hparams?.nLayer,
    '| add_bos:', addBos);

  if (arch !== 'mamba2' && arch !== 'mamba') {
    onBanner('warn', `<b>Unexpected architecture: <code>${arch}</code></b> (expected mamba2).`);
  }
  if (addBos) {
    onBanner('warn', `<b>Model wants a BOS token.</b> This model was trained without one.`);
  }

  return { wllama, meta, supportsGPU, addBos, arch };
}

// ══════════════════════════════════════════════════════════════════
//  Completion wrapper
// ══════════════════════════════════════════════════════════════════

/**
 * Greedy completion with top-k logprobs.
 * Returns the raw wllama response (OAI-style: choices[0].text + .logprobs).
 *
 * wllama logprobs format (chat-completion style):
 *   choices[0].logprobs.content[i] = {
 *     token: string,          // decoded token text
 *     logprob: number,        // log P(token | prefix)
 *     top_logprobs: [         // top-k alternatives at this position
 *       { token: string, logprob: number, bytes?: number[] }, ...
 *     ]
 *   }
 */
export async function complete(prompt, { maxTokens = 3, abortSignal = null } = {}) {
  return wllama.createCompletion({
    prompt,
    max_tokens: maxTokens,
    temperature: 0,
    top_p: 1.0,
    top_k: 0,
    penalty_repeat: CONFIG.PENALTY_REPEAT,
    n_probs: CONFIG.N_PROBS,
    logprobs: CONFIG.N_PROBS,
    stream: false,
    // Forwarded to llama-server (wllama spreads all options into the request
    // JSON, so untyped fields pass through). cache_prompt enables slot prefix
    // reuse; timings_per_token returns the ms/token breakdown.
    cache_prompt: CONFIG.CACHE_PROMPT,
    timings_per_token: CONFIG.TIMINGS_PER_TOKEN,
    abortSignal: abortSignal ?? undefined,
  });
}

/**
 * Extract a flat timings summary from a wllama completion response.
 * Returns null if the server didn't emit timings.
 *
 * Fields (from llama-server ResultTimings):
 *   prompt_n            tokens evaluated for the prompt
 *   prompt_ms           wall time for prompt eval
 *   predicted_n         tokens generated
 *   predicted_ms        wall time for generation
 *   predicted_per_token_ms
 *   cached_tokens       prompt tokens served from cache (usage.prompt_tokens_details)
 */
export function extractTimings(resp) {
  if (!resp) return null;
  const t = resp.timings;
  const cached = resp.usage?.prompt_tokens_details?.cached_tokens ?? 0;
  if (!t) return { cachedTokens: cached, hasTimings: false };
  return {
    hasTimings:      true,
    promptN:         t.prompt_n,
    promptMs:        t.prompt_ms,
    predictedN:      t.predicted_n,
    predictedMs:     t.predicted_ms,
    predictedPerTokenMs: t.predicted_per_token_ms,
    predictedPerSecond:  t.predicted_per_second,
    promptPerSecond:    t.prompt_n ? (t.prompt_n / (t.prompt_ms / 1000)) : 0,
    cachedTokens:    cached,
  };
}

// ══════════════════════════════════════════════════════════════════
//  Keyboard candidates — ported from demo/server.py::_keyboard_candidates
// ══════════════════════════════════════════════════════════════════

/**
 * Generate word-completion / next-word candidates for the predictive
 * keyboard.
 *
 * Two modes (auto-detected from cursor position):
 *
 * 1. Next-word (cursor after space):
 *    Single completion call. Returns top-k first-token words + the
 *    greedy full word.
 *
 * 2. Word completion (cursor mid-word):
 *    Tries N fallback prefix lengths (0, 1, 2 chars shorter) in
 *    parallel. For each: generates with top-k logprobs, extracts
 *    greedy multi-token word completion + top-k single-token
 *    alternatives at position 0. Also detects next-word candidates
 *    (when the model predicts a space token = current word is complete).
 *    Merges + dedups by word, keeps highest prob, sorts, returns top_k.
 *
 * @returns {Promise<Array<{text, prob, is_next_word?}>>}
 */
export async function keyboardCandidates(text, {
  maxTokens = 5,
  topK = 5,
  abortSignal = null,
} = {}) {
  const [textBeforeWord, currentWord] = extractCurrentWord(text);

  // ── Next-word prediction (cursor after space) ──
  if (!currentWord) {
    return _nextWordCandidates(text, maxTokens, topK, abortSignal);
  }

  // ── Word completion (cursor mid-word) ──
  return _wordCompletionCandidates(textBeforeWord, currentWord, maxTokens, topK, abortSignal);
}

/**
 * Next-word mode: generate from full text, return top-k first-token
 * words plus the greedy full word.
 */
async function _nextWordCandidates(text, maxTokens, topK, abortSignal) {
  const resp = await complete(text, { maxTokens, abortSignal });
  const choice   = resp.choices?.[0] ?? {};
  const content  = choice.text ?? '';
  const lpContent = choice.logprobs?.content ?? [];

  const candidates = [];
  const seen = new Set();

  // Greedy full word first
  if (content) {
    const firstWord = extractFirstWord(content.trimStart());
    if (firstWord && !tokenHasDigit(firstWord) && !hasByteFallbackGarbage(firstWord)) {
      const prob = lpContent[0] ? Math.exp(lpContent[0].logprob) : 0.5;
      candidates.push({ text: firstWord, prob });
      seen.add(firstWord);
    }
  }

  // Top-k first-token alternatives
  const alts = lpContent[0]?.top_logprobs ?? [];
  for (const tok of alts) {
    const tokText = tok.token || '';
    if (!tokText.trim()) continue;
    if (tokenHasDigit(tokText)) continue;
    if (hasByteFallbackGarbage(tokText)) continue;
    const word = tokText.trim();
    if (word && !seen.has(word)) {
      candidates.push({ text: word, prob: Math.exp(tok.logprob) });
      seen.add(word);
    }
    if (candidates.length >= topK) break;
  }

  return candidates.slice(0, topK);
}

/**
 * Word-completion mode: try multiple fallback prefix lengths in
 * parallel, merge candidates.
 */
async function _wordCompletionCandidates(textBeforeWord, currentWord, maxTokens, topK, abortSignal) {
  // Build fallback paths: progressively shorter prefixes
  const maxFallback = Math.min(2, currentWord.length - 1);
  const fallbackPaths = [];

  for (let fallback = 0; fallback <= maxFallback; fallback++) {
    const shorterLen = currentWord.length - fallback;
    if (shorterLen < 1) break;
    const shorterWord = currentWord.slice(0, shorterLen);
    const prefix = textBeforeWord + shorterWord;
    fallbackPaths.push({ shorterWord, prefix, isFromScratch: false });
  }

  // Always add the from-scratch path if there's preceding context
  // (predicts the NEXT word from text_before_word only, then filters
  // by current_word prefix — rescues single-token words)
  if (textBeforeWord.trim()) {
    fallbackPaths.push({ shorterWord: '', prefix: textBeforeWord, isFromScratch: true });
  }

  // Fire all fallback paths in parallel. wllama 3.6.0+ (PR #270) fixed the
  // concurrent-completion bug that previously forced sequential calls, and
  // loadModel() sets n_parallel + kv_unified:false so each slot gets its
  // own SSM state — required for a recurrent (Mamba) model.
  const results = await Promise.allSettled(
    fallbackPaths.map(p => complete(p.prefix, { maxTokens, abortSignal }))
  );

  // candidates_map: word -> {text, prob, is_next_word?}
  // Next-word candidates use a separate key namespace ("__next__" + word)
  const candidatesMap = new Map();

  for (let i = 0; i < fallbackPaths.length; i++) {
    const { shorterWord, isFromScratch } = fallbackPaths[i];
    const result = results[i];
    if (result.status !== 'fulfilled') continue;

    const resp = result.value;
    if (!resp) continue;
    const choice    = resp.choices?.[0] ?? {};
    const content   = choice.text ?? '';
    const lpContent = choice.logprobs?.content ?? [];
    const firstLogprob = lpContent[0]?.logprob;

    // 1. Greedy multi-token word completion
    if (content) {
      const raw = isFromScratch ? content.trimStart() : content;
      const wordCompletion = extractFirstWord(raw);
      if (wordCompletion) {
        const fullWord = shorterWord + wordCompletion;
        if (fullWord.startsWith(currentWord)
            && fullWord.length >= currentWord.length
            && !tokenHasDigit(fullWord)
            && !hasByteFallbackGarbage(fullWord)) {
          const prob = firstLogprob !== undefined ? Math.exp(firstLogprob) : 0.5;
          _upsert(candidatesMap, fullWord, { text: fullWord, prob });
        }
      }

      // 1b. Next-word candidate (model thinks current word is complete)
      if (!isFromScratch && content[0] && /\s/.test(content[0])) {
        const nextWord = extractFirstWord(content.trimStart());
        if (nextWord && !tokenHasDigit(nextWord) && !hasByteFallbackGarbage(nextWord)) {
          const prob = firstLogprob !== undefined ? Math.exp(firstLogprob) : 0.5;
          _upsert(candidatesMap, '__next__' + nextWord,
            { text: nextWord, prob, is_next_word: true });
        }
      }
    }

    // 2. Top-k single-token alternatives at position 0
    const alts = lpContent[0]?.top_logprobs ?? [];
    const greedyFirstToken = lpContent[0]?.token;

    for (const tok of alts) {
      // Skip the greedy first token — already represented by the greedy
      // multi-token word above (including it again produces subword noise)
      if (tok.token === greedyFirstToken) continue;

      let tokText = tok.token || '';
      if (!tokText.trim()) continue;
      if (tokenHasDigit(tokText)) continue;
      if (hasByteFallbackGarbage(tokText)) continue;

      if (isFromScratch) {
        // At from-scratch level, tokens are ▁-prefixed — lstrip to get word
        tokText = tokText.trimStart();
        if (!tokText) continue;
      } else {
        // Whitespace-prefixed token = model thinks current word is complete
        // → include as next-word candidate
        if (/\s/.test(tokText[0])) {
          const nextWord = tokText.trim();
          if (nextWord && !tokenHasDigit(nextWord) && !hasByteFallbackGarbage(nextWord)) {
            _upsert(candidatesMap, '__next__' + nextWord,
              { text: nextWord, prob: Math.exp(tok.logprob), is_next_word: true });
          }
          continue;
        }
      }

      const fullWord = shorterWord + tokText;
      if (fullWord.startsWith(currentWord) && fullWord.length >= currentWord.length) {
        _upsert(candidatesMap, fullWord,
          { text: fullWord, prob: Math.exp(tok.logprob) });
      }
    }
  }

  // Sort by prob desc, filter byte-fallback garbage, return top_k
  return [...candidatesMap.values()]
    .filter(c => !hasByteFallbackGarbage(c.text))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, topK);
}

/** Insert or update candidate, keeping the highest prob. */
function _upsert(map, key, candidate) {
  const existing = map.get(key);
  if (!existing || candidate.prob > existing.prob) {
    map.set(key, candidate);
  }
}
