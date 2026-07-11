// ──────────────────────────────────────────────────────────────────
// Morpheus-WASM — Predictive keyboard mode
//
// A phone-style chat UI with word suggestion chips + virtual keyboard.
// All inference runs in-browser via wllama (shared morpheus.js module).
//
// Strategies ported from demo/predictive-keyboard.html + demo/server.py:
//   - keyboardCandidates(): word-completion + next-word prediction with
//     retokenization fallback (progressively shorter prefixes in parallel)
//   - Sticky merge: carries forward previous candidates that match the
//     new prefix (prevents predictions vanishing on tokenization switch)
//   - Android-style chip layout: highest-probability word in the CENTER
//   - Chip acceptance: word completion, next-word, punctuation handling
//   - Virtual keyboard: QWERTY + ñ, shift, symbols, long-press accents
// ──────────────────────────────────────────────────────────────────

import {
  CONFIG, loadModel, keyboardCandidates,
  isPurePunct,
} from './morpheus.js';

// ── Mode-specific config ──────────────────────────────────────────
const DEBOUNCE_MS  = 150;
const KB_TOP_K     = 5;       // fetch 5 candidates; sticky pool keeps extras
const MAX_TOKENS   = 5;       // keyboard mode generates more tokens

// ── DOM ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
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
const statusDot    = $('statusDot');
const editor       = $('editor');
const sendBtn      = $('sendBtn');
const chatArea     = $('chatArea');
const suggestionBar = $('suggestionBar');
const mLatency     = $('mLatency');

// ── State ─────────────────────────────────────────────────────────
let modelReady = false;
let shiftActive = false;
let symbolsMode = false;
let debounceTimer = null;
let abortCtrl = null;
let lastQuery = '';

// Sticky merge state: carry forward candidates that match new prefix
let stickyPool = [];
let stickyWord = '';

// ── Logging ───────────────────────────────────────────────────────
function log(...args) {
  console.log('[morpheus-kb]', ...args);
}

// ══════════════════════════════════════════════════════════════════
//  UI helpers (loading panel)
// ══════════════════════════════════════════════════════════════════
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
function failLoading(title, err) {
  log('ERROR:', err);
  loadIcon.className = 'icon-wrap error';
  loadIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
  setLoad(title, 'See the console for details.');
  const msg = (err && err.message) ? err.message : String(err);
  showBanner('error', `<b>Failed to load the model.</b><br><code>${msg}</code>`);
}

// ══════════════════════════════════════════════════════════════════
//  Model loading
// ══════════════════════════════════════════════════════════════════
async function init() {
  try {
    await loadModel({
      onLog: log,
      onProgress: (pct) => setProgress(pct, 'Downloading model'),
      onStatus: (title, html) => setLoad(title, html),
      onBackendBadge: setBackendBadge,
      onBanner: (type, html) => { if (type === 'warn') showBanner('warn', html); },
    });

    modelReady = true;
    loadingPanel.classList.add('hidden');
    statusDot.className = 'status-dot online';
    log('ready. keyboard mode active.');
    editor.focus();
    checkAutoShift();
  } catch (e) {
    failLoading('Initialization failed', e);
  }
}

// ══════════════════════════════════════════════════════════════════
//  Virtual keyboard
// ══════════════════════════════════════════════════════════════════

const LETTER_LAYOUT = {
  1: [
    {l:'q', a:'1'}, {l:'w', a:'2'}, {l:'e', a:'3', alt:'éêëè'},
    {l:'r', a:'4'}, {l:'t', a:'5'}, {l:'y', a:'6'}, {l:'u', a:'7', alt:'ûüùú'},
    {l:'i', a:'8', alt:'îïìí'}, {l:'o', a:'9', alt:'ôöòó'}, {l:'p', a:'0'},
  ],
  2: [
    {l:'a', alt:'áàâäã'}, {l:'s', alt:'šß'}, {l:'d'}, {l:'f'}, {l:'g'},
    {l:'h'}, {l:'j'}, {l:'k'}, {l:'l'}, {l:'ñ', alt:'ñ'},
  ],
  3: [
    {l:'shift'}, {l:'z'}, {l:'x'}, {l:'c', alt:'ç'}, {l:'v'},
    {l:'b'}, {l:'n', alt:'ñ'}, {l:'m'}, {l:'backspace'},
  ],
  4: [
    {l:'symbols'}, {l:','}, {l:'space', extraWide:true}, {l:'.'}, {l:'enter'},
  ],
};

const SYMBOL_LAYOUT = {
  1: [{l:'1'},{l:'2'},{l:'3'},{l:'4'},{l:'5'},{l:'6'},{l:'7'},{l:'8'},{l:'9'},{l:'0'}],
  2: [{l:'@'},{l:'#'},{l:'$'},{l:'%'},{l:'&'},{l:'*'},{l:'-'},{l:'+'},{l:'('},{l:')'}],
  3: [{l:'shift'},{l:'!'},{l:'"'},{l:"'"},{l:':'},{l:';'},{l:'/'},{l:'?'},{l:'backspace'}],
  4: [{l:'ABC'},{l:','},{l:'space', extraWide:true},{l:'.'},{l:'enter'}],
};

function shiftSVG() {
  return '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>';
}
function backspaceSVG() {
  return '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 4H8l-7 8 7 8h13a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z"/><line x1="18" y1="9" x2="12" y2="15"/><line x1="12" y1="9" x2="18" y2="15"/></svg>';
}
function enterSVG() {
  return '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 10 4 15 9 20"/><path d="M20 4v7a4 4 0 0 1-4 4H4"/></svg>';
}

function renderKeyboard() {
  const layout = symbolsMode ? SYMBOL_LAYOUT : LETTER_LAYOUT;
  for (let rowNum = 1; rowNum <= 4; rowNum++) {
    const rowEl = document.querySelector(`.kb-row[data-row="${rowNum}"]`);
    rowEl.innerHTML = '';
    for (const key of layout[rowNum]) {
      const btn = document.createElement('button');
      btn.className = 'kb-key';
      btn.dataset.key = key.l;

      if (key.l === 'shift') {
        btn.classList.add('mod');
        if (shiftActive) btn.classList.add('shift-active');
        btn.innerHTML = shiftSVG();
        btn.dataset.action = 'shift';
      } else if (key.l === 'backspace') {
        btn.classList.add('mod');
        btn.innerHTML = backspaceSVG();
        btn.dataset.action = 'backspace';
      } else if (key.l === 'enter') {
        btn.classList.add('mod');
        btn.innerHTML = enterSVG();
        btn.dataset.action = 'enter';
      } else if (key.l === 'symbols') {
        btn.classList.add('mod');
        btn.textContent = '123';
        btn.dataset.action = 'symbols';
      } else if (key.l === 'ABC') {
        btn.classList.add('mod');
        btn.textContent = 'ABC';
        btn.dataset.action = 'symbols';
      } else if (key.l === 'space') {
        btn.classList.add('extra-wide');
        btn.innerHTML = '<svg width="28" height="22" viewBox="0 0 28 22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="11" x2="22" y2="11"/></svg>';
        btn.dataset.action = 'space';
      } else {
        let display = key.l;
        if (shiftActive && !symbolsMode) display = display.toUpperCase();
        btn.textContent = display;
        if (key.a) {
          const alt = document.createElement('span');
          alt.className = 'alt-chars';
          alt.textContent = key.a;
          btn.appendChild(alt);
        }
        btn.dataset.action = 'type';
        btn.dataset.char = display;
      }

      // Long-press for alternate characters
      if (key.alt) {
        let pressTimer = null;
        btn.addEventListener('touchstart', () => {
          pressTimer = setTimeout(() => showAltChars(btn, key.alt, key.l), 400);
        });
        btn.addEventListener('touchend', () => clearTimeout(pressTimer));
        btn.addEventListener('touchmove', () => clearTimeout(pressTimer));
      }

      btn.addEventListener('click', (e) => {
        e.preventDefault();
        handleKey(btn.dataset.action, btn.dataset.char);
      });
      btn.addEventListener('mousedown', (e) => e.preventDefault());
      rowEl.appendChild(btn);
    }
  }
}

function showAltChars(btn, altChars, baseChar) {
  const popup = document.createElement('div');
  popup.className = 'alt-popup';
  const rect = btn.getBoundingClientRect();
  popup.style.left = rect.left + 'px';
  popup.style.top = (rect.top - 50) + 'px';

  const chars = altChars.split('');
  chars.unshift(baseChar);
  for (const c of chars) {
    const item = document.createElement('div');
    item.className = 'alt-item';
    item.textContent = c;
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      typeChar(c);
      popup.remove();
    });
    popup.appendChild(item);
  }
  document.body.appendChild(popup);

  setTimeout(() => {
    const handler = (e) => {
      if (!popup.contains(e.target)) {
        popup.remove();
        document.removeEventListener('click', handler);
      }
    };
    document.addEventListener('click', handler);
  }, 100);
}

function handleKey(action, char) {
  switch (action) {
    case 'type': typeChar(char); break;
    case 'shift':
      shiftActive = !shiftActive;
      renderKeyboard();
      break;
    case 'backspace': {
      const pos = editor.selectionStart;
      if (pos > 0) {
        editor.value = editor.value.substring(0, pos - 1) + editor.value.substring(pos);
        editor.setSelectionRange(pos - 1, pos - 1);
      }
      editor.focus();
      checkAutoShift();
      onInput();
      break;
    }
    case 'enter': sendMessage(); break;
    case 'symbols':
      symbolsMode = !symbolsMode;
      renderKeyboard();
      break;
    case 'space':
      typeChar(' ');
      editor.focus();
      break;
  }
}

function typeChar(c) {
  const pos = editor.selectionStart;
  editor.value = editor.value.substring(0, pos) + c + editor.value.substring(pos);
  editor.setSelectionRange(pos + 1, pos + 1);
  editor.focus();
  // Shift is one-shot: consume after typing, then re-evaluate
  shiftActive = false;
  renderKeyboard();
  checkAutoShift();
  onInput();
  autoResize();
}

function autoResize() {
  editor.style.height = 'auto';
  editor.style.height = Math.min(editor.scrollHeight, 120) + 'px';
}

/** Auto-shift: uppercase after sentence-ending punctuation + space. */
function checkAutoShift() {
  const prefix = getPrefix();
  const lastIsSpace = prefix.length > 0 && /\s/.test(prefix[prefix.length - 1]);
  const shouldShift = prefix.trim() === '' || (lastIsSpace && /[.!?]["')\]]?\s+$/.test(prefix));
  if (shouldShift !== shiftActive) {
    shiftActive = shouldShift;
    renderKeyboard();
  }
}

// ══════════════════════════════════════════════════════════════════
//  Chat
// ══════════════════════════════════════════════════════════════════

function sendMessage() {
  const text = editor.value.trim();
  if (!text) return;
  // Reset sticky merge — new message starts fresh
  stickyPool = [];
  stickyWord = '';
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.textContent = text;
  chatArea.appendChild(bubble);
  chatArea.scrollTop = chatArea.scrollHeight;
  editor.value = '';
  autoResize();
  sendBtn.disabled = true;
  clearSuggestions();
  shiftActive = false;
  checkAutoShift();
  onInput();
}

// ══════════════════════════════════════════════════════════════════
//  Suggestions (sticky merge + chip rendering + acceptance)
// ══════════════════════════════════════════════════════════════════

function clearSuggestions() {
  suggestionBar.innerHTML = '<div class="chip placeholder">Idatzi letra bat gutxienez...</div>';
}

function getCurrentWord() {
  const prefix = getPrefix();
  let i = prefix.length - 1;
  while (i >= 0 && !/\s/.test(prefix[i])) i--;
  return prefix.substring(i + 1);
}

function getCurrentWordStart() {
  const prefix = getPrefix();
  let i = prefix.length - 1;
  while (i >= 0 && !/\s/.test(prefix[i])) i--;
  return i + 1;
}

/**
 * Sticky merge: carry forward previous candidates that match the
 * current prefix. Prevents predictions from vanishing when the user
 * types the first letter of a predicted word and the tokenization
 * path switches.
 */
function mergeCandidates(freshCandidates) {
  const currentWord = getCurrentWord().toLowerCase();

  // Filter previous candidates by the current prefix
  let survivors = [];
  if (stickyPool.length > 0 && currentWord.length > 0) {
    survivors = stickyPool
      .filter(c => c.text.toLowerCase().startsWith(currentWord))
      .map(c => ({ ...c, _sticky: true }));
  }

  // Fresh candidates not already in survivors
  const survivorTexts = new Set(survivors.map(c => c.text));
  const freshOnly = freshCandidates
    .filter(c => !survivorTexts.has(c.text))
    .map(c => ({ ...c, _sticky: false }));

  // Merge and sort. Sticky survivors get a small boost because
  // cross-path probabilities aren't directly comparable.
  const STICKY_BOOST = 0.1;
  const merged = [...survivors, ...freshOnly];
  merged.sort((a, b) =>
    (b.prob + (b._sticky ? STICKY_BOOST : 0)) -
    (a.prob + (a._sticky ? STICKY_BOOST : 0))
  );

  // Update sticky state for next iteration
  stickyPool = merged.slice(0, 5).map(c => ({
    text: c.text, prob: c.prob, is_next_word: c.is_next_word || false,
  }));
  stickyWord = currentWord;

  // Return top 3 (strip internal _sticky flag)
  return merged.slice(0, 3).map(c => ({
    text: c.text, prob: c.prob, is_next_word: c.is_next_word || false,
  }));
}

function renderChips(candidates) {
  if (!candidates.length) {
    suggestionBar.innerHTML = '<div class="chip placeholder">Ez dago iradokizunik...</div>';
    return;
  }

  const wordStart = getCurrentWordStart();

  // Android-style layout: highest-probability word in the CENTER chip.
  // 3 chips: [2nd, 1st, 3rd]  2 chips: [2nd, 1st]  1 chip: [1st]
  let ordered;
  if (candidates.length >= 3) {
    ordered = [candidates[1], candidates[0], candidates[2]];
  } else if (candidates.length === 2) {
    ordered = [candidates[1], candidates[0]];
  } else {
    ordered = candidates;
  }

  suggestionBar.innerHTML = '';
  for (let i = 0; i < ordered.length; i++) {
    const c = ordered[i];
    const chip = document.createElement('div');
    chip.className = 'chip';
    if (i === Math.floor(ordered.length / 2)) chip.classList.add('chip-primary');
    chip.textContent = c.text;
    chip.addEventListener('click', () => acceptChip({
      text: c.text,
      replaceFrom: wordStart,
      is_next_word: c.is_next_word || false,
    }));
    chip.addEventListener('mousedown', (e) => e.preventDefault());
    suggestionBar.appendChild(chip);
  }
}

function acceptChip(data) {
  const pos = editor.selectionStart;
  let newText = data.text;
  const isPunct = isPurePunct(newText);

  // Reset sticky merge — accepted word starts a fresh prediction context
  stickyPool = [];
  stickyWord = '';

  if (data.is_next_word) {
    // Next-word candidate: insert with leading + trailing space
    const insert = ' ' + newText + ' ';
    editor.value = editor.value.substring(0, pos) + insert + editor.value.substring(pos);
    const newPos = pos + insert.length;
    editor.setSelectionRange(newPos, newPos);
  } else if (isPunct) {
    // Punctuation: attach directly to previous word, no space before
    let insertPos = pos;
    let before = editor.value.substring(0, insertPos);
    let after = editor.value.substring(pos);
    if (before.endsWith(' ')) { before = before.slice(0, -1); insertPos--; }
    const insert = newText.trim() + ' ';
    editor.value = before + insert + after;
    const newPos = insertPos + insert.length;
    editor.setSelectionRange(newPos, newPos);
  } else {
    // Normal word: auto-add trailing space
    if (!newText.endsWith(' ') && !newText.endsWith('\n')) newText += ' ';
    if (data.replaceFrom !== null) {
      editor.value = editor.value.substring(0, data.replaceFrom) + newText + editor.value.substring(pos);
      const newPos = data.replaceFrom + newText.length;
      editor.setSelectionRange(newPos, newPos);
    } else {
      editor.value = editor.value.substring(0, pos) + newText + editor.value.substring(pos);
      const newPos = pos + newText.length;
      editor.setSelectionRange(newPos, newPos);
    }
  }
  editor.focus();
  autoResize();
  shiftActive = false;
  checkAutoShift();
  onInput();
}

// ══════════════════════════════════════════════════════════════════
//  Completion orchestration
// ══════════════════════════════════════════════════════════════════

function getPrefix() { return editor.value.substring(0, editor.selectionStart); }

async function doComplete(prefix) {
  if (!modelReady) return;
  lastQuery = prefix;

  let candidates;
  let latency;
  try {
    const t0 = performance.now();
    candidates = await keyboardCandidates(prefix, {
      maxTokens: MAX_TOKENS,
      topK: KB_TOP_K,
      abortSignal: abortCtrl?.signal,
    });
    latency = performance.now() - t0;
  } catch (e) {
    if (e?.name === 'AbortError' || /abort/i.test(e?.message || '')) return;
    log('completion error:', e);
    return;
  }

  // Staleness guard
  if (lastQuery !== prefix) return;
  if (getPrefix() !== prefix) return;

  mLatency.textContent = Math.round(latency) + ' ms';

  // Sticky merge
  const merged = mergeCandidates(candidates);
  renderChips(merged);
}

function onInput() {
  sendBtn.disabled = !editor.value.trim();
  const prefix = getPrefix();
  if (!prefix.trim()) {
    clearSuggestions();
    stickyPool = [];
    stickyWord = '';
    return;
  }

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    if (abortCtrl) { try { abortCtrl.abort(); } catch {} }
    abortCtrl = new AbortController();
    doComplete(prefix);
  }, DEBOUNCE_MS);
}

// ══════════════════════════════════════════════════════════════════
//  Event listeners
// ══════════════════════════════════════════════════════════════════

editor.addEventListener('input', () => { autoResize(); onInput(); });

editor.addEventListener('keydown', (e) => {
  // Enter (without Shift) → send message
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
    return;
  }
  // Tab → accept the center (most probable) suggestion
  if (e.key === 'Tab') {
    const primaryChip = suggestionBar.querySelector('.chip-primary');
    if (primaryChip && !primaryChip.classList.contains('placeholder')) {
      e.preventDefault();
      primaryChip.click();
    }
  }
});

editor.addEventListener('keyup', (e) => {
  if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) {
    onInput();
    checkAutoShift();
    return;
  }
  checkAutoShift();
});

sendBtn.addEventListener('click', sendMessage);

chatArea.addEventListener('click', () => editor.focus());

// Prevent double-tap zoom
let lastTouch = 0;
document.addEventListener('touchend', (e) => {
  const now = Date.now();
  if (now - lastTouch <= 300) e.preventDefault();
  lastTouch = now;
}, { passive: false });

// ── Go ────────────────────────────────────────────────────────────
renderKeyboard();
init();
autoResize();
