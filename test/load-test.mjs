// Headless test harness for morpheus-wasm Phase 1.
// Loads the page, captures console + page errors, waits for the model to
// become ready, then screenshots and reports metrics.
//
// Run: NODE_PATH=$(npm root -g) node test/load-test.mjs

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
// Resolve the globally-installed playwright via NODE_PATH-style lookup.
const globalRoot = require('child_process')
  .execSync('npm root -g', { encoding: 'utf8' }).trim();
const { chromium } = require(globalRoot + '/playwright');

const URL = 'http://127.0.0.1:8765/';
const TIMEOUT_MS = 180_000; // 3 min — model download + wasm init + first completion

const log = (...a) => console.log('[test]', ...a);

const browser = await chromium.launch({
  headless: true,
  args: [
    '--enable-unsafe-swiftshader',        // software WebGPU
    '--enable-features=Vulkan,WebGPU',
    '--ignore-gpu-blocklist',
  ],
});

const page = await browser.newPage({ viewport: { width: 820, height: 900 } });

const consoleLines = [];
const errors = [];

page.on('console', (msg) => {
  const type = msg.type();
  const text = msg.text();
  consoleLines.push(`[${type}] ${text}`);
  if (type === 'error') errors.push(text);
});
page.on('pageerror', (err) => { errors.push('PAGEERROR: ' + err.message); consoleLines.push('[pageerror] ' + err.message); });
page.on('requestfailed', (req) => {
  consoleLines.push(`[reqfail] ${req.url()} — ${req.failure()?.errorText}`);
});

log('navigating to', URL);
await page.goto(URL, { waitUntil: 'domcontentloaded' });

// Wait for either: editor card visible (success) or error banner shown (failure)
log('waiting for model load (up to', TIMEOUT_MS/1000, 's)...');
const start = Date.now();

let outcome = 'timeout';
try {
  await Promise.race([
    // success: editor card becomes visible
    page.waitForSelector('#editorCard:not(.hidden)', { timeout: TIMEOUT_MS })
      .then(() => { outcome = 'ready'; }),
    // failure: error banner appears
    page.waitForSelector('#loadBanner.error.show', { timeout: TIMEOUT_MS })
      .then(() => { outcome = 'error'; }),
  ]);
} catch (e) {
  outcome = 'timeout';
}

const elapsed = ((Date.now() - start)/1000).toFixed(1);
log(`outcome = ${outcome} after ${elapsed}s`);

// Give first completion a moment to render ghost text
if (outcome === 'ready') {
  try {
    await page.waitForTimeout(4000);
  } catch {}
}

// Capture state
const state = await page.evaluate(() => {
  const $ = (id) => document.getElementById(id);
  return {
    editorVisible: !$('editorCard').classList.contains('hidden'),
    editorValue: $('editor')?.value || '',
    editorSelStart: $('editor')?.selectionStart,
    editorSelEnd: $('editor')?.selectionEnd,
    ghostActive: $('editor')?.classList.contains('ghost-active'),
    mBackend: $('mBackend')?.textContent,
    mLatency: $('mLatency')?.textContent,
    mConfidence: $('mConfidence')?.textContent,
    loadTitle: $('loadTitle')?.textContent,
    loadStatus: $('loadStatus')?.textContent,
    badgeBackend: $('badgeBackend')?.textContent,
    bannerHTML: $('loadBanner')?.innerHTML,
    progressPct: $('progressPct')?.textContent,
    debugLog: $('debugLog')?.textContent,
  };
});

log('=== PAGE STATE ===');
console.log(JSON.stringify(state, null, 2));

log('=== CONSOLE LOG (last 40 lines) ===');
console.log(consoleLines.slice(-40).join('\n'));

log('=== ERRORS ===');
console.log(errors.length ? errors.join('\n') : '(none)');

// Screenshots
await page.screenshot({ path: 'test/screenshot-full.png', fullPage: true });
log('screenshot saved → test/screenshot-full.png');

// Editor-only crop
const editorCard = await page.$('#editorCard');
if (editorCard && state.editorVisible) {
  await editorCard.screenshot({ path: 'test/screenshot-editor.png' });
  log('editor screenshot saved → test/screenshot-editor.png');
}

await browser.close();
process.exit(outcome === 'ready' ? 0 : 1);
