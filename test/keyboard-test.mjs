// ──────────────────────────────────────────────────────────────────
// Headless test for the predictive keyboard mode.
// Loads keyboard.html, types partial words, checks suggestion chips
// appear, clicks a chip, verifies acceptance.
// ──────────────────────────────────────────────────────────────────

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const globalRoot = require('child_process')
  .execSync('npm root -g').toString().trim();
const { chromium } = require(globalRoot + '/playwright');

const URL = 'http://127.0.0.1:8765/keyboard.html';
const TIMEOUT = 180_000;

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 400, height: 800 } });

const errors = [];
page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
page.on('console', msg => {
  if (msg.type() === 'error') errors.push(`console.error: ${msg.text()}`);
});

console.log('[kb-test] navigating to', URL);
await page.goto(URL, { waitUntil: 'domcontentloaded' });

// Wait for model to load (loading panel gets .hidden)
console.log('[kb-test] waiting for model load (up to 180 s)...');
await page.waitForFunction(() => {
  const panel = document.getElementById('loadingPanel');
  return panel && panel.classList.contains('hidden');
}, { timeout: TIMEOUT });
console.log('[kb-test] model loaded — keyboard mode ready');

// Check the virtual keyboard rendered
const kbKeyCount = await page.locator('.kb-key').count();
console.log('[kb-test] virtual keyboard keys:', kbKeyCount);

// Helper: get current suggestion chips
async function getChips() {
  return page.evaluate(() => {
    const chips = document.querySelectorAll('.suggestion-bar .chip:not(.placeholder)');
    return [...chips].map(c => c.textContent);
  });
}

// Helper: type into editor and wait for suggestions
async function typeText(text) {
  await page.click('#editor');
  await page.fill('#editor', '');
  await page.type('#editor', text, { delay: 10 });
  // Wait for either chips to appear or timeout
  try {
    await page.waitForSelector('.suggestion-bar .chip:not(.placeholder)', { timeout: 5000 });
    await sleep(100);  // settle
  } catch {
    await sleep(500);  // fallback wait
  }
}

// ── Test 1: Next-word prediction (cursor after space) ──
console.log('[kb-test] === Test 1: Next-word prediction ===');
await typeText('Kaixo, zer ');
const chips1 = await getChips();
console.log('[kb-test] typed "Kaixo, zer " → chips:', chips1);
if (chips1.length === 0) {
  console.log('[kb-test] ✗ FAIL: no suggestion chips appeared');
} else {
  const hasModuz = chips1.some(c => c.toLowerCase().includes('moduz'));
  console.log('[kb-test] ' + (hasModuz ? '✓' : '✗') + ' ' +
    (hasModuz ? 'PASS: "moduz" found in suggestions' : `FAIL: expected "moduz" in ${JSON.stringify(chips1)}`));
}

// ── Test 2: Word completion (cursor mid-word) ──
console.log('[kb-test] === Test 2: Word completion ===');
await typeText('Kaixo, mo');
const chips2 = await getChips();
console.log('[kb-test] typed "Kaixo, mo" → chips:', chips2);
if (chips2.length === 0) {
  console.log('[kb-test] ✗ FAIL: no word completion chips');
} else {
  // All chips should be word completions starting with "mo"
  const allStartWithMo = chips2.every(c => c.toLowerCase().startsWith('mo'));
  console.log('[kb-test] ' + (allStartWithMo ? '✓' : '✗') + ' ' +
    (allStartWithMo ? `PASS: all chips start with "mo" (${JSON.stringify(chips2)})` : `FAIL: expected all chips to start with "mo" in ${JSON.stringify(chips2)}`));
}

// ── Test 3: Chip click acceptance ──
console.log('[kb-test] === Test 3: Chip click acceptance ===');
// Re-type and click the primary chip
await typeText('Kaixo, zer ');
await sleep(300);
const primaryChip = page.locator('.suggestion-bar .chip-primary');
const chipText = await primaryChip.textContent().catch(() => null);
if (chipText) {
  console.log('[kb-test] primary chip:', chipText);
  await primaryChip.click();
  await sleep(200);
  const editorValue = await page.inputValue('#editor');
  console.log('[kb-test] after click, editor =', JSON.stringify(editorValue));
  const accepted = editorValue.includes(chipText);
  console.log('[kb-test] ' + (accepted ? '✓' : '✗') + ' ' +
    (accepted ? 'PASS: chip text inserted into editor' : 'FAIL: chip text not found in editor'));
} else {
  console.log('[kb-test] ✗ FAIL: no primary chip to click');
}

// ── Test 4: Virtual keyboard type ──
console.log('[kb-test] === Test 4: Virtual keyboard type ===');
// Clear editor, type "K" via virtual keyboard
await page.fill('#editor', '');
await sleep(200);
await page.locator('.kb-key[data-action="type"][data-char="K"]').click();
await sleep(400);
const afterK = await page.inputValue('#editor');
console.log('[kb-test] after clicking "K" key, editor =', JSON.stringify(afterK));
const kTyped = afterK === 'K';
console.log('[kb-test] ' + (kTyped ? '✓' : '✗') + ' ' +
  (kTyped ? 'PASS: virtual keyboard types correctly' : 'FAIL: expected "K"'));

// ── Test 5: Send message creates chat bubble ──
console.log('[kb-test] === Test 5: Send message ===');
await page.fill('#editor', 'Kaixo');
await sleep(100);
await page.click('#sendBtn');
await sleep(200);
const bubbleCount = await page.locator('.chat-bubble').count();
const editorCleared = (await page.inputValue('#editor')) === '';
console.log('[kb-test] chat bubbles:', bubbleCount, '| editor cleared:', editorCleared);
const sentOk = bubbleCount > 0 && editorCleared;
console.log('[kb-test] ' + (sentOk ? '✓' : '✗') + ' ' +
  (sentOk ? 'PASS: message sent, bubble created' : 'FAIL: message not sent'));

// ── Page state ──
const state = await page.evaluate(() => ({
  mLatency: document.getElementById('mLatency')?.textContent,
  statusDot: document.getElementById('statusDot')?.className,
  editorValue: document.getElementById('editor')?.value,
  chipCount: document.querySelectorAll('.suggestion-bar .chip:not(.placeholder)').length,
}));
console.log('[kb-test] === FINAL STATE ===');
console.log(JSON.stringify(state, null, 2));

console.log('[kb-test] === ERRORS ===');
if (errors.length) {
  errors.forEach(e => console.log(' ', e));
} else {
  console.log('(none)');
}

await page.screenshot({ path: 'test/screenshot-keyboard.png' });
console.log('[kb-test] screenshot → test/screenshot-keyboard.png');

await browser.close();
