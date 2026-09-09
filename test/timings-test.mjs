// Tier 2 measurement: load the page, type several prompts, capture the
// per-token timings + cached_tokens that morpheus.js now logs.
//
// Run: NODE_PATH=$(npm root -g) node test/timings-test.mjs

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const globalRoot = require('child_process')
  .execSync('npm root -g', { encoding: 'utf8' }).trim();
const { chromium } = require(globalRoot + '/playwright');

const URL = process.argv[2] || 'http://127.0.0.1:8765/';
const log = (...a) => console.log('[timings]', ...a);

const browser = await chromium.launch({
  headless: true,
  args: [
    '--enable-unsafe-swiftshader',
    '--enable-features=Vulkan,WebGPU',
    '--ignore-gpu-blocklist',
  ],
});
const page = await browser.newPage({ viewport: { width: 820, height: 900 } });

// Capture console lines that contain our timings log
const timingsLines = [];
page.on('console', (msg) => {
  const text = msg.text();
  if (text.includes('[morpheus] timings:')) timingsLines.push(text);
});

log('navigating to', URL);
await page.goto(URL, { waitUntil: 'domcontentloaded' });

// Wait for the model to load (editor card visible)
log('waiting for model load...');
await page.waitForSelector('#editorCard:not(.hidden)', { timeout: 240_000 });
log('model ready');

const editor = page.locator('#editor');

// Test prompts — each is an "append" case (prefix grows by a word each time),
// which is the scenario where cache_prompt might help Mamba.
const prompts = [
  'Kaixo, zer ',
  'Kaixo, zer moduz ',
  'Kaixo, zer moduz? Nik ',
  'Euskal Herria ',
  'Euskal Herria herri ',
  'Nire izena ',
];

for (const p of prompts) {
  timingsLines.length = 0;
  await editor.fill('');
  await editor.click();
  await page.keyboard.type(p, { delay: 30 });
  // Wait for a completion to land (ghost appears or debug log updates)
  await page.waitForTimeout(2500);
  const tLine = timingsLines[timingsLines.length - 1];
  if (tLine) {
    log(`typed ${JSON.stringify(p)}`);
    console.log('   ' + tLine.replace('[morpheus] ', ''));
  } else {
    log(`typed ${JSON.stringify(p)} → (no timings line captured)`);
  }
}

// Also read the full debug log from the DOM for a sanity check
const debugText = await page.evaluate(() => document.getElementById('debugLog')?.textContent || '');
const lastFew = debugText.trim().split('\n').slice(-12);
log('=== debug log (last 12 lines) ===');
console.log(lastFew.join('\n'));

// Backend info
const backend = await page.evaluate(() => ({
  mBackend: document.getElementById('mBackend')?.textContent,
  crossOriginIsolated: window.crossOriginIsolated,
  badgeBackend: document.getElementById('badgeBackend')?.textContent,
}));
log('=== backend ===');
console.log(JSON.stringify(backend, null, 2));

await page.screenshot({ path: 'test/screenshot-timings.png', fullPage: true });
log('screenshot → test/screenshot-timings.png');
await browser.close();
