// Interactive UX smoke test: type a prompt, wait for ghost, Tab to accept.
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const globalRoot = require('child_process').execSync('npm root -g', { encoding: 'utf8' }).trim();
const { chromium } = require(globalRoot + '/playwright');

const log = (...a) => console.log('[ux]', ...a);
const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader', '--enable-features=Vulkan,WebGPU', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
await page.goto('http://127.0.0.1:8765/', { waitUntil: 'domcontentloaded', timeout: 60_000 });

// Wait for editor to become visible (model loaded).
await page.waitForSelector('#editor:not([style*="none"])', { timeout: 180_000 });
await page.waitForFunction(() => document.getElementById('editorCard') && !document.getElementById('editorCard').classList.contains('hidden'), { timeout: 180_000 });
log('editor ready — model loaded');

const ed = page.locator('#editor');
// Clear and type a fresh prompt.
await ed.fill('');
await ed.click();
await page.keyboard.type('Euskal Herria ', { delay: 40 });
log('typed: "Euskal Herria "');

// Wait for ghost text (a selection longer than the typed prefix appears).
try {
  await page.waitForFunction(() => {
    const t = document.getElementById('editor');
    return t.classList.contains('ghost-active') && t.selectionEnd > t.selectionStart;
  }, { timeout: 15_000 });
  const state = await page.evaluate(() => {
    const t = document.getElementById('editor');
    return { value: t.value, selStart: t.selectionStart, selEnd: t.selectionEnd, ghost: t.classList.contains('ghost-active') };
  });
  const ghost = state.value.slice(state.selStart, state.selEnd);
  log('ghost appeared:', JSON.stringify(state.value), '| ghost part:', JSON.stringify(ghost));

  // Press Tab to accept.
  await page.keyboard.press('Tab');
  await page.waitForTimeout(200);
  const after = await page.evaluate(() => document.getElementById('editor').value);
  log('after Tab, editor =', JSON.stringify(after));
  if (after.includes(ghost) && after.endsWith(ghost)) {
    log('✓ Tab accepted the ghost — interactive loop works');
  } else {
    log('✗ Tab did not accept as expected');
  }
} catch (e) {
  log('no ghost within 15s (model may be slow on WASM):', e.message);
}

await page.screenshot({ path: 'test/screenshot-ux.png', fullPage: true });
log('screenshot → test/screenshot-ux.png');
await browser.close();
