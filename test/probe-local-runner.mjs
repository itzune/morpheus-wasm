// Run probe-local.html against the locally-served patched GGUF.
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const globalRoot = require('child_process').execSync('npm root -g', { encoding: 'utf8' }).trim();
const { chromium } = require(globalRoot + '/playwright');

const v = process.argv[2] || '3.5.1';
const url = 'http://127.0.0.1:8766/model/morpheus-v2-mamba.Q4_K_M.wllama.gguf';
const pageUrl = `http://127.0.0.1:8766/test/probe-local.html?v=${v}&url=${encodeURIComponent(url)}`;

const log = (...a) => console.log('[probe]', ...a);
const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader', '--enable-features=Vulkan,WebGPU', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', (m) => console.log('  [' + m.type() + '] ' + m.text()));
page.on('pageerror', (e) => console.log('  [pageerror] ' + e.message));

log('opening', pageUrl);
await page.goto(pageUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
try {
  await page.waitForFunction(() => document.getElementById('out').textContent.includes('RESULT:'), { timeout: 150_000 });
  const text = await page.evaluate(() => document.getElementById('out').textContent);
  console.log(text);
} catch (e) {
  log('TIMEOUT/ERROR:', e.message);
  const text = await page.evaluate(() => document.getElementById('out')?.textContent || '(empty)');
  console.log(text);
}
await browser.close();
