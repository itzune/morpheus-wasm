// Run probe.html across candidate wllama versions, report which loads the model.
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const globalRoot = require('child_process').execSync('npm root -g', { encoding: 'utf8' }).trim();
const { chromium } = require(globalRoot + '/playwright');

const VERSIONS = process.argv.slice(2);
const URL = 'http://localhost:8765/test/probe.html?v=';

const log = (...a) => console.log('[probe]', ...a);

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader', '--enable-features=Vulkan,WebGPU', '--ignore-gpu-blocklist'],
});

for (const v of VERSIONS) {
  const page = await browser.newPage();
  const lines = [];
  page.on('console', (m) => lines.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => lines.push('[pageerror] ' + e.message));
  log('--- wllama@' + v + ' ---');
  try {
    await page.goto(URL + v, { waitUntil: 'domcontentloaded', timeout: 60_000 });
    // wait for RESULT line
    await page.waitForFunction(() => document.getElementById('out').textContent.includes('RESULT:'), { timeout: 120_000 });
    const text = await page.evaluate(() => document.getElementById('out').textContent);
    // print the meaningful lines
    text.split('\n').filter(l => /probing|libllama|WebGPU|LOADED|add_bos|COMPLETION|RESULT|ERROR/.test(l)).forEach(l => console.log('  ' + l));
  } catch (e) {
    log('  PROBE FAILED/TIMEOUT:', e.message);
    lines.filter(l => /error|shape|tensor/i.test(l)).slice(-5).forEach(l => console.log('  ' + l));
  }
  await page.close();
}

await browser.close();
