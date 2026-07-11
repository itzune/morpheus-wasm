import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const globalRoot = require('child_process').execSync('npm root -g', { encoding: 'utf8' }).trim();
const { chromium } = require(globalRoot + '/playwright');

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-swiftshader', '--enable-features=Vulkan,WebGPU', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', (m) => { if (m.text().includes('STRING') || m.text()==='DONE') console.log('  ' + m.text()); });
await page.goto('http://127.0.0.1:8765/test/compare-probe.html', { waitUntil: 'domcontentloaded', timeout: 60_000 });
await page.waitForFunction(() => document.getElementById('out').textContent.includes('DONE'), { timeout: 150_000 });
await browser.close();
