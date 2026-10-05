const puppeteer = require('puppeteer');
const path = require('path');

const FILE = 'file:///' + path.join(path.resolve(__dirname, '..'), 'dist', 'index.html').replace(/\\/g, '/');

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: process.env.CHROME_PATH,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--allow-file-access-from-files']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000 });
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  page.on('console', m => { if (m.type() === 'error') console.log('[console.error]', m.text()); });

  await page.goto(FILE, { waitUntil: 'networkidle0', timeout: 60000 });
  await new Promise(r => setTimeout(r, 2000));

  console.log('driver shape:', await page.evaluate(() => {
    const d = window.driver;
    return {
      typeofDriver: typeof d,
      topKeys: d ? Object.keys(d) : null,
      jsType: d?.js ? typeof d.js : 'n/a',
      jsKeys: d?.js ? Object.keys(d.js) : null,
      jsDriverType: typeof d?.js?.driver,
      jsDriverIsFn: typeof d?.js?.driver === 'function'
    };
  }));

  await page.evaluate(() => { const b = document.querySelector('.driver-popover-close-btn'); if (b) b.click(); });
  await new Promise(r => setTimeout(r, 400));

  // Tooltip de nodo con coordenadas explicitas del circle
  const nodePos = await page.evaluate(() => {
    const c = document.querySelector('#svg-nodes .node-body');
    const r = c.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, r: r.toJSON() };
  });
  console.log('nodo pos', JSON.stringify(nodePos));
  await page.mouse.move(nodePos.x, nodePos.y);
  await new Promise(r => setTimeout(r, 600));
  console.log('tooltip tras mouse.move:', JSON.stringify(await page.evaluate(() => {
    const t = document.querySelector('.tt');
    if (!t) return { exists: false };
    return { exists: true, show: t.dataset.show, text: t.textContent.slice(0, 60) };
  })));

  // Tour via el boton real de la app
  const tourResult = await page.evaluate(() => {
    document.querySelector('#btn-tour').click();
    return { clicked: true };
  });
  console.log('tour click:', JSON.stringify(tourResult));
  await new Promise(r => setTimeout(r, 1200));
  console.log('tour dom:', JSON.stringify(await page.evaluate(() => ({
    popover: !!document.querySelector('.driver-popover'),
    title: document.querySelector('.driver-popover-title')?.textContent || null,
    body: document.querySelector('.driver-popover-description')?.textContent?.slice(0, 40) || null,
    footerBtns: [...document.querySelectorAll('.driver-popover-footer button')].map(b => b.textContent.trim()),
    activeEl: document.querySelector('.driver-active-element')?.tagName || null
  }))));

  await browser.close();
})().catch(e => { console.error('FALLO:', e.message); process.exit(1); });