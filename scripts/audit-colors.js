/* Verificacion visual por muestreo de pixeles: renderizo el screenshot en un
   canvas dentro del navegador y cuento los colores reales por familia.
   Asi confirmo que la paleta cian/naranja se aplica de verdad. */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const SHOTS = path.join(ROOT, '.shots');
const FILE = 'file:///' + path.join(ROOT, 'dist', 'index.html').replace(/\\/g, '/');

const PALETTE = {
  cian:   [34, 211, 238],   // #22d3ee
  cian2:  [103, 232, 249],  // #67e8f9
  cian3:  [6, 182, 212],    // #06b6d4
  naranja:[251, 146, 60],   // #fb923c
  naranja2:[249, 115, 22],  // #f97316
  naranja3:[253, 186, 116], // #fdba74
  rojo:   [251, 113, 133],  // fb7185 (backprop)
  violeta:[167, 139, 250]   // a78bfa
};

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: process.env.CHROME_PATH,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--allow-file-access-from-files']
  });
  const page = await browser.newPage();
  await page.goto(FILE, { waitUntil: 'networkidle0', timeout: 60000 });
  await new Promise(r => setTimeout(r, 1800));
  await page.evaluate(() => { const b = document.querySelector('.driver-popover-close-btn'); if (b) b.click(); });
  await new Promise(r => setTimeout(r, 400));

  const shots = ['01-inicial.png', '03-paso-loss.png', '04-gradientes.png', '06-chart-tras-epocas.png', '07-xor.png', '11-quiz.png', '14-tour.png', '16-mobile.png'];
  const missing = shots.filter(s => !fs.existsSync(path.join(SHOTS, s)));
  if (missing.length) console.log('capturas ausentes:', missing.join(', '));

  for (const shot of shots) {
    const p = path.join(SHOTS, shot);
    if (!fs.existsSync(p)) continue;
    const b64 = fs.readFileSync(p).toString('base64');

    const counts = await page.evaluate(async (b64, PALETTE) => {
      const img = new Image();
      img.src = 'data:image/png;base64,' + b64;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const { data } = ctx.getImageData(0, 0, c.width, c.height);

      const acc = {};
      for (const k in PALETTE) acc[k] = 0;
      let lit = 0, dark = 0, total = 0;

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const sum = r + g + b;
        total++;
        if (sum > 500) lit++; else if (sum < 120) dark++;
        // tolerancia de 60 por canal
        for (const k in PALETTE) {
          const p = PALETTE[k];
          if (Math.abs(r - p[0]) < 60 && Math.abs(g - p[1]) < 60 && Math.abs(b - p[2]) < 60) { acc[k]++; break; }
        }
      }
      return {
        size: img.width + 'x' + img.height,
        litPct: (lit / total * 100).toFixed(2),
        darkPct: (dark / total * 100).toFixed(2),
        palette: acc
      };
    }, b64, PALETTE);

    const total = Object.values(counts.palette).reduce((a, b) => a + b, 0);
    console.log('\n' + shot, counts.size, 'luz=' + counts.litPct + '%', 'oscuro=' + counts.darkPct + '%');
    console.log('  paleta:', Object.entries(counts.palette)
      .filter(([, v]) => v > 0)
      .map(([k, v]) => k + '=' + v + '(' + (v / total * 100).toFixed(1) + '%)').join(' '));
  }

  await browser.close();
})().catch(e => { console.error('FALLO:', e.message); process.exit(1); });