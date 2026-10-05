/* Dos diagnose-directed:
   1. Dónde viven realmente las etiquetas de borde (el selector anterior
      apuntaba a #svg-edges y por eso reportaba cero deltas).
   2. Por qué XOR se queda en loss ~0.1: imprimo activaciones y deltas por
      patrón para ver si las dos sigmoides ocultas se apagan. */
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
  const errors = [];
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });

  await page.goto(FILE, { waitUntil: 'networkidle0', timeout: 60000 });
  await new Promise(r => setTimeout(r, 1800));
  await page.evaluate(() => { const b = document.querySelector('.driver-popover-close-btn'); if (b) b.click(); });
  await new Promise(r => setTimeout(r, 300));

  // --- 1. Localizacion de las etiquetas ---
  for (let i = 0; i < 6; i++) await page.click('#btn-next');
  await new Promise(r => setTimeout(r, 300));
  const loc = await page.evaluate(() => {
    const all = [...document.querySelectorAll('text.edge-label')];
    const deltas = all.filter(t => /^[+-]\d/.test(t.textContent.trim()));
    const where = el => {
      for (let p = el; p; p = p.parentElement) if (p.id) return '#' + p.id;
      return '?';
    };
    return {
      total: all.length,
      deltas: deltas.length,
      deltaParents: [...new Set(deltas.map(where))],
      samples: deltas.slice(0, 8).map(t => t.textContent.trim()),
      weightSamples: all.filter(t => !/^[+-]\d/.test(t.textContent.trim())).slice(0, 8).map(t => t.textContent.trim()),
      updatingLit: document.querySelectorAll('#svg-edges .edge-lit').length
    };
  });
  console.log('--- Etiquetas de borde en el paso 6 ---');
  console.log('  total edge-label:', loc.total, ' deltas:', loc.deltas, ' en:', loc.deltaParents.join(','));
  console.log('  deltas:', loc.samples.join(' '));
  console.log('  pesos :', loc.weightSamples.join(' '));
  console.log('  aristas iluminadas:', loc.updatingLit);

  // --- 2. Diagnostico XOR ---
  const diag = await page.evaluate(async () => {
    const sel = document.querySelector('#preset-select');
    sel.value = 'xor';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    const out = { hidden: document.querySelector('#num-hidden').value, activation: document.querySelector('#hidden-act').value, lr: document.querySelector('#lr-slider').value, steps: [] };

    for (let cycle = 1; cycle <= 12; cycle++) {
      const row = { cycle, losses: [] };
      for (let p = 0; p < 4; p++) {
        const ps = document.querySelector('#pattern-select');
        ps.value = String(p);
        ps.dispatchEvent(new Event('change', { bubbles: true }));
        document.querySelector('#btn-epoch').click();
        row.losses.push(Number(document.querySelector('#hud-loss').textContent));
      }
      // Tras el ciclo leemos el estado interno de la ultima pasada
      row.h = document.querySelectorAll('#svg-nodes .node-g[data-key^="h"]').length;
      out.steps.push(row);
    }
    out.finalLoss = out.steps[out.steps.length - 1].losses.map(v => v.toFixed(4)).join(' ');
    out.best = Math.min(...out.steps[out.steps.length - 1].losses).toFixed(4);
    return out;
  });

  console.log('\n--- XOR: config y evolucion (4 patrones x 12 ciclos) ---');
  console.log('  ocultas=' + diag.hidden, ' activacion=' + diag.activation, ' lr=' + diag.lr);
  diag.steps.forEach(s => console.log(`   ciclo ${String(s.cycle).padStart(2)}: ${s.losses.map(v => v.toFixed(4)).join('  ')}`));
  console.log('  mejor loss del ultimo ciclo:', diag.best);

  // --- 3. XOR con mas flujo: 400 ciclos ---
  const long = await page.evaluate(async () => {
    let best = 1, bestAt = 0;
    const trail = [];
    for (let c = 0; c < 400; c++) {
      const ps = document.querySelector('#pattern-select');
      ps.value = String(c % 4);
      ps.dispatchEvent(new Event('change', { bubbles: true }));
      document.querySelector('#btn-epoch').click();
      const l = Number(document.querySelector('#hud-loss').textContent);
      if (l < best) { best = l; bestAt = c; }
      if (c % 100 === 99) trail.push(l.toFixed(4));
    }
    return { best: best.toFixed(5), bestAt, trail, epochs: document.querySelector('#hud-epoch').textContent };
  });
  console.log('\n--- XOR: 1600 pasos (400 ciclos) ---');
  console.log('  mejor loss:', long.best, 'en el paso', long.bestAt, '| loss cada 400:', long.trail.join(' '));

  // --- 4. XOR con lr mas alto ---
  const hi = await page.evaluate(async () => {
    const s = document.querySelector('#lr-slider');
    s.value = '1';
    s.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('#btn-reset').click();
    sel = document.querySelector('#preset-select');
    sel.value = 'xor';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    let best = 1, bestAt = 0;
    for (let c = 0; c < 400; c++) {
      const ps = document.querySelector('#pattern-select');
      ps.value = String(c % 4);
      ps.dispatchEvent(new Event('change', { bubbles: true }));
      document.querySelector('#btn-epoch').click();
      const l = Number(document.querySelector('#hud-loss').textContent);
      if (l < best) { best = l; bestAt = c; }
    }
    return { lr: document.querySelector('#lr-slider').value, best: best.toFixed(5), bestAt };
  });
  console.log('\n--- XOR con lr=1.0 ---');
  console.log('  mejor loss:', hi.best, 'en el paso', hi.bestAt);

  // --- 5. Aumento de neuronas ocultas ---
  const wide = await page.evaluate(async () => {
    const s = document.querySelector('#lr-slider');
    s.value = '0.5';
    s.dispatchEvent(new Event('input', { bubbles: true }));
    const nh = document.querySelector('#num-hidden');
    nh.value = '4';
    nh.dispatchEvent(new Event('change', { bubbles: true }));
    document.querySelector('#btn-reset').click();
    let best = 1;
    for (let c = 0; c < 400; c++) {
      const ps = document.querySelector('#pattern-select');
      ps.value = String(c % 4);
      ps.dispatchEvent(new Event('change', { bubbles: true }));
      document.querySelector('#btn-epoch').click();
      const l = Number(document.querySelector('#hud-loss').textContent);
      if (l < best) best = l;
    }
    return { hidden: document.querySelector('#num-hidden').value, best: best.toFixed(5) };
  });
  console.log('\n--- XOR con 4 ocultas, lr=0.5 ---');
  console.log('  ocultas=' + wide.hidden, ' mejor loss:', wide.best);

  console.log('\n=== ERRORES ===');
  console.log(errors.length ? errors.join('\n') : 'ninguno');
  await browser.close();
})().catch(e => { console.error('FALLO:', e.message); process.exit(1); });