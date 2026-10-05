/* Verifica los estados visuales dinamicos: flechas de flujo, brillo de nodos,
   etiquetas de delta en el paso de actualizacion, y que el canvas del chart
   no este en blanco. Tambien comprueba la convergencia del entrenamiento. */
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

  const visual = () => page.evaluate(() => {
    const flows = [...document.querySelectorAll('#svg-edges .edge-flow')];
    const lit = [...document.querySelectorAll('#svg-edges .edge.edge-lit')].length;
    const dim = [...document.querySelectorAll('#svg-edges .edge.edge-dim')].length;
    const activeNodes = [...document.querySelectorAll('#svg-nodes .node-g[data-active="true"]')];
    const halos = document.querySelectorAll('#svg-nodes circle').length;
    const deltaLabels = [...document.querySelectorAll('#svg-edges text.edge-label')]
      .filter(t => /^[+-]\d/.test(t.textContent.trim())).map(t => t.textContent.trim());
    const glows = [...document.querySelectorAll('#svg-nodes circle')].filter(c => c.getAttribute('filter')).length;
    return {
      flowCount: flows.length,
      flowColors: [...new Set(flows.map(f => f.getAttribute('stroke')))],
      flowAnimated: flows.filter(f => (f.getAttribute('style') || '').includes('animation')).length,
      litEdges: lit, dimEdges: dim,
      activeNodes: activeNodes.length,
      nodeLabels: activeNodes.map(n => n.dataset.nodeKey),
      glowFilters: glows,
      totalCircles: halos,
      deltaLabels,
      phase: document.querySelector('#epoch-counter').textContent
    };
  });

  const expect = {
    1: { flowCount: 4, flowColors: ['#22d3ee'], activeNodes: 2 },
    2: { flowCount: 4, flowColors: ['#22d3ee'], activeNodes: 2 },
    3: { flowCount: 0, flowColors: [], activeNodes: 2 },
    4: { flowCount: 4, flowColors: ['#f97316'], activeNodes: 2 },
    5: { flowCount: 4, flowColors: ['#f97316'], activeNodes: 2 },
    6: { flowCount: 0, flowColors: [], activeNodes: 2 }
  };

  console.log('--- Estados visuales por fase ---');
  for (let step = 1; step <= 6; step++) {
    await page.click('#btn-next');
    await new Promise(r => setTimeout(r, 300));
    const v = await visual();
    const e = expect[step];
    const flowOk = v.flowCount === e.flowCount;
    const colorOk = JSON.stringify(v.flowColors) === JSON.stringify(e.flowColors);
    const nodeOk = v.activeNodes === e.activeNodes;
    console.log(`  paso ${step}: flow=${v.flowCount} (${flowColorsStr(v.flowColors)}) lit=${v.litEdges} dim=${v.dimEdges} nodos=${v.activeNodes}[${v.nodeLabels}] glow=${v.glowFilters} deltaLabels=[${v.deltaLabels}]  ${flowOk && colorOk && nodeOk ? 'OK' : 'REVISAR'}`);
  }

  // Canvas del chart: verificar que tiene pixeles
  await page.evaluate(() => document.querySelector('[data-tab="chart"]').click());
  await new Promise(r => setTimeout(r, 500));
  const chartOk = await page.evaluate(() => {
    const c = document.querySelector('#loss-chart');
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let nonEmpty = 0, orange = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] > 0) nonEmpty++;
      if (d[i] > 180 && d[i + 1] > 90 && d[i + 1] < 200 && d[i + 2] < 120) orange++;
    }
    const ds = window.Chart ? Object.values(Chart.instances || {}) : [];
    return { w: c.width, h: c.height, nonEmptyPx: nonEmpty, orangePx: orange, total: c.width * c.height };
  });
  console.log('\n--- Chart ---');
  console.log('  canvas', chartOk.w + 'x' + chartOk.h, 'pintados=', chartOk.nonEmptyPx, 'naranja=', chartOk.orangePx, chartOk.nonEmptyPx > 500 ? 'OK dibuja' : 'VACIO');

  // Convergencia: 200 epocas en el ejemplo simple
  console.log('\n--- Convergencia (200 epocas, sigmoid lr=0.5) ---');
  const conv = await page.evaluate(() => {
    const out = [];
    for (let i = 0; i < 200; i++) document.querySelector('#btn-epoch').click();
    return {
      loss: document.querySelector('#hud-loss').textContent,
      epoch: document.querySelector('#hud-epoch').textContent,
      delta: document.querySelector('#hud-delta').textContent,
      note: document.querySelector('#hud-delta-note').textContent
    };
  });
  console.log('  ' + JSON.stringify(conv));
  const l0 = parseFloat(conv.loss);
  console.log('  perdida final < 0.01 ?', l0 < 0.01 ? 'SI converge' : 'NO converge (' + l0 + ')');

  // Divergencia con lr maxima
  console.log('\n--- Divergencia (60 epocas, lr=1.0) ---');
  await page.evaluate(() => {
    const s = document.querySelector('#lr-slider');
    s.value = '1';
    s.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('#btn-reset').click();
    for (let i = 0; i < 60; i++) document.querySelector('#btn-epoch').click();
  });
  await new Promise(r => setTimeout(r, 300));
  const div = await page.evaluate(() => ({
    loss: document.querySelector('#hud-loss').textContent,
    note: document.querySelector('#hud-delta-note').textContent
  }));
  console.log('  ' + JSON.stringify(div));

  // Auto-entrenar hasta convergencia
  console.log('\n--- Auto-entrenar hasta convergencia ---');
  await page.evaluate(() => {
    const s = document.querySelector('#lr-slider');
    s.value = '0.5';
    s.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('#btn-reset').click();
    document.querySelector('#btn-auto').click();
  });
  await new Promise(r => setTimeout(r, 9000));
  const auto = await page.evaluate(() => ({
    text: document.querySelector('#auto-text').textContent,
    epoch: document.querySelector('#hud-epoch').textContent,
    loss: document.querySelector('#hud-loss').textContent,
    mode: document.querySelector('#mode-label').textContent
  }));
  console.log('  ' + JSON.stringify(auto));

  // XOR: debe aprender los 4 patrones
  console.log('\n--- XOR: 4 patrones, 120 epocas ---');
  const xorRes = await page.evaluate(async () => {
    document.querySelector('#btn-reset').click();
    const sel = document.querySelector('#preset-select');
    sel.value = 'xor';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    const results = [];
    for (let i = 0; i < 30; i++) {
      for (const p of [0, 1, 2, 3]) {
        const ps = document.querySelector('#pattern-select');
        ps.value = String(p);
        ps.dispatchEvent(new Event('change', { bubbles: true }));
        document.querySelector('#btn-epoch').click();
      }
    }
    for (let p = 0; p < 4; p++) {
      const ps = document.querySelector('#pattern-select');
      ps.value = String(p);
      ps.dispatchEvent(new Event('change', { bubbles: true }));
      await new Promise(r => setTimeout(r, 30));
      const out = document.querySelector('#hud-loss').textContent;
      results.push({ patron: p, loss: out, texto: document.querySelector('#pattern-select').selectedOptions[0].textContent.trim() });
    }
    return { epoch: document.querySelector('#hud-epoch').textContent, results };
  });
  console.log('  epocas totales:', xorRes.epoch);
  xorRes.results.forEach(r => {
    const ok = parseFloat(r.loss) < 0.02;
    console.log(`   patron ${r.patron}: loss=${r.loss} ${ok ? 'OK aprendido' : 'sin aprender'}  "${r.texto}"`);
  });

  console.log('\n=== ERRORES ===');
  console.log(errors.length ? errors.join('\n') : 'ninguno');
  await browser.close();
})().catch(e => { console.error('FALLO:', e.message); process.exit(1); });

function flowColorsStr(a) { return a.length ? a.join(',') : '-'; }