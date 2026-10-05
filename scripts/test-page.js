const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'dist');
const FILE = 'file:///' + path.join(OUT, 'index.html').replace(/\\/g, '/');
const SHOTS = path.join(ROOT, '.shots');

(async () => {
  if (!fs.existsSync(SHOTS)) fs.mkdirSync(SHOTS, { recursive: true });

  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: process.env.CHROME_PATH || undefined,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--allow-file-access-from-files'
    ]
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });
  page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));
  page.on('requestfailed', r => errors.push('REQFAIL: ' + r.url().slice(0, 90) + ' :: ' + r.failure()?.errorText));

  await page.goto(FILE, { waitUntil: 'networkidle0', timeout: 60000 });
  await new Promise(r => setTimeout(r, 2500));

  // Cierra el tour automatico si aparecio
  await page.evaluate(() => {
    const d = document.querySelector('.driver-popover-close-btn');
    if (d) d.click();
  });
  await new Promise(r => setTimeout(r, 400));

  const step = async name => {
    await page.screenshot({ path: path.join(SHOTS, name + '.png'), fullPage: false });
  };

  const probe = async label => {
    const state = await page.evaluate(() => ({
      epoch: document.querySelector('#hud-epoch')?.textContent,
      loss: document.querySelector('#hud-loss')?.textContent,
      step: document.querySelector('#epoch-counter')?.textContent,
      mathLen: document.querySelector('#math-container')?.textContent.trim().length,
      mathHtml: document.querySelector('#math-container')?.innerHTML.length,
      nodes: document.querySelectorAll('#svg-nodes .node-g').length,
      edges: document.querySelectorAll('#svg-edges .edge-group').length,
      pills: document.querySelectorAll('.phase-pill').length,
      activePill: document.querySelector('.phase-pill[data-active="true"]')?.textContent.trim(),
      wCells: document.querySelectorAll('[data-mw]').length,
      bCells: document.querySelectorAll('[data-mb]').length,
      katex: document.querySelectorAll('#math-container .katex').length,
      quizOpts: document.querySelectorAll('.quiz-opt').length,
      gloss: document.querySelectorAll('#theory-glossary .katex').length,
      gradCells: document.querySelectorAll('#tab-gradients .tnum').length,
      inputs: document.querySelectorAll('[data-inp]').length,
      targets: document.querySelectorAll('[data-tgt]').length,
      comment: document.querySelector('#prof-comment')?.textContent.slice(0, 45),
      tourBtn: !!document.querySelector('#btn-tour'),
      ttNodes: document.querySelectorAll('[data-tip-title]').length
    }));
    console.log(label, JSON.stringify(state));
    return state;
  };

  console.log('=== ESTADO INICIAL ===');
  await probe('init');
  await step('01-inicial');

  // Recorre los 7 pasos
  for (let i = 1; i <= 7; i++) {
    await page.click('#btn-next');
    await new Promise(r => setTimeout(r, 320));
    await probe('paso' + i);
  }
  await step('02-tras-7-pasos');

  // Vuelve a paso 3 (loss) y captura el desglose matematico
  await page.evaluate(() => { for (let i = 0; i < 4; i++) document.querySelector('#btn-prev').click(); });
  await new Promise(r => setTimeout(r, 400));
  await probe('tras-prev');
  await step('03-paso-loss');

  // Tab de gradientes
  await page.evaluate(() => document.querySelector('[data-tab="gradients"]').click());
  await new Promise(r => setTimeout(r, 350));
  await step('04-gradientes');

  // Tab de chart
  await page.evaluate(() => document.querySelector('[data-tab="chart"]').click());
  await new Promise(r => setTimeout(r, 350));
  await step('05-chart');

  // Corre varias epocas
  await page.evaluate(() => { for (let i = 0; i < 40; i++) document.querySelector('#btn-epoch').click(); });
  await new Promise(r => setTimeout(r, 600));
  const afterEpochs = await probe('tras-40-epocas');
  await page.evaluate(() => document.querySelector('[data-tab="chart"]').click());
  await new Promise(r => setTimeout(r, 350));
  await step('06-chart-tras-epocas');

  // Preset XOR
  await page.select('#preset-select', 'xor');
  await new Promise(r => setTimeout(r, 500));
  const xorState = await probe('xor');
  await step('07-xor');

  // Pattern switch
  await page.select('#pattern-select', '1');
  await new Promise(r => setTimeout(r, 400));
  await probe('xor-patron1');

  // Entrada editable
  await page.evaluate(() => {
    const el = document.querySelector('[data-inp="0"]');
    el.value = '0.75';
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await new Promise(r => setTimeout(r, 400));
  await probe('xor-x-editado');

  // Activacion ReLU
  await page.select('#activation-select', 'relu');
  await new Promise(r => setTimeout(r, 400));
  await probe('relu');

  // tanh
  await page.select('#activation-select', 'tanh');
  await new Promise(r => setTimeout(r, 400));
  await probe('tanh');

  // Peso manual
  await page.select('#preset-select', 'simple');
  await new Promise(r => setTimeout(r, 400));
  await page.evaluate(() => {
    const el = document.querySelector('[data-mw="w1:0:0"]');
    el.value = '0.9';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await new Promise(r => setTimeout(r, 400));
  await probe('peso-manual');

  // Nodo clicable -> detalle
  await page.evaluate(() => {
    const n = [...document.querySelectorAll('#svg-nodes .node-g')].find(g => g.dataset.nodeKey === 'h1');
    n.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
  await new Promise(r => setTimeout(r, 450));
  const detail = await page.evaluate(() => ({
    visible: !document.querySelector('#node-detail').hidden,
    len: document.querySelector('#node-detail')?.textContent.length,
    katex: document.querySelectorAll('#nd-eq .katex').length,
    btnClear: !document.querySelector('#btn-clear-sel').hidden
  }));
  console.log('detalle-nodo', JSON.stringify(detail));
  await step('08-detalle-nodo');

  // Tooltip: hover sobre un nodo
  await page.hover('#svg-nodes .node-g');
  await new Promise(r => setTimeout(r, 500));
  const tip = await page.evaluate(() => {
    const t = document.querySelector('.tt');
    return t ? { show: t.dataset.show, text: t.textContent.slice(0, 70) } : null;
  });
  console.log('tooltip-nodo', JSON.stringify(tip));

  // Tooltip: hover sobre una conexion (usando coordenadas del linea)
  const edgeBox = await page.evaluate(() => {
    const l = document.querySelector('#svg-edges line[stroke="transparent"]');
    if (!l) return null;
    const r = l.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  });
  if (edgeBox) {
    await page.mouse.move(edgeBox.x, edgeBox.y);
    await new Promise(r => setTimeout(r, 550));
  }
  const tip2 = await page.evaluate(() => {
    const t = document.querySelector('.tt');
    return t ? { show: t.dataset.show, text: t.textContent.slice(0, 70) } : null;
  });
  console.log('tooltip-edge', JSON.stringify(tip2));

  // Tooltip en boton
  await page.hover('#lr-slider');
  await new Promise(r => setTimeout(r, 500));
  const tip3 = await page.evaluate(() => {
    const t = document.querySelector('.tt');
    return t ? { show: t.dataset.show, text: t.textContent.slice(0, 70) } : null;
  });
  console.log('tooltip-slider', JSON.stringify(tip3));
  await step('09-tooltip');

  // Atajos de teclado
  await page.evaluate(() => document.body.focus());
  await page.keyboard.press('ArrowRight');
  await new Promise(r => setTimeout(r, 300));
  const afterArrow = await page.evaluate(() => document.querySelector('#epoch-counter').textContent);
  console.log('tecla ArrowRight ->', afterArrow);

  await page.keyboard.press('Digit5');
  await new Promise(r => setTimeout(r, 400));
  const after5 = await page.evaluate(() => document.querySelector('#epoch-counter').textContent);
  console.log('tecla 5 ->', after5);

  await page.keyboard.press('KeyR');
  await new Promise(r => setTimeout(r, 400));
  const afterR = await page.evaluate(() => document.querySelector('#epoch-counter').textContent);
  console.log('tecla R ->', afterR);

  // Modal ayuda
  await page.click('#btn-help');
  await new Promise(r => setTimeout(r, 500));
  const modal = await page.evaluate(() => ({
    visible: !document.querySelector('#modal-help').hidden,
    keys: document.querySelectorAll('#modal-body kbd').length
  }));
  console.log('modal', JSON.stringify(modal));
  await step('10-modal');
  await page.keyboard.press('Escape');
  await new Promise(r => setTimeout(r, 400));
  const modalClosed = await page.evaluate(() => document.querySelector('#modal-help').hidden);
  console.log('modal-tras-Escape-cerrado=', modalClosed);

  // Quiz
  await page.evaluate(() => document.querySelector('[data-theory="quiz"]').click());
  await new Promise(r => setTimeout(r, 300));
  await page.evaluate(() => {
    const groups = document.querySelectorAll('[data-quiz]');
    groups.forEach(g => {
      const btns = g.querySelectorAll('.quiz-opt');
      const correct = [...btns].find(b => b.dataset.opt === 'B') || btns[0];
      correct.click();
    });
  });
  await new Promise(r => setTimeout(r, 400));
  const quizRes = await page.evaluate(() => ({
    answered: document.querySelectorAll('[data-result]').length,
    marks: [...document.querySelectorAll('[data-result]')].filter(r => r.textContent.trim()).length,
    green: [...document.querySelectorAll('[data-result]')].filter(r => r.style.color === 'rgb(103, 232, 249)').length
  }));
  console.log('quiz', JSON.stringify(quizRes));
  await step('11-quiz');

  // Glosario
  await page.evaluate(() => document.querySelector('[data-theory="glossary"]').click());
  await new Promise(r => setTimeout(r, 300));
  await step('12-glosario');

  // Auto-entrenar
  await page.evaluate(() => document.querySelector('[data-theory="derivation"]').click());
  await page.click('#btn-auto');
  await new Promise(r => setTimeout(r, 2200));
  const autoState = await page.evaluate(() => ({
    text: document.querySelector('#auto-text').textContent,
    epoch: document.querySelector('#hud-epoch').textContent,
    mode: document.querySelector('#mode-label').textContent
  }));
  console.log('auto-2.2s', JSON.stringify(autoState));
  await step('13-auto-entrenando');
  await page.click('#btn-auto');
  await new Promise(r => setTimeout(r, 300));
  console.log('auto-despues', await page.evaluate(() => document.querySelector('#auto-text').textContent));

  // Tour
  await page.evaluate(() => localStorage.removeItem('bp-tour-done'));
  await page.click('#btn-tour');
  await new Promise(r => setTimeout(r, 900));
  const tourState = await page.evaluate(() => ({
    popover: !!document.querySelector('.driver-popover'),
    title: document.querySelector('.driver-popover-title')?.textContent,
    highlighted: !!document.querySelector('.driver-active-element, [data-driver-popover-target]')
  }));
  console.log('tour', JSON.stringify(tourState));
  await step('14-tour');

  // Tour: avanzar varios pasos
  for (let i = 0; i < 5; i++) {
    await page.evaluate(() => {
      const btns = document.querySelectorAll('.driver-popover-footer button');
      const next = [...btns].find(b => /next|siguiente/i.test(b.textContent));
      if (next) next.click();
    });
    await new Promise(r => setTimeout(r, 450));
  }
  console.log('tour-paso6', await page.evaluate(() => document.querySelector('.driver-popover-title')?.textContent));
  await step('15-tour-paso');

  await page.evaluate(() => { const b = document.querySelector('.driver-popover-close-btn'); if (b) b.click(); });
  await new Promise(r => setTimeout(r, 400));

  // Responsive
  await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
  await new Promise(r => setTimeout(r, 700));
  await step('16-mobile');
  const mobile = await page.evaluate(() => ({
    overflowX: document.documentElement.scrollWidth > window.innerWidth + 2,
    sw: document.documentElement.scrollWidth,
    iw: window.innerWidth
  }));
  console.log('mobile', JSON.stringify(mobile));

  await page.setViewport({ width: 820, height: 1100 });
  await new Promise(r => setTimeout(r, 600));
  await step('17-tablet');

  await page.setViewport({ width: 1600, height: 1000 });
  await new Promise(r => setTimeout(r, 500));

  // Full page
  await page.screenshot({ path: path.join(SHOTS, '18-fullpage.png'), fullPage: true });

  console.log('\n=== ERRORES ===');
  console.log(errors.length ? errors.join('\n') : 'ninguno');

  await browser.close();
})().catch(e => { console.error('FALLO:', e.message); process.exit(1); });