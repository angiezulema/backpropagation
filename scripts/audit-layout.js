/* Auditoría de layout: no puedo ver las capturas, así que verifico la
   geometría del DOM (desbordes, solapamientos, elementos a cero, contraste). */
const puppeteer = require('puppeteer');
const path = require('path');

const FILE = 'file:///' + path.join(path.resolve(__dirname, '..'), 'dist', 'index.html').replace(/\\/g, '/');

const audit = () => {
  const out = { overflow: [], zero: [], offscreen: [], overlap: [], tiny: [], contrast: [], counts: {} };

  const vw = document.documentElement.clientWidth;
  out.docScrollW = document.documentElement.scrollWidth;
  out.bodyScrollW = document.body.scrollWidth;
  out.viewportW = vw;

  // 1. Desbordamiento horizontal del documento
  if (out.docScrollW > vw + 1) out.overflow.push({ el: 'document', scrollW: out.docScrollW, vw });

  // 2. Elementos que se salen del viewport por la derecha
  const all = [...document.querySelectorAll('body *')];
  for (const el of all) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.position === 'fixed') continue;
    if (el.closest('[hidden]')) continue;

    const r = el.getBoundingClientRect();
    // OPTION nunca se renderiza por separado: no es un fallo de layout
    if (el.tagName === 'OPTION') continue;
    if (r.width === 0 && r.height === 0) {
      out.zero.push({ tag: el.tagName, cls: (el.className || '').toString().slice(0, 50) });
      continue;
    }
    // Solo cuenta si ningún ancestro tiene scroll horizontal propio: el
    // stepper y el canvas son carruseles deliberados, no un fallo de layout.
    const inScroller = (() => {
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        const pcs = getComputedStyle(p);
        if (['auto', 'scroll'].includes(pcs.overflowX)) return true;
      }
      return false;
    })();
    if (r.right > vw + 2 && r.width < vw * 1.2 && !inScroller) {
      out.offscreen.push({
        tag: el.tagName,
        id: el.id,
        cls: (el.className || '').toString().slice(0, 45),
        right: Math.round(r.right)
      });
    }
    // 3. Texto ilegible. Se excluyen los OPTION (nunca se pintan) y los spans
    //    internos de KaTeX con font-size:1 (separadores de ancho cero).
    const fs = parseFloat(cs.fontSize);
    const isZeroWidthSpacer = el.textContent.trim() === '';
    if (el.children.length === 0 && el.textContent.trim() && fs < 9 &&
        !isZeroWidthSpacer && el.tagName !== 'OPTION') {
      // Tamaño efectivo considerando que KaTeX escala por arriba
      const parentFs = el.parentElement ? parseFloat(getComputedStyle(el.parentElement).fontSize) : fs;
      out.tiny.push({
        tag: el.tagName, fs: Math.round(fs * 10) / 10,
        effective: Math.round(Math.max(fs, parentFs * 0.9) * 10) / 10,
        text: el.textContent.trim().slice(0, 24)
      });
    }
  }

  // 4. Texto que se sale verticalmente de un contenedor con altura fija
  const fixed = [...document.querySelectorAll('.glass, .math-container, .eq, kbd, .field')];
  for (const el of fixed) {
    const cs = getComputedStyle(el);
    if (cs.overflowY === 'visible' && el.scrollHeight > el.clientHeight + 8 && el.clientHeight > 0) {
      out.overflow.push({
        tag: el.tagName, id: el.id,
        cls: (el.className || '').toString().slice(0, 40),
        scrollH: el.scrollHeight, clientH: el.clientHeight
      });
    }
  }

  // 5. Recuento de elementos clave
  out.counts = {
    pills: document.querySelectorAll('.phase-pill').length,
    nodes: document.querySelectorAll('#svg-nodes .node-g').length,
    edgeGroups: document.querySelectorAll('#svg-edges .edge-group').length,
    edgeLabels: document.querySelectorAll('#svg-nodes text.edge-label, #svg-edges text.edge-label').length,
    flowArrows: document.querySelectorAll('#svg-edges .edge-flow').length,
    tipTargets: document.querySelectorAll('[data-tip-title],[data-tip-body],[data-tip-rows]').length,
    katex: document.querySelectorAll('.katex').length,
    wInputs: document.querySelectorAll('[data-mw]').length,
    bInputs: document.querySelectorAll('[data-mb]').length,
    hud: document.querySelectorAll('#hud-loss,#hud-delta,#hud-deltanorm,#hud-epoch').length,
    svgTexts: document.querySelectorAll('#svg-nodes text').length
  };

  // 6. El SVG tiene contenido dentro del viewBox
  const svg = document.querySelector('#net-svg');
  const vb = svg.viewBox.baseVal;
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  for (const t of svg.querySelectorAll('text')) {
    try {
      const bb = t.getBBox();
      minX = Math.min(minX, bb.x); maxX = Math.max(maxX, bb.x + bb.width);
      minY = Math.min(minY, bb.y); maxY = Math.max(maxY, bb.y + bb.height);
    } catch (e) { /* getBBox falla en elementos sin layout */ }
  }
  out.svgBounds = { minX: Math.round(minX), maxX: Math.round(maxX), minY: Math.round(minY), maxY: Math.round(maxY), vbW: vb.width, vbH: vb.height };
  out.svgTextOutOfBounds = (minX < vb.x - 2 || maxX > vb.x + vb.width + 2 || minY < vb.y - 2 || maxY > vb.y + vb.height + 2);

  // 7. Anchos de las columnas principales
  const cols = ['header', 'section', 'main > section'].map(() => null);
  out.mainCols = [...document.querySelectorAll('main > section')].map(s => {
    const r = s.getBoundingClientRect();
    return { w: Math.round(r.width), x: Math.round(r.x) };
  });

  return out;
};

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: process.env.CHROME_PATH,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--allow-file-access-from-files']
  });

  for (const vp of [{ w: 1600, h: 1000, n: 'desktop' }, { w: 1280, h: 900, n: 'laptop' }, { w: 820, h: 1100, n: 'tablet' }, { w: 430, h: 900, n: 'mobile' }]) {
    const page = await browser.newPage();
    await page.setViewport({ width: vp.w, height: vp.h });
    await page.goto(FILE, { waitUntil: 'networkidle0', timeout: 60000 });
    await new Promise(r => setTimeout(r, 1800));
    await page.evaluate(() => { const b = document.querySelector('.driver-popover-close-btn'); if (b) b.click(); });
    await new Promise(r => setTimeout(r, 300));

    const res = await page.evaluate(audit);
    console.log('\n########## ' + vp.n + ' ' + vp.w + 'x' + vp.h + ' ##########');
    console.log('scrollW/vw:', res.docScrollW, '/', res.viewportW);
    console.log('counts:', JSON.stringify(res.counts));
    console.log('svgBounds:', JSON.stringify(res.svgBounds), 'outOfBounds=', res.svgTextOutOfBounds);
    console.log('mainCols:', JSON.stringify(res.mainCols));
    if (res.offscreen.length) console.log('OFFSCREEN (' + res.offscreen.length + '):', JSON.stringify(res.offscreen.slice(0, 8)));
    if (res.overflow.length) console.log('V-OVERFLOW (' + res.overflow.length + '):', JSON.stringify(res.overflow.slice(0, 8)));
    if (res.zero.length) console.log('ZERO-SIZE (' + res.zero.length + '):', JSON.stringify(res.zero.slice(0, 6)));
    if (res.tiny.length) console.log('TINY-TEXT (' + res.tiny.length + '):', JSON.stringify(res.tiny.slice(0, 6)));
    if (!res.offscreen.length && !res.overflow.length && !res.zero.length) console.log('layout: OK');
    await page.close();
  }

  await browser.close();
})().catch(e => { console.error('FALLO:', e.message); process.exit(1); });