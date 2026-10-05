/* ==========================================================================
   app.js — Motor del simulador de Backpropagation
   --------------------------------------------------------------------------
   Responsabilidades:
     1. Estado de la red y calculo matematico (forward, backprop, update)
     2. Render de la interfaz (stepper, SVG, matrices, HUD, charts)
     3. Tooltips propios con posicionamiento automatique
     4. Tour guiado con driver.js
     5. Atajos de teclado e interactividad del diagrama
   ========================================================================== */

(() => {
  'use strict';

  const C = window.CONTENT;
  const SVG_NS = 'http://www.w3.org/2000/svg';

  /* =====================================================================
     1. ESTADO
     ===================================================================== */

  const config = {
    preset: 'simple',
    activation: 'sigmoid',
    learningRate: 0.50,
    numInputs: 2,
    numHidden: 2,
    numOutputs: 2,
    patternIndex: 0
  };

  const net = {
    inputs: [0, 0],
    targets: [0],
    w1: [], b1: [], z1: [], a1: [], d1: [], gw1: [], gb1: [],
    w2: [], b2: [], z2: [], a2: [], d2: [], gw2: [], gb2: [],
    losses: [],
    lossTotal: 0
  };

  let currentStep = 0;
  let epochCount = 0;
  let prevLoss = null;
  let lossHistory = [];
  let weightHistory = [];
  let lossChart = null;
  let autoTimer = null;
  let autoPlaying = false;
  let autoSpeed = 100;
  let selectedNode = null;
  let lastFocusedEl = null;

  /* Copia profunda de los parametros antes de una actualizacion, para poder
     dibujar las flechas de "peso nuevo" y detectar el signo del cambio. */
  let wSnapshot = null;

  /* =====================================================================
     2. UTILIDADES
     ===================================================================== */

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const fmt = (n, d = 5) => (Number.isFinite(n) ? n.toFixed(d) : '—');
  const svgEl = (tag, attrs = {}) => {
    const el = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) if (attrs[k] != null) el.setAttribute(k, attrs[k]);
    return el;
  };

  /* Render de KaTeX tolerante a fallos: si KaTeX no esta disponible,
     devuelve el texto plano en vez de romper la vista. */
  function katexInto(el, tex, display = true) {
    if (!el) return;
    const raw = tex || '';
    if (window.katex) {
      try {
        katex.render(raw, el, {
          displayMode: display,
          throwOnError: false,
          strict: false,
          output: 'html'
        });
        return;
      } catch (e) { /* cae al texto plano */ }
    }
    el.textContent = raw.replace(/\\[a-zA-Z]+|[{}$]/g, '');
  }

  /* Resuelve el marcado en los comentarios del profesor: <kbd>, <em>, <strong>
     y formulas inline $...$ las pasan por KaTeX. */
  function renderRich(el, html) {
    if (!el) return;
    el.innerHTML = html || '';
    // Convierte $...$ en spans katex, ignorando los que ya estan dentro de
    // bloques display (KaTeX no debe anidarse).
    $$('.math-inline', el).forEach(katexInto.bind(null, null, null, false));
  }

  /* Convierte $...$ en KaTeX dentro de un contenedor, respetando <em>/<strong>. */
  function hydrateInlineMath(root) {
    if (!root || !window.katex) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue || !node.nodeValue.includes('$')) return NodeFilter.FILTER_REJECT;
        const p = node.parentElement;
        if (!p || p.closest('.katex, [data-katex], script, style')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    for (const node of nodes) {
      const parts = node.nodeValue.split(/(\$[^$]+\$)/g);
      const frag = document.createDocumentFragment();
      for (const part of parts) {
        if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
          const span = document.createElement('span');
          span.className = 'math-inline';
          try {
            katex.render(part.slice(1, -1), span, { throwOnError: false, strict: false, output: 'html' });
          } catch (e) {
            span.textContent = part;
          }
          frag.appendChild(span);
        } else if (part) {
          frag.appendChild(document.createTextNode(part));
        }
      }
      node.parentNode.replaceChild(frag, node);
    }
  }

  /* =====================================================================
     3. MATEMATICA DE LA RED
     ===================================================================== */

  function activate(z) {
    switch (config.activation) {
      case 'tanh':   return Math.tanh(z);
      case 'relu':   return Math.max(0, z);
      default:       return 1 / (1 + Math.exp(-z));
    }
  }

  function activateDeriv(a, z) {
    switch (config.activation) {
      case 'tanh': return 1 - a * a;
      case 'relu': return z > 0 ? 1 : 0;
      default:     return a * (1 - a);
    }
  }

  function shapeNet() {
    const { numInputs: nI, numHidden: nH, numOutputs: nO } = config;
    const mkMat = (r, c) => Array.from({ length: r }, () => Array(c).fill(0));
    net.w1 = mkMat(nI, nH); net.gw1 = mkMat(nI, nH);
    net.w2 = mkMat(nH, nO); net.gw2 = mkMat(nH, nO);
    net.b1 = new Array(nH).fill(0); net.gb1 = new Array(nH).fill(0);
    net.z1 = new Array(nH).fill(0); net.a1 = new Array(nH).fill(0); net.d1 = new Array(nH).fill(0);
    net.b2 = new Array(nO).fill(0); net.gb2 = new Array(nO).fill(0);
    net.z2 = new Array(nO).fill(0); net.a2 = new Array(nO).fill(0); net.d2 = new Array(nO).fill(0);
    net.losses = new Array(nO).fill(0);
    net.inputs = new Array(nI).fill(0);
    net.targets = new Array(nO).fill(0);
  }

  function forward() {
    for (let j = 0; j < config.numHidden; j++) {
      let s = net.b1[j];
      for (let i = 0; i < config.numInputs; i++) s += net.inputs[i] * net.w1[i][j];
      net.z1[j] = s;
      net.a1[j] = activate(s);
    }
    net.lossTotal = 0;
    for (let k = 0; k < config.numOutputs; k++) {
      let s = net.b2[k];
      for (let j = 0; j < config.numHidden; j++) s += net.a1[j] * net.w2[j][k];
      net.z2[k] = s;
      net.a2[k] = activate(s);
      const e = net.targets[k] - net.a2[k];
      net.losses[k] = 0.5 * e * e;
      net.lossTotal += net.losses[k];
    }
  }

  function backprop() {
    for (let k = 0; k < config.numOutputs; k++) {
      net.d2[k] = (net.a2[k] - net.targets[k]) * activateDeriv(net.a2[k], net.z2[k]);
      for (let j = 0; j < config.numHidden; j++) net.gw2[j][k] = net.a1[j] * net.d2[k];
      net.gb2[k] = net.d2[k];
    }
    for (let j = 0; j < config.numHidden; j++) {
      let acc = 0;
      for (let k = 0; k < config.numOutputs; k++) acc += net.d2[k] * net.w2[j][k];
      net.d1[j] = acc * activateDeriv(net.a1[j], net.z1[j]);
      for (let i = 0; i < config.numInputs; i++) net.gw1[i][j] = net.inputs[i] * net.d1[j];
      net.gb1[j] = net.d1[j];
    }
  }

  function snapshotWeights() {
    wSnapshot = {
      w1: net.w1.map(r => r.slice()),
      w2: net.w2.map(r => r.slice()),
      b1: net.b1.slice(),
      b2: net.b2.slice()
    };
  }

  function update() {
    const eta = config.learningRate;
    snapshotWeights();
    for (let j = 0; j < config.numHidden; j++) {
      for (let k = 0; k < config.numOutputs; k++) net.w2[j][k] -= eta * net.gw2[j][k];
    }
    for (let k = 0; k < config.numOutputs; k++) net.b2[k] -= eta * net.gb2[k];
    for (let i = 0; i < config.numInputs; i++) {
      for (let j = 0; j < config.numHidden; j++) net.w1[i][j] -= eta * net.gw1[i][j];
    }
    for (let j = 0; j < config.numHidden; j++) net.b1[j] -= eta * net.gb1[j];
  }

  function deltaNorm() {
    let s = 0;
    net.d1.forEach(v => { s += v * v; });
    net.d2.forEach(v => { s += v * v; });
    return Math.sqrt(s);
  }

  function fullCycle() {
    prevLoss = net.lossTotal;
    forward();
    backprop();
    update();
    epochCount++;
    lossHistory.push({ epoch: epochCount, loss: net.lossTotal });
    weightHistory.push(epochCount);
    pushChart();
    forward();
    backprop();
  }

  /* =====================================================================
     4. TOOLTIPS
     ===================================================================== */

  const Tip = (() => {
    let el = null, hideTimer = null;

    function node() {
      if (!el) {
        el = document.createElement('div');
        el.className = 'tt';
        el.setAttribute('role', 'tooltip');
        document.body.appendChild(el);
      }
      return el;
    }

    function show(target) {
      if (!target || !target.dataset) return;
      const title = target.dataset.tipTitle;
      const body  = target.dataset.tipBody;
      const rowHtml = target.dataset.tipRows;
      const tex  = target.dataset.tipFormula;
      if (!title && !body && !rowHtml && !tex) return;

      hideTimer && clearTimeout(hideTimer);
      const t = node();

      let formulaBlock = '';
      if (tex) {
        const holder = document.createElement('span');
        katexInto(holder, tex, false);
        formulaBlock =
          `<dl class="mt-1.5 border-t border-slate-700/60 pt-1.5">
             <div class="tt-row"><dt>formula</dt><dd>${holder.innerHTML}</dd></div>
           </dl>`;
      }

      t.innerHTML =
        (title ? `<div class="tt-title">${title}</div>` : '') +
        (body ? `<div>${body}</div>` : '') +
        (rowHtml ? `<dl class="mt-1.5 space-y-0.5">${rowHtml}</dl>` : '') +
        formulaBlock;

      t.dataset.show = 'true';
      position(target);
    }

    function position(target) {
      const t = node();
      const r = target.getBoundingClientRect();
      // Se mide el tamaño con el tooltip ya visible: si se mide con opacity:0
      // durante la transicion puede dar un ancho erroneo y descolocarlo.
      t.style.left = '0px';
      t.style.top = '0px';
      const tr = t.getBoundingClientRect();
      const gap = 10;

      let left = r.left + r.width / 2 - tr.width / 2;
      let top  = r.bottom + gap;

      // Voltea a arriba si no cabe abajo
      if (top + tr.height > innerHeight - 8) {
        top = r.top - tr.height - gap;
      }
      // Voltea a la derecha si no cabe a la izquierda
      if (left < 8) left = 8;
      if (left + tr.width > innerWidth - 8) left = innerWidth - tr.width - 8;
      // Si tampoco cabe arriba, lo pega dentro de la ventana
      if (top < 8) top = r.top + r.height / 2 - tr.height / 2;

      t.style.left = left + 'px';
      t.style.top  = top + 'px';
    }

    function hide() {
      hideTimer = setTimeout(() => {
        if (!el) return;
        el.dataset.show = 'false';
      }, 60);
    }

    /* mouseout tambien se dispara al moverse entre hijos del mismo elemento
       (por ejemplo del <circle> al <text> de un nodo SVG). Comprobamos que el
       destino no siga dentro del mismo elemento con tooltip. */
    function onOut(e) {
      const from = e.target.closest?.('[data-tip-title], [data-tip-body], [data-tip-rows]');
      if (!from) return;
      const to = e.relatedTarget;
      if (to && from.contains(to)) return;
      hide();
    }

    function refresh() {
      if (el && el.dataset.show === 'true' && lastFocusedEl) {
        show(lastFocusedEl);
      }
    }

    /* Delegacion: cualquier elemento con data-tip-body / data-tip-title / data-tip-rows */
    function bind() {
      document.addEventListener('mouseover', e => {
        const t = e.target.closest?.('[data-tip-title], [data-tip-body], [data-tip-rows]');
        if (!t) return;
        lastFocusedEl = t;
        show(t);
      });
      document.addEventListener('mouseout', onOut);
      // Los elementos SVG con eventos propios necesitan refresco manual
      document.addEventListener('focusin', e => {
        const t = e.target.closest?.('[data-tip-title], [data-tip-body], [data-tip-rows]');
        if (t) { lastFocusedEl = t; show(t); }
      });
      document.addEventListener('focusout', hide);
      addEventListener('scroll', () => lastFocusedEl && hide(), true);
      addEventListener('resize', hide);
    }

    return { bind, refresh, hide, show };
  })();

  /* Tooltips con contenido dinamico: se reescriben los dataset en cada render
     y se refresca el tooltip visible. */
  function setTip(el, title, body, rows) {
    if (!el) return;
    if (title != null) el.dataset.tipTitle = title;
    if (body != null) el.dataset.tipBody = body;
    if (rows != null) el.dataset.tipRows = rows;
  }

  const rows = pairs => pairs.map(([k, v]) =>
    `<div class="tt-row"><dt>${k}</dt><dd>${v}</dd></div>`).join('');

  /* =====================================================================
     5. RENDER · Stepper de fases
     ===================================================================== */

  function buildStepper() {
    const ol = $('#phase-stepper');
    ol.innerHTML = '';

    C.phases.forEach((ph, i) => {
      if (i > 0) {
        const arrow = document.createElement('li');
        arrow.className = 'text-slate-700';
        arrow.setAttribute('aria-hidden', 'true');
        arrow.innerHTML = '<svg viewBox="0 0 24 24" class="h-3 w-3" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="m9 5 7 7-7 7"/></svg>';
        ol.appendChild(arrow);
      }

      const li = document.createElement('li');
      li.id = 'phase-' + i;
      li.className = 'phase-pill';
      li.dataset.index = String(i);
      li.style.cursor = 'pointer';
      li.innerHTML = `
        <span class="grid h-4.5 w-4.5 shrink-0 place-items-center rounded-full bg-abyss-700 font-mono text-[9px] font-bold text-slate-400 transition">${i}</span>
        <span class="phase-label">${ph.short}</span>`;

      // El dataset solo admite cadenas: se guarda el LaTeX crudo y Tip.show()
      // lo compila con KaTeX en el momento de pintarlo.
      setTip(li,
        ph.title,
        `${ph.desc}<div class="tt-note">${ph.tip}</div>`,
        ph.formula || ''
      );
      if (ph.formula) li.dataset.tipFormula = ph.formula;

      li.addEventListener('click', () => {
        if (i > currentStep) {
          // Avanza hasta esa fase ejecutando los pasos intermedios
          while (currentStep < i) nextStep(false);
        } else {
          currentStep = i;
          render();
        }
      });

      ol.appendChild(li);
    });
  }

  function updateStepper() {
    C.phases.forEach((ph, i) => {
      const el = $('#phase-' + i);
      if (!el) return;
      const active = i === currentStep;
      const done = i < currentStep;
      el.dataset.active = String(active);
      el.dataset.done = String(done);

      const phaseColor = [
        'text-slate-400', 'text-cyan-glow-400', 'text-cyan-glow-300',
        'text-ember-400', 'text-ember-600', 'text-rose-400', 'text-cyan-glow-400'
      ][i];
      el.style.color = active ? phaseColor : '';
      const badge = el.querySelector('span');
      if (badge) {
        badge.style.background = active
          ? 'currentColor'
          : done ? 'rgba(103,232,249,0.16)' : '';
        badge.style.color = active ? '#04121a' : done ? '#67e8f9' : '';
      }
    });
  }

  /* =====================================================================
     6. RENDER · Diagrama SVG
     ===================================================================== */

  const LAYOUT = { inputX: 92, hiddenX: 380, outputX: 668, r: 27 };

  function layerPositions(n, x, hTop, hBot) {
    if (n === 1) return [{ x, y: (hTop + hBot) / 2 }];
    const gap = (hBot - hTop) / (n - 1);
    return Array.from({ length: n }, (_, i) => ({ x, y: hTop + i * gap }));
  }

  function nodeMeta() {
    const inP  = layerPositions(config.numInputs,  LAYOUT.inputX,  86, 254);
    const hidP = layerPositions(config.numHidden, LAYOUT.hiddenX, 86, 254);
    const outP = layerPositions(config.numOutputs, LAYOUT.outputX, 86, 254);
    return { inP, hidP, outP };
  }

  function renderSVG() {
    const gE = $('#svg-edges');
    const gN = $('#svg-nodes');
    gE.innerHTML = '';
    gN.innerHTML = '';

    const { inP, hidP, outP } = nodeMeta();
    const { r } = LAYOUT;

    const forwardHidden = currentStep === 1;
    const forwardOut    = currentStep === 2;
    const bpOut         = currentStep === 4;
    const bpHid         = currentStep === 5;
    const updating      = currentStep === 6;

    /* ---- Conexiones ---- */
    const addEdge = (from, to, w, grad, layer, weightRef, labelRef, fromKey, toKey) => {
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const x1 = from.x + ux * (r + 5);
      const y1 = from.y + uy * (r + 5);
      const x2 = to.x - ux * (r + 11);
      const y2 = to.y - uy * (r + 11);

      const isHiddenLayer = layer === 1;
      const active =
        (forwardHidden && isHiddenLayer) || (forwardOut && !isHiddenLayer) ||
        (bpOut && !isHiddenLayer) || (bpHid && isHiddenLayer) || updating;
      const direction = (forwardHidden && isHiddenLayer) || (forwardOut && !isHiddenLayer) ? 'fwd'
                      : (bpOut && !isHiddenLayer) || (bpHid && isHiddenLayer) ? 'bwd' : '';

      const base = document.createElementNS(SVG_NS, 'g');
      base.setAttribute('class', 'edge-group');

      const line = svgEl('line', {
        x1, y1, x2, y2,
        class: 'edge ' + (active ? 'edge-lit' : 'edge-dim'),
        stroke: '#3b5c7a',
        'stroke-width': clamp(Math.abs(w) * 2.6, 1.1, 5.4)
      });
      base.appendChild(line);

      if (direction) {
        const isFwd = direction === 'fwd';
        const flow = svgEl('line', {
          x1, y1, x2, y2,
          class: 'edge-flow',
          stroke: isFwd ? '#22d3ee' : '#f97316',
          'stroke-width': clamp(Math.abs(w) * 1.4 + 1.6, 2, 4.6),
          'stroke-dasharray': isFwd ? '7 15' : '7 15',
          'marker-end': isFwd ? 'url(#arrow-cyan)' : 'url(#arrow-ember)',
          style: `animation: ${isFwd ? 'flow-forward' : 'flow-backprop'} 1.1s linear infinite`
        });
        base.appendChild(flow);
      }

      // Etiqueta del peso
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      const wBefore = wSnapshot ? weightSnapshotValue(weightRef) : w;
      const deltaW = updating && wBefore != null ? w - wBefore : null;

      const label = svgEl('text', {
        x: mx, y: my - 5,
        class: 'edge-label',
        'text-anchor': 'middle',
        fill: active ? (direction === 'bwd' ? '#fdba74' : '#67e8f9') : '#64748b',
        opacity: active ? 1 : 0.42
      });
      label.textContent = w.toFixed(2);
      base.appendChild(label);

      if (updating && deltaW != null && Math.abs(deltaW) > 1e-9) {
        const up = deltaW > 0;
        const dLabel = svgEl('text', {
          x: mx, y: my + 8,
          class: 'edge-label',
          'text-anchor': 'middle',
          fill: up ? '#67e8f9' : '#fdba74',
          opacity: 0.95
        });
        dLabel.textContent = (up ? '+' : '') + deltaW.toFixed(3);
        base.appendChild(dLabel);
      }

      gE.appendChild(base);

      // Zona de hover generosa para facilitar el tooltip
      const hit = svgEl('line', {
        x1, y1, x2, y2,
        stroke: 'transparent',
        'stroke-width': 16,
        style: 'cursor:pointer'
      });
      hit.dataset.edgeLayer = String(layer);
      hit.dataset.edgeRef = weightRef;
      hit.dataset.edgeFrom = fromKey;
      hit.dataset.edgeTo = toKey;
      const gradTxt = grad != null ? fmt(grad) : '—';
      const wNewTxt = updating && deltaW != null
        ? fmt(w - config.learningRate * (grad ?? 0))
        : fmt(w);
      setTip(hit,
        labelRef,
        `${layer === 1 ? 'Conexion de la capa oculta' : 'Conexion de la capa de salida'}. El grosor de la linea es proporcional a |w| y el color indica el signo del peso.`,
        rows([
          ['peso w', fmt(w)],
          ['∂E/∂w', gradTxt],
          ['w nuevo', wNewTxt]
        ]));
      gE.appendChild(hit);
    };

    for (let i = 0; i < config.numInputs; i++) {
      for (let j = 0; j < config.numHidden; j++) {
        addEdge(inP[i], hidP[j], net.w1[i][j], net.gw1[i][j], 1,
          `w1:${i}:${j}`, `w<sub>${i + 1}${j + 1}</sub><sup>(1)</sup>`, `x${i}`, `h${j}`);
      }
    }
    for (let j = 0; j < config.numHidden; j++) {
      for (let k = 0; k < config.numOutputs; k++) {
        addEdge(hidP[j], outP[k], net.w2[j][k], net.gw2[j][k], 2,
          `w2:${j}:${k}`, `w<sub>${j + 1}${k + 1}</sub><sup>(2)</sup>`, `h${j}`, `o${k}`);
      }
    }

    /* ---- Nodos ---- */
    const layerName = ['Entradas', 'Capa oculta', 'Capa de salida'];
    const layerFormula = [
      'x_i = valor de entrada',
      'a_j = σ(z_j)',
      'a_k = σ(z_k)'
    ];

    const addNode = (pos, layer, idx, key) => {
      const g = document.createElementNS(SVG_NS, 'g');
      const isInput = layer === 0;
      const isOut = layer === 2;

      const active =
        (layer === 0 && currentStep === 0) ||
        (layer === 1 && (forwardHidden || bpHid)) ||
        (layer === 2 && (forwardOut || currentStep === 3 || bpOut || updating));

      const isSel = selectedNode === key;

      const fill = isInput ? '#0d1526' : isOut ? '#1a1030' : '#082430';
      const stroke = isInput ? '#22d3ee' : isOut ? '#f97316' : '#67e8f9';

      const circle = svgEl('circle', {
        cx: pos.x, cy: pos.y, r,
        class: 'node-body',
        fill,
        stroke: active ? stroke : '#2a3d5e',
        'stroke-width': active ? 2.6 : 1.8,
        filter: active ? (isOut ? 'url(#glow-ember)' : 'url(#glow-cyan)') : 'none'
      });
      g.appendChild(circle);

      // Nombre de la neurona
      const nm = svgEl('text', {
        x: pos.x, y: pos.y - 4,
        'text-anchor': 'middle',
        'font-size': 12,
        'font-weight': 700,
        fill: isInput ? '#a5f3fc' : isOut ? '#fed7aa' : '#a5f3fc',
        'font-family': 'JetBrains Mono, monospace'
      });
      nm.textContent = key;
      g.appendChild(nm);

      // Valor principal
      let valText;
      if (isInput) valText = fmt(net.inputs[idx], 3);
      else if (layer === 1) valText = 'a=' + fmt(net.a1[idx], 3);
      else valText = 'a=' + fmt(net.a2[idx], 3);

      const vt = svgEl('text', {
        x: pos.x, y: pos.y + 11,
        'text-anchor': 'middle',
        'font-size': 9.5,
        fill: '#94a3b8',
        'font-family': 'JetBrains Mono, monospace'
      });
      vt.textContent = valText;
      g.appendChild(vt);

      g.dataset.nodeKey = key;
      g.dataset.active = String(active);
      g.dataset.selected = String(isSel);
      g.classList.add('node-g', 'node-hit');

      setTip(g,
        key,
        `${layerName[layer]} · indice ${idx + 1}<div class="tt-note">Haz clic para fijar el detalle de esta neurona.</div>`,
        isInput
          ? rows([[layerFormula[0], fmt(net.inputs[idx], 5)]])
          : rows([
              ['z (preactivacion)', fmt(layer === 1 ? net.z1[idx] : net.z2[idx])],
              ['a (activacion)',    fmt(layer === 1 ? net.a1[idx] : net.a2[idx])],
              ['delta',             fmt(layer === 1 ? net.d1[idx] : net.d2[idx])],
              ['b (sesgo)',         fmt(layer === 1 ? net.b1[idx] : net.b2[idx])],
              [layer === 2 ? 'y (objetivo)' : '', layer === 2 ? fmt(net.targets[idx]) : '']
            ].filter(r => r[1])
          )
      );

      g.addEventListener('click', () => {
        selectedNode = (selectedNode === key) ? null : key;
        render();
      });

      gN.appendChild(g);
    };

    inP.forEach((p, i)  => addNode(p, 0, i, 'x' + (i + 1)));
    hidP.forEach((p, j) => addNode(p, 1, j, 'h' + (j + 1)));
    outP.forEach((p, k) => addNode(p, 2, k, 'o' + (k + 1)));

    // Titulos de capa
    const titles = [
      { x: LAYOUT.inputX,  t: 'ENTRADA',      c: '#22d3ee' },
      { x: LAYOUT.hiddenX, t: 'OCULTA',       c: '#67e8f9' },
      { x: LAYOUT.outputX, t: 'SALIDA',       c: '#fb923c' }
    ];
    for (const { x, t, c } of titles) {
      const tx = svgEl('text', {
        x, y: 34, 'text-anchor': 'middle',
        'font-size': 10, 'font-weight': 700,
        fill: c, 'font-family': 'Inter, sans-serif',
        'letter-spacing': 1.6, opacity: 0.8
      });
      tx.textContent = t;
      gN.appendChild(tx);
    }
  }

  function weightSnapshotValue(ref) {
    if (!wSnapshot || !ref) return null;
    const [layer, i, j] = ref.split(':');
    return Number(layer) === 1 ? wSnapshot.w1[+i][+j] : wSnapshot.w2[+i][+j];
  }

  /* =====================================================================
     7. RENDER · Detalle de neurona
     ===================================================================== */

  function renderNodeDetail() {
    const box = $('#node-detail');
    const btn = $('#btn-clear-sel');
    if (!selectedNode) { box.hidden = true; btn.hidden = true; return; }

    btn.hidden = false;
    const m = /^([xho])(\d+)$/.exec(selectedNode);
    if (!m) { box.hidden = true; btn.hidden = true; return; }
    const [, kind, numStr] = m;
    const i = Number(numStr) - 1;

    let html = '';
    if (kind === 'x') {
      html = `
        <div class="flex items-center justify-between gap-3">
          <div>
            <h3 class="text-sm font-bold text-cyan-glow-300">Entrada x${i + 1}</h3>
            <p class="text-[11px] text-slate-400">Valor que recibe la red del patron actual.</p>
          </div>
          <div class="flex items-center gap-2">
            <label class="text-[10px] text-slate-400 uppercase">valor</label>
            <input id="detail-x" type="number" step="0.05" class="field w-24 !text-center font-bold text-cyan-glow-300" value="${net.inputs[i]}">
          </div>
        </div>
        <div class="mt-3 grid gap-2 text-[11px] sm:grid-cols-2">
          ${net.w1[i].map((w, j) => `
            <div class="flex items-center justify-between rounded-lg border border-slate-700/50 bg-abyss-950/50 px-2.5 py-1.5">
              <span class="font-mono text-cyan-glow-300">x${i + 1} → h${j + 1}</span>
              <span class="tnum text-slate-300">w = ${fmt(w, 3)}</span>
            </div>`).join('')}
        </div>`;
    } else if (kind === 'h') {
      const wOut = net.w2[i].map((w, k) => `${fmt(w, 3)}`).join(' | ');
      html = `
        <div class="flex items-center justify-between gap-3">
          <div>
            <h3 class="text-sm font-bold text-cyan-glow-300">Neurona oculta h${i + 1}</h3>
            <p class="text-[11px] text-slate-400">Combina las entradas y aplica la activacion para formar su señal hacia la capa de salida.</p>
          </div>
        </div>
        <div class="mt-3 grid gap-3 text-[11px] sm:grid-cols-2">
          <div class="rounded-lg border border-slate-700/50 bg-abyss-950/50 p-2.5">
            <div class="mb-1.5 font-semibold text-slate-300">Pesos de entrada</div>
            ${net.w1.map((row, ii) => `
              <div class="flex justify-between py-0.5">
                <span class="font-mono text-slate-400">w<sub>${ii + 1}${i + 1}</sub><sup>(1)</sup></span>
                <span class="tnum ${row[i] >= 0 ? 'text-cyan-glow-300' : 'text-ember-300'}">${fmt(row[i], 4)}</span>
              </div>`).join('')}
            <div class="mt-1.5 flex justify-between border-t border-slate-700/50 pt-1.5">
              <span class="font-mono text-slate-400">b<sub>${i + 1}</sub><sup>(1)</sup></span>
              <span class="tnum text-slate-200">${fmt(net.b1[i], 4)}</span>
            </div>
          </div>
          <div class="rounded-lg border border-slate-700/50 bg-abyss-950/50 p-2.5">
            <div class="mb-1.5 font-semibold text-slate-300">Estado de activacion</div>
            <div class="flex justify-between py-0.5"><span class="text-slate-400">z</span><span class="tnum text-cyan-glow-300">${fmt(net.z1[i])}</span></div>
            <div class="flex justify-between py-0.5"><span class="text-slate-400">a = sigma(z)</span><span class="tnum text-cyan-glow-300">${fmt(net.a1[i])}</span></div>
            <div class="flex justify-between py-0.5"><span class="text-slate-400">delta</span><span class="tnum text-ember-300">${fmt(net.d1[i])}</span></div>
            <div class="mt-1.5 flex justify-between border-t border-slate-700/50 pt-1.5">
              <span class="text-slate-400">y hacia salidas</span><span class="tnum text-slate-300">w = ${wOut}</span>
            </div>
          </div>
        </div>
        <div class="mt-3">
          <div class="mb-1 text-[10px] font-semibold tracking-wide text-slate-400 uppercase">Suma ponderada reconstruida</div>
          <div id="nd-eq" class="eq"></div>
        </div>`;
    } else {
      html = `
        <div class="flex items-center justify-between gap-3">
          <div>
            <h3 class="text-sm font-bold text-ember-300">Neurona de salida o${i + 1}</h3>
            <p class="text-[11px] text-slate-400">Produce la prediccion de la red. Su error local delta dirige el primer paso de la retropropagacion.</p>
          </div>
          <div class="flex items-center gap-2">
            <label class="text-[10px] text-slate-400 uppercase">objetivo y</label>
            <input id="detail-y" type="number" step="0.05" min="0" max="1" class="field w-24 !text-center font-bold text-ember-300" value="${net.targets[i]}">
          </div>
        </div>
        <div class="mt-3 grid gap-3 text-[11px] sm:grid-cols-2">
          <div class="rounded-lg border border-slate-700/50 bg-abyss-950/50 p-2.5">
            <div class="mb-1.5 font-semibold text-slate-300">Pesos desde la capa oculta</div>
            ${net.w2.map((row, jj) => `
              <div class="flex justify-between py-0.5">
                <span class="font-mono text-slate-400">w<sub>${jj + 1}${i + 1}</sub><sup>(2)</sup></span>
                <span class="tnum ${row[i] >= 0 ? 'text-cyan-glow-300' : 'text-ember-300'}">${fmt(row[i], 4)}</span>
              </div>`).join('')}
            <div class="mt-1.5 flex justify-between border-t border-slate-700/50 pt-1.5">
              <span class="font-mono text-slate-400">b<sub>${i + 1}</sub><sup>(2)</sup></span>
              <span class="tnum text-slate-200">${fmt(net.b2[i], 4)}</span>
            </div>
          </div>
          <div class="rounded-lg border border-slate-700/50 bg-abyss-950/50 p-2.5">
            <div class="mb-1.5 font-semibold text-slate-300">Prediccion y error</div>
            <div class="flex justify-between py-0.5"><span class="text-slate-400">z</span><span class="tnum text-cyan-glow-300">${fmt(net.z2[i])}</span></div>
            <div class="flex justify-between py-0.5"><span class="text-slate-400">a = sigma(z)</span><span class="tnum text-cyan-glow-300">${fmt(net.a2[i])}</span></div>
            <div class="flex justify-between py-0.5"><span class="text-slate-400">y (objetivo)</span><span class="tnum text-ember-300">${fmt(net.targets[i])}</span></div>
            <div class="mt-1.5 flex justify-between border-t border-slate-700/50 pt-1.5">
              <span class="text-slate-400">delta</span><span class="tnum text-ember-300">${fmt(net.d2[i])}</span>
            </div>
          </div>
        </div>
        <div class="mt-3">
          <div class="mb-1 text-[10px] font-semibold tracking-wide text-slate-400 uppercase">Error local reconstruido</div>
          <div id="nd-eq" class="eq"></div>
        </div>`;
    }

    box.innerHTML = html;
    box.hidden = false;

    if (kind === 'h') {
      const terms = net.w1.map((row, ii) => `${fmt(net.inputs[ii], 3)} × ${fmt(row[i], 3)}`);
      terms.push(fmt(net.b1[i], 3));
      katexInto($('#nd-eq'), `z_{${i + 1}}^{(1)} = ${terms.join(' + ')} = ${fmt(net.z1[i])}`);
    } else if (kind === 'o') {
      const terms = net.w2.map((row, jj) => `${fmt(net.a1[jj], 4)} × ${fmt(row[i], 3)}`);
      terms.push(fmt(net.b2[i], 3));
      katexInto($('#nd-eq'),
        `\\delta_{${i + 1}}^{(2)} = (${fmt(net.a2[i], 4)} - ${fmt(net.targets[i], 3)}) \\times ${fmt(activateDeriv(net.a2[i], net.z2[i]), 4)} = ${fmt(net.d2[i])}`);
    }

    const dx = $('#detail-x');
    if (dx) dx.addEventListener('change', e => {
      const v = parseFloat(e.target.value);
      if (Number.isFinite(v)) { net.inputs[i] = v; refresh(); }
    });
    const dy = $('#detail-y');
    if (dy) dy.addEventListener('change', e => {
      const v = clamp(parseFloat(e.target.value) || 0, 0, 1);
      net.targets[i] = v; refresh();
    });
  }

  /* =====================================================================
     8. RENDER · Paneles de entradas y objetivos
     ===================================================================== */

  function renderInputsPanel() {
    const ph = $('#inputs-panel');
    if (!ph) return;
    const p = C.inputsPanel;

    const xFields = net.inputs.map((v, i) => `
      <div class="min-w-0">
        <label class="mb-1 block text-[10px] font-semibold text-slate-400 uppercase">x${i + 1}</label>
        <input type="number" step="0.05" class="field !text-center font-bold text-cyan-glow-300"
               data-inp="${i}" value="${v}" aria-label="Entrada ${i + 1}">
      </div>`).join('');

    const yFields = net.targets.map((v, k) => `
      <div class="min-w-0">
        <label class="mb-1 block text-[10px] font-semibold text-slate-400 uppercase">y${k + 1}</label>
        <input type="number" step="0.05" min="0" max="1" class="field !text-center font-bold text-ember-300"
               data-tgt="${k}" value="${v}" aria-label="Objetivo ${k + 1}">
      </div>`).join('');

    ph.innerHTML = `
      <div class="min-w-0">
        <div class="mb-1.5 text-[10px] font-semibold text-cyan-glow-400 uppercase">${p.xLabel}</div>
        <div class="grid gap-2" style="grid-template-columns:repeat(${Math.max(net.inputs.length, 1)},minmax(0,1fr))">${xFields}</div>
      </div>
      <div class="min-w-0">
        <div class="mb-1.5 text-[10px] font-semibold text-ember-400 uppercase">${p.yLabel}</div>
        <div class="grid gap-2" style="grid-template-columns:repeat(${Math.max(net.targets.length, 1)},minmax(0,1fr))">${yFields}</div>
      </div>`;

    $$('[data-inp]', ph).forEach(el => el.addEventListener('change', e => {
      const v = parseFloat(e.target.value);
      if (Number.isFinite(v)) { net.inputs[+el.dataset.inp] = v; refresh(); }
    }));
    $$('[data-tgt]', ph).forEach(el => el.addEventListener('change', e => {
      const v = clamp(parseFloat(e.target.value) || 0, 0, 1);
      net.targets[+el.dataset.tgt] = v; refresh();
    }));
  }

  /* =====================================================================
     9. RENDER · Tabla de matrices W, B
     ===================================================================== */

  function buildMatrices() {
    const ph = $('#tab-matrices');
    ph.innerHTML = `
      <div class="rounded-xl border border-slate-700/50 bg-abyss-950/50 p-3">
        <h4 class="mb-1 flex items-center justify-between text-xs font-bold text-slate-200">
          <span>Entradas y objetivos</span>
          <span class="text-[10px] font-normal text-slate-400">editable</span>
        </h4>
        <p class="mb-2.5 text-[10px] leading-relaxed text-slate-500">Modifica estos valores para observar como cambia todo el calculo.</p>
        <div id="inputs-panel" class="grid grid-cols-1 gap-3 sm:grid-cols-2"></div>
      </div>

      <div class="rounded-xl border border-cyan-glow-600/25 bg-cyan-glow-900/15 p-3">
        <h4 class="mb-2 flex items-center justify-between text-xs font-bold text-cyan-glow-300">
          <span>Pesos de la capa oculta · W<sup>(1)</sup></span>
          <span class="text-[10px] font-normal text-slate-400">entrada → oculta</span>
        </h4>
        <div id="m-w1"></div>
        <div class="mt-2.5 border-t border-slate-700/40 pt-2" id="m-b1"></div>
      </div>

      <div class="rounded-xl border border-ember-600/25 bg-ember-900/15 p-3">
        <h4 class="mb-2 flex items-center justify-between text-xs font-bold text-ember-300">
          <span>Pesos de la capa de salida · W<sup>(2)</sup></span>
          <span class="text-[10px] font-normal text-slate-400">oculta → salida</span>
        </h4>
        <div id="m-w2"></div>
        <div class="mt-2.5 border-t border-slate-700/40 pt-2" id="m-b2"></div>
      </div>`;

    renderMatrixTables();
  }

  function renderMatrixTables() {
    const w1 = $('#m-w1'), w2 = $('#m-w2');
    const b1 = $('#m-b1'), b2 = $('#m-b2');
    if (!w1 || !w2 || !b1 || !b2) return;
    if (!net.w1.length || !net.w2.length || !net.b1.length || !net.b2.length) return;

    /* --- W(1): filas = entradas, columnas = ocultas --- */
    w1.innerHTML = `
      <div class="overflow-x-auto">
        <table class="w-full text-center text-[11px]">
          <thead>
            <tr class="border-b border-slate-700/50 text-slate-500">
              <th class="p-1 text-left font-normal">de \\ a</th>
              ${Array.from({ length: config.numHidden }, (_, j) =>
                `<th class="p-1 font-semibold text-cyan-glow-400">h${j + 1}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${net.w1.map((row, i) => `
              <tr class="border-b border-slate-800/60 last:border-0">
                <td class="p-1 text-left font-mono text-slate-400">x${i + 1}</td>
                ${row.map((v, j) => matrixCell('w1', i, j, v, 'cyan')).join('')}
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;

    /* --- W(2): filas = ocultas, columnas = salidas --- */
    w2.innerHTML = `
      <div class="overflow-x-auto">
        <table class="w-full text-center text-[11px]">
          <thead>
            <tr class="border-b border-slate-700/50 text-slate-500">
              <th class="p-1 text-left font-normal">de \\ a</th>
              ${Array.from({ length: config.numOutputs }, (_, k) =>
                `<th class="p-1 font-semibold text-ember-400">o${k + 1}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${net.w2.map((row, j) => `
              <tr class="border-b border-slate-800/60 last:border-0">
                <td class="p-1 text-left font-mono text-slate-400">h${j + 1}</td>
                ${row.map((v, k) => matrixCell('w2', j, k, v, 'ember')).join('')}
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;

    b1.innerHTML = biasRow('b1', net.b1, config.numHidden, 'cyan');
    b2.innerHTML = biasRow('b2', net.b2, config.numOutputs, 'ember');

    /* listeners */
    $$('[data-mw]', w1).concat($$('[data-mw]', w2)).forEach(inp => {
      inp.addEventListener('change', () => {
        const [layer, i, j] = inp.dataset.mw.split(':');
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) return;
        if (layer === 'w1') net.w1[+i][+j] = v; else net.w2[+i][+j] = v;
        refresh();
      });
      inp.addEventListener('input', () => {
        const [layer, i, j] = inp.dataset.mw.split(':');
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) return;
        if (layer === 'w1') net.w1[+i][+j] = v; else net.w2[+i][+j] = v;
        forward(); backprop(); renderHUD(); renderSVG(); renderMath();
      });
    });

    $$('[data-mb]', b1).concat($$('[data-mb]', b2)).forEach(inp => {
      inp.addEventListener('change', () => {
        const [layer, j] = inp.dataset.mb.split(':');
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) return;
        if (layer === 'b1') net.b1[+j] = v; else net.b2[+j] = v;
        refresh();
      });
      inp.addEventListener('input', () => {
        const [layer, j] = inp.dataset.mb.split(':');
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) return;
        if (layer === 'b1') net.b1[+j] = v; else net.b2[+j] = v;
        forward(); backprop(); renderHUD(); renderSVG(); renderMath();
      });
    });
  }

  function matrixCell(layer, i, j, v, tone) {
    const before = wSnapshot ? weightSnapshotValue(`${layer}:${i}:${j}`) : null;
    const changed = before != null && Math.abs(before - v) > 1e-9;
    const flash = changed ? (v > before ? 'up' : 'down') : '';
    const color = tone === 'cyan' ? 'text-cyan-glow-300' : 'text-ember-300';
    const label = layer === 'w1' ? `w<sub>${i + 1}${j + 1}</sub><sup>(1)</sup>` : `w<sub>${i + 1}${j + 1}</sub><sup>(2)</sup>`;
    return `
      <td class="p-1">
        <input type="number" step="0.05" data-mw="${layer}:${i}:${j}"
               class="matrix-cell field !px-1.5 !py-1 !text-center !text-[11px] font-mono ${color}"
               data-flash="${flash}" value="${v.toFixed(3)}"
               aria-label="Peso ${layer} fila ${i + 1} columna ${j + 1}">
      </td>`;
  }

  function biasRow(layer, arr, n, tone) {
    const color = tone === 'cyan' ? 'text-cyan-glow-300' : 'text-ember-300';
    return `
      <div class="flex flex-wrap items-center gap-2">
        <span class="text-[10px] font-semibold tracking-wide text-slate-400 uppercase">Sesgos b<sup>(${layer === 'b1' ? 1 : 2})</sup></span>
        ${Array.from({ length: n }, (_, i) => `
          <div class="flex items-center gap-1">
            <span class="font-mono text-[10px] text-slate-500">${layer === 'b1' ? 'h' : 'o'}${i + 1}</span>
            <input type="number" step="0.05" data-mb="${layer}:${i}"
                   class="field !w-20 !px-1.5 !py-1 !text-center !text-[11px] font-mono ${color}"
                   value="${arr[i].toFixed(3)}">
          </div>`).join('')}
      </div>`;
  }

  /* =====================================================================
     10. RENDER · Tabla de gradientes
     ===================================================================== */

  function renderGradients() {
    const box = $('#tab-gradients');
    if (!box) return;

    const w = matToTable('∂E/∂w<sup>(1)</sup>', net.gw1, config.numInputs, config.numHidden, 'x', 'h', '#f97316');
    const w2m = matToTable('∂E/∂w<sup>(2)</sup>', net.gw2, config.numHidden, config.numOutputs, 'h', 'o', '#f97316');
    const d1 = vectorToTable('delta<sup>(1)</sup> (oculta)', net.d1, 'h');
    const d2 = vectorToTable('delta<sup>(2)</sup> (salida)', net.d2, 'o');

    box.innerHTML = `
      <div class="rounded-xl border border-slate-700/50 bg-abyss-950/50 p-3">
        <h4 class="mb-2 text-xs font-bold text-slate-200">Gradiente de los pesos ocultos</h4>
        ${w}
        <p class="mt-2 text-[10px] leading-relaxed text-slate-500">Se multiplica por la tasa de aprendizaje eta y se resta al peso. El signo indica en que direccion hay que moverlo.</p>
      </div>
      <div class="rounded-xl border border-slate-700/50 bg-abyss-950/50 p-3">
        <h4 class="mb-2 text-xs font-bold text-slate-200">Gradiente de los pesos de salida</h4>
        ${w2m}
        <p class="mt-2 text-[10px] leading-relaxed text-slate-500">Un gradiente cercano a cero significa que ese peso apenas afecta al error actual.</p>
      </div>
      <div class="rounded-xl border border-cyan-glow-600/25 bg-cyan-glow-900/15 p-3">
        <h4 class="mb-2 text-xs font-bold text-cyan-glow-300">Deltas de la capa oculta</h4>
        ${d1}
        <p class="mt-2 text-[10px] leading-relaxed text-slate-500">Se acumulan sumando la contribucion de cada neurona de salida, ponderada por su peso.</p>
      </div>
      <div class="rounded-xl border border-ember-600/25 bg-ember-900/15 p-3">
        <h4 class="mb-2 text-xs font-bold text-ember-300">Deltas de la capa de salida</h4>
        ${d2}
        <p class="mt-2 text-[10px] leading-relaxed text-slate-500">Si el delta es negativo, la neurona produce una salida mayor que el objetivo: hay que bajar su peso.</p>
      </div>`;
  }

  function matToTable(title, mat, rows, cols, rowPrefix, colPrefix, color) {
    return `
      <div class="mb-1.5 text-[10px] font-semibold text-slate-400">${title}</div>
      <div class="overflow-x-auto">
        <table class="w-full text-center text-[11px]">
          <thead>
            <tr class="border-b border-slate-700/50 text-slate-500">
              <th class="p-1 text-left font-normal">de \\ a</th>
              ${Array.from({ length: cols }, (_, j) => `<th class="p-1 text-slate-500">${colPrefix}${j + 1}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${mat.map((r, i) => `
              <tr class="border-b border-slate-800/60 last:border-0">
                <td class="p-1 text-left font-mono text-slate-500">${rowPrefix}${i + 1}</td>
                ${r.map(v => `<td class="tnum p-1" style="color:${v >= 0 ? '#fdba74' : '#67e8f9'}">${fmt(v, 4)}</td>`).join('')}
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  }

  function vectorToTable(title, arr, prefix) {
    return `
      <div class="mb-1.5 text-[10px] font-semibold text-slate-400">${title}</div>
      <div class="flex flex-wrap gap-2">
        ${arr.map((v, i) => `
          <div class="rounded-lg border border-slate-700/50 bg-abyss-900/60 px-2.5 py-1.5 text-center">
            <div class="font-mono text-[10px] text-slate-500">${prefix}${i + 1}</div>
            <div class="tnum text-[13px] font-bold" style="color:${v >= 0 ? '#fdba74' : '#67e8f9'}">${fmt(v, 5)}</div>
          </div>`).join('')}
      </div>`;
  }

  /* =====================================================================
     11. RENDER · HUD de metricas
     ===================================================================== */

  function renderHUD() {
    const loss = net.lossTotal;
    $('#hud-loss').textContent = fmt(loss);

    // Escala logaritmica para que la perdida siga siendo legible al bajar
    const pct = clamp((1 - Math.log10(Math.max(loss, 1e-6)) / Math.log10(Math.max(prevLoss ?? loss, 1e-6) || 1e-6, 10)) * 100, 0, 100);
    $('#hud-loss-meter').style.width = (isFinite(pct) ? pct : 100) + '%';

    const dEl = $('#hud-delta');
    const dNote = $('#hud-delta-note');
    if (prevLoss == null || epochCount === 0) {
      dEl.textContent = '—';
      dEl.className = 'tnum mt-1 text-xl font-bold text-slate-500';
      dNote.textContent = 'aun sin comparar';
    } else {
      const d = loss - prevLoss;
      dEl.textContent = (d >= 0 ? '+' : '') + fmt(d);
      const good = d < 0;
      dEl.className = 'tnum mt-1 text-xl font-bold ' + (good ? 'text-cyan-glow-300' : 'text-ember-300');
      dNote.textContent = good ? 'la perdida baja · aprendiendo' : 'la perdida sube · revisa eta';
    }

    const dn = deltaNorm();
    $('#hud-deltanorm').textContent = fmt(dn);
    $('#hud-deltanorm-meter').style.width = clamp(dn * 300, 0, 100) + '%';

    $('#hud-epoch').textContent = String(epochCount);

    const status = $('#hud-status');
    if (autoPlaying) status.textContent = 'entrenando automaticamente…';
    else if (epochCount === 0) status.textContent = 'sin entrenar todavia';
    else status.textContent = `${lossHistory.length} puntos en la curva`;

    $('#epoch-counter').textContent = `Epoca ${epochCount} · Paso ${currentStep}/6`;
  }

  /* =====================================================================
     12. RENDER · Desglose matematico por paso
     ===================================================================== */

  function renderMath() {
    const box = $('#math-container');
    const badge = $('#step-badge');
    const comment = $('#prof-comment');
    const ph = C.phases[currentStep];

    badge.textContent = ph.title.split('·')[0].trim();
    badge.style.borderColor = ['#2a3d5e', '#22d3ee', '#67e8f9', '#fb923c', '#f97316', '#fb7185', '#22d3ee'][currentStep] + '66';
    badge.style.color = ['#94a3b8', '#67e8f9', '#a5f3fc', '#fdba74', '#fb923c', '#fda4af', '#67e8f9'][currentStep];
    badge.style.background = 'rgba(5,8,15,0.7)';

    const card = (tone, title, body, eqId) => `
      <div class="math-card math-card-${tone}">
        <span class="mb-1 block text-[11px] font-bold ${
          tone === 'cyan' ? 'text-cyan-glow-300' : tone === 'ember' ? 'text-ember-300' : tone === 'rose' ? 'text-rose-300' : 'text-violet-300'
        }">${title}</span>
        <div class="eq">${body}</div>
        ${eqId ? `<div id="${eqId}" class="mt-1.5"></div>` : ''}
      </div>`;

    let html = '';

    if (currentStep === 0) {
      const xs = net.inputs.map((v, i) => `x${i + 1} = ${fmt(v, 3)}`).join(' &nbsp;·&nbsp; ');
      const ys = net.targets.map((v, k) => `y${k + 1} = ${fmt(v, 3)}`).join(' &nbsp;·&nbsp; ');
      html = `
        ${card('cyan', 'Entradas del modelo', `
          <div class="text-slate-400">vector de entrada</div>
          <div class="mt-1 text-[13px] font-bold text-cyan-glow-300">${xs}</div>
          <div class="mt-2 text-slate-400">objetivos deseados</div>
          <div class="mt-1 text-[13px] font-bold text-ember-300">${ys}</div>`)}
        ${card('violet', 'Arquitectura de la red',
          `<div class="text-slate-400">entradas <span class="text-slate-200">${config.numInputs}</span> → ocultas <span class="text-cyan-glow-300">${config.numHidden}</span> → salidas <span class="text-ember-300">${config.numOutputs}</span></div>
           <div class="text-slate-400 mt-1">parametros entrenables: <span class="text-slate-200">${config.numInputs * config.numHidden + config.numHidden + config.numHidden * config.numOutputs + config.numOutputs}</span></div>`)}
        ${card('cyan', 'Activacion elegida',
          `<div>sigma(z) = ${C.activations[config.activation].fx}</div>
           <div class="mt-0.5 text-slate-400">sigma&#39;(z) = ${C.activations[config.activation].dfx}</div>
           <div class="mt-0.5 text-slate-400">rango de salida: ${C.activations[config.activation].range}</div>`)}`;
    }

    else if (currentStep === 1) {
      html = net.a1.map((a, j) => {
        const terms = net.w1.map((row, i) => `${fmt(net.inputs[i], 3)} × ${fmt(row[j], 3)}`);
        return card('cyan', `Neurona oculta h${j + 1}`, `
          <div class="text-slate-400">z${j + 1} = (x1·w1${j + 1}) + (x2·w2${j + 1}) + b${j + 1}</div>
          <div class="mt-1 text-cyan-glow-300">z${j + 1} = (${terms.join(') + (')}) + ${fmt(net.b1[j], 3)}</div>
          <div class="mt-1.5 font-bold text-cyan-glow-200">= ${fmt(net.z1[j])}</div>
          <div class="mt-2 border-t border-slate-700/50 pt-1.5 text-slate-400">a${j + 1} = sigma(z${j + 1})</div>
          <div class="mt-0.5 font-bold text-cyan-glow-300">a${j + 1} = ${fmt(a)}</div>`, 'eq-h' + j);
      }).join('');
    }

    else if (currentStep === 2) {
      html = net.a2.map((a, k) => {
        const terms = net.w2.map((row, j) => `${fmt(net.a1[j], 4)} × ${fmt(row[k], 3)}`);
        return card('ember', `Neurona de salida o${k + 1}`, `
          <div class="text-slate-400">z${k + 1} = sum(a · w) + b${k + 1}</div>
          <div class="mt-1 text-ember-300">z${k + 1} = (${terms.join(') + (')}) + ${fmt(net.b2[k], 3)}</div>
          <div class="mt-1.5 font-bold text-ember-200">= ${fmt(net.z2[k])}</div>
          <div class="mt-2 border-t border-slate-700/50 pt-1.5 text-slate-400">prediccion a${k + 1} = sigma(z${k + 1})</div>
          <div class="mt-0.5 font-bold text-ember-300">a${k + 1} = ${fmt(a)}</div>`, 'eq-o' + k);
      }).join('');
    }

    else if (currentStep === 3) {
      const perOut = net.losses.map((l, k) => `
        <div class="border-b border-slate-700/50 py-1 last:border-0">
          <div class="text-slate-400">E${k + 1} = ½ · (y${k + 1} − a${k + 1})²</div>
          <div class="text-ember-300">= ½ · (${fmt(net.targets[k], 4)} − ${fmt(net.a2[k])})² = <span class="font-bold">${fmt(l)}</span></div>
        </div>`).join('');
      html = `
        ${card('ember', 'Error cuadratico por salida', perOut)}
        ${card('ember', 'Perdida total', `
          <div>E<sub>total</sub> = ${net.losses.map((_, i) => `E${i + 1}`).join(' + ')}</div>
          <div class="mt-1 text-xl font-bold text-ember-300">E = ${fmt(net.lossTotal)}</div>
          ${prevLoss != null ? `<div class="mt-2 border-t border-slate-700/50 pt-1.5">epoca anterior: <span class="tnum ${net.lossTotal < prevLoss ? 'text-cyan-glow-300' : 'text-ember-300'}">${fmt(prevLoss)}</span> · cambio: <span class="tnum">${(net.lossTotal - prevLoss >= 0 ? '+' : '') + fmt(net.lossTotal - prevLoss)}</span></div>` : ''}`)}`;
    }

    else if (currentStep === 4) {
      html = net.a2.map((a, k) => {
        const dv = activateDeriv(a, net.z2[k]);
        const gTerms = net.w2.map((row, j) =>
          `∂E/∂w<sup>(2)</sup><sub>${j + 1}${k + 1}</sub> = ${fmt(net.a1[j], 4)} × ${fmt(net.d2[k])} = <span class="font-bold text-white">${fmt(net.gw2[j][k])}</span>`);
        return card('rose', `Delta de salida delta${k + 1}`, `
          <div class="text-slate-400">delta${k + 1} = (a${k + 1} − y${k + 1}) · sigma&#39;(z${k + 1})</div>
          <div class="mt-1 text-rose-300">= (${fmt(a)} − ${fmt(net.targets[k])}) × ${fmt(dv)}</div>
          <div class="mt-0.5 font-bold text-rose-200">delta${k + 1} = ${fmt(net.d2[k])}</div>
          <div class="mt-2 border-t border-slate-700/50 pt-1.5 text-slate-400">gradientes de los pesos que llegan a o${k + 1}</div>
          <div class="mt-1 space-y-0.5 text-indigo-200">${gTerms.join('<br>')}</div>`);
      }).join('');
    }

    else if (currentStep === 5) {
      html = net.a1.map((a, j) => {
        const dv = activateDeriv(a, net.z1[j]);
        const sumTerms = net.w2.map((row, k) =>
          `${fmt(net.d2[k], 4)} × ${fmt(row[j], 3)}`);
        const gTerms = net.w1.map((row, i) =>
          `∂E/∂w<sup>(1)</sup><sub>${i + 1}${j + 1}</sub> = ${fmt(net.inputs[i], 4)} × ${fmt(net.d1[j])} = <span class="font-bold text-white">${fmt(net.gw1[i][j])}</span>`);
        const full = config.numOutputs > 1
          ? `(${sumTerms.join(') + (')})`
          : `(${sumTerms[0]})`;
        return card('rose', `Delta oculto delta${j + 1}`, `
          <div class="text-slate-400">delta${j + 1} = [sum de delta${''} · w] · sigma&#39;(z${j + 1})</div>
          <div class="mt-1 text-rose-300">= ${full} × ${fmt(dv)}</div>
          <div class="mt-0.5 font-bold text-rose-200">delta${j + 1} = ${fmt(net.d1[j])}</div>
          <div class="mt-2 border-t border-slate-700/50 pt-1.5 text-slate-400">gradientes de los pesos de entrada</div>
          <div class="mt-1 space-y-0.5 text-cyan-200">${gTerms.join('<br>')}</div>`);
      }).join('');
    }

    else if (currentStep === 6) {
      const eta = config.learningRate;
      const rowsUp = [];
      for (let j = 0; j < config.numHidden; j++) {
        for (let k = 0; k < config.numOutputs; k++) {
          const before = weightSnapshotValue(`w2:${j}:${k}`);
          if (before == null) continue;
          rowsUp.push(`w<sup>(2)</sup><sub>${j + 1}${k + 1}</sub>: ${fmt(before, 4)} − (${eta} × ${fmt(net.gw2[j][k], 4)}) = <span class="font-bold text-white">${fmt(net.w2[j][k])}</span>`);
        }
      }
      for (let i = 0; i < config.numInputs; i++) {
        for (let j = 0; j < config.numHidden; j++) {
          const before = weightSnapshotValue(`w1:${i}:${j}`);
          if (before == null) continue;
          rowsUp.push(`w<sup>(1)</sup><sub>${i + 1}${j + 1}</sub>: ${fmt(before, 4)} − (${eta} × ${fmt(net.gw1[i][j], 4)}) = <span class="font-bold text-white">${fmt(net.w1[i][j])}</span>`);
        }
      }
      html = `
        ${card('cyan', `Descenso de gradiente con eta = ${eta}`, `
          <div class="text-slate-400">w<sup>(nuevo)</sup> = w<sup>(viejo)</sup> − eta · ∂E/∂w</div>
          <div class="mt-1.5 space-y-0.5 text-cyan-200">${rowsUp.join('<br>')}</div>`)}
        ${card('ember', 'Resultado de la epoca', `
          <div class="text-slate-400">epoca completada: <span class="text-white">${epochCount}</span></div>
          <div class="mt-0.5 text-slate-400">perdida: <span class="font-bold text-ember-300">${fmt(net.lossTotal)}</span></div>
          ${prevLoss != null ? `<div class="mt-0.5 text-slate-400">variacion: <span class="tnum ${net.lossTotal <= prevLoss ? 'text-cyan-glow-300' : 'text-ember-300'}">${(net.lossTotal - prevLoss >= 0 ? '+' : '') + fmt(net.lossTotal - prevLoss)}</span></div>` : ''}
          <div class="mt-2 border-t border-slate-700/50 pt-1.5 text-slate-400">La red ha aprendido un poco mas. Pulsa de nuevo para iniciar la epoca siguiente.</div>`)}`;
    }

    box.innerHTML = html;
    hydrateInlineMath(box);

    renderRich(comment, C.comments[currentStep]);
    hydrateInlineMath(comment);
  }

  /* =====================================================================
     13. RENDER · Teoria, quiz y glosario
     ===================================================================== */

  function renderTheory() {
    const d = $('#theory-derivation');
    d.innerHTML = `
      <h3 class="flex items-center gap-2 text-sm font-bold text-white">
        <span class="h-1 w-1 rounded-full bg-cyan-glow-400"></span>
        Fundamentos del algoritmo de retropropagacion
      </h3>
      <p>
        El algoritmo de backpropagation calcula el gradiente del error global respecto a cada peso
        de la red aplicando iterativamente la <strong>regla de la cadena</strong> del calculo diferencial.
        No es un algoritmo distinto para cada capa: es la misma idea encadenada en dos niveles.
      </p>

      <div class="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div class="rounded-xl border border-cyan-glow-600/25 bg-cyan-glow-900/15 p-4">
          <h4 class="mb-2 text-xs font-bold text-cyan-glow-300">1. Capa de salida · gradiente directo</h4>
          <p class="mb-2">Dado el error cuadratico medio para una neurona de salida k, buscamos la derivada parcial del error respecto a su peso de la segunda capa.</p>
          <div id="eq-1" class="my-2"></div>
          <p class="mb-2">Aplicando la regla de la cadena sobre las tres dependencias encadenadas:</p>
          <div id="eq-2" class="my-2"></div>
          <p class="mb-2">Definimos el error local de la salida, y el gradiente quedaReduced a un producto:</p>
          <div id="eq-3" class="my-2"></div>
        </div>

        <div class="rounded-xl border border-ember-600/25 bg-ember-900/15 p-4">
          <h4 class="mb-2 text-xs font-bold text-ember-300">2. Capa oculta · propagacion hacia atras</h4>
          <p class="mb-2">Para un peso de la capa oculta, la influencia en el error pasa por <em>todas</em> las neuronas de salida a la vez. La cadena es mas larga:</p>
          <div id="eq-4" class="my-2"></div>
          <p class="mb-2">Agrupando, la senal de error acumulada de la capa oculta se escribe:</p>
          <div id="eq-5" class="my-2"></div>
          <p class="mb-2">Y la actualizacion por descenso de gradiente es:</p>
          <div id="eq-6" class="my-2"></div>
        </div>
      </div>

      <div class="rounded-xl border border-slate-700/50 bg-abyss-950/50 p-4">
        <h4 class="mb-2 text-xs font-bold text-slate-200">Intuicion del algoritmo</h4>
        <p>
          Piensa en la red como una maquina que ajusta un knob en cada conexion. La regla de la cadena nos dice
          en que direccion mover cada knob para reducir el error, y la tasa de aprendizaje nos dice cuanto
          girarlo en esa direccion. El descenso de gradiente se limita a seguir esa downhill local, sin necesidad
          de conocer la forma global de la superficie de perdida.
        </p>
      </div>`;

    katexInto($('#eq-1'), 'E = \\frac{1}{2} \\sum_k (y_k - a_k^{(2)})^2');
    katexInto($('#eq-2'), '\\frac{\\partial E}{\\partial w_{jk}^{(2)}} = \\frac{\\partial E}{\\partial a_k^{(2)}} \\cdot \\frac{\\partial a_k^{(2)}}{\\partial z_k^{(2)}} \\cdot \\frac{\\partial z_k^{(2)}}{\\partial w_{jk}^{(2)}}');
    katexInto($('#eq-3'), '\\frac{\\partial E}{\\partial w_{jk}^{(2)}} = a_j^{(1)} \\cdot \\delta_k^{(2)}');
    katexInto($('#eq-4'), '\\frac{\\partial E}{\\partial w_{ij}^{(1)}} = \\left( \\sum_{k} \\frac{\\partial E}{\\partial z_k^{(2)}} \\frac{\\partial z_k^{(2)}}{\\partial a_j^{(1)}} \\right) \\frac{\\partial a_j^{(1)}}{\\partial z_j^{(1)}} \\frac{\\partial z_j^{(1)}}{\\partial w_{ij}^{(1)}}');
    katexInto($('#eq-5'), '\\delta_j^{(1)} = \\left( \\sum_{k} \\delta_k^{(2)} \\cdot w_{jk}^{(2)} \\right) \\cdot \\sigma\'(z_j^{(1)})');
    katexInto($('#eq-6'), 'w_{ij}^{(new)} = w_{ij}^{(old)} - \\eta \\frac{\\partial E}{\\partial w_{ij}}');
    hydrateInlineMath(d);

    /* Quiz */
    const q = $('#theory-quiz');
    q.innerHTML = `
      <div>
        <h3 class="mb-1 text-sm font-bold text-white">Comprueba tu comprension del algoritmo</h3>
        <p class="text-[11px] text-slate-500">Cinco preguntas sobre los fundamentos que acabas de ver en el simulador.</p>
      </div>
      <div class="grid grid-cols-1 gap-4 md:grid-cols-2">
        ${C.quiz.map((item, qi) => `
          <div class="rounded-xl border border-slate-700/50 bg-abyss-950/50 p-4">
            <p class="mb-2.5 font-semibold text-slate-200"><span class="text-cyan-glow-400">${qi + 1}.</span> ${item.q}</p>
            <div class="space-y-1.5" data-quiz="${qi}">
              ${item.options.map(o => `
                <button class="quiz-opt" type="button" data-opt="${o.id}">${o.id}) ${o.text}</button>`).join('')}
            </div>
            <div class="mt-2 text-[11px] font-semibold" data-result="${qi}"></div>
          </div>`).join('')}
      </div>`;

    $$('[data-quiz]', q).forEach(group => {
      const qi = +group.dataset.quiz;
      const item = C.quiz[qi];
      const resultEl = $(`[data-result="${qi}"]`, q);

      $$('.quiz-opt', group).forEach(btn => {
        btn.addEventListener('click', () => {
          const chosen = btn.dataset.opt;
          const ok = chosen === item.correct;
          $$('.quiz-opt', group).forEach(b => {
            b.disabled = true;
            if (b.dataset.opt === item.correct) b.dataset.state = 'correct';
            else if (b === btn) b.dataset.state = 'wrong';
          });
          resultEl.style.color = ok ? '#67e8f9' : '#fdba74';
          resultEl.textContent = ok ? 'Correcto. ' + item.ok : 'Incorrecto. ' + item.ko;
        });
      });
    });

    /* Glosario */
    const g = $('#theory-glossary');
    g.innerHTML = `
      <h3 class="mb-1 text-sm font-bold text-white">Glosario de notacion</h3>
      <p class="text-[11px] text-slate-500">Los simbolos que aparecen en el diagrama, en las tablas y en las formulas del simulador.</p>
      <div class="overflow-x-auto rounded-xl border border-slate-700/50">
        <table class="w-full text-[11px]">
          <thead>
            <tr class="border-b border-slate-700/60 bg-abyss-950/60 text-slate-400">
              <th class="p-2.5 text-left font-semibold">Simbolo</th>
              <th class="p-2.5 text-left font-semibold">Nombre</th>
              <th class="p-2.5 text-left font-semibold">Significado</th>
            </tr>
          </thead>
          <tbody>
            ${C.glossary.map(item => `
              <tr class="border-b border-slate-800/60 last:border-0">
                <td class="p-2.5 text-cyan-glow-300" data-g="${item.sym}"></td>
                <td class="p-2.5 font-semibold text-slate-200">${item.name}</td>
                <td class="p-2.5 text-slate-400">${item.desc}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
    $$('[data-g]', g).forEach(td => katexInto(td, td.dataset.g, false));
  }

  /* =====================================================================
     14. RENDER · Modal de ayuda
     ===================================================================== */

  function buildHelp() {
    const b = $('#modal-body');
    const h = C.help;
    b.innerHTML = `
      <p>${h.intro}</p>
      <ul class="space-y-2.5">
        ${h.steps.map(s => `
          <li class="flex gap-2.5">
            <span class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-gradient-to-r from-cyan-glow-400 to-ember-500"></span>
            <span><strong class="${s.color}">${s.name}:</strong> ${s.body}</span>
          </li>`).join('')}
      </ul>
      <div class="rounded-lg border border-cyan-glow-600/25 bg-cyan-glow-900/15 p-3 text-[11px] leading-relaxed text-slate-300">
        ${h.extras}
      </div>
      <div>
        <div class="mb-2 text-[10px] font-semibold tracking-wide text-slate-400 uppercase">Atajos de teclado</div>
        <div class="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
          ${h.keys.map(([k, v]) => `
            <div class="flex items-center justify-between gap-3 rounded-lg border border-slate-700/50 bg-abyss-950/50 px-2.5 py-1.5">
              <kbd class="rounded border border-slate-600 bg-abyss-800 px-1.5 py-0.5 font-mono text-[10px] text-cyan-glow-300">${k}</kbd>
              <span class="text-[11px] text-slate-400">${v}</span>
            </div>`).join('')}
        </div>
      </div>`;
    hydrateInlineMath(b);
  }

  function openModal() {
    lastFocusedEl = document.activeElement;
    $('#modal-help').hidden = false;
    $('#modal-close').focus();
  }
  function closeModal() {
    $('#modal-help').hidden = true;
    if (lastFocusedEl && lastFocusedEl.focus) lastFocusedEl.focus();
  }

  /* =====================================================================
     15. TOUR GUIADO (driver.js)
     ===================================================================== */

  let tourDriver = null;

  /* driver.js espera el contenido dentro de `popover`, y los textos de los
     botones en `popover.footerButtons`. Se traducen todos al español. */
  function buildTourSteps() {
    const accent = c => c === 'ember' ? '#fb923c' : '#67e8f9';

    const steps = C.tour.map((s, i) => ({
      el: s.el,
      popover: {
        title: `<span style="color:${accent(s.accent)}">${s.title}</span>`,
        description: s.body,
        showProgress: true
      },
      onHighlightStarted: (el) => {
        if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }));

    steps.push({
      popover: {
        title: `<span style="color:#67e8f9">${C.tourOutro.title}</span>`,
        description: C.tourOutro.body,
        doneText: 'Empezar a practicar',
        closeText: 'Cerrar'
      },
      onLastClick: () => {
        try { tourDriver?.destroy(); } catch (e) { /* noop */ }
        tourDriver = null;
        localStorage.setItem('bp-tour-done', '1');
      }
    });

    return steps;
  }

  /* El bundle IIFE de driver.js se expone como window.driver.js.driver.
   Se aceptan tambien las formas de UMD y de ESM por si cambia el CDN. */
  function driverFactory() {
    const d = window.driver;
    if (typeof d === 'function') return d;
    if (typeof d?.driver === 'function') return d.driver;
    if (typeof d?.js?.driver === 'function') return d.js.driver;
    if (typeof d?.js === 'function') return d.js;
    return null;
  }

  function startTour() {
    const create = driverFactory();
    if (!create) { openModal(); return; }
    try { tourDriver?.destroy(); } catch (e) { /* noop */ }

    /* onDestroyed cubre tanto el boton de cerrar como onLastClick: sin esto
       tourDriver queda con referencia y los atajos de teclado se bloquean. */
    tourDriver = create({
      showProgress: true,
      allowClose: true,
      overlayColor: 'rgba(5, 8, 15, 0.78)',
      smoothScroll: true,
      animate: true,
      nextBtnText: 'Siguiente',
      prevBtnText: 'Atras',
      doneBtnText: 'Listo',
      closeBtnText: 'Cerrar tour',
      steps: buildTourSteps(),
      onDestroyed: () => { tourDriver = null; }
    });
    tourDriver.drive();
  }

  /* =====================================================================
     16. CHART
     ===================================================================== */

  function initChart() {
    const ctx = $('#loss-chart').getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(251, 146, 60, 0.28)');
    gradient.addColorStop(1, 'rgba(251, 146, 60, 0)');

    lossChart = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [{
        label: 'Perdida total (MSE)',
        data: [],
        borderColor: '#fb923c',
        backgroundColor: gradient,
        borderWidth: 2,
        fill: true,
        tension: 0.32,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: '#67e8f9'
      }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 220 },
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            grid: { color: 'rgba(148,163,184,0.07)' },
            ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' }, maxTicksLimit: 12 },
            title: { display: true, text: 'Epocas', color: '#64748b', font: { size: 10 } }
          },
          y: {
            type: 'logarithmic',
            grid: { color: 'rgba(148,163,184,0.07)' },
            ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } },
            title: { display: true, text: 'Perdida (escala log)', color: '#64748b', font: { size: 10 } }
          }
        },
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { size: 10, family: 'Inter' }, boxWidth: 12 } },
          tooltip: {
            backgroundColor: 'rgba(5,8,15,0.96)',
            borderColor: 'rgba(34,211,238,0.3)',
            borderWidth: 1,
            titleColor: '#e2e8f0',
            bodyColor: '#94a3b8',
            padding: 9,
            callbacks: {
              label: c => '  perdida: ' + Number(c.parsed.y).toExponential(3)
            }
          }
        }
      }
    });
  }

  function pushChart() {
    if (!lossChart) return;
    lossChart.data.labels.push(epochCount);
    lossChart.data.datasets[0].data.push(Math.max(net.lossTotal, 1e-9));
    lossChart.update('none');
  }

  /* =====================================================================
     17. PRESETS Y PATRONES
     ===================================================================== */

  function applyPreset(key) {
    config.preset = key;
    const p = C.presets[key];

    config.numInputs = p.w1.length;
    config.numHidden = p.w1[0].length;
    config.numOutputs = p.w2[0].length;
    config.patternIndex = 0;

    shapeNet();

    net.w1 = p.w1.map(r => r.slice());
    net.b1 = p.b1.slice();
    net.w2 = p.w2.map(r => r.slice());
    net.b2 = p.b2.slice();

    if (p.patterns && p.patterns.length) {
      net.inputs = p.patterns[0].x.slice();
      net.targets = p.patterns[0].y.slice();
    } else {
      net.inputs = p.inputs.slice();
      net.targets = p.targets.slice();
    }

    epochCount = 0;
    prevLoss = null;
    lossHistory = [];
    wSnapshot = null;
    currentStep = 0;
    selectedNode = null;

    $('#pattern-wrap').hidden = !(p.patterns && p.patterns.length);
    if (p.patterns && p.patterns.length) {
      const sel = $('#pattern-select');
      sel.innerHTML = p.patterns.map((pt, i) =>
        `<option value="${i}">${pt.x.join(' , ')} → ${pt.y[0]}   (${pt.label})</option>`).join('');
      sel.value = '0';
    }

    renderInputsPanel();
    refresh();
  }

  function applyPattern(idx) {
    const p = C.presets[config.preset];
    if (!p.patterns || !p.patterns[idx]) return;
    config.patternIndex = idx;
    net.inputs = p.patterns[idx].x.slice();
    net.targets = p.patterns[idx].y.slice();
    renderInputsPanel();
    refresh();
  }

  /* =====================================================================
     18. CONTROL DE PASOS
     ===================================================================== */

  function nextStep(doRender = true) {
    if (currentStep === 0)      { forward();  currentStep = 1; }
    else if (currentStep === 1) { currentStep = 2; }
    else if (currentStep === 2) { currentStep = 3; }
    else if (currentStep === 3) { backprop(); currentStep = 4; }
    else if (currentStep === 4) { currentStep = 5; }
    else if (currentStep === 5) { currentStep = 6; }
    else if (currentStep === 6) {
      fullCycle();
      currentStep = 0;
    }
    if (doRender) render();
  }

  function prevStep() {
    if (currentStep === 6) {
      // Volver desde "actualizacion" ya aplico el cambio: se recalcula el estado
      // anterior para que la vista no mienta.
      currentStep = 5;
      forward(); backprop();
      render();
      return;
    }
    if (currentStep > 0) { currentStep--; render(); }
  }

  function runEpoch() {
    fullCycle();
    currentStep = 6;
    render();
  }

  function toggleAuto() {
    if (autoPlaying) { stopAuto(); return; }
    autoPlaying = true;
    $('#auto-icon').innerHTML = '<path d="M7 4.5h3.5v15H7zM13.5 4.5H17v15h-3.5z"/>';
    $('#auto-text').textContent = 'Pausar';
    $('#btn-auto').dataset.active = 'true';
    $('#mode-dot').className = 'relative inline-block h-2 w-2 rounded-full bg-ember-400';
    $('#mode-label').textContent = 'Entrenando';

    autoTimer = setInterval(() => {
      if (config.preset === 'xor' && C.presets.xor.patterns.length > 1) {
        // En XOR se recorren los 4 patrones por ciclo
        const next = (config.patternIndex + 1) % C.presets.xor.patterns.length;
        $('#pattern-select').value = String(next);
        applyPattern(next);
      }
      fullCycle();
      currentStep = 6;
      render();

      if (net.lossTotal < 0.0005 || epochCount >= 500) stopAuto();
    }, autoSpeed);
  }

  function stopAuto() {
    clearInterval(autoTimer);
    autoTimer = null;
    autoPlaying = false;
    $('#auto-icon').innerHTML = '<path d="M7 4.5v15l13-7.5z"/>';
    $('#auto-text').textContent = 'Auto-entrenar';
    delete $('#btn-auto').dataset.active;
    $('#mode-dot').className = 'relative inline-block h-2 w-2 rounded-full bg-cyan-glow-400';
    $('#mode-label').textContent = 'Listo';
  }

  function resetAll() {
    stopAuto();
    applyPreset($('#preset-select').value);
    if (lossChart) {
      lossChart.data.labels = [];
      lossChart.data.datasets[0].data = [];
      lossChart.update();
    }
  }

  /* Recalcula todo sin perder el paso actual (tras editar un parametro) */
  function refresh() {
    forward();
    backprop();
    wSnapshot = null;
    render();
  }

  /* =====================================================================
     19. RENDER MAESTRO
     ===================================================================== */

  function render() {
    updateStepper();
    renderSVG();
    renderMath();
    renderHUD();
    renderMatrixTables();
    renderGradients();
    renderNodeDetail();
    Tip.refresh();
  }

  /* =====================================================================
     20. INTERACTIVIDAD
     ===================================================================== */

  function bindRipple() {
    document.addEventListener('pointerdown', e => {
      const btn = e.target.closest('.btn');
      if (!btn || btn.disabled) return;
      const r = btn.getBoundingClientRect();
      const span = document.createElement('span');
      span.className = 'ripple';
      span.style.left = (e.clientX - r.left) + 'px';
      span.style.top = (e.clientY - r.top) + 'px';
      span.style.width = '0px';
      span.style.height = '0px';
      btn.appendChild(span);
      setTimeout(() => span.remove(), 560);
    });
  }

  function bindTabs() {
    $$('.tab-btn[data-tab]').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        $$('.tab-btn[data-tab]').forEach(b => b.dataset.active = String(b === btn));
        ['matrices', 'chart', 'gradients'].forEach(t => { $('#tab-' + t).hidden = t !== tab; });
        if (tab === 'chart' && lossChart) lossChart.resize();
      });
    });

    $$('.tab-btn[data-theory]').forEach(btn => {
      btn.addEventListener('click', () => {
        const t = btn.dataset.theory;
        $$('.tab-btn[data-theory]').forEach(b => b.dataset.active = String(b === btn));
        ['derivation', 'quiz', 'glossary'].forEach(k => { $('#theory-' + k).hidden = k !== t; });
      });
    });
  }

  function bindKeys() {
    document.addEventListener('keydown', e => {
      if (tourDriver) return;
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      if (!$('#modal-help').hidden) {
        if (e.key === 'Escape') { e.preventDefault(); closeModal(); }
        return;
      }

      switch (e.key) {
        case 'ArrowRight': e.preventDefault(); stopAuto(); nextStep(); break;
        case 'ArrowLeft':  e.preventDefault(); stopAuto(); prevStep(); break;
        case ' ':
        case 'Spacebar':   e.preventDefault(); toggleAuto(); break;
        case 'r': case 'R': e.preventDefault(); resetAll(); break;
        case 't': case 'T': e.preventDefault(); startTour(); break;
        case 'Escape':    closeModal(); break;
        default:
          if (/^[1-7]$/.test(e.key)) {
            e.preventDefault();
            stopAuto();
            const target = +e.key - 1;
            while (currentStep < target) nextStep(false);
            if (currentStep > target) { currentStep = target; }
            render();
          }
      }
    });
  }

  function bindControls() {
    $('#preset-select').addEventListener('change', e => {
      stopAuto();
      applyPreset(e.target.value);
    });

    $('#activation-select').addEventListener('change', e => {
      config.activation = e.target.value;
      refresh();
    });

    $('#lr-slider').addEventListener('input', e => {
      config.learningRate = parseFloat(e.target.value);
      $('#lr-val').textContent = config.learningRate.toFixed(2);
    });

    $('#pattern-select').addEventListener('change', e => {
      stopAuto();
      applyPattern(+e.target.value);
    });

    $('#btn-next').addEventListener('click', () => { stopAuto(); nextStep(); });
    $('#btn-prev').addEventListener('click', () => { stopAuto(); prevStep(); });
    $('#btn-epoch').addEventListener('click', () => { stopAuto(); runEpoch(); });
    $('#btn-auto').addEventListener('click', toggleAuto);
    $('#btn-reset').addEventListener('click', resetAll);
    $('#btn-tour').addEventListener('click', startTour);
    $('#btn-help').addEventListener('click', openModal);
    $('#modal-close').addEventListener('click', closeModal);
    $('#modal-action').addEventListener('click', closeModal);
    $('#modal-backdrop').addEventListener('click', closeModal);
    $('#btn-clear-sel').addEventListener('click', () => { selectedNode = null; render(); });
  }

  /* =====================================================================
     21. ARRANQUE
     ===================================================================== */

  function init() {
    buildStepper();
    buildMatrices();
    renderInputsPanel();
    renderTheory();
    buildHelp();
    initChart();
    Tip.bind();
    bindRipple();
    bindTabs();
    bindKeys();
    bindControls();

    applyPreset($('#preset-select').value);

    // Tour automatico la primera visita
    if (!localStorage.getItem('bp-tour-done')) {
      setTimeout(startTour, 900);
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();