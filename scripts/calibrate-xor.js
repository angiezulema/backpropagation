/* Replica la red del simulador en Node para calibrar los pesos iniciales de
   XOR. Los pesos fijos del preset eran degenerados: w1 = [[1,1],[-1,-1]] hace
   que ambas ocultas calculen x1-x2, con lo que la salida es monotonica en un
   solo eje y XOR resulta inseparable por construccion.
   Aqui se buscan inicializaciones asimetricas, vivas (sin sigmoides
   saturadas) y que converjan rapido con la lr por defecto del simulador. */

const sig = z => 1 / (1 + Math.exp(-z));
const dsig = a => a * (1 - a);

const patterns = [
  { x: [0, 0], y: 0 }, { x: [0, 1], y: 1 },
  { x: [1, 0], y: 1 }, { x: [1, 1], y: 0 }
];

const mk = (w1, b1, w2, b2) => ({
  w1: w1.map(r => r.slice()), b1: b1.slice(),
  w2: w2.map(r => r.slice()), b2: b2.slice()
});
const clone = n => mk(n.w1, n.b1, n.w2, n.b2);

function fwd(net, x, y) {
  const z1 = [0, 0].map(j => net.b1[j] + x[0] * net.w1[0][j] + x[1] * net.w1[1][j]);
  const a1 = z1.map(sig);
  const z2 = net.b2[0] + a1[0] * net.w2[0][0] + a1[1] * net.w2[1][0];
  const a2 = sig(z2);
  return { z1, a1, a2, loss: 0.5 * (a2 - y) ** 2 };
}

/* Una epoca = visitar los 4 patrones en orden, como hace el auto-entrenar. */
function epoch(net, lr) {
  for (const p of patterns) {
    const f = fwd(net, p.x, p.y);
    const d2 = (f.a2 - p.y) * dsig(f.a2);
    const gw2 = [f.a1[0] * d2, f.a1[1] * d2];
    const gb2 = d2;
    const d1 = [0, 1].map(j => d2 * net.w2[j][0] * dsig(f.a1[j]));
    for (let j = 0; j < 2; j++) {
      net.w2[j][0] -= lr * gw2[j];
      net.b2[0] -= lr * gb2;
      for (let i = 0; i < 2; i++) net.w1[i][j] -= lr * (p.x[i] * d1[j]);
      net.b1[j] -= lr * d1[j];
    }
  }
}

const worst = net => Math.max(...patterns.map(p => fwd(net, p.x).loss));
const losses = net => patterns.map(p => fwd(net, p.x).loss);

/* Una inicializacion es "viva" si ninguna preactivacion oculta se satura:
   si |z1| es enorme la derivada a(1-a) se anula y la neurona no aprende. */
function saturation(net) {
  let worstDer = 1;
  for (const p of patterns) {
    const f = fwd(net, p.x);
    for (let j = 0; j < 2; j++) worstDer = Math.min(worstDer, dsig(f.a1[j]));
  }
  return { worstDer: +worstDer.toFixed(4), dead: worstDer < 0.02 };
}

function run(cfg, lr, epochs) {
  const net = clone(cfg);
  const marks = {};
  for (let e = 1; e <= epochs; e++) {
    epoch(net, lr);
    if (e === 1 || e === 5 || e === 20 || e === 50 || e === 100 || e === 200) marks[e] = +worst(net).toFixed(4);
  }
  return { net, marks, final: +worst(net).toFixed(5) };
}

/* ---- Deterministico: LCG para que el resultado sea reproducible ---- */
let seed = 20251005;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

const cands = {
  'actual (degen)': { w1: [[1, 1], [-1, -1]], b1: [-0.5, 0.5], w2: [[1], [1]], b2: [-0.5] },
  'AND/OR clasico': { w1: [[4, 2], [4, 2]], b1: [-6, -1], w2: [[-1], [1]], b2: [-0.5] },
  'AND/OR suave': { w1: [[3, 1.5], [3, 1.5]], b1: [-4, -0.8], w2: [[-1], [1]], b2: [-0.4] }
};

/* Candidatos deterministas de la familia "dos proyecciones distintas". */
cands['dos planos vivos'] = { w1: [[1.0, 0.6], [-1.0, 0.9]], b1: [-0.3, -0.6], w2: [[-1.0], [1.0]], b2: [-0.5] };
cands['x1-x2 / x1+x2'] = { w1: [[1, 1], [-1, 1]], b1: [-0.5, -0.5], w2: [[1], [1]], b2: [-1.0] };

/* Barrido aleatorio: pesos pequenos en [-s, s], sesgos en [-b, b]. */
let bestRandom = null;
for (let trial = 0; trial < 4000; trial++) {
  const s = 0.4 + (trial % 5) * 0.3;
  const b = 0.2 + (trial % 3) * 0.25;
  const cfg = {
    w1: [[-s + 2 * s * rnd(), -s + 2 * s * rnd()], [-s + 2 * s * rnd(), -s + 2 * s * rnd()]],
    b1: [-b + 2 * b * rnd(), -b + 2 * b * rnd()],
    w2: [[-1.2 + 1.4 * rnd()], [-1.2 + 1.4 * rnd()]],
    b2: [-0.5 + rnd()]
  };
  if (saturation(cfg).dead) continue;
  const r = run(cfg, 0.5, 100);
  const score = r.marks[20] * 2 + r.marks[50];
  if (!bestRandom || score < bestRandom.score) bestRandom = { cfg, score, r, s, b };
}

console.log('--- deterministas ---');
console.log('candidato            lr   ini    @1     @5     @20    @50    @100   @200   sat(der)');
for (const [name, cfg] of Object.entries(cands)) {
  for (const lr of [0.5, 0.8, 1.0]) {
    const r = run(cfg, lr, 200);
    const sat = saturation(cfg);
    console.log(
      `${name.padEnd(20)} ${String(lr).padEnd(4)} ${r.marks[1].toFixed(4).padEnd(6)} ` +
      `${String(r.marks[5]).padEnd(6)} ${String(r.marks[20]).padEnd(6)} ${String(r.marks[50]).padEnd(6)} ` +
      `${String(r.marks[100]).padEnd(6)} ${String(r.marks[200]).padEnd(6)} ${sat.dead ? 'MUERTA ' + sat.worstDer : 'viva ' + sat.worstDer}`
    );
  }
}

console.log('\n--- mejor aleatorio del barrido (4000 intentos) ---');
console.log('  rango w1 ±' + bestRandom.s.toFixed(1) + '  b1 ±' + bestRandom.b.toFixed(2));
console.log('  cfg:', JSON.stringify({
  w1: bestRandom.cfg.w1.map(r => r.map(v => +v.toFixed(3))),
  b1: bestRandom.cfg.b1.map(v => +v.toFixed(3)),
  w2: bestRandom.cfg.w2.map(r => r.map(v => +v.toFixed(3))),
  b2: bestRandom.cfg.b2.map(v => +v.toFixed(3))
}));
for (const lr of [0.5, 0.8, 1.0]) {
  const r = run(bestRandom.cfg, lr, 200);
  console.log(`  lr=${lr}: @1=${r.marks[1]} @5=${r.marks[5]} @20=${r.marks[20]} @50=${r.marks[50]} @100=${r.marks[100]} @200=${r.marks[200]}`);
}

console.log('\n--- detalle del ganador, lr=0.5 ---');
{
  const net = clone(bestRandom.cfg);
  for (let e = 0; e <= 8; e++) {
    const l = losses(net).map(v => v.toFixed(4).padStart(7)).join(' ');
    const a = patterns.map(p => fwd(net, p.x).a2.toFixed(3).padStart(6)).join(' ');
    console.log(`  epoca ${String(e).padStart(2)}  loss:${l}   salida:${a}`);
    epoch(net, 0.5);
  }
  console.log('\n--- el mismo ganador, 400 epochs ---');
  for (let e = 0; e < 400; e++) epoch(net, 0.5);
  console.log('  loss final por patron:', losses(net).map(v => v.toFixed(5)).join('  '));
  console.log('  pesos w1:', JSON.stringify(net.w1.map(r => r.map(v => +v.toFixed(3)))));
  console.log('  sesgos b1:', net.b1.map(v => +v.toFixed(3)));
  console.log('  pesos w2:', net.w2.map(r => r.map(v => +v.toFixed(3))));
  console.log('  sesgo b2:', +net.b2[0].toFixed(3));
}