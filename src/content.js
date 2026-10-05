/* ==========================================================================
   content.js — Todo el texto del simulador vive aqui para que pueda
   editarse sin tocar la logica. Exporta un objeto global CONTENT.
   ========================================================================== */

window.CONTENT = (() => {
  'use strict';

  /* ---------------------------------------------------------------
     Las 7 fases del algoritmo
     --------------------------------------------------------------- */
  const phases = [
    {
      n: 0,
      key: 'init',
      short: 'Entrada',
      title: 'Paso 0 · Datos de entrada',
      formula: null,
      desc: 'Se presentan los patrones X que entran a la red y los objetivos Y que la red deberia producir. No hay calculo todavia: es el estado inicial de los parametros.',
      tip: 'Edita X e Y en el panel de entradas. Los objetivos son la respuesta correcta contra la que se mide el error.'
    },
    {
      n: 1,
      key: 'hidden',
      short: 'Forward oculto',
      title: 'Paso 1 · Forward pass, capa oculta',
      formula: 'z_j^{(1)} = \\sum_i w_{ij}^{(1)} x_i + b_j^{(1)}',
      desc: 'Cada neurona oculta combina sus entradas con pesos, suma el sesgo y aplica la funcion de activacion. De ahi salen los valores que alimentaran la capa de salida.',
      tip: 'z es la suma ponderada previa a la activacion; a = sigma(z) es la salida ya no lineal. Si quitas sigma, la red solo podria resolver problemas linealmente separables.'
    },
    {
      n: 2,
      key: 'output',
      short: 'Forward salida',
      title: 'Paso 2 · Forward pass, capa de salida',
      formula: 'z_k^{(2)} = \\sum_j w_{jk}^{(2)} a_j^{(1)} + b_k^{(2)}',
      desc: 'La ultima capa mezcla las activaciones ocultas y produce la prediccion a_k. Con esto termina el forward pass: ya existe una respuesta de la red.',
      tip: 'Compara a_k con el objetivo y_k. La diferencia entre ambos es lo que mide el siguiente paso.'
    },
    {
      n: 3,
      key: 'loss',
      short: 'Error',
      title: 'Paso 3 · Funcion de perdida',
      formula: 'E_{total} = \\frac{1}{2} \\sum_k (y_k - a_k)^2',
      desc: 'Se cuantifica la discrepancia entre prediccion y objetivo mediante el error cuadratico medio. Entrenar consiste en minimizar este numero.',
      tip: 'El factor 1/2 existe para que al derivar desaparezca el 2 de la potencia del cuadrado. El MSE penaliza mas los errores grandes que los pequenos.'
    },
    {
      n: 4,
      key: 'bp-output',
      short: 'Backprop salida',
      title: 'Paso 4 · Backpropagation, capa de salida',
      formula: "\\delta_k^{(2)} = (a_k - y_k)\\,\\sigma'(z_k^{(2)})",
      desc: 'El error local de la capa de salida indica cuanto responsable es cada neurona del error total. La derivada de la activacion modula esa sensibilidad.',
      tip: 'delta es la derivada del error respecto a la preactivacion z. Si delta se acerca a cero, esa neurona es virtually irrelevante para la salida.'
    },
    {
      n: 5,
      key: 'bp-hidden',
      short: 'Backprop oculto',
      title: 'Paso 5 · Backpropagation, capa oculta',
      formula: "\\delta_j^{(1)} = \\left(\\sum_k \\delta_k^{(2)} w_{jk}^{(2)}\\right)\\sigma'(z_j^{(1)})",
      desc: 'La culpa del error se reparte hacia atras: cada neurona oculta recibe una fraccion proporcional a los pesos que la conectan con la salida.',
      tip: 'Aqui aparece la regla de la cadena en toda su potencia. La suma es indispensable: el error de una neurona oculta pasa por todas las de salida.'
    },
    {
      n: 6,
      key: 'update',
      short: 'Actualizacion',
      title: 'Paso 6 · Actualizacion de pesos y sesgos',
      formula: 'w^{(new)} = w^{(old)} - \\eta \\frac{\\partial E}{\\partial w}',
      desc: 'Se aplica el descenso de gradiente: cada peso se mueve en direccion opuesta a su gradiente, con un tamano de paso dado por eta. Con esto la red aprende.',
      tip: 'eta controla el tamano del paso. Si es muy grande los pesos oscilan y la perdida diverge; si es muy pequeno el aprendizaje es lentisimo.'
    }
  ];

  /* ---------------------------------------------------------------
     Comentarios del profesor, uno por fase
     --------------------------------------------------------------- */
  const comments = [
    'Estado inicial cargado. Pulsa <kbd>Siguiente paso</kbd> para empezar a evaluar las combinaciones lineales <em>z</em> y las activaciones <em>a</em> de la capa oculta.',
    'Las entradas se ponderan y se transfieren mediante la funcion de activacion elegida. La salida de cada neurona oculta sera la entrada de la capa de salida.',
    'El <strong>forward pass</strong> ha terminado: ya tenemos la prediccion actual de la red. Ahora hay que medir que tan lejos quedo del objetivo.',
    'El error total cuantifica la distancia entre la prediccion y la respuesta correcta. Todo el resto del algoritmo existe para reducir este numero.',
    'Comienza la <strong>retropropagacion</strong>. El delta local mide la sensibilidad del error total respecto a las activaciones de la capa final.',
    'Observa la <strong>regla de la cadena</strong>: la culpa del error se distribuye hacia atras de forma proporcional a los pesos conectores.',
    'Los pesos y sesgos han sido ajustados: la red ha aprendido un poco mas. Pulsa de nuevo para iniciar la epoca siguiente y comprobar como baja el error.'
  ];

  /* ---------------------------------------------------------------
     Datos de entrada (tambien en targets)
     --------------------------------------------------------------- */
  const inputsPanel = {
    title: 'Patron de entrada y objetivo',
    xLabel: 'Entradas',
    yLabel: 'Objetivos',
    hint: 'Modifica estos valores para observar como cambia todo el calculo.'
  };

  /* ---------------------------------------------------------------
     Autoevaluacion
     --------------------------------------------------------------- */
  const quiz = [
    {
      q: 'Por que es necesaria la derivada de la funcion de activacion (sigma prima) durante el backpropagation?',
      options: [
        { id: 'A', text: 'Para calcular la tasa de aprendizaje eta.' },
        { id: 'B', text: 'Por la regla de la cadena, al derivar la activacion respecto a su suma ponderada z.' },
        { id: 'C', text: 'Para convertir pesos negativos en positivos.' }
      ],
      correct: 'B',
      ok: 'Correcto. La derivada indica como responde la activacion a los cambios en z, y ese factor modula toda la sensibilidad del error.',
      ko: 'No es eso. Piensa en la regla de la cadena: para derivar E respecto a w hay que atravesar sigma, y ahi aparece sigma prima.'
    },
    {
      q: 'Si la tasa de aprendizaje es demasiado alta (eta = 10.0), que suele ocurrir?',
      options: [
        { id: 'A', text: 'El algoritmo converge mas rapido y de forma estable.' },
        { id: 'B', text: 'Los pesos oscilan drásticamente y la funcion de perdida diverge.' },
        { id: 'C', text: 'Los deltas se vuelven exactamente cero.' }
      ],
      correct: 'B',
      ok: 'Exacto. Pasos demasiado grandes producen sobrepasos (overshooting) sobre la superficie de perdida y el entrenamiento se vuelve inestable.',
      ko: 'Piensa en el tamano del paso. Con eta muy grande se "salta" el minimo y se acaba al otro lado, cada vez mas lejos.'
    },
    {
      q: 'Por que un perceptron de una sola capa (sin capa oculta) no puede resolver XOR?',
      options: [
        { id: 'A', text: 'Porque solo admite pesos negativos.' },
        { id: 'B', text: 'Porque su frontera de decision es una unica recta y XOR no es linealmente separable.' },
        { id: 'C', text: 'Porque no puede aprender con una sola neurona de salida.' }
      ],
      correct: 'B',
      ok: 'Muy bien. Una sola capa produce una frontera lineal. La capa oculta aporta la no linealidad que permite combinar varias rectas y resolver XOR.',
      ko: 'La clave esta en la forma de la frontera de decision. Sin capa oculta solo puedes trazar una recta, y XOR no se separa asi.'
    },
    {
      q: 'Que papel juega el sesgo b en la neurona?',
      options: [
        { id: 'A', text: 'Regulariza la red para evitar el sobreajuste.' },
        { id: 'B', text: 'Desplaza la frontera de decision, permitiendo que el origen no sea un caso limite.' },
        { id: 'C', text: 'Aumenta la velocidad de la funcion de activacion.' }
      ],
      correct: 'B',
      ok: 'Exacto. El sesgo actua como punto de corte: sin el, la neurona esta obligada a pasar por el origen de coordenadas.',
      ko: 'No regula ni acelera nada. Es un termino constante que traslada la suma ponderada, dando libertad a la frontera de decision.'
    },
    {
      q: 'Si delta de una neurona oculta es cero, que implica?',
      options: [
        { id: 'A', text: 'Que esa neurona no contribuye al error actual, aunque siga conectada al resto de la red.' },
        { id: 'B', text: 'Que el peso de esa neurona debe eliminarse de la red.' },
        { id: 'C', text: 'Que la funcion de activacion esta mal configurada.' }
      ],
      correct: 'A',
      ok: 'Correcto. El gradiente es cero, asi que el descenso de gradiente no modificara esos pesos en esta iteracion.',
      ko: 'Cuidado con la palabra "debe". Delta cero solo significa que el gradiente no fluye por ahi en esta iteracion, no que la conexion sea inservible.'
    }
  ];

  /* ---------------------------------------------------------------
     Glosario
     --------------------------------------------------------------- */
  const glossary = [
    { sym: 'x_i', name: 'Entrada', desc: 'Valor de la neurona de entrada i del patron actual.' },
    { sym: 'w_{ij}^{(1)}', name: 'Peso oculto', desc: 'Fuerza con que la entrada i influye en la neurona oculta j.' },
    { sym: 'b_j^{(1)}', name: 'Sesgo oculto', desc: 'Termino constante de la neurona oculta j, desplaza su umbral.' },
    { sym: 'z_j^{(1)}', name: 'Preactivacion', desc: 'Suma ponderada antes de aplicar la activacion.' },
    { sym: '\\sigma(z)', name: 'Activacion', desc: 'Funcion no lineal aplicada a z. Aporta la capacidad de aprender relaciones complejas.' },
    { sym: 'a_j^{(1)}', name: 'Activacion oculta', desc: 'Salida de la neurona oculta j: sigma(z). Es la entrada de la capa de salida.' },
    { sym: '\\delta_k^{(2)}', name: 'Delta de salida', desc: 'Sensibilidad del error respecto a la preactivacion de la neurona de salida k.' },
    { sym: '\\delta_j^{(1)}', name: 'Delta oculto', desc: 'Sensibilidad del error respecto a la neurona oculta j, ya acumulada a traves de la capa de salida.' },
    { sym: '\\frac{\\partial E}{\\partial w}', name: 'Gradiente', desc: 'Tasa de cambio del error al variar el peso. Indica la direccion de mayor crecimiento del error.' },
    { sym: '\\eta', name: 'Tasa de aprendizaje', desc: 'Tamano del paso del descenso de gradiente.' },
    { sym: 'E_{total}', name: 'Perdida', desc: 'Error cuadratico medio entre prediccion y objetivo. Se busca minimizarlo.' },
    { sym: '\\eta', name: 'Epoca', desc: 'Una pasada completa por todos los patrones de entrenamiento con actualizacion de pesos.' }
  ];

  /* ---------------------------------------------------------------
     Guia rapida (modal)
     --------------------------------------------------------------- */
  const help = {
    intro: 'Este simulador descompone el funcionamiento interno de un perceptron multicapa con exactitud matematica. Avanza paso a paso y observa como se transforman los valores en cada conexion.',
    steps: [
      { color: 'text-cyan-glow-300', name: 'Forward pass', body: 'Calcula la combinacion lineal <em>z = sum w·x + b</em> y despues la activacion <em>a = sigma(z)</em>.' },
      { color: 'text-ember-300', name: 'Funcion de perdida', body: 'Evalua la discrepancia entre la salida real <em>a</em> y el valor deseado <em>y</em>.' },
      { color: 'text-rose-300', name: 'Backpropagation', body: 'Propaga los gradientes de error <em>delta</em> desde la salida hacia la capa oculta.' },
      { color: 'text-cyan-glow-300', name: 'Actualizacion', body: 'Ajusta los pesos con el descenso de gradiente: <em>w(nuevo) = w(viejo) - eta·gradiente</em>.' }
    ],
    extras: 'Puedes cambiar la tasa de aprendizaje, editar pesos manualmente en las tablas, modificar las entradas y los objetivos, o cargar el problema <strong>XOR</strong> con sus cuatro patrones para ver por que la capa oculta es imprescindible. Pasa el cursor sobre cualquier neurona, conexion o control para obtener una explicacion en detalle.',
    keys: [
      ['Flecha izquierda', 'Paso anterior'],
      ['Flecha derecha', 'Paso siguiente'],
      ['Espacio', 'Auto-entrenar o pausar'],
      ['1 a 7', 'Saltar directamente a una fase'],
      ['T', 'Abrir el recorrido guiado'],
      ['R', 'Reiniciar la red'],
      ['Esc', 'Cerrar el dialogo']
    ]
  };

  /* ---------------------------------------------------------------
     Recorrido guiado (driver.js)
     --------------------------------------------------------------- */
  const tour = [
    {
      el: '#phase-stepper',
      title: 'Las 7 fases del algoritmo',
      body: 'Todo el recorrido esta dividido en siete pasos. El stepper siempre muestra en cual estas, y los pasos completados quedan marcados para no perder el hilo.',
      accent: 'cyan'
    },
    {
      el: '#net-svg',
      title: 'El diagrama de la red',
      body: 'Cada neurona y cada conexion son interactivos: pasa el cursor para ver z, a, delta y el gradiente de cada peso. El grosor de una conexion es proporcional a |w|, y el color indica el signo.',
      accent: 'cyan'
    },
    {
      el: '#tab-matrices',
      title: 'Edita los parametros',
      body: 'Pesos y sesgos son editables. Cambia un valor y observa de inmediato como se recalcula toda la red: es la mejor forma de entender el papel de cada parametro.',
      accent: 'ember'
    },
    {
      el: '#lr-slider',
      title: 'Controla la velocidad de aprendizaje',
      body: 'La tasa eta define el tamano de cada paso del descenso de gradiente. Sube mucho el valor y observa como la perdida deja de converger y empieza a divergir.',
      accent: 'ember'
    },
    {
      el: '#math-container',
      title: 'El calculo, paso a paso',
      body: 'Aqui se desglosa cada operacion con los valores numericos reales. Pulsa <kbd>Siguiente paso</kbd> para avanzar y ver como se forma el gradiente.',
      accent: 'cyan'
    },
    {
      el: '#btn-auto',
      title: 'Entrena y observa la convergencia',
      body: 'Pulsa auto-entrenar para ejecutar epocas seguidas y ver como desciende la curva de perdida. El entrenamiento se detiene solo cuando el modelo converge.',
      accent: 'ember'
    },
    {
      el: '#tab-chart',
      title: 'La curva de perdida',
      body: 'Cada punto es una epoca. Una buena configuracion produce una curva que baja de forma sostenida; una tasa mal elegida produce oscilaciones o divergencia.',
      accent: 'cyan'
    },
    {
      el: '#theory-derivation',
      title: 'Demostracion y autoevaluacion',
      body: 'Consulta la derivacion formal con la regla de la cadena, pon a prueba lo aprendido en la autoevaluacion y consulta el glosario de notacion.',
      accent: 'cyan'
    }
  ];

  const tourOutro = {
    title: 'Listo para explorar',
    body: 'Ya conoces la interfaz. Empieza con el <strong>Ejemplo numérico estándar</strong>, avanza con <kbd>Siguiente paso</kbd> y no olvides mirar el problema <strong>XOR</strong>: es donde la capa oculta demuestra por que existe.'
  };

  /* ---------------------------------------------------------------
     Presets
     --------------------------------------------------------------- */
  const presets = {
    simple: {
      label: 'Numerico estandar (2-2-2)',
      inputs: [0.05, 0.10],
      targets: [0.01, 0.99],
      w1: [[0.15, 0.25], [0.20, 0.30]],
      b1: [0.35, 0.35],
      w2: [[0.40, 0.50], [0.45, 0.55]],
      b2: [0.60, 0.60],
      patterns: null
    },
    xor: {
      label: 'Logica XOR (2-2-1, 4 patrones)',
      inputs: [0, 0],
      targets: [0],
      w1: [[1.0, 1.0], [-1.0, -1.0]],
      b1: [-0.5, 0.5],
      w2: [[1.0], [1.0]],
      b2: [-0.5],
      patterns: [
        { x: [0, 0], y: [0], label: 'Falso ∧ Falso → Falso' },
        { x: [0, 1], y: [1], label: 'Falso ∧ Verdadero → Verdadero' },
        { x: [1, 0], y: [1], label: ' Verdadero ∧ Falso → Verdadero' },
        { x: [1, 1], y: [0], label: 'Verdadero ∧ Verdadero → Falso' }
      ]
    },
    custom: {
      label: 'Personalizada',
      inputs: [0.30, -0.20],
      targets: [0.80],
      w1: [[0.45, -0.30], [0.25, 0.55]],
      b1: [0.10, -0.15],
      w2: [[0.90], [-0.50]],
      b2: [0.05],
      patterns: null
    }
  };

  return {
    phases, comments, inputsPanel, quiz, glossary, help, tour, tourOutro, presets,
    activations: {
      sigmoid: { label: 'Sigmoide', fx: '1 / (1 + e^-z)', dfx: 'a(1 - a)', range: '(0, 1)' },
      tanh:   { label: 'Tangente hiperbolica', fx: 'tanh(z)', dfx: '1 - a^2', range: '(-1, 1)' },
      relu:   { label: 'ReLU', fx: 'max(0, z)', dfx: '1 si z > 0', range: '[0, ∞)' }
    }
  };
})();