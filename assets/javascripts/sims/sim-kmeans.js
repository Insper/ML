/* Interactive k-means: watch Lloyd's algorithm alternate assign/update. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-kmeans',
    S.t("k-means — Lloyd's algorithm live", 'k-means — algoritmo de Lloyd ao vivo'),
    S.t('Step alternates ASSIGN (color points by nearest centroid) and UPDATE (move centroids to the mean). Try k = 2 or 5 on 3 real blobs.',
        'Passo alterna ATRIBUIR (colore pontos pelo centróide mais próximo) e ATUALIZAR (move centróides para a média). Experimente k = 2 ou 5 com 3 grupos reais.'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 620, 320);
  const ctx = cv.ctx;
  const PALETTE = [S.colors.orange, S.colors.blue, S.colors.green, S.colors.red, S.colors.yellow];

  // three gaussian blobs, fixed seed
  const rand = S.rng(11);
  const pts = [];
  [[150, 90], [430, 110], [300, 240]].forEach(c => {
    for (let i = 0; i < 32; i++) {
      pts.push({ x: c[0] + S.gauss(rand) * 34, y: c[1] + S.gauss(rand) * 30, a: -1 });
    }
  });

  let k = 3, cents = [], phase = 'assign', iter = 0, timer = null, seedCounter = 1;

  const kCtl = S.slider(ui.controls, {
    label: 'k', min: 2, max: 5, step: 1, value: 3, width: 90, fmt: v => v.toFixed(0),
    oninput: v => { k = v; reset(); }
  });
  S.button(ui.controls, S.t('Step', 'Passo'), step, true);
  S.button(ui.controls, S.t('▶ Play', '▶ Rodar'), () => { stop(); timer = setInterval(step, 600); });
  S.button(ui.controls, S.t('↺ New start', '↺ Novo início'), reset);

  function reset() {
    stop(); iter = 0; phase = 'assign';
    const r = S.rng(100 + seedCounter++);
    cents = [];
    for (let j = 0; j < k; j++) cents.push({ x: 60 + r() * 500, y: 40 + r() * 240 });
    pts.forEach(p => p.a = -1);
    draw();
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }

  function inertia() {
    let s = 0;
    pts.forEach(p => { if (p.a >= 0) { const c = cents[p.a]; s += (p.x - c.x) ** 2 + (p.y - c.y) ** 2; } });
    return s;
  }

  function step() {
    if (phase === 'assign') {
      let changed = 0;
      pts.forEach(p => {
        let best = 0, bd = Infinity;
        cents.forEach((c, j) => {
          const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
          if (d < bd) { bd = d; best = j; }
        });
        if (p.a !== best) changed++;
        p.a = best;
      });
      phase = 'update';
      if (changed === 0 && iter > 0) { stop(); phase = 'done'; }
    } else if (phase === 'update') {
      cents.forEach((c, j) => {
        const mine = pts.filter(p => p.a === j);
        if (mine.length) {
          c.x = mine.reduce((s, p) => s + p.x, 0) / mine.length;
          c.y = mine.reduce((s, p) => s + p.y, 0) / mine.length;
        }
      });
      iter++;
      phase = 'assign';
    }
    draw();
  }

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    pts.forEach(p => {
      ctx.fillStyle = p.a >= 0 ? PALETTE[p.a] : S.colors.muted;
      ctx.globalAlpha = 0.85;
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, 7); ctx.fill();
    });
    ctx.globalAlpha = 1;
    cents.forEach((c, j) => {
      ctx.strokeStyle = PALETTE[j]; ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(c.x - 8, c.y - 8); ctx.lineTo(c.x + 8, c.y + 8);
      ctx.moveTo(c.x + 8, c.y - 8); ctx.lineTo(c.x - 8, c.y + 8);
      ctx.stroke();
      ctx.strokeStyle = '#000'; ctx.lineWidth = 1;
      ctx.strokeRect(c.x - 9, c.y - 9, 18, 18);
    });
    const nextLabel = phase === 'assign'
      ? S.t('next: ASSIGN points', 'próximo: ATRIBUIR pontos')
      : phase === 'update'
        ? S.t('next: UPDATE centroids', 'próximo: ATUALIZAR centróides')
        : `<b style="color:${S.colors.green}">${S.t('CONVERGED — assignments stable', 'CONVERGIU — atribuições estáveis')}</b>`;
    const hasAssign = pts.some(p => p.a >= 0);
    ui.readout.innerHTML =
      `${S.t('iteration', 'iteração')} <b style="color:${S.colors.orange}">${iter}</b>` +
      ` · ${S.t('inertia', 'inércia')}: <b style="color:${S.colors.orange}">${hasAssign ? Math.round(inertia()).toLocaleString() : '—'}</b>` +
      ` · ${nextLabel}`;
  }
  reset();
})();
