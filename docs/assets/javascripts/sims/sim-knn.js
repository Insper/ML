/* Interactive k-NN: drag the query point, change k, watch the vote flip. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-knn',
    S.t('k-NN — the vote of the neighbors', 'k-NN — o voto dos vizinhos'),
    S.t('Drag the ★ query point and change k. The background shows the decision regions for the current k.',
        'Arraste o ponto ★ de consulta e mude k. O fundo mostra as regiões de decisão para o k atual.'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 620, 320);
  const ctx = cv.ctx;
  const C0 = S.colors.orange, C1 = S.colors.blue;

  // two overlapping gaussian classes, fixed seed
  const rand = S.rng(21);
  const pts = [];
  for (let i = 0; i < 35; i++) pts.push({ x: 200 + S.gauss(rand) * 62, y: 140 + S.gauss(rand) * 58, c: 0 });
  for (let i = 0; i < 35; i++) pts.push({ x: 400 + S.gauss(rand) * 62, y: 190 + S.gauss(rand) * 58, c: 1 });

  let k = 5, q = { x: 310, y: 160 }, dragging = false;

  S.slider(ui.controls, {
    label: 'k', min: 1, max: 25, step: 2, value: 5, fmt: v => v.toFixed(0),
    oninput: v => { k = v; draw(); }
  });

  function neighbors(x, y, kk) {
    return pts
      .map(p => ({ p, d: (p.x - x) ** 2 + (p.y - y) ** 2 }))
      .sort((a, b) => a.d - b.d)
      .slice(0, kk);
  }
  function vote(x, y, kk) {
    const nn = neighbors(x, y, kk);
    const v1 = nn.filter(n => n.p.c === 1).length;
    return { cls: v1 * 2 > kk ? 1 : 0, v1, v0: kk - v1, nn };
  }

  cv.el.addEventListener('pointerdown', e => { dragging = true; q = cv.pos(e); draw(); });
  cv.el.addEventListener('pointermove', e => { if (dragging) { q = cv.pos(e); draw(); } });
  window.addEventListener('pointerup', () => { dragging = false; });

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    // decision regions on a coarse grid
    const cell = 10;
    for (let gx = 0; gx < cv.W; gx += cell) {
      for (let gy = 0; gy < cv.H; gy += cell) {
        const r = vote(gx + cell / 2, gy + cell / 2, k);
        ctx.fillStyle = r.cls === 1 ? 'rgba(88,166,255,0.10)' : 'rgba(240,136,62,0.10)';
        ctx.fillRect(gx, gy, cell, cell);
      }
    }
    const res = vote(q.x, q.y, k);
    // lines to the k neighbors
    ctx.lineWidth = 1;
    res.nn.forEach(n => {
      ctx.strokeStyle = n.p.c === 1 ? C1 : C0;
      ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(q.x, q.y); ctx.lineTo(n.p.x, n.p.y); ctx.stroke();
    });
    ctx.globalAlpha = 1;
    // training points
    pts.forEach(p => {
      ctx.fillStyle = p.c === 1 ? C1 : C0;
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 7); ctx.fill();
    });
    // highlight the neighbors
    res.nn.forEach(n => {
      ctx.strokeStyle = S.colors.neutral; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(n.p.x, n.p.y, 7, 0, 7); ctx.stroke();
    });
    // query star
    ctx.fillStyle = res.cls === 1 ? C1 : C0;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    star(q.x, q.y, 10);
    ui.readout.innerHTML =
      `${S.t('vote', 'votação')}: <b style="color:${C0}">${res.v0}</b> × <b style="color:${C1}">${res.v1}</b>` +
      ` → ${S.t('predicted class', 'classe prevista')}: <b style="color:${res.cls === 1 ? C1 : C0}">${res.cls === 1 ? S.t('blue', 'azul') : S.t('orange', 'laranja')}</b>` +
      ` · <span style="font-size:.75rem">${S.t('Small k = jagged regions (variance); large k = smooth regions (bias).', 'k pequeno = regiões recortadas (variância); k grande = regiões suaves (viés).')}</span>`;
  }

  function star(x, y, r) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const rr = i % 2 === 0 ? r : r * 0.45;
      const a = -Math.PI / 2 + i * Math.PI / 5;
      const px = x + rr * Math.cos(a), py = y + rr * Math.sin(a);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }
  draw();
})();
