/* Interactive perceptron learning on linearly separable data. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-perceptron',
    S.t('Perceptron learning (1958)', 'Aprendizado do perceptron (1958)'),
    S.t('Each step visits one point; on a mistake, the boundary w·x + b = 0 tilts toward fixing it. Guaranteed to converge on separable data.',
        'Cada passo visita um ponto; em caso de erro, a fronteira w·x + b = 0 se ajusta para corrigi-lo. Convergência garantida em dados separáveis.'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 620, 320);
  const ctx = cv.ctx;
  const C0 = S.colors.orange, C1 = S.colors.blue;

  // linearly separable classes (labels -1 / +1), features scaled to ~[-1,1]
  const rand = S.rng(31);
  const pts = [];
  for (let i = 0; i < 30; i++) pts.push({ x: -0.45 + S.gauss(rand) * 0.22, y: 0.35 + S.gauss(rand) * 0.22, t: -1 });
  for (let i = 0; i < 30; i++) pts.push({ x: 0.45 + S.gauss(rand) * 0.22, y: -0.35 + S.gauss(rand) * 0.22, t: 1 });

  const PX = x => (x + 1.4) / 2.8 * cv.W;
  const PY = y => cv.H - (y + 1.4) / 2.8 * cv.H;

  let w1, w2, b, idx, epoch, mistakesThisEpoch, converged, timer = null, lastUpdated = -1;

  S.button(ui.controls, S.t('Step', 'Passo'), step, true);
  S.button(ui.controls, S.t('▶ Play', '▶ Rodar'), () => { stop(); timer = setInterval(step, 120); });
  S.button(ui.controls, S.t('↺ Reset', '↺ Reiniciar'), reset);

  function reset() {
    stop();
    w1 = 0.2; w2 = 0.8; b = 0.3;   // deliberately bad start
    idx = 0; epoch = 0; mistakesThisEpoch = 0; converged = false; lastUpdated = -1;
    draw();
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }

  function errors() {
    return pts.filter(p => Math.sign(w1 * p.x + w2 * p.y + b) !== p.t).length;
  }

  function step() {
    if (converged) { stop(); return; }
    const p = pts[idx];
    const pred = w1 * p.x + w2 * p.y + b >= 0 ? 1 : -1;
    if (pred !== p.t) {
      const eta = 0.5;
      w1 += eta * p.t * p.x;
      w2 += eta * p.t * p.y;
      b += eta * p.t;
      mistakesThisEpoch++;
      lastUpdated = idx;
    } else {
      lastUpdated = -1;
    }
    idx++;
    if (idx >= pts.length) {
      if (mistakesThisEpoch === 0) { converged = true; stop(); }
      idx = 0; epoch++; mistakesThisEpoch = 0;
    }
    draw();
  }

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    // decision boundary: w1*x + w2*y + b = 0
    if (Math.abs(w2) > 1e-9) {
      ctx.strokeStyle = S.colors.neutral; ctx.lineWidth = 2;
      ctx.beginPath();
      const y1 = (-b - w1 * (-1.4)) / w2, y2 = (-b - w1 * 1.4) / w2;
      ctx.moveTo(PX(-1.4), PY(y1)); ctx.lineTo(PX(1.4), PY(y2));
      ctx.stroke();
    }
    pts.forEach((p, i) => {
      const wrong = Math.sign(w1 * p.x + w2 * p.y + b) !== p.t;
      ctx.fillStyle = p.t === 1 ? C1 : C0;
      ctx.beginPath(); ctx.arc(PX(p.x), PY(p.y), 4.5, 0, 7); ctx.fill();
      if (wrong) {
        ctx.strokeStyle = S.colors.red; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(PX(p.x), PY(p.y), 8, 0, 7); ctx.stroke();
      }
      if (i === lastUpdated) {
        ctx.strokeStyle = S.colors.yellow; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(PX(p.x), PY(p.y), 11, 0, 7); ctx.stroke();
      }
    });
    const e = errors();
    ui.readout.innerHTML =
      `${S.t('epoch', 'época')} <b style="color:${S.colors.orange}">${epoch}</b>` +
      ` · ${S.t('misclassified', 'mal classificados')}: <b style="color:${e ? S.colors.red : S.colors.green}">${e}/${pts.length}</b>` +
      (converged ? ` · <b style="color:${S.colors.green}">${S.t('CONVERGED', 'CONVERGIU')}</b>` : '') +
      ` · <span style="font-size:.75rem">${S.t('red rings = currently misclassified; yellow ring = point that just moved the boundary', 'anéis vermelhos = mal classificados; anel amarelo = ponto que acabou de mover a fronteira')}</span>`;
  }
  reset();
})();
