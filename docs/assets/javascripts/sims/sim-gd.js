/* Interactive gradient descent on a 1-D convex loss. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-gd',
    S.t('Gradient descent, step by step', 'Gradiente descendente, passo a passo'),
    S.t('Pick a learning rate and watch the updates w ← w − η·J′(w). Try η = 1.05 to see divergence.',
        'Escolha uma taxa de aprendizado e observe as atualizações w ← w − η·J′(w). Experimente η = 1.05 para ver a divergência.'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 620, 300);
  const ctx = cv.ctx;

  // Loss J(w) = 0.3(w-2)^2 + 0.5, gradient J'(w) = 0.6(w-2)
  const J = w => 0.3 * (w - 2) * (w - 2) + 0.5;
  const dJ = w => 0.6 * (w - 2);
  const WMIN = -6, WMAX = 10, JMAX = J(WMIN);

  const X = w => (w - WMIN) / (WMAX - WMIN) * (cv.W - 40) + 20;
  const Y = j => cv.H - 30 - (j / JMAX) * (cv.H - 60);

  let w0 = -4.5, path = [w0], timer = null;

  const eta = S.slider(ui.controls, {
    label: 'η', min: 0.02, max: 1.15, step: 0.01, value: 0.3,
    fmt: v => v.toFixed(2),
    oninput: () => reset()
  });
  const stepBtn = S.button(ui.controls, S.t('Step', 'Passo'), step, true);
  S.button(ui.controls, S.t('▶ Play', '▶ Rodar'), play);
  S.button(ui.controls, S.t('↺ Reset', '↺ Reiniciar'), reset);

  function step() {
    const w = path[path.length - 1];
    if (Math.abs(dJ(w)) < 1e-4 || Math.abs(w) > 50 || path.length > 60) { stop(); return; }
    path.push(w - eta.get() * dJ(w));
    draw();
  }
  function play() { stop(); timer = setInterval(step, 350); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }
  function reset() { stop(); path = [w0]; draw(); }

  cv.el.addEventListener('pointerdown', (e) => {
    const p = cv.pos(e);
    w0 = WMIN + (p.x - 20) / (cv.W - 40) * (WMAX - WMIN);
    w0 = Math.max(WMIN, Math.min(WMAX, w0));
    reset();
  });

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    // axes
    ctx.strokeStyle = S.colors.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(20, Y(0)); ctx.lineTo(cv.W - 20, Y(0)); ctx.stroke();
    // loss curve
    ctx.strokeStyle = S.colors.blue; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 20; px <= cv.W - 20; px += 2) {
      const w = WMIN + (px - 20) / (cv.W - 40) * (WMAX - WMIN);
      const y = Math.min(cv.H - 5, Y(J(w)));
      px === 20 ? ctx.moveTo(px, y) : ctx.lineTo(px, y);
    }
    ctx.stroke();
    // minimum marker
    ctx.fillStyle = S.colors.green;
    ctx.beginPath(); ctx.arc(X(2), Y(J(2)), 4, 0, 7); ctx.fill();
    ctx.fillStyle = S.colors.muted; ctx.font = '11px Inter,sans-serif';
    ctx.fillText('w* = 2', X(2) - 18, Y(J(2)) + 18);
    // GD path
    for (let i = 0; i < path.length; i++) {
      const wv = Math.max(WMIN, Math.min(WMAX, path[i]));
      const x = X(wv), y = Math.min(cv.H - 5, Y(J(path[i])));
      if (i > 0) {
        const pwv = Math.max(WMIN, Math.min(WMAX, path[i - 1]));
        ctx.strokeStyle = S.colors.orange; ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(X(pwv), Math.min(cv.H - 5, Y(J(path[i - 1]))));
        ctx.lineTo(x, y); ctx.stroke();
      }
      ctx.fillStyle = i === path.length - 1 ? S.colors.red : S.colors.orange;
      ctx.beginPath(); ctx.arc(x, y, i === path.length - 1 ? 6 : 3.5, 0, 7); ctx.fill();
    }
    // readout
    const w = path[path.length - 1];
    const diverging = Math.abs(w) > 50 || (path.length > 3 && J(w) > J(path[0]) * 1.5);
    ui.readout.innerHTML =
      `${S.t('iteration', 'iteração')} <b style="color:${S.colors.orange}">${path.length - 1}</b>` +
      ` · w = <b style="color:${S.colors.orange}">${w.toFixed(3)}</b>` +
      ` · J(w) = <b style="color:${S.colors.orange}">${J(w) > 1e6 ? '∞' : J(w).toFixed(3)}</b>` +
      (diverging ? ` · <b style="color:${S.colors.red}">${S.t('DIVERGING — η too large!', 'DIVERGINDO — η muito grande!')}</b>` : '') +
      `<br><span style="font-size:.75rem">${S.t('Click on the curve to change the starting point.', 'Clique na curva para mudar o ponto inicial.')}</span>`;
    stepBtn.disabled = false;
  }
  draw();
})();
