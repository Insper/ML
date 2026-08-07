/* Interactive PCA: rotate a projection axis, maximize the projected variance. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-pca',
    S.t('PCA — find the direction of maximum variance', 'PCA — encontre a direção de máxima variância'),
    S.t('Rotate the projection axis with the slider. PCA is the angle that maximizes the variance of the projections (ticks on the axis).',
        'Gire o eixo de projeção com o controle. PCA é o ângulo que maximiza a variância das projeções (marcas sobre o eixo).'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 620, 320);
  const ctx = cv.ctx;
  const CX = cv.W / 2, CY = cv.H / 2;

  // correlated 2-D gaussian cloud (centered)
  const rand = S.rng(71);
  const pts = [];
  for (let i = 0; i < 80; i++) {
    const u = S.gauss(rand) * 95, v = S.gauss(rand) * 28;
    const rot = 0.5; // true principal direction ≈ 28.6°
    pts.push({
      x: u * Math.cos(rot) - v * Math.sin(rot),
      y: u * Math.sin(rot) + v * Math.cos(rot)
    });
  }
  // total variance and true PC1 angle from the covariance matrix
  const n = pts.length;
  let sxx = 0, syy = 0, sxy = 0;
  pts.forEach(p => { sxx += p.x * p.x; syy += p.y * p.y; sxy += p.x * p.y; });
  sxx /= n; syy /= n; sxy /= n;
  const totalVar = sxx + syy;
  const pc1deg = Math.atan2(2 * sxy, sxx - syy) / 2 * 180 / Math.PI;

  let angle = 90;
  const ctl = S.slider(ui.controls, {
    label: S.t('axis angle', 'ângulo do eixo'), min: 0, max: 180, step: 1, value: 90,
    width: 200, fmt: v => v.toFixed(0) + '°',
    oninput: v => { angle = v; draw(); }
  });
  S.button(ui.controls, S.t('Snap to PC1', 'Ir para PC1'), () => {
    angle = (pc1deg + 180) % 180;
    ctl.set(Math.round(angle));
    draw();
  }, true);

  function projVar(deg) {
    const t = deg * Math.PI / 180;
    const ux = Math.cos(t), uy = Math.sin(t);
    // note: screen y points down, but variance is symmetric so it is fine
    let s = 0;
    pts.forEach(p => { const d = p.x * ux + p.y * uy; s += d * d; });
    return s / n;
  }

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    const t = angle * Math.PI / 180;
    const ux = Math.cos(t), uy = -Math.sin(t);   // screen coords (y down)
    // axis line
    ctx.strokeStyle = S.colors.neutral; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(CX - ux * 300, CY - uy * 300);
    ctx.lineTo(CX + ux * 300, CY + uy * 300);
    ctx.stroke();
    // points + projections
    pts.forEach(p => {
      const px = CX + p.x, py = CY - p.y;              // math → screen
      const d = (px - CX) * ux + (py - CY) * uy;       // projection along axis
      const qx = CX + d * ux, qy = CY + d * uy;
      ctx.strokeStyle = S.colors.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(qx, qy); ctx.stroke();
      ctx.fillStyle = S.colors.blue;
      ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 7); ctx.fill();
      ctx.fillStyle = S.colors.orange;
      ctx.beginPath(); ctx.arc(qx, qy, 2.5, 0, 7); ctx.fill();
    });
    const pv = projVar(angle), ratio = pv / totalVar;
    // variance bar
    const bx = 20, by = cv.H - 24, bw = 220;
    ctx.fillStyle = S.colors.grid; ctx.fillRect(bx, by, bw, 10);
    ctx.fillStyle = ratio > 0.85 ? S.colors.green : S.colors.orange;
    ctx.fillRect(bx, by, bw * ratio, 10);
    ctx.fillStyle = S.colors.muted; ctx.font = '10px Inter,sans-serif';
    ctx.fillText(S.t('captured variance', 'variância capturada'), bx, by - 4);
    ui.readout.innerHTML =
      `${S.t('projected variance', 'variância projetada')}: <b style="color:${S.colors.orange}">${Math.round(pv).toLocaleString()}</b>` +
      ` = <b style="color:${ratio > 0.85 ? S.colors.green : S.colors.orange}">${(ratio * 100).toFixed(1)}%</b> ${S.t('of total', 'do total')}` +
      (ratio > 0.85 ? ` · <b style="color:${S.colors.green}">${S.t('≈ PC1 found!', '≈ PC1 encontrada!')}</b>` : '') +
      `<br><span style="font-size:.75rem">${S.t('blue = data · orange = projections onto the axis', 'azul = dados · laranja = projeções sobre o eixo')}</span>`;
  }
  draw();
})();
