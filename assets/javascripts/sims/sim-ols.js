/* Interactive OLS: drag points, watch the least-squares line and R² react. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-ols',
    S.t('Least squares, hands on', 'Mínimos quadrados, na prática'),
    S.t('Drag any point. The line always minimizes the sum of squared residuals (gray segments). Drag one point far away and see its leverage.',
        'Arraste qualquer ponto. A reta sempre minimiza a soma dos resíduos ao quadrado (segmentos cinzas). Arraste um ponto para longe e veja sua alavancagem.'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 620, 320);
  const ctx = cv.ctx;

  const rand = S.rng(61);
  let pts = [];
  function initPts() {
    pts = [];
    for (let i = 0; i < 12; i++) {
      const x = 60 + i * 45 + rand() * 18;
      pts.push({ x, y: 250 - 0.32 * x + S.gauss(rand) * 22 });
    }
  }

  S.button(ui.controls, S.t('↺ Reset points', '↺ Reiniciar pontos'), () => { initPts(); draw(); });

  let dragging = null;
  cv.el.addEventListener('pointerdown', e => {
    const p = cv.pos(e);
    dragging = pts.reduce((best, pt) =>
      ((pt.x - p.x) ** 2 + (pt.y - p.y) ** 2) < ((best.x - p.x) ** 2 + (best.y - p.y) ** 2) ? pt : best, pts[0]);
    if ((dragging.x - p.x) ** 2 + (dragging.y - p.y) ** 2 > 40 * 40) dragging = null;
  });
  cv.el.addEventListener('pointermove', e => {
    if (dragging) {
      const p = cv.pos(e);
      dragging.x = Math.max(5, Math.min(cv.W - 5, p.x));
      dragging.y = Math.max(5, Math.min(cv.H - 5, p.y));
      draw();
    }
  });
  window.addEventListener('pointerup', () => { dragging = null; });

  function ols() {
    const n = pts.length;
    const mx = pts.reduce((s, p) => s + p.x, 0) / n;
    const my = pts.reduce((s, p) => s + p.y, 0) / n;
    let sxy = 0, sxx = 0, syy = 0;
    pts.forEach(p => {
      sxy += (p.x - mx) * (p.y - my);
      sxx += (p.x - mx) ** 2;
      syy += (p.y - my) ** 2;
    });
    const b = sxx > 1e-9 ? sxy / sxx : 0;
    const a = my - b * mx;
    let sse = 0;
    pts.forEach(p => { sse += (p.y - (a + b * p.x)) ** 2; });
    const r2 = syy > 1e-9 ? 1 - sse / syy : 0;
    return { a, b, sse, r2 };
  }

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    const { a, b, sse, r2 } = ols();
    // residual segments
    ctx.strokeStyle = S.colors.muted; ctx.lineWidth = 1; ctx.globalAlpha = 0.7;
    pts.forEach(p => {
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x, a + b * p.x); ctx.stroke();
    });
    ctx.globalAlpha = 1;
    // OLS line
    ctx.strokeStyle = S.colors.blue; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(0, a); ctx.lineTo(cv.W, a + b * cv.W); ctx.stroke();
    // points
    pts.forEach(p => {
      ctx.fillStyle = S.colors.orange;
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.fill(); ctx.stroke();
    });
    // slope shown in math coordinates (screen y is flipped)
    ui.readout.innerHTML =
      `ŷ = <b style="color:${S.colors.blue}">${a.toFixed(1)}</b> ${b >= 0 ? '+' : '−'} <b style="color:${S.colors.blue}">${Math.abs(b).toFixed(3)}</b>·x` +
      ` · SSE = <b style="color:${S.colors.orange}">${Math.round(sse).toLocaleString()}</b>` +
      ` · R² = <b style="color:${r2 > 0.6 ? S.colors.green : S.colors.yellow}">${r2.toFixed(3)}</b>` +
      ` · <span style="font-size:.75rem">${S.t('(screen coordinates: y grows downward)', '(coordenadas de tela: y cresce para baixo)')}</span>`;
  }
  initPts(); draw();
})();
