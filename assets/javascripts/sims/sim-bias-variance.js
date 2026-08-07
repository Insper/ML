/* Interactive bias-variance: polynomial degree slider, live train/val MSE. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-bias-variance',
    S.t('Underfit ↔ overfit, live', 'Subajuste ↔ sobreajuste, ao vivo'),
    S.t('Slide the polynomial degree. Training error (orange) always falls; validation error (blue) is U-shaped.',
        'Deslize o grau do polinômio. O erro de treino (laranja) sempre cai; o erro de validação (azul) tem forma de U.'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 620, 320);
  const ctx = cv.ctx;

  // data: y = sin(2πx) + noise, x in [0,1]; separate train/val samples
  const rand = S.rng(41);
  const f = x => Math.sin(2 * Math.PI * x);
  function sample(n) {
    const a = [];
    for (let i = 0; i < n; i++) {
      const x = rand();
      a.push({ x, y: f(x) + S.gauss(rand) * 0.28 });
    }
    return a.sort((p, q) => p.x - q.x);
  }
  const train = sample(22), val = sample(22);

  let degree = 3, coef = [];

  S.slider(ui.controls, {
    label: S.t('degree', 'grau'), min: 0, max: 12, step: 1, value: 3,
    fmt: v => v.toFixed(0),
    oninput: v => { degree = v; fit(); draw(); }
  });

  // polynomial least squares via normal equations + tiny ridge for stability;
  // features scaled to [-1, 1] to keep the Gram matrix well-conditioned
  const z = x => 2 * x - 1;
  function phi(x, d) {
    const row = [1];
    for (let j = 1; j <= d; j++) row.push(Math.pow(z(x), j));
    return row;
  }
  function fit() {
    const d = degree, m = d + 1;
    const A = Array.from({ length: m }, () => new Array(m).fill(0));
    const bv = new Array(m).fill(0);
    train.forEach(p => {
      const r = phi(p.x, d);
      for (let i = 0; i < m; i++) {
        bv[i] += r[i] * p.y;
        for (let j = 0; j < m; j++) A[i][j] += r[i] * r[j];
      }
    });
    for (let i = 0; i < m; i++) A[i][i] += 1e-8;
    coef = solve(A, bv);
  }
  function solve(A, b) { // gaussian elimination with partial pivoting
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) continue;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const t = M[r][c] / M[c][c];
        for (let cc = c; cc <= n; cc++) M[r][cc] -= t * M[c][cc];
      }
    }
    return M.map((row, i) => Math.abs(row[i]) < 1e-12 ? 0 : row[n] / row[i]);
  }
  function predict(x) {
    const r = phi(x, degree);
    return r.reduce((s, v, i) => s + v * (coef[i] || 0), 0);
  }
  function mse(data) {
    return data.reduce((s, p) => s + (p.y - predict(p.x)) ** 2, 0) / data.length;
  }

  const PX = x => 20 + x * (cv.W - 40);
  const PY = y => cv.H / 2 - y * (cv.H / 3.2);

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    // true function
    ctx.strokeStyle = S.colors.grid; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let px = 20; px <= cv.W - 20; px += 3) {
      const x = (px - 20) / (cv.W - 40);
      px === 20 ? ctx.moveTo(px, PY(f(x))) : ctx.lineTo(px, PY(f(x)));
    }
    ctx.stroke(); ctx.setLineDash([]);
    // fitted polynomial
    ctx.strokeStyle = S.colors.neutral; ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let px = 20; px <= cv.W - 20; px += 2) {
      const x = (px - 20) / (cv.W - 40);
      const y = PY(predict(x));
      if (y < -50 || y > cv.H + 50) { started = false; continue; }
      started ? ctx.lineTo(px, y) : ctx.moveTo(px, y);
      started = true;
    }
    ctx.stroke();
    // points
    train.forEach(p => {
      ctx.fillStyle = S.colors.orange;
      ctx.beginPath(); ctx.arc(PX(p.x), PY(p.y), 4, 0, 7); ctx.fill();
    });
    val.forEach(p => {
      ctx.strokeStyle = S.colors.blue; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(PX(p.x), PY(p.y), 4, 0, 7); ctx.stroke();
    });
    const tr = mse(train), va = mse(val);
    const verdict = degree <= 1
      ? S.t('underfitting: too rigid for the sine', 'subajuste: rígido demais para a senoide')
      : va > tr * 3 && degree >= 8
        ? S.t('overfitting: chasing the noise', 'sobreajuste: perseguindo o ruído')
        : S.t('reasonable fit', 'ajuste razoável');
    ui.readout.innerHTML =
      `${S.t('train MSE', 'MSE treino')}: <b style="color:${S.colors.orange}">${tr.toFixed(3)}</b>` +
      ` · ${S.t('validation MSE', 'MSE validação')}: <b style="color:${S.colors.blue}">${va.toFixed(3)}</b>` +
      ` · ${verdict}` +
      `<br><span style="font-size:.75rem">● ${S.t('filled = training points', 'cheios = pontos de treino')} · ○ ${S.t('hollow = validation points', 'vazados = pontos de validação')} · ${S.t('dashed = true function', 'tracejada = função verdadeira')}</span>`;
  }
  fit(); draw();
})();
