/* Interactive decision threshold: score distributions → metrics + ROC point. */
(function () {
  const S = window.mlSim;
  const ui = S.card('sim-threshold',
    S.t('The threshold is a choice', 'O limiar é uma escolha'),
    S.t('Slide the decision threshold over the score distributions (5:1 imbalance) and watch precision, recall and the ROC point move.',
        'Deslize o limiar de decisão sobre as distribuições de score (desbalanceamento 5:1) e veja precisão, recall e o ponto ROC se moverem.'));
  if (!ui) return;

  const cv = S.canvas(ui.canvasWrap, 640, 300);
  const ctx = cv.ctx;

  // scores in [0,1]: negatives centered 0.35, positives centered 0.65 (imbalanced)
  const rand = S.rng(51);
  const clamp = v => Math.max(0.001, Math.min(0.999, v));
  const neg = [], pos = [];
  for (let i = 0; i < 500; i++) neg.push(clamp(0.35 + S.gauss(rand) * 0.13));
  for (let i = 0; i < 100; i++) pos.push(clamp(0.65 + S.gauss(rand) * 0.13));

  let thr = 0.5;
  S.slider(ui.controls, {
    label: S.t('threshold', 'limiar'), min: 0.02, max: 0.98, step: 0.01, value: 0.5,
    width: 220, fmt: v => v.toFixed(2),
    oninput: v => { thr = v; draw(); }
  });

  function confusion(t) {
    const TP = pos.filter(s => s >= t).length;
    const FN = pos.length - TP;
    const FP = neg.filter(s => s >= t).length;
    const TN = neg.length - FP;
    return { TP, FN, FP, TN };
  }

  // histogram bins
  const BINS = 40;
  function hist(data) {
    const h = new Array(BINS).fill(0);
    data.forEach(s => h[Math.min(BINS - 1, Math.floor(s * BINS))]++);
    return h;
  }
  const hNeg = hist(neg), hPos = hist(pos);
  const hMax = Math.max(...hNeg, ...hPos);

  // panel layout: left = distributions, right = ROC
  const L = { x: 15, y: 15, w: 380, h: 250 };
  const R = { x: 430, y: 15, w: 195, h: 250 };

  // precompute the ROC curve
  const roc = [];
  for (let i = 0; i <= 100; i++) {
    const c = confusion(i / 100);
    roc.push({
      fpr: c.FP / (c.FP + c.TN),
      tpr: c.TP + c.FN > 0 ? c.TP / (c.TP + c.FN) : 0
    });
  }

  function draw() {
    ctx.clearRect(0, 0, cv.W, cv.H);
    ctx.font = '10px Inter,sans-serif';

    // --- left: score distributions ---
    const bw = L.w / BINS;
    for (let i = 0; i < BINS; i++) {
      const x = L.x + i * bw;
      const nH = hNeg[i] / hMax * L.h, pH = hPos[i] / hMax * L.h;
      ctx.fillStyle = 'rgba(240,136,62,0.55)';
      ctx.fillRect(x, L.y + L.h - nH, bw - 1, nH);
      ctx.fillStyle = 'rgba(88,166,255,0.55)';
      ctx.fillRect(x, L.y + L.h - pH, bw - 1, pH);
    }
    // threshold line
    const tx = L.x + thr * L.w;
    ctx.strokeStyle = S.colors.neutral; ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(tx, L.y); ctx.lineTo(tx, L.y + L.h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = S.colors.muted;
    ctx.fillText(S.t('← predict negative', '← prever negativo'), tx - 105, L.y + 10);
    ctx.fillText(S.t('predict positive →', 'prever positivo →'), tx + 6, L.y + 10);
    ctx.fillStyle = S.colors.orange;
    ctx.fillText(S.t('negatives (500)', 'negativos (500)'), L.x + 4, L.y + L.h + 14);
    ctx.fillStyle = S.colors.blue;
    ctx.fillText(S.t('positives (100)', 'positivos (100)'), L.x + 110, L.y + L.h + 14);

    // --- right: ROC curve ---
    ctx.strokeStyle = S.colors.grid;
    ctx.strokeRect(R.x, R.y, R.w, R.h);
    ctx.beginPath();
    ctx.moveTo(R.x, R.y + R.h); ctx.lineTo(R.x + R.w, R.y); ctx.stroke();
    ctx.strokeStyle = S.colors.blue; ctx.lineWidth = 2;
    ctx.beginPath();
    roc.forEach((p, i) => {
      const x = R.x + p.fpr * R.w, y = R.y + R.h - p.tpr * R.h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // current operating point
    const c = confusion(thr);
    const fpr = c.FP / (c.FP + c.TN), tpr = c.TP / (c.TP + c.FN);
    ctx.fillStyle = S.colors.red;
    ctx.beginPath();
    ctx.arc(R.x + fpr * R.w, R.y + R.h - tpr * R.h, 6, 0, 7);
    ctx.fill();
    ctx.fillStyle = S.colors.muted;
    ctx.fillText('ROC', R.x + R.w / 2 - 10, R.y - 4);
    ctx.fillText('FPR →', R.x + R.w - 34, R.y + R.h + 12);
    ctx.save();
    ctx.translate(R.x - 5, R.y + 40); ctx.rotate(-Math.PI / 2);
    ctx.fillText('TPR →', 0, 0);
    ctx.restore();

    // --- metrics readout ---
    const prec = c.TP + c.FP > 0 ? c.TP / (c.TP + c.FP) : 0;
    const rec = tpr;
    const f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
    const acc = (c.TP + c.TN) / 600;
    ui.readout.innerHTML =
      `TP <b style="color:${S.colors.green}">${c.TP}</b> · FP <b style="color:${S.colors.red}">${c.FP}</b>` +
      ` · FN <b style="color:${S.colors.red}">${c.FN}</b> · TN <b style="color:${S.colors.green}">${c.TN}</b>` +
      ` &nbsp;|&nbsp; ${S.t('precision', 'precisão')} <b style="color:${S.colors.orange}">${prec.toFixed(2)}</b>` +
      ` · recall <b style="color:${S.colors.orange}">${rec.toFixed(2)}</b>` +
      ` · F1 <b style="color:${S.colors.orange}">${f1.toFixed(2)}</b>` +
      ` · ${S.t('accuracy', 'acurácia')} <b>${acc.toFixed(2)}</b>` +
      `<br><span style="font-size:.75rem">${S.t('Note: at threshold 0.98 accuracy is still ~0.83 while recall collapses — the accuracy paradox in action.', 'Note: com limiar 0.98 a acurácia ainda é ~0.83 enquanto o recall desaba — o paradoxo da acurácia em ação.')}</span>`;
  }
  draw();
})();
