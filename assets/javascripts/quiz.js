/**
 * ML Interactive Quiz Engine
 * Usage: buildQuiz('unique-id', 'Topic Title', questionsArray)
 * Each question: { q, opts: [string...], ans: index, exp: string }
 *
 * UI language follows the page locale (mkdocs-static-i18n sets <html lang>).
 */
(function () {
  const I18N = {
    en: {
      subtitle: (n) => `${n} questions · pick an option, then press Check`,
      check: 'Check Answers',
      retry: '↺ Retry',
      explanation: '💡 Explanation:',
      great: '🎉 Excellent!',
      good: '👍 Good job!',
      review: '📚 Review the material'
    },
    pt: {
      subtitle: (n) => `${n} questões · clique em uma opção e depois em Verificar`,
      check: 'Verificar Respostas',
      retry: '↺ Refazer',
      explanation: '💡 Explicação:',
      great: '🎉 Excelente!',
      good: '👍 Bom trabalho!',
      review: '📚 Revise o conteúdo'
    }
  };

  function strings() {
    const lang = (document.documentElement.lang || 'en').toLowerCase();
    return lang.startsWith('pt') ? I18N.pt : I18N.en;
  }

  const colors = {
    bg: '#0d1117', card: '#161b22', border: '#30363d',
    correctBg: '#1f3d1f', wrongBg: '#3d1a1a',
    correct: '#3fb950', wrong: '#ff7b72', unanswered: '#d29922',
    neutral: '#c9d1d9', muted: '#8b949e', accent: '#f0883e'
  };

  window.buildQuiz = function (id, title, qs) {
    const wrap = document.getElementById('quiz-' + id);
    if (!wrap) return;
    const t = strings();

    wrap.innerHTML = '';
    wrap.style.cssText = `background:${colors.bg};border-radius:12px;padding:1.5rem;margin:2rem 0;font-family:Inter,sans-serif;`;

    // Header
    const header = document.createElement('div');
    header.style.cssText = `display:flex;align-items:center;gap:.8rem;margin-bottom:1.2rem;padding-bottom:.8rem;border-bottom:1px solid ${colors.border};`;
    const badge = document.createElement('div');
    badge.style.cssText = 'margin-left:auto;display:none;padding:4px 14px;border-radius:20px;font-weight:bold;font-size:.9rem;';
    header.innerHTML = `
      <span style="font-size:1.3rem;">🧠</span>
      <div>
        <div style="color:${colors.accent};font-weight:bold;font-size:1rem;">Quiz — ${title}</div>
        <div style="color:${colors.muted};font-size:.8rem;">${t.subtitle(qs.length)}</div>
      </div>`;
    header.appendChild(badge);
    wrap.appendChild(header);

    // Questions
    const cards = [];
    qs.forEach((q, qi) => {
      const qDiv = document.createElement('div');
      qDiv.style.cssText = `margin-bottom:1rem;padding:1rem;background:${colors.card};border-radius:8px;border:1px solid ${colors.border};transition:border-color .2s;`;

      const qText = document.createElement('div');
      qText.style.cssText = `color:${colors.neutral};font-weight:600;margin-bottom:.7rem;font-size:.95rem;`;
      qText.textContent = `${qi + 1}. ${q.q}`;
      qDiv.appendChild(qText);

      const labels = [];
      q.opts.forEach((opt, oi) => {
        const label = document.createElement('label');
        label.style.cssText = 'display:flex;align-items:flex-start;gap:.6rem;padding:.45rem .6rem;border-radius:5px;cursor:pointer;margin-bottom:.2rem;transition:background .15s;border:1px solid transparent;';
        label.onmouseover = () => { if (!label.dataset.locked) label.style.background = '#21262d'; };
        label.onmouseout = () => { if (!label.dataset.locked) label.style.background = 'transparent'; };

        const input = document.createElement('input');
        input.type = 'radio';
        input.name = `quiz-${id}-q${qi}`;
        input.value = String(oi);
        input.style.cssText = `margin-top:3px;accent-color:${colors.accent};flex-shrink:0;`;

        const span = document.createElement('span');
        span.style.cssText = `color:${colors.muted};font-size:.9rem;`;
        span.textContent = opt;

        label.appendChild(input);
        label.appendChild(span);
        labels.push({ label, input, span });
        qDiv.appendChild(label);
      });

      const exp = document.createElement('div');
      exp.style.cssText = 'display:none;margin-top:.6rem;padding:.6rem .8rem;background:#0d2137;border-left:3px solid #58a6ff;border-radius:0 5px 5px 0;color:#8b949e;font-size:.83rem;line-height:1.5;';
      qDiv.appendChild(exp);

      cards.push({ qDiv, labels, exp, ans: q.ans, expText: q.exp });
      wrap.appendChild(qDiv);
    });

    // Footer: buttons + result
    const footer = document.createElement('div');
    footer.style.cssText = 'display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.5rem;';

    const checkBtn = document.createElement('button');
    checkBtn.textContent = t.check;
    checkBtn.style.cssText = `padding:8px 26px;background:${colors.accent};color:#0d1117;border:none;border-radius:6px;cursor:pointer;font-weight:bold;font-size:.9rem;`;

    const retryBtn = document.createElement('button');
    retryBtn.textContent = t.retry;
    retryBtn.style.cssText = `padding:8px 18px;background:#21262d;color:${colors.neutral};border:1px solid ${colors.border};border-radius:6px;cursor:pointer;font-size:.9rem;`;

    const result = document.createElement('div');
    result.style.cssText = `color:${colors.muted};font-size:.85rem;`;

    footer.appendChild(checkBtn);
    footer.appendChild(retryBtn);
    footer.appendChild(result);
    wrap.appendChild(footer);

    checkBtn.addEventListener('click', () => {
      let score = 0;
      checkBtn.disabled = true;

      cards.forEach((card) => {
        const sel = card.labels.find(({ input }) => input.checked);
        const chosen = sel ? parseInt(sel.input.value, 10) : -1;
        const correct = chosen === card.ans;
        if (correct) score++;

        card.labels.forEach(({ label, input, span }, oi) => {
          label.dataset.locked = '1';
          label.style.cursor = 'default';
          input.disabled = true;
          if (oi === card.ans) {
            label.style.background = colors.correctBg;
            label.style.borderColor = colors.correct;
            span.style.color = colors.correct;
          } else if (oi === chosen && !correct) {
            label.style.background = colors.wrongBg;
            label.style.borderColor = colors.wrong;
            span.style.color = colors.wrong;
          }
        });

        card.qDiv.style.borderColor = correct ? colors.correct : (chosen === -1 ? colors.unanswered : colors.wrong);

        if (card.expText) {
          card.exp.style.display = 'block';
          card.exp.innerHTML = `<strong style="color:#58a6ff;">${t.explanation}</strong> ${card.expText}`;
        }
      });

      const total = cards.length;
      const pct = Math.round((score / total) * 100);
      const color = pct >= 80 ? colors.correct : pct >= 50 ? colors.unanswered : colors.wrong;
      const msg = pct >= 80 ? t.great : pct >= 50 ? t.good : t.review;

      badge.style.display = 'block';
      badge.style.background = color + '22';
      badge.style.color = color;
      badge.style.border = `1px solid ${color}`;
      badge.textContent = `${score}/${total}`;
      result.innerHTML = `<span style="color:${color};font-weight:bold;">${pct}% — ${msg}</span>`;
    });

    retryBtn.addEventListener('click', () => {
      cards.forEach((card) => {
        card.labels.forEach(({ label, input, span }) => {
          label.style.background = 'transparent';
          label.style.borderColor = 'transparent';
          label.dataset.locked = '';
          label.style.cursor = 'pointer';
          span.style.color = colors.muted;
          input.checked = false;
          input.disabled = false;
        });
        card.qDiv.style.borderColor = colors.border;
        card.exp.style.display = 'none';
      });
      checkBtn.disabled = false;
      badge.style.display = 'none';
      result.innerHTML = '';
    });
  };
})();
