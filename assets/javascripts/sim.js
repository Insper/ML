/**
 * ML Interactive Simulation helpers
 * Shared by the per-topic sim-*.js widgets. Styling matches quiz.js.
 * UI language follows the page locale (mkdocs-static-i18n sets <html lang>).
 */
(function () {
  const isPT = (document.documentElement.lang || 'en').toLowerCase().startsWith('pt');

  const colors = {
    bg: '#0d1117', card: '#161b22', border: '#30363d',
    orange: '#f0883e', blue: '#58a6ff', green: '#3fb950',
    red: '#ff7b72', yellow: '#d29922',
    neutral: '#c9d1d9', muted: '#8b949e', grid: '#21262d'
  };

  function t(en, pt) { return isPT && pt !== undefined ? pt : en; }

  // Deterministic RNG so every visitor sees the same dataset
  function rng(seed) {
    let s = seed >>> 0;
    return function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let x = Math.imul(s ^ (s >>> 15), 1 | s);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rand) {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function card(mountId, title, subtitle) {
    const wrap = document.getElementById(mountId);
    if (!wrap) return null;
    wrap.innerHTML = '';
    wrap.style.cssText = `background:${colors.bg};border-radius:12px;padding:1.2rem;margin:1.5rem 0;font-family:Inter,sans-serif;`;

    const header = document.createElement('div');
    header.style.cssText = `display:flex;align-items:center;gap:.8rem;margin-bottom:.9rem;padding-bottom:.7rem;border-bottom:1px solid ${colors.border};`;
    header.innerHTML = `
      <span style="font-size:1.2rem;">🕹️</span>
      <div>
        <div style="color:${colors.orange};font-weight:bold;font-size:.95rem;">${title}</div>
        <div style="color:${colors.muted};font-size:.78rem;">${subtitle || ''}</div>
      </div>`;
    wrap.appendChild(header);

    const canvasWrap = document.createElement('div');
    wrap.appendChild(canvasWrap);

    const controls = document.createElement('div');
    controls.style.cssText = 'display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.7rem;';
    wrap.appendChild(controls);

    const readout = document.createElement('div');
    readout.style.cssText = `color:${colors.muted};font-size:.82rem;margin-top:.6rem;line-height:1.5;min-height:1.2em;`;
    wrap.appendChild(readout);

    return { wrap, canvasWrap, controls, readout };
  }

  // Canvas with devicePixelRatio scaling and logical-coordinate mouse mapping
  function canvas(parent, W, H) {
    const c = document.createElement('canvas');
    const dpr = window.devicePixelRatio || 1;
    c.width = W * dpr;
    c.height = H * dpr;
    c.style.cssText = `width:100%;max-width:${W}px;display:block;margin:0 auto;background:${colors.card};border:1px solid ${colors.border};border-radius:8px;touch-action:none;`;
    parent.appendChild(c);
    const ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    function pos(evt) {
      const r = c.getBoundingClientRect();
      const e = evt.touches ? evt.touches[0] : evt;
      return { x: (e.clientX - r.left) * (W / r.width), y: (e.clientY - r.top) * (H / r.height) };
    }
    return { el: c, ctx, W, H, pos };
  }

  function slider(parent, opts) {
    const box = document.createElement('label');
    box.style.cssText = `display:flex;align-items:center;gap:.5rem;color:${colors.neutral};font-size:.82rem;`;
    const name = document.createElement('span');
    name.textContent = opts.label;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = opts.min; input.max = opts.max;
    input.step = opts.step; input.value = opts.value;
    input.style.cssText = `accent-color:${colors.orange};width:${opts.width || 140}px;`;
    const val = document.createElement('span');
    val.style.cssText = `color:${colors.orange};font-weight:bold;min-width:3.2em;font-size:.82rem;`;
    const fmt = opts.fmt || (v => v);
    val.textContent = fmt(parseFloat(input.value));
    input.addEventListener('input', () => {
      val.textContent = fmt(parseFloat(input.value));
      if (opts.oninput) opts.oninput(parseFloat(input.value));
    });
    box.appendChild(name); box.appendChild(input); box.appendChild(val);
    parent.appendChild(box);
    return { get: () => parseFloat(input.value), set: (v) => { input.value = v; val.textContent = fmt(parseFloat(input.value)); } };
  }

  function button(parent, label, onclick, primary) {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText = primary
      ? `padding:6px 18px;background:${colors.orange};color:#0d1117;border:none;border-radius:6px;cursor:pointer;font-weight:bold;font-size:.82rem;`
      : `padding:6px 14px;background:#21262d;color:${colors.neutral};border:1px solid ${colors.border};border-radius:6px;cursor:pointer;font-size:.82rem;`;
    b.addEventListener('click', onclick);
    parent.appendChild(b);
    return b;
  }

  window.mlSim = { colors, t, rng, gauss, card, canvas, slider, button, isPT };
})();
