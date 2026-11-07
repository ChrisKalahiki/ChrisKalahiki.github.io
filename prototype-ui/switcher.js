// PROTOTYPE: throwaway variant + theme switcher for the UI options ticket. Never merge to main.
(function () {
  const VARIANTS = [
    { key: 'a', name: 'Minimal theme + dark mode' },
    { key: 'b', name: 'Custom editorial layout' },
    { key: 'c', name: 'al-folio style' },
  ];

  const parts = location.pathname.split('/');
  const page = parts.pop() || 'index.html';
  const current = parts.pop();
  const index = Math.max(0, VARIANTS.findIndex((v) => v.key === current));

  function readTheme() {
    const fromUrl = new URLSearchParams(location.search).get('theme');
    if (fromUrl === 'light' || fromUrl === 'dark') return fromUrl;
    try {
      const stored = localStorage.getItem('proto-theme');
      if (stored) return stored;
    } catch (err) {
      console.warn('theme storage unavailable', err);
    }
    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem('proto-theme', theme);
    } catch (err) {
      console.warn('theme storage unavailable', err);
    }
    const btn = document.getElementById('proto-theme');
    if (btn) btn.textContent = theme === 'dark' ? '☾ dark' : '☀ light';
  }

  function go(offset) {
    const next = VARIANTS[(index + offset + VARIANTS.length) % VARIANTS.length];
    location.href = `../${next.key}/${page}?theme=${document.documentElement.dataset.theme}`;
  }

  applyTheme(readTheme());

  const bar = document.createElement('div');
  bar.id = 'proto-bar';
  bar.innerHTML = `
    <button data-step="-1" aria-label="Previous variant">←</button>
    <span><b>${VARIANTS[index].key.toUpperCase()}</b> ${VARIANTS[index].name}</span>
    <button data-step="1" aria-label="Next variant">→</button>
    <span class="sep"></span>
    <button id="proto-theme"></button>
    <span class="sep"></span>
    <a href="index.html">Home</a><a href="publications.html">Pubs</a>`;
  const style = document.createElement('style');
  style.textContent = `
    #proto-bar{position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:99999;display:flex;align-items:center;gap:8px;
      padding:6px 10px;border-radius:999px;background:#111;color:#fff;font:13px/1 ui-monospace,Menlo,monospace;
      box-shadow:0 6px 24px rgba(0,0,0,.35);outline:2px solid #ff0;white-space:nowrap;max-width:calc(100vw - 32px);overflow-x:auto}
    #proto-bar button,#proto-bar a{all:unset;cursor:pointer;padding:6px 8px;border-radius:999px;color:#fff}
    #proto-bar button:hover,#proto-bar a:hover{background:#333}
    #proto-bar .sep{width:1px;height:16px;background:#444}`;
  document.head.appendChild(style);
  document.body.appendChild(bar);
  applyTheme(document.documentElement.dataset.theme);

  bar.addEventListener('click', (e) => {
    const step = e.target.closest('[data-step]');
    if (step) go(Number(step.dataset.step));
    if (e.target.id === 'proto-theme') {
      applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea, [contenteditable]')) return;
    if (e.key === 'ArrowLeft') go(-1);
    if (e.key === 'ArrowRight') go(1);
    if (e.key === 't') applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });
})();
