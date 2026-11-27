// Usage: start the app (streamlit run app.py), then: node scripts/layout_audit.js 1440,1000
// Reports overlapping blocks and vertical gaps under 12px on every page and tab.
// Layout audit: overlaps and tight vertical gaps between visual blocks, per page and viewport.
const { chromium } = require('playwright');
const PAGES = ["", "match-forecaster", "national-teams", "leaderboard", "opponent-report", "match-analysis",
               "leagues", "pl-player-forecaster", "market-value", "player-scout", "archetypes", "datasets"];
const SEL = '.hf-kpi, [class*="st-key-spcard"], .hf-note, .hf-hero, .hf-repo, .hf-tags, .hf-section, .hf-step, ' +
            '[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stTabs"] > div:first-child, ' +
            'div[data-testid="stVerticalBlockBorderWrapper"]:not([class*="st-key-spcard"]) , [data-testid="stSelectbox"], .hf-bullets, .hf-kv';
(async () => {
  const widths = (process.argv[2] || "1440,1000").split(",").map(Number);
  const b = await chromium.launch();
  let total = 0;
  for (const w of widths) {
    const p = await b.newPage({ viewport: { width: w, height: 1000 } });
    for (const path of PAGES) {
      await p.goto('http://localhost:8501/' + path, { waitUntil: 'networkidle' });
      await p.waitForTimeout(6500);
      const nTabs = await p.locator('[data-baseweb="tab"]').count();
      const tabIdx = nTabs ? [...Array(nTabs).keys()] : [-1];
      let issues = [];
      for (const ti of tabIdx) {
      if (ti >= 0) { await p.locator('[data-baseweb="tab"]').nth(ti).click(); await p.waitForTimeout(1800); }
      const tabName = ti >= 0 ? (await p.locator('[data-baseweb="tab"]').nth(ti).innerText()).trim() : "";
      const found = await p.evaluate((SEL) => {
        const els = [...document.querySelectorAll(SEL)].filter(e => {
          const r = e.getBoundingClientRect(); return r.width > 20 && r.height > 8 && e.offsetParent !== null;
        });
        const boxes = els.map(e => { const r = e.getBoundingClientRect();
          return { e, t: r.top + scrollY, b: r.bottom + scrollY, l: r.left, r: r.right,
                   n: (e.className && String(e.className).match(/hf-[a-z]+|st-key-spcard|stPlotlyChart|stDataFrame/) || [e.getAttribute('data-testid') || e.tagName])[0] }; });
        const out = [];
        for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
          const A = boxes[i], B = boxes[j];
          if (A.e.contains(B.e) || B.e.contains(A.e)) continue;
          const ox = Math.min(A.r, B.r) - Math.max(A.l, B.l), oy = Math.min(A.b, B.b) - Math.max(A.t, B.t);
          if (ox > 2 && oy > 2) out.push(`OVERLAP ${A.n} / ${B.n} by ${Math.round(oy)}px at y=${Math.round(Math.max(A.t, B.t))}`);
          else if (ox > 40) { // horizontally aligned -> check vertical gap
            const gap = B.t >= A.b ? B.t - A.b : A.t >= B.b ? A.t - B.b : null;
            if (gap !== null && gap < 12) out.push(`TIGHT ${A.n} / ${B.n} gap ${Math.round(gap)}px at y=${Math.round(Math.min(A.b, B.b))}`);
          }
        }
        return [...new Set(out)];
      }, SEL);
      issues.push(...found.map(x => (tabName ? `[${tabName}] ` : "") + x));
      }
      total += issues.length;
      console.log(`[${w}] ${path || "hub"}: ${issues.length} issue(s)` + (issues.length ? "\n   " + issues.slice(0, 8).join("\n   ") : ""));
    }
    await p.close();
  }
  console.log("TOTAL", total); await b.close();
})();
