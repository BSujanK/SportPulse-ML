"""Hugging Face Hub–style design system, in the SportPulse black & plasma-orange theme.

Visual language borrowed from huggingface.co: thin-bordered rounded cards,
pill tags, monospace repo ids, a model-card + "Inference API" widget layout,
dataset-viewer tables and leaderboard tables. Colours are the original
SportPulse palette (core black + plasma orange).
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from webapp.icons import has as _has_icon
from webapp.icons import svg

# ── Palette (unchanged from the original theme) ──────────────────────────────
BLACK = "#0a0a0a"
CARD = "#111111"
CARD_2 = "#161616"
BORDER = "#262626"
BORDER_HI = "#3a3a3a"
ORANGE = "#FF6B00"
ORANGE_2 = "#FF8C00"
ORANGE_3 = "#FFB347"
TEXT = "#F0F0F0"
MUTED = "#9a9a9a"
DIM = "#5c5c5c"
WIN = "#22c55e"
DRAW = "#f59e0b"
LOSS = "#ef4444"
SEQ = ["#FF6B00", "#FFB347", "#22c55e", "#38bdf8", "#a78bfa", "#f472b6", "#f59e0b",
       "#94a3b8", "#ef4444", "#14b8a6", "#eab308", "#fb7185", "#818cf8", "#4ade80"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {{ --orange:{ORANGE}; --card:{CARD}; --border:{BORDER}; --muted:{MUTED}; }}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
  font-family: 'Source Sans 3', 'Source Sans Pro', ui-sans-serif, system-ui, sans-serif !important; }}
code, pre, .mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }}
.stApp {{ background: {BLACK}; }}
.block-container {{ padding-top: 4.2rem !important; max-width: 1380px; }}
header[data-testid="stHeader"] {{ background: rgba(10,10,10,.92); backdrop-filter: blur(8px);
  border-bottom: 1px solid {BORDER}; }}
/* top navigation links, HF navbar style */
header [data-testid="stTopNavLink"], [data-testid="stNavSectionHeader"] {{ border-radius: 6px; }}
header [data-testid="stTopNavLink"]:hover {{ background: rgba(255,107,0,.08); }}
header [data-testid="stTopNavLink"][aria-current="page"] span,
header a[aria-current="page"] p {{ color: {ORANGE} !important; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
hr {{ border-color: {BORDER} !important; }}

/* ── Cards ─────────────────────────────────────────── */
.hf-card {{ background: linear-gradient(180deg, {CARD_2} 0%, {CARD} 100%);
  border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; height: 100%;
  transition: border-color .15s, box-shadow .15s, transform .15s; }}
.hf-card:hover {{ border-color: {BORDER_HI}; box-shadow: 0 4px 20px rgba(0,0,0,.45); }}
.hf-card.accent:hover {{ border-color: rgba(255,107,0,.55); }}
.hf-card-title {{ font-family: 'IBM Plex Mono', monospace; font-size: .92rem; font-weight: 600;
  color: {TEXT}; display:flex; align-items:center; gap:8px; min-width:0; }}
.hf-card-title > span:last-child {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }}
.hf-card-title .org {{ color: {MUTED}; font-weight: 500; }}
.hf-card-meta {{ color: {MUTED}; font-size: .78rem; margin-top: 8px; display:flex; flex-wrap:wrap;
  gap: 4px 12px; align-items:center; }}
.hf-card-desc {{ color: #c8c8c8; font-size: .86rem; margin-top: 8px; line-height: 1.45; }}
.hf-dot {{ color: {DIM}; }}
.hf-icon {{ width: 26px; height: 26px; border-radius: 7px; display:inline-flex; align-items:center;
  justify-content:center; font-size: 14px; flex-shrink:0;
  background: rgba(255,107,0,.12); border: 1px solid rgba(255,107,0,.35); }}
.hf-icon.ds {{ background: rgba(56,189,248,.10); border-color: rgba(56,189,248,.35); }}
.hf-icon.sp {{ background: rgba(167,139,250,.12); border-color: rgba(167,139,250,.35); }}

/* ── Tags / pills ─────────────────────────────────── */
.hf-tags {{ display:flex; flex-wrap:wrap; gap:6px; margin: 10px 0 4px 0; }}
.hf-tag {{ display:inline-flex; align-items:center; gap:5px; padding: 2px 9px; border-radius: 6px;
  font-size: .76rem; border: 1px solid {BORDER}; background: {CARD_2}; color: #d4d4d4; white-space:nowrap; }}
.hf-tag.task {{ border-color: rgba(255,107,0,.45); color: {ORANGE_3}; background: rgba(255,107,0,.08); }}
.hf-tag.lib {{ border-color: rgba(56,189,248,.35); color: #7dd3fc; background: rgba(56,189,248,.06); }}
.hf-tag.ds  {{ border-color: rgba(34,197,94,.35); color: #86efac; background: rgba(34,197,94,.06); }}
.hf-tag.lic {{ border-color: rgba(167,139,250,.35); color: #c4b5fd; background: rgba(167,139,250,.06); }}

/* ── Repo header ───────────────────────────────────── */
.hf-repo {{ display:flex; align-items:center; gap:10px; font-family:'IBM Plex Mono',monospace;
  font-size: 1.35rem; font-weight: 600; margin: 4px 0 2px 0; flex-wrap: wrap; }}
.hf-repo .org {{ color: {MUTED}; font-weight: 500; }}
.hf-repo .sep {{ color: {DIM}; }}
.hf-badge {{ font-family:'Source Sans 3',sans-serif; font-size:.72rem; font-weight:600; padding: 1px 8px;
  border-radius: 6px; border:1px solid {BORDER}; color:{MUTED}; }}
.hf-badge.live {{ color:{WIN}; border-color: rgba(34,197,94,.4); background: rgba(34,197,94,.07); }}

/* ── Widget box ("Inference API") ──────────────────── */
.hf-widget-head {{ display:flex; align-items:center; gap:8px; font-weight:700; font-size:.92rem;
  color:{TEXT}; padding-bottom: 8px; border-bottom: 1px solid {BORDER}; margin-bottom: 10px; }}
.hf-widget-head .bolt {{ color:{ORANGE}; }}
.hf-widget-sub {{ color:{MUTED}; font-size:.78rem; font-weight:500; margin-left:auto; }}
.hf-label-row {{ display:flex; align-items:center; gap:10px; margin: 7px 0; font-size:.88rem; }}
.hf-label-row .lbl {{ width: 128px; color:{TEXT}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.hf-label-row .bar {{ flex:1; height: 18px; background:{CARD_2}; border-radius: 5px; overflow:hidden;
  border:1px solid {BORDER}; }}
.hf-label-row .bar > div {{ height:100%; border-radius: 4px; }}
.hf-label-row .val {{ width: 56px; text-align:right; font-family:'IBM Plex Mono',monospace; color:{MUTED};
  font-size:.82rem; }}

/* ── KPI tiles ─────────────────────────────────────── */
.hf-kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius: 10px; padding: 12px 14px; }}
.hf-kpi .k {{ color:{MUTED}; font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
.hf-kpi .v {{ font-size: clamp(1.05rem, 1.6vw, 1.55rem); white-space:nowrap; font-weight: 700; color:{TEXT}; line-height:1.15; margin-top:2px; }}
.hf-kpi .v .u {{ font-size:.9rem; color:{MUTED}; font-weight:500; margin-left:3px; }}
.hf-kpi .d {{ font-size:.76rem; margin-top:2px; color:{MUTED}; }}
.hf-kpi .d.good {{ color:{WIN}; }} .hf-kpi .d.bad {{ color:{LOSS}; }} .hf-kpi .d.orange {{ color:{ORANGE_2}; }}

/* ── Hero / section ────────────────────────────────── */
.hf-hero {{ border:1px solid {BORDER}; border-radius: 14px; padding: 26px 30px; margin-bottom: 18px;
  background: radial-gradient(1100px 260px at 88% 0%, rgba(255,107,0,.13), transparent 60%),
              linear-gradient(180deg, #121212, {BLACK}); }}
.hf-hero h1 {{ font-size: 2.05rem; font-weight: 800; margin: 0; padding: 0; color: {TEXT} !important; }}
.hf-hero h1 span {{ color:{ORANGE} !important; }}
.hf-hero-title {{ font-size: 2.05rem; font-weight: 800; letter-spacing:-0.02em; color:{TEXT}; line-height:1.15; }}
.hf-hero-title span {{ color:{ORANGE}; }}
.hf-hero p {{ color:{MUTED}; margin: 6px 0 0 0; font-size: 1rem; max-width: 900px; }}
.hf-section {{ display:flex; align-items:center; gap:10px; margin: 22px 0 10px 0; font-weight:700;
  font-size: 1.08rem; }}
.hf-section .count {{ font-weight: 500; color:{MUTED}; font-size:.9rem; }}
.hf-note {{ border:1px solid {BORDER}; border-left: 3px solid {ORANGE}; border-radius: 8px;
  padding: 10px 14px; background:{CARD}; color:#d4d4d4; font-size:.86rem; margin: 8px 0; }}
.hf-mono {{ font-family:'IBM Plex Mono',monospace; font-size:.82rem; color:{MUTED}; }}

/* ── Section icon, steps, kv, bullets ─────────────── */
.hf-section-icon {{ color:{ORANGE}; display:inline-flex; }}
.hf-steps {{ display:flex; align-items:stretch; gap:8px; flex-wrap:wrap; margin: 6px 0 4px 0; }}
.hf-step {{ flex:1 1 0; min-width:140px; min-height:118px; background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:12px 14px; }}
.hf-step-icon {{ width:28px; height:28px; border-radius:7px; display:flex; align-items:center; justify-content:center;
  color:{ORANGE}; background: rgba(255,107,0,.10); border:1px solid rgba(255,107,0,.30); margin-bottom:8px; }}
.hf-step-t {{ font-weight:600; font-size:.9rem; }}
.hf-step-c {{ color:{MUTED}; font-size:.78rem; margin-top:2px; line-height:1.4; }}
.hf-step-arrow {{ display:flex; align-items:center; color:{DIM}; }}
.hf-kv {{ display:grid; grid-template-columns: minmax(0,1fr) auto; gap:8px 16px; font-size:.84rem; align-items:baseline; }}
.hf-kv .k {{ color:{MUTED}; min-width:0; }}
.hf-kv .v {{ font-family:'IBM Plex Mono',monospace; color:{TEXT}; text-align:right; white-space:nowrap; }}
.hf-bullets .b {{ display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid {BORDER};
  font-size:.88rem; line-height:1.45; }}
.hf-bullets .b:last-child {{ border-bottom:none; }}
.hf-bullets .i {{ margin-top:2px; display:inline-flex; }}
.hf-bullets .good .i {{ color:{WIN}; }} .hf-bullets .bad .i {{ color:{LOSS}; }} .hf-bullets .info .i {{ color:{ORANGE}; }}
.hf-icon {{ color:{ORANGE_3}; }} .hf-icon.ds {{ color:#7dd3fc; }} .hf-icon.sp {{ color:#c4b5fd; }}

/* page links under cards: quiet, HF-style */
[data-testid="stPageLink"] a {{ padding: 2px 4px !important; }}
[data-testid="stPageLink"] a p {{ color:{MUTED} !important; font-size:.8rem !important; }}
[data-testid="stPageLink"] a:hover p {{ color:{ORANGE_2} !important; }}
.hf-card {{ min-height: 128px; }}

/* ── Uniform cards (fixed-height containers) ──────── */
[class*="st-key-spcard"] {{ border-radius:12px !important; overflow:hidden !important;
  background: linear-gradient(180deg, {CARD_2} 0%, {CARD} 100%) !important;
  transition: transform .35s cubic-bezier(.34,1.56,.64,1), border-color .2s ease, box-shadow .3s ease; }}
[class*="st-key-spcard"]:hover {{ transform: translateY(-3px); border-color: rgba(255,107,0,.55) !important;
  box-shadow: 0 10px 28px rgba(0,0,0,.55), 0 0 0 1px rgba(255,107,0,.12); }}
[class*="st-key-spcard"] > div {{ overflow:hidden !important; }}
[class*="st-key-spcard"] [data-testid="stElementContainer"] {{ overflow:visible !important; }}
.hf-cardbody {{ padding-bottom: 1rem; }}
[class*="st-key-spcard"] [data-testid="stVerticalBlock"] {{ height:100%; justify-content:space-between; gap:.25rem; }}
.hf-cardbody {{ display:flex; flex-direction:column; gap:6px; }}
.hf-cardbody .hf-card-desc {{ margin-top:2px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
  overflow:hidden; min-height:2.9em; }}
.hf-cardbody .hf-card-meta {{ margin-top:2px; flex-wrap:nowrap; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; display:block; }}
.hf-card-extra {{ color:{MUTED}; font-size:.78rem; display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical;
  overflow:hidden; }}
[class*="st-key-spcard"] [data-testid="stPageLink"] a {{ padding:0 !important; }}
[class*="st-key-spcard"] [data-testid="stPageLink"] a p {{ font-size:.8rem !important; font-weight:600; }}

/* ── KPI tiles: equal height, single-line captions ─── */
.hf-kpi {{ min-height: 98px; display:flex; flex-direction:column; justify-content:space-between; }}
.hf-kpi .k, .hf-kpi .d {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

/* ── Type scale (Hugging Face: Source Sans + IBM Plex Mono) ── */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ font-size:.95rem; line-height:1.62; }}
[data-testid="stMarkdownContainer"] h3 {{ font-size:1.3rem !important; font-weight:700 !important; margin:.4rem 0 .6rem 0 !important; }}
[data-testid="stMarkdownContainer"] h4 {{ font-size:1.05rem !important; font-weight:600 !important; margin:1.1rem 0 .5rem 0 !important;
  color:{TEXT} !important; }}
[data-testid="stMarkdownContainer"] table {{ border-collapse:separate; border-spacing:0; width:100%; font-size:.86rem;
  border:1px solid {BORDER}; border-radius:10px; overflow:hidden; margin:.4rem 0 1rem 0; }}
[data-testid="stMarkdownContainer"] th {{ background:{CARD_2}; color:{MUTED}; font-weight:600; text-align:left; white-space:nowrap;
  padding:8px 12px !important; border-bottom:1px solid {BORDER} !important; border-top:none !important;
  border-left:none !important; border-right:none !important; }}
[data-testid="stMarkdownContainer"] td {{ padding:7px 12px !important; border:none !important;
  border-bottom:1px solid #1c1c1c !important; }}
[data-testid="stMarkdownContainer"] tr:last-child td {{ border-bottom:none !important; }}
[data-testid="stMarkdownContainer"] code {{ font-size:.82em; padding:1px 5px; border-radius:5px; background:#1a1a1a; color:{ORANGE_3}; }}
label[data-testid="stWidgetLabel"] p {{ font-size:.82rem !important; color:{MUTED} !important; font-weight:600; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color:{DIM} !important; font-size:.78rem !important; }}

/* ── Motion: Framer-Motion-style spring/ease-out (CSS) ── */
@keyframes sp-enter {{ from {{ opacity:0; transform: translateY(14px); }} to {{ opacity:1; transform:none; }} }}
@keyframes sp-rise {{ from {{ opacity:0; transform: translateY(18px) scale(.985); }} to {{ opacity:1; transform:none; }} }}
@keyframes sp-fade {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
[class*="st-key-page_"] {{ animation: sp-enter .55s cubic-bezier(.22,1,.36,1) both; }}
[class*="st-key-page_"] [class*="st-key-spcard"], [class*="st-key-page_"] .hf-kpi, [class*="st-key-page_"] .hf-step {{ animation: sp-rise .6s cubic-bezier(.22,1,.36,1) both; }}
[class*="st-key-page_"] [data-testid="stColumn"]:nth-child(2) [class*="st-key-spcard"], [class*="st-key-page_"] [data-testid="stColumn"]:nth-child(2) .hf-kpi {{ animation-delay:.06s; }}
[class*="st-key-page_"] [data-testid="stColumn"]:nth-child(3) [class*="st-key-spcard"], [class*="st-key-page_"] [data-testid="stColumn"]:nth-child(3) .hf-kpi {{ animation-delay:.12s; }}
[class*="st-key-page_"] [data-testid="stColumn"]:nth-child(4) .hf-kpi {{ animation-delay:.18s; }}
[class*="st-key-page_"] [data-testid="stColumn"]:nth-child(5) .hf-kpi {{ animation-delay:.24s; }}
[class*="st-key-page_"] [data-testid="stColumn"]:nth-child(6) .hf-kpi {{ animation-delay:.30s; }}
.hf-step:nth-child(3) {{ animation-delay:.05s; }} .hf-step:nth-child(5) {{ animation-delay:.10s; }}
.hf-step:nth-child(7) {{ animation-delay:.15s; }} .hf-step:nth-child(9) {{ animation-delay:.20s; }}
.hf-step:nth-child(11) {{ animation-delay:.25s; }}
[data-baseweb="tab-panel"] > div {{ animation: sp-fade .35s cubic-bezier(.22,1,.36,1) both; }}
.stPlotlyChart {{ animation: sp-fade .5s ease both; }}
.hf-kpi, .hf-step {{ transition: border-color .2s ease, transform .35s cubic-bezier(.34,1.56,.64,1); }}
.hf-kpi:hover, .hf-step:hover {{ border-color:{BORDER_HI}; transform: translateY(-2px); }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation:none !important; transition:none !important; }} }}

/* ── Streamlit component polish ───────────────────── */
[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:10px 14px; }}
[data-testid="stMetricValue"] {{ font-weight:700; }}
[data-testid="stExpander"] details {{ background:{CARD}; border:1px solid {BORDER} !important; border-radius:10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:{TEXT} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{ORANGE} !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:10px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{BORDER} !important; border-radius: 12px !important;
  background: {CARD}; }}
.stButton > button[kind="primary"] {{ background:{ORANGE}; border-color:{ORANGE}; color:#000; font-weight:700; }}
.stButton > button[kind="primary"]:hover {{ background:{ORANGE_2}; border-color:{ORANGE_2}; color:#000; }}
a {{ color: {ORANGE_2}; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }} ::-webkit-scrollbar-thumb {{ background:#333; border-radius:3px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ── Plotly template ──────────────────────────────────────────────────────────
_tpl = go.layout.Template()
_tpl.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans 3, sans-serif", color=TEXT, size=13),
    colorway=SEQ,
    xaxis=dict(gridcolor="#1f1f1f", zerolinecolor="#2a2a2a", linecolor=BORDER, tickcolor=BORDER),
    yaxis=dict(gridcolor="#1f1f1f", zerolinecolor="#2a2a2a", linecolor=BORDER, tickcolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor=CARD_2, bordercolor=BORDER, font=dict(color=TEXT)),
    title=dict(font=dict(size=15)),
)
pio.templates["sportpulse"] = _tpl
pio.templates.default = "sportpulse"


def plot(fig, height: int | None = None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── HTML components ─────────────────────────────────────────────────────────
def esc(s) -> str:
    return html.escape(str(s))


def _icon(icon: str, size: int = 15) -> str:
    return svg(icon, size) if _has_icon(icon) else esc(icon)


def tags(items: list[tuple[str, str]]) -> str:
    """items: (kind, text) with kind in task|lib|ds|lic|''."""
    return '<div class="hf-tags">' + "".join(
        f'<span class="hf-tag {k}">{esc(t)}</span>' for k, t in items) + "</div>"


def repo_header(org: str, name: str, icon: str = "box", badge: str | None = None,
                tag_items: list | None = None) -> None:
    b = f'<span class="hf-badge live">{esc(badge)}</span>' if badge else ""
    st.markdown(
        f'<div class="hf-repo"><span class="hf-icon">{_icon(icon, 15)}</span><span class="org">{esc(org)}</span>'
        f'<span class="sep">/</span><span>{esc(name)}</span>{b}</div>'
        + (tags(tag_items) if tag_items else ""), unsafe_allow_html=True)


def card(title: str, org: str = "sportpulse", icon: str = "box", kind: str = "",
         desc: str = "", meta: list[str] | None = None) -> str:
    meta_html = ' <span class="hf-dot">•</span> '.join(esc(m) for m in (meta or []))
    return (f'<div class="hf-card accent"><div class="hf-card-title"><span class="hf-icon {kind}">{_icon(icon, 14)}</span>'
            f'<span title="{esc(org)}/{esc(title)}"><span class="org">{esc(org)}/</span>{esc(title)}</span></div>'
            + (f'<div class="hf-card-desc">{esc(desc)}</div>' if desc else "")
            + f'<div class="hf-card-meta">{meta_html}</div></div>')


def card_body(title: str, org: str = "sportpulse", icon: str = "box", kind: str = "",
              desc: str = "", meta: list[str] | None = None, extra: str = "") -> str:
    """Card content without its own frame (the frame is the fixed-height container)."""
    meta_html = ' <span class="hf-dot">•</span> '.join(esc(m) for m in (meta or []))
    return (f'<div class="hf-cardbody"><div class="hf-card-title"><span class="hf-icon {kind}">{_icon(icon, 14)}</span>'
            f'<span title="{esc(org)}/{esc(title)}"><span class="org">{esc(org)}/</span>{esc(title)}</span></div>'
            + (f'<div class="hf-card-desc">{esc(desc)}</div>' if desc else "")
            + (f'<div class="hf-card-extra">{extra}</div>' if extra else "")
            + f'<div class="hf-card-meta">{meta_html}</div></div>')


_GRID_N = [0]


def card_grid(items: list[dict], cols: int = 3, height: int = 176) -> None:
    """Uniform grid of equal-size cards. item keys: title, org, icon, kind, desc, meta, extra, page, link."""
    for row in range(0, len(items), cols):
        columns = st.columns(cols, gap="medium")
        for c, it in zip(columns, items[row:row + cols]):
            _GRID_N[0] += 1
            with c, st.container(border=True, height=height, key=f"spcard_{_GRID_N[0]}"):
                st.markdown(card_body(it["title"], it.get("org", "sportpulse"), it.get("icon", "box"),
                                      it.get("kind", ""), it.get("desc", ""), it.get("meta"), it.get("extra", "")),
                            unsafe_allow_html=True)
                if it.get("page"):
                    st.page_link(it["page"], label=it.get("link", "Open"), icon=":material/arrow_forward:",
                                 icon_position="right")


def kpi(label: str, value: str, unit: str = "", delta: str = "", tone: str = "") -> str:
    return (f'<div class="hf-kpi"><div class="k" title="{esc(label)}">{esc(label)}</div><div class="v">{esc(value)}'
            f'<span class="u">{esc(unit)}</span></div>'
            + f'<div class="d {tone}" title="{esc(delta)}">{esc(delta) or "&nbsp;"}</div>' + "</div>")


def kpi_grid(items: list[str], cols: int = 2) -> None:
    """KPI tiles wrapped into rows of `cols` (for narrow columns such as the inference widget)."""
    for i in range(0, len(items), cols):
        cs = st.columns(cols, gap="small")
        for c, h in zip(cs, items[i:i + cols]):
            c.markdown(h, unsafe_allow_html=True)


_FEATURE_WORDS = {
    "elo": "Elo", "pi": "pi-rating", "gd": "goal difference", "sot": "shots on target", "pts": "points",
    "gf": "goals for", "ga": "goals against", "sf": "shots for", "sa": "shots against", "stf": "shots on target for",
    "sta": "shots on target against", "cf": "corners for", "ca": "corners against", "lg": "league",
    "hg": "home goals", "ag": "away goals", "hw": "home-win rate", "mkt": "market", "str": "strength",
}


def readable_feature(name: str) -> str:
    """Turn engine feature ids into plain labels (pi_gd -> 'pi-rating expected goal difference')."""
    special = {"pi_gd": "pi-rating expected goal difference", "elo_diff": "Elo difference (incl. home advantage)",
               "elo_p_home": "Elo home-win probability", "elo_h": "Home Elo", "elo_a": "Away Elo",
               "pi_h_home": "Home team pi-rating at home", "pi_a_away": "Away team pi-rating away",
               "pi_h_away": "Home team pi-rating away", "pi_a_home": "Away team pi-rating at home",
               "games_h": "Home team experience", "games_a": "Away team experience",
               "rest_h": "Home rest days", "rest_a": "Away rest days", "division": "League", "tier": "League tier",
               "month": "Month", "lg_n": "League sample size", "lg_hg": "League home goals per match",
               "lg_ag": "League away goals per match", "lg_draw": "League draw rate", "lg_hw": "League home-win rate",
               "xh_s": "Expected home goals (short form)", "xa_s": "Expected away goals (short form)",
               "xh_l": "Expected home goals (long form)", "xa_l": "Expected away goals (long form)"}
    if name in special:
        return special[name]
    parts = name.split("_")
    gap = parts[0] == "d"
    side = {"h": "Home", "a": "Away"}.get(parts[0], "")
    memory = {"s": "short form", "l": "long form"}.get(parts[1] if len(parts) > 1 else "", "")
    rest = [_FEATURE_WORDS.get(x, x) for x in parts[2:]] if memory else [_FEATURE_WORDS.get(x, x) for x in parts[1:]]
    text = " ".join(w for w in ([side] + rest) if w)
    if gap:
        text = text[:1].upper() + text[1:] + " gap"
    return f"{text} ({memory})" if memory else (text or name)


def kpi_row(items: list[str]) -> None:
    cols = st.columns(len(items))
    for c, h in zip(cols, items):
        c.markdown(h, unsafe_allow_html=True)


def label_bars(rows: list[tuple[str, float, str]]) -> str:
    """HF text-classification widget output: label, probability bar, score."""
    out = []
    for lbl, p, color in rows:
        out.append(f'<div class="hf-label-row"><div class="lbl">{esc(lbl)}</div>'
                   f'<div class="bar"><div style="width:{p * 100:.1f}%;background:{color}"></div></div>'
                   f'<div class="val">{p:.3f}</div></div>')
    return "".join(out)


def widget_head(title: str, sub: str = "") -> str:
    return (f'<div class="hf-widget-head"><span class="bolt">{svg("zap", 15)}</span>{esc(title)}'
            f'<span class="hf-widget-sub">{esc(sub)}</span></div>')


def section(title: str, count: str | None = None, icon: str | None = None) -> None:
    c = f'<span class="count">{esc(count)}</span>' if count else ""
    i = f'<span class="hf-section-icon">{svg(icon, 16)}</span>' if icon else ""
    st.markdown(f'<div class="hf-section">{i}{esc(title)}{c}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str, str]]) -> None:
    """Horizontal pipeline diagram: (icon, title, caption)."""
    cells = "".join(
        f'<div class="hf-step"><div class="hf-step-icon">{svg(i, 16)}</div><div class="hf-step-t">{esc(t)}</div>'
        f'<div class="hf-step-c">{esc(c)}</div></div>' + ('<div class="hf-step-arrow">' + svg("arrow", 14) + '</div>'
                                                         if n < len(items) - 1 else "")
        for n, (i, t, c) in enumerate(items))
    st.markdown(f'<div class="hf-steps">{cells}</div>', unsafe_allow_html=True)


def kv(rows: list[tuple[str, str]]) -> str:
    """Two-column key / value list (replaces raw feature dumps)."""
    return '<div class="hf-kv">' + "".join(
        f'<div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>' for k, v in rows) + "</div>"


def bullets(rows: list[tuple[str, str]]) -> str:
    """Scouting-note bullets: (tone, text) with tone in good|bad|info."""
    icon = {"good": "check", "bad": "alert", "info": "activity"}
    return '<div class="hf-bullets">' + "".join(
        f'<div class="b {t}"><span class="i">{svg(icon.get(t, "activity"), 14)}</span><span>{x}</span></div>'
        for t, x in rows) + "</div>"


def note(text: str) -> None:
    st.markdown(f'<div class="hf-note">{text}</div>', unsafe_allow_html=True)


def fmt_eur(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    if v >= 1e6:
        return f"€{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"€{v / 1e3:.0f}K"
    return f"€{v:.0f}"
