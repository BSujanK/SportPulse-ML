"""SportPulse ML — Streamlit entry point (Hugging Face Space / Streamlit Cloud compatible).

    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="SportPulse ML", page_icon=str(ROOT / "webapp" / "assets" / "mark.svg"),
                   layout="wide", initial_sidebar_state="collapsed")

from webapp import ui  # noqa: E402
from webapp.state import artifacts_ready  # noqa: E402

ui.inject_css()
st.logo(str(ROOT / "webapp" / "assets" / "logo.svg"), size="large")

if not artifacts_ready():
    st.error("Model artifacts not found. Train them first with `python -m sportpulse.train all`.")
    st.stop()

V = ROOT / "webapp" / "views"


def page(file, title, icon, url=None, default=False):
    return st.Page(str(V / file), title=title, icon=f":material/{icon}:", url_path=url, default=default)


nav = st.navigation({
    "": [page("hub.py", "Hub", "home", default=True)],
    "Forecasting": [
        page("match_forecaster.py", "Match Forecaster", "bolt", "match-forecaster"),
        page("international.py", "National Teams", "public", "national-teams"),
        page("leaderboard.py", "Leaderboard", "leaderboard", "leaderboard"),
    ],
    "Analysis": [
        page("opponent_report.py", "Opponent Report", "shield", "opponent-report"),
        page("match_analysis.py", "Match Analysis", "sports_soccer", "match-analysis"),
        page("leagues.py", "Leagues", "table_chart", "leagues"),
    ],
    "Players": [
        page("fpl.py", "PL Player Forecaster", "query_stats", "pl-player-forecaster"),
        page("market_value.py", "Market Value", "payments", "market-value"),
        page("player_scout.py", "EA FIFA Scout", "person_search", "player-scout"),
        page("archetypes.py", "Archetypes", "hub", "archetypes"),
    ],
    "Data": [page("datasets.py", "Datasets", "database", "datasets")],
}, position="top")
# Each page renders inside a container keyed by its path: switching pages mounts a fresh
# container, which plays the CSS enter animation (reruns within a page keep the key).
with st.container(key=f"page_{(nav.url_path or 'hub').replace('-', '_')}"):
    nav.run()

st.markdown(
    f'<div class="hf-footer"><span><b>SportPulse ML</b> · football forecasting and analysis on real data · MIT</span>'
    '<span>Data: football-data.co.uk &amp; clubelo (xgabora) · StatsBomb open data · Fantasy Premier League (vaastav) · '
    'international results (martj42) · Transfermarkt (dcaribou) · EA Sports FIFA (sofifa)</span></div>'
    f'<style>.hf-footer{{margin-top:48px;padding:18px 0;border-top:1px solid {ui.BORDER};color:{ui.DIM};'
    'font-size:.8rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}'
    '.hf-footer b{color:#ccc}</style>',
    unsafe_allow_html=True)
