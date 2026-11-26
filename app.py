"""SportPulse ML — Streamlit entry point (Hugging Face Space compatible).

    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="SportPulse ML · Football forecasting hub", page_icon="⚽",
                   layout="wide", initial_sidebar_state="collapsed")

from webapp import ui  # noqa: E402
from webapp.state import artifacts_ready  # noqa: E402

ui.inject_css()

if not artifacts_ready():
    st.error("Model artifacts not found. Train them first:")
    st.code("python -m sportpulse.train all", language="bash")
    st.stop()

V = ROOT / "webapp" / "views"
pages = [
    st.Page(str(V / "hub.py"), title="Hub", icon="🏠", default=True),
    st.Page(str(V / "match_forecaster.py"), title="Match Forecaster", icon="⚡", url_path="match-forecaster"),
    st.Page(str(V / "leaderboard.py"), title="Leaderboard", icon="🏆", url_path="leaderboard"),
    st.Page(str(V / "leagues.py"), title="Leagues", icon="📊", url_path="leagues"),
    st.Page(str(V / "fpl.py"), title="PL Player Forecaster", icon="🎯", url_path="pl-player-forecaster"),
    st.Page(str(V / "international.py"), title="National Teams", icon="🌍", url_path="national-teams"),
    st.Page(str(V / "player_scout.py"), title="Player Scout", icon="🔎", url_path="player-scout"),
    st.Page(str(V / "archetypes.py"), title="Archetypes", icon="🧬", url_path="archetypes"),
    st.Page(str(V / "datasets.py"), title="Datasets", icon="🗂️", url_path="datasets"),
]
nav = st.navigation(pages, position="top")
nav.run()

st.markdown(
    f'<div style="margin-top:48px;padding:18px 0;border-top:1px solid {ui.BORDER};'
    f'color:{ui.DIM};font-size:.8rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">'
    '<span>⚽ <b style="color:#ccc">SportPulse ML</b> · real-data football forecasting · MIT</span>'
    '<span>Data: football-data.co.uk & clubelo (xgabora) · Fantasy Premier League (vaastav) · '
    'international results (martj42) · EA Sports FIFA (sofifa)</span></div>',
    unsafe_allow_html=True)
