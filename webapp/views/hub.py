import datetime as dt

import streamlit as st

from sportpulse import config
from webapp import state, ui

import json as _json

mm, pm = state.match_metrics(), state.player_metrics()
fm = _json.loads((config.ARTIFACTS_DIR / "fpl/metrics.json").read_text())
im = _json.loads((config.ARTIFACTS_DIR / "international/metrics.json").read_text())
flb = {r["model"]: r for r in fm["leaderboard"]}
ilb = {r["model"]: r for r in im["competitive"]}
lb = {r["model"]: r for r in mm["leaderboard"]}
prod = lb["SportPulse Market-Residual (production)"]
mkt = lb["Bookmaker market (Shin)"]
fund = lb["SportPulse Fundamentals (no odds)"]
updated = dt.datetime.fromtimestamp((config.MATCH_ARTIFACTS / "metrics.json").stat().st_mtime)
upd = f"Updated {updated:%b %d, %Y}"

st.markdown(f"""
<div class="hf-hero">
  <div style="display:flex;align-items:center;gap:16px">
    <div style="width:58px;height:58px;border-radius:14px;background:linear-gradient(135deg,#FF6B00,#9D0208);
         display:flex;align-items:center;justify-content:center;font-size:30px">⚽</div>
    <div>
      <h1>SportPulse <span>ML</span></h1>
      <p>Production football forecasting trained on <b>{mm['data']['matches_total']:,}</b> real club matches
      ({mm['data']['first_date'][:4]}–{mm['data']['last_date'][:4]}, {mm['data']['divisions']} leagues, data through
      <b>{mm['data']['last_date']}</b>) and <b>{pm['data']['player_seasons']:,}</b> EA Sports player-seasons.
      Every metric below is measured on data the models never saw.</p>
    </div>
  </div>
  {ui.tags([("task", "Tabular Classification"), ("task", "Probabilistic Forecasting"),
            ("lib", "LightGBM"), ("lib", "scikit-learn"), ("ds", "football-data.co.uk"),
            ("ds", "EA Sports FIFA"), ("lic", "MIT")])}
</div>""", unsafe_allow_html=True)

edge = (mkt["rps"] - prod["rps"]) / mkt["rps"] * 100
ui.kpi_row([
    ui.kpi("Out-of-sample RPS", f"{prod['rps']:.4f}", "", f"{edge:+.2f}% vs bookmaker", "good" if edge > 0 else "bad"),
    ui.kpi("Log-loss", f"{prod['log_loss']:.4f}", "", f"market {mkt['log_loss']:.4f}", "orange"),
    ui.kpi("1X2 accuracy", f"{prod['accuracy'] * 100:.1f}", "%", f"on {prod['n']:,} test matches"),
    ui.kpi("Calibration error", f"{prod['ece'] * 100:.2f}", "%", "class-wise ECE", "good"),
    ui.kpi("Player value R²", f"{pm['value']['model']['r2']:.3f}", "", "log €, unseen players", "orange"),
])

ui.section("Models", "8")
v, d, a = pm["value"]["model"], pm["development"]["model"], pm["archetypes"]
cards = [
    ("premier-league-player-forecaster", "🎯", "PL Player Forecaster",
     f"Next-gameweek FPL points, P(60′), P(goal), P(assist) for every PL player. Live through GW{fm['data']['latest_gameweek']}.",
     ["Tabular Regression", f"best-XI {flb['SportPulse LightGBM (production)']['top_xi']:.2f} vs FPL xP {flb['FPL official xP']['top_xi']:.2f}", upd]),
    ("national-team-forecaster", "🌍", "National Teams",
     f"Tournament-weighted World Football Elo + LightGBM. Data through {im['data']['last']}, incl. World Cup 2026.",
     ["Tabular Classification", f"RPS {ilb['SportPulse International (LightGBM)']['rps']:.4f}", upd]),
    ("match-outcome-market-residual", "⚡", "Match Forecaster",
     "LightGBM boosted from the de-vigged bookmaker prior. Production 1X2 model.",
     ["Tabular Classification", f"RPS {prod['rps']:.4f}", upd]),
    ("match-outcome-fundamentals", "📐", "Match Forecaster",
     "Pure football model: Elo, pi-ratings, form & league context. No betting data.",
     ["Tabular Classification", f"RPS {fund['rps']:.4f}", upd]),
    ("scoreline-poisson-dixon-coles", "🎯", "Match Forecaster",
     "Poisson LightGBM goal rates + Dixon-Coles low-score dependence. O/U, BTTS, correct score.",
     ["Count Regression", f"O/U LL {mm['over_under_25']['logloss_model']:.4f}", upd]),
    ("player-market-value", "💶", "Player Scout",
     "LightGBM regression of market value from 60+ attributes, contract & league.",
     ["Tabular Regression", f"R² {v['r2']:.3f}", upd]),
    ("player-development-quantile", "📈", "Player Scout",
     "Next-season rating with 80% quantile intervals. Beats persistence & potential-pull.",
     ["Quantile Regression", f"MAE {d['mae']:.2f}", upd]),
    ("player-archetypes-gmm", "🧬", "Archetypes",
     f"Gaussian mixture on quality-normalised style; k={a['k']} chosen by BIC.",
     ["Clustering", f"{a['k']} roles", upd]),
]
page_for = {"PL Player Forecaster": "webapp/views/fpl.py", "National Teams": "webapp/views/international.py",
            "Match Forecaster": "webapp/views/match_forecaster.py", "Player Scout": "webapp/views/player_scout.py",
            "Archetypes": "webapp/views/archetypes.py"}
for row in range(0, len(cards), 3):
    cols = st.columns(3)
    for c, (name, icon, page, desc, meta) in zip(cols, cards[row:row + 3]):
        with c:
            st.markdown(ui.card(name, icon=icon, desc=desc, meta=meta), unsafe_allow_html=True)
            st.page_link(page_for[page], label=f"Open in {page} →")

ui.section("Datasets", "4")
c1, c2 = st.columns(2)
with c1:
    st.markdown(ui.card("club-matches-2000-2026", icon="🗂️", kind="ds",
                        desc="Results, shots, corners, cards and pre-match 1X2 / O-U / Asian handicap odds.",
                        meta=["Tabular", f"{mm['data']['matches_total'] / 1000:.0f}k rows",
                              f"{mm['data']['divisions']} leagues", f"through {mm['data']['last_date']}"]),
                unsafe_allow_html=True)
    st.page_link("webapp/views/datasets.py", label="Open dataset viewer →")
with c2:
    st.markdown(ui.card("ea-fifa-players-15-22", icon="🗂️", kind="ds",
                        desc="Every player in EA Sports FIFA 15–22: 60+ attributes, value, wage, contract.",
                        meta=["Tabular", f"{pm['data']['player_seasons'] / 1000:.0f}k rows",
                              f"{pm['data']['players']:,} players", "8 editions"]),
                unsafe_allow_html=True)
    st.page_link("webapp/views/datasets.py", label="Open dataset viewer →")

c3, c4 = st.columns(2)
with c3:
    st.markdown(ui.card("fantasy-premier-league-2016-2027", icon="🗂️", kind="ds",
                        desc="Per-player, per-fixture minutes, goals, assists, xG, xA, ICT, bonus and FPL points.",
                        meta=["Tabular", f"{fm['data']['rows'] / 1000:.0f}k rows", f"{fm['data']['players']:,} players",
                              f"through GW{fm['data']['latest_gameweek']} 2026-27"]), unsafe_allow_html=True)
with c4:
    st.markdown(ui.card("international-results-1872-2026", icon="🗂️", kind="ds",
                        desc="Every men's international since 1872 with tournament and venue.",
                        meta=["Tabular", f"{im['data']['matches'] / 1000:.0f}k rows", f"through {im['data']['last']}"]),
                unsafe_allow_html=True)

ui.section("How it works")
st.markdown("""
```text
 raw data ─► cleaning ─► sequential feature engine (leak-free, one chronological pass)
              │            ├─ Elo (goal-difference weighted)   ├─ pi-ratings (home/away)
              │            ├─ EWM form: goals, shots, SoT, corners, points (short + long memory)
              │            └─ league scoring context            └─ Shin de-vigged market prices
              ▼
   chronological split:  train 2002-07 → 2023-06 │ valid 2023-24 │ test 2024-07 → latest
              ▼
   Stage 1  LightGBM fundamentals (Optuna-tuned)  ──► prior when odds are missing
   Stage 2  LightGBM boosted from log(p_market)   ──► learns corrections to the market
   Stage 3  temperature calibration (valid season)
   Goals    Poisson LightGBM λ_home, λ_away + Dixon-Coles ρ
              ▼
   production refit on every match up to the latest date ─► live fixture inference
```
""")
