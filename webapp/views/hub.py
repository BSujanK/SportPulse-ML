import datetime as dt

import streamlit as st

from sportpulse import config
from webapp import state, ui
from webapp.icons import MARK_SVG as MARK

import json as _json

mm, pm = state.match_metrics(), state.player_metrics()
fm = _json.loads((config.ARTIFACTS_DIR / "fpl/metrics.json").read_text())
im = _json.loads((config.ARTIFACTS_DIR / "international/metrics.json").read_text())
mv = state.market_value_metrics()
mvr = {t: {r["model"]: r for r in mv[t]["leaderboard"]} for t in ("forecast", "fair_value", "fees")}
fee_sp, fee_tm = mvr["fees"]["SportPulse fee model (LightGBM)"], mvr["fees"]["Transfermarkt value"]
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
         display:flex;align-items:center;justify-content:center;font-size:30px">{MARK}</div>
    <div>
      <div class="hf-hero-title">SportPulse <span>ML</span></div>
      <p>Production football forecasting trained on <b>{mm['data']['matches_total']:,}</b> real club matches
      ({mm['data']['first_date'][:4]}–{mm['data']['last_date'][:4]}, {mm['data']['divisions']} leagues, data through
      <b>{mm['data']['last_date']}</b>) and <b>{pm['data']['player_seasons']:,}</b> EA Sports player-seasons.
      Plus an analyst layer on <b>{state.analyst_metrics()['data']['matches']:,}</b> StatsBomb event-data matches.
      Every metric below is measured on data the models never saw.</p>
    </div>
  </div>
  {ui.tags([("task", "Tabular Classification"), ("task", "Probabilistic Forecasting"),
            ("lib", "LightGBM"), ("lib", "scikit-learn"), ("ds", "football-data.co.uk"),
            ("ds", "StatsBomb open data"), ("ds", "EA Sports FIFA"), ("lic", "MIT")])}
</div>""", unsafe_allow_html=True)

edge = (mkt["rps"] - prod["rps"]) / mkt["rps"] * 100
ui.kpi_row([
    ui.kpi("Out-of-sample RPS", f"{prod['rps']:.4f}", "", f"{edge:+.2f}% vs bookmaker", "good" if edge > 0 else "bad"),
    ui.kpi("Log-loss", f"{prod['log_loss']:.4f}", "", f"market {mkt['log_loss']:.4f}", "orange"),
    ui.kpi("1X2 accuracy", f"{prod['accuracy'] * 100:.1f}", "%", f"on {prod['n']:,} test matches"),
    ui.kpi("Calibration error", f"{prod['ece'] * 100:.2f}", "%", "class-wise ECE", "good"),
    ui.kpi("Transfer-fee error", f"{fee_sp['median_ape'] * 100:.0f}", "%",
           f"Transfermarkt value {fee_tm['median_ape'] * 100:.0f}%", "good"),
])

v, d, a = pm["value"]["model"], pm["development"]["model"], pm["archetypes"]
am = state.analyst_metrics()
axg = am["xg"]["non_penalty_cv"]
V = "webapp/views/"
models = [
    dict(title="expected-goals-freeze-frame", icon="crosshair", page=V + "match_analysis.py", link="Match Analysis",
         desc="Shot-level xG with freeze-frame geometry; beats StatsBomb's commercial xG on unseen 2022–24 tournaments.",
         meta=["Binary classification", f"log-loss {axg['SportPulse xG (LightGBM)']['log_loss']:.4f}"]),
    dict(title="premier-league-player-forecaster", icon="target", page=V + "fpl.py", link="PL Player Forecaster",
         desc="Next-gameweek points and involvement probabilities for every Premier League player.",
         meta=["Regression", f"best XI {flb['SportPulse LightGBM (production)']['top_xi']:.2f} vs xP "
                             f"{flb['FPL official xP']['top_xi']:.2f}"]),
    dict(title="national-team-forecaster", icon="globe", page=V + "international.py", link="National Teams",
         desc="Tournament-weighted World Football Elo with a LightGBM layer, including World Cup 2026.",
         meta=["Classification", f"RPS {ilb['SportPulse International (LightGBM)']['rps']:.4f}"]),
    dict(title="match-outcome-market-residual", icon="zap", page=V + "match_forecaster.py", link="Match Forecaster",
         desc="LightGBM boosted from the de-vigged bookmaker prior. The production 1X2 model.",
         meta=["Classification", f"RPS {prod['rps']:.4f}"]),
    dict(title="match-outcome-fundamentals", icon="scale", page=V + "match_forecaster.py", link="Match Forecaster",
         desc="Pure football model: Elo, pi-ratings, form and league context. No betting data.",
         meta=["Classification", f"RPS {fund['rps']:.4f}"]),
    dict(title="scoreline-poisson-dixon-coles", icon="grid", page=V + "match_forecaster.py", link="Match Forecaster",
         desc="Poisson goal rates with Dixon-Coles low-score dependence: O/U, BTTS, correct score.",
         meta=["Count regression", f"O/U log-loss {mm['over_under_25']['logloss_model']:.4f}"]),
    dict(title="transfer-fee", icon="euro", page=V + "market_value.py", link="Market Value",
         desc="Fee for a real transfer from form, minutes, both clubs and the valuation history.",
         meta=["Regression", f"error {fee_sp['median_ape'] * 100:.0f}% vs TM {fee_tm['median_ape'] * 100:.0f}%"]),
    dict(title="market-value-forecast", icon="trending", page=V + "market_value.py", link="Market Value",
         desc="Predicts each player's next Transfermarkt valuation from everything known today.",
         meta=["Regression", f"error {mvr['forecast']['SportPulse forecast (LightGBM)']['median_ape'] * 100:.1f}% "
                             f"vs {mvr['forecast']['Persistence (last Transfermarkt value)']['median_ape'] * 100:.1f}%"]),
    dict(title="performance-fair-value", icon="search", page=V + "market_value.py", link="Market Value",
         desc="Values a player from minutes, output, club and league alone, to flag mispriced players.",
         meta=["Regression", f"R² {mvr['fair_value']['SportPulse fair value (LightGBM)']['r2_log']:.2f}"]),
    dict(title="ea-fifa-development", icon="activity", page=V + "player_scout.py", link="EA FIFA Scout",
         desc="Next-season EA FIFA rating with 80% quantile intervals (FIFA 15–22 data).",
         meta=["Quantile regression", f"MAE {d['mae']:.2f}"]),
    dict(title="player-archetypes-gmm", icon="component", page=V + "archetypes.py", link="Archetypes",
         desc=f"Gaussian mixture on EA FIFA playing-style attributes, {a['k']} roles chosen by BIC.",
         meta=["Clustering", f"{a['k']} roles"]),
]
ui.section("Models", f"{len(models)} · {upd.lower()}", icon="box")
ui.card_grid(models, cols=3, height=172)

D = V + "datasets.py"
datasets = [
    dict(title="club-matches-2000-2026", icon="database", kind="ds", page=D, link="Dataset viewer",
         desc="Results, shots, corners, cards and pre-match 1X2, over/under and Asian handicap odds.",
         meta=[f"{mm['data']['matches_total'] / 1000:.0f}k rows", f"{mm['data']['divisions']} leagues",
               f"to {mm['data']['last_date']}"]),
    dict(title="statsbomb-open-events", icon="database", kind="ds", page=D, link="Dataset viewer",
         desc=f"Event data with pitch locations and shot freeze-frames from {len(am['data']['competitions'])} competition-seasons.",
         meta=[f"{am['data']['matches']:,} matches", f"{am['data']['shots'] / 1000:.0f}k shots",
               f"{(am['data']['passes'] + am['data']['carries']) / 1e6:.1f}M actions"]),
    dict(title="fantasy-premier-league", icon="database", kind="ds", page=V + "fpl.py", link="PL Player Forecaster",
         desc="Per-player, per-fixture minutes, goals, assists, xG, xA, ICT, bonus and FPL points.",
         meta=[f"{fm['data']['rows'] / 1000:.0f}k rows", f"{fm['data']['players']:,} players",
               f"to GW{fm['data']['latest_gameweek']} 2026-27"]),
    dict(title="international-results", icon="database", kind="ds", page=V + "international.py", link="National Teams",
         desc="Every men's international since 1872, with tournament and venue.",
         meta=[f"{im['data']['matches'] / 1000:.0f}k rows", f"to {im['data']['last']}"]),
    dict(title="transfermarkt-2013-2026", icon="database", kind="ds", page=V + "market_value.py", link="Market Value",
         desc="Market valuations, appearances, games and transfers with fees (dcaribou snapshot).",
         meta=[f"{mv['data']['valuations'] / 1000:.0f}k valuations", f"{mv['data']['players']:,} players",
               f"to {mv['data']['last']}"]),
    dict(title="ea-fifa-players-15-22", icon="database", kind="ds", page=D, link="Dataset viewer",
         desc="Every player in EA Sports FIFA 15–22: 60+ attributes, value, wage and contract.",
         meta=[f"{pm['data']['player_seasons'] / 1000:.0f}k rows", f"{pm['data']['players']:,} players"]),
]
ui.section("Datasets", str(len(datasets)), icon="database")
ui.card_grid(datasets, cols=3, height=172)

ui.section("How it works", icon="layers")
ui.steps([("database", "Ingest", "Club results & odds, StatsBomb events, FPL, internationals. Nothing imputed."),
          ("activity", "Feature engine", "One chronological pass: Elo, pi-ratings, form, Shin market prices"),
          ("clock", "Chronological split", "Train → validate → untouched test window"),
          ("layers", "Models", "LightGBM stacks, Poisson-Dixon-Coles, xG, expected threat"),
          ("gauge", "Calibrate & benchmark", "vs bookmakers, FPL xP, World Elo, StatsBomb xG"),
          ("zap", "Serve", "Refit on all data → live fixture, gameweek and match reports")])
