import json

import pandas as pd
import streamlit as st

from sportpulse import config
from sportpulse.international import InternationalPredictor
from webapp import ui

D = config.ARTIFACTS_DIR / "international"


@st.cache_resource(show_spinner=False)
def predictor():
    return InternationalPredictor()


@st.cache_data(show_spinner=False)
def load():
    return json.loads((D / "metrics.json").read_text()), pd.read_parquet(D / "test_predictions.parquet")


m, tp = load()
P = predictor()
d = m["data"]
lb = {r["model"]: r for r in m["competitive"]}
mine, elo = lb["SportPulse International (LightGBM)"], lb["World Football Elo (standard, logistic)"]

ui.repo_header("sportpulse", "national-team-forecaster", "globe", badge="● live",
               tag_items=[("task", "Tabular Classification"), ("lib", "LightGBM"), ("", "World Football Elo"),
                          ("ds", "international-results 1872–2026"), ("", f"data through {d['last']}")])

ui.kpi_row([ui.kpi("Competitive RPS", f"{mine['rps']:.4f}", "", f"Elo {elo['rps']:.4f}", "good"),
            ui.kpi("Accuracy", f"{mine['accuracy'] * 100:.1f}", "%", f"{mine['n']:,} competitive matches"),
            ui.kpi("Log-loss", f"{mine['log_loss']:.4f}", "", f"Elo {elo['log_loss']:.4f}"),
            ui.kpi("Nations rated", f"{len(P.ranking):,}"),
            ui.kpi("Matches", f"{d['matches'] / 1000:.0f}k", "", f"{d['first'][:4]}–{d['last'][:4]}")])

left, right = st.columns([1.5, 1], gap="large")
with right:
    with st.container(border=True):
        st.markdown(ui.widget_head("Inference API", "national teams"), unsafe_allow_html=True)
        teams = P.ranking.team.tolist()
        c1, c2 = st.columns(2)
        h = c1.selectbox("Team A", teams, index=teams.index("Spain") if "Spain" in teams else 0)
        a = c2.selectbox("Team B", teams, index=teams.index("Argentina") if "Argentina" in teams else 1)
        neutral = st.toggle("Neutral venue", True)
        tourn = st.selectbox("Competition", ["FIFA World Cup", "UEFA Euro", "Copa América",
                                             "FIFA World Cup qualification", "UEFA Nations League", "Friendly"])
        if h != a:
            p, f = P.predict(h, a, neutral=neutral, tournament=tourn)
            st.markdown(ui.label_bars([(f"{h} win", p[0], ui.WIN), ("Draw", p[1], ui.DRAW),
                                       (f"{a} win", p[2], ui.LOSS)]), unsafe_allow_html=True)
            st.caption(f"Elo {f['elo_h']:.0f} vs {f['elo_a']:.0f}"
                       f"{'' if neutral else ' (+100 home advantage)'} · fair odds "
                       f"{1 / p[0]:.2f} / {1 / p[1]:.2f} / {1 / p[2]:.2f}")
        else:
            st.warning("Pick two different teams.")

with left:
    t1, t2, t3 = st.tabs([":material/military_tech: World ranking", ":material/analytics: Evaluation", ":material/history: Recent predictions"])
    with t1:
        r = P.ranking.head(60).copy()
        r.insert(0, "#", range(1, len(r) + 1))
        st.dataframe(r[["#", "team", "elo", "form_pts", "gf", "ga", "games", "last"]], hide_index=True,
                     width="stretch", height=560,
                     column_config={"team": "Nation",
                                    "elo": st.column_config.ProgressColumn("Elo", min_value=float(r.elo.min()) - 30,
                                                                           max_value=float(r.elo.max()), format="%.0f"),
                                    "form_pts": st.column_config.NumberColumn("Form PPG", format="%.2f"),
                                    "gf": st.column_config.NumberColumn("GF/g", format="%.2f"),
                                    "ga": st.column_config.NumberColumn("GA/g", format="%.2f"),
                                    "last": st.column_config.DateColumn("Last match")})
    with t2:
        for title, key in (("All test matches", "leaderboard"), ("Competitive only (no friendlies)", "competitive"),
                           ("FIFA World Cup 2026", "world_cup_2026"),
                           ("Competitive, both squads covered by Transfermarkt", "competitive_covered")):
            if key not in m:
                continue
            ui.section(title, f"n = {m[key][0]['n']:,}" if m[key] else "")
            st.dataframe(pd.DataFrame(m[key])[["model", "rps", "log_loss", "brier", "accuracy", "ece"]],
                         hide_index=True, width="stretch",
                         column_config={"model": st.column_config.TextColumn("Model", width="large")} | {
                             c: st.column_config.NumberColumn(lbl, format="%.4f") for c, lbl in
                             {"rps": "RPS ↓", "log_loss": "Log-loss ↓", "brier": "Brier ↓", "accuracy": "Accuracy ↑",
                              "ece": "ECE ↓"}.items()})
        ui.note(f"Chronological protocol: ratings run from {d['first']}; the model trains on 1994 → 2022, early-stops "
                f"on 2023 → mid-2024 and is tested on <b>{d['test_window'][0]} → {d['test_window'][1]}</b> "
                f"({d['test']:,} matches, including Euro 2024 qualifiers onward and the 2026 World Cup). On the 104 "
                "World Cup matches the gap to Elo is small relative to its noise. The squad-value logistic replicates Peeters "
                "(2018); where a squad has no Transfermarkt coverage its value difference is set to zero, so read it "
                "on the covered-squads table.")
    with t3:
        x = tp.sort_values("date", ascending=False).head(80).copy()
        x["Match"] = x.home_team + " – " + x.away_team
        x["Score"] = x.hg.astype(str) + "–" + x.ag.astype(str)
        st.dataframe(x[["date", "tournament", "Match", "Score", "p_h", "p_d", "p_a"]], hide_index=True,
                     width="stretch", height=560,
                     column_config={"date": st.column_config.DateColumn("Date"),
                                    "p_h": st.column_config.ProgressColumn("P(A)", min_value=0, max_value=1, format="%.2f"),
                                    "p_d": st.column_config.ProgressColumn("P(draw)", min_value=0, max_value=1, format="%.2f"),
                                    "p_a": st.column_config.ProgressColumn("P(B)", min_value=0, max_value=1, format="%.2f")})
