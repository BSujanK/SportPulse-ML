import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportpulse import config
from sportpulse.inference import league_table
from webapp import state, ui

from sportpulse.inference import CALENDAR_YEAR_LEAGUES as CALENDAR_YEAR

P = state.predictor()
rec = state.recent_matches()

ui.repo_header("sportpulse", "league-intelligence", "chart",
               tag_items=[("", "Space"), ("task", "Standings · Power ratings"),
                          ("ds", "club-matches-2000-2026"), ("", f"latest match {rec.date.max():%Y-%m-%d}")])

divs = [d for d in config.DIVISIONS if d in set(rec.division)]
c1, c2 = st.columns([2, 1])
div = c1.selectbox("League", divs, index=divs.index("E0"), format_func=config.division_label)
r = rec[rec.division == div].copy()
if div in CALENDAR_YEAR:
    r["season"] = r.date.dt.year.astype(str)
else:
    y0 = r.date.dt.year - (r.date.dt.month < 7)
    r["season"] = y0.astype(str) + "-" + (y0 + 1).astype(str).str[-2:]
seasons = sorted(r.season.unique(), reverse=True)
season = c2.selectbox("Season", seasons)
rs = r[r.season == season]

t1, t2, t3 = st.tabs([":material/format_list_numbered: Standings", ":material/bolt: Power ratings (live model state)", ":material/timeline: Elo history"])
with t1:
    tbl = league_table(rs)
    ui.kpi_row([ui.kpi("Matches", f"{len(rs):,}"),
                ui.kpi("Goals / match", f"{(rs.hg + rs.ag).mean():.2f}"),
                ui.kpi("Home win", f"{(rs.result == 'H').mean() * 100:.1f}", "%"),
                ui.kpi("Draws", f"{(rs.result == 'D').mean() * 100:.1f}", "%"),
                ui.kpi("Last match", f"{rs.date.max():%d %b %Y}")])
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=min(38 * (len(tbl) + 1), 820),
                 column_config={"Pts": st.column_config.ProgressColumn(
                     "Pts", min_value=0, max_value=int(tbl.Pts.max()) if len(tbl) else 1, format="%d"),
                     "Form": st.column_config.TextColumn("Last 5")})

with t2:
    pr = P.power_rankings(div)
    pr.insert(0, "#", range(1, len(pr) + 1))
    st.dataframe(
        pr, hide_index=True, use_container_width=True, height=min(38 * (len(pr) + 1), 820),
        column_config={
            "team": "Team",
            "elo": st.column_config.ProgressColumn("Elo", min_value=float(pr.elo.min()) - 20,
                                                   max_value=float(pr.elo.max()), format="%.0f"),
            "pi_overall": st.column_config.NumberColumn("π overall", format="%.2f"),
            "pi_home": st.column_config.NumberColumn("π home", format="%.2f"),
            "pi_away": st.column_config.NumberColumn("π away", format="%.2f"),
            "form_ppg": st.column_config.NumberColumn("Form PPG", format="%.2f"),
            "form_gd": st.column_config.NumberColumn("Form GD", format="%+.2f"),
            "l_gf": st.column_config.NumberColumn("GF/90 (long)", format="%.2f"),
            "l_ga": st.column_config.NumberColumn("GA/90 (long)", format="%.2f"),
            "s_stf": st.column_config.NumberColumn("SoT for", format="%.1f"),
            "s_sta": st.column_config.NumberColumn("SoT against", format="%.1f"),
            "mkt_strength": st.column_config.NumberColumn("Market xPts", format="%.2f"),
            "games": "Games", "last_date": st.column_config.DateColumn("Last match")})
    st.caption("This is the exact state the forecaster reads for the next fixture of each team.")

with t3:
    eh = state.elo_history()
    eh = eh[eh.division == div]
    if eh.empty:
        st.info("Elo history is stored for the featured leagues: "
                + ", ".join(config.division_label(d) for d in config.FEATURED_DIVISIONS))
    else:
        long = pd.concat([eh[["date", "home_id", "elo_h"]].set_axis(["date", "team", "elo"], axis=1),
                          eh[["date", "away_id", "elo_a"]].set_axis(["date", "team", "elo"], axis=1)])
        long["team"] = long.team.str.split("|").str[1]
        top = P.power_rankings(div).team.head(6).tolist()
        pick = st.multiselect("Teams", sorted(long.team.unique()), default=top)
        fig = go.Figure()
        for i, t in enumerate(pick):
            s = long[long.team == t].sort_values("date")
            fig.add_trace(go.Scatter(x=s.date, y=s.elo, name=t, mode="lines",
                                     line=dict(width=2, color=ui.SEQ[i % len(ui.SEQ)])))
        fig.update_layout(title="Pre-match Elo rating", height=480, yaxis_title="Elo")
        ui.plot(fig)
