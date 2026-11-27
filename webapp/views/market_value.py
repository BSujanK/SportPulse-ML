import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from webapp import state, ui

mm = state.market_value_metrics()
cur = state.market_value("current_players")
cur["label"] = cur["name"] + " · " + cur["club"].fillna("—")
fc, fv, fe = mm["forecast"], mm["fair_value"], mm["fees"]


def row(board, model):
    return next(r for r in board["leaderboard"] if r["model"] == model)


ui.repo_header("sportpulse", "market-value", "euro", badge="snapshot · Jun 2026",
               tag_items=[("task", "Tabular Regression"), ("lib", "LightGBM"), ("ds", "transfermarkt-2013-2026"),
                          ("", f"{mm['data']['valuations']:,} valuations"), ("", f"as of {mm['asof']}")])

f_sp, f_tm = row(fe, "SportPulse fee model (LightGBM)"), row(fe, "Transfermarkt value")
n_sp, n_pe = row(fc, "SportPulse forecast (LightGBM)"), row(fc, "Persistence (last Transfermarkt value)")
v_sp = row(fv, "SportPulse fair value (LightGBM)")
ui.kpi_row([
    ui.kpi("Transfer-fee error", f"{f_sp['median_ape'] * 100:.0f}", "%",
           f"Transfermarkt value {f_tm['median_ape'] * 100:.0f}%", "good"),
    ui.kpi("Next-valuation error", f"{n_sp['median_ape'] * 100:.1f}", "%",
           f"persistence {n_pe['median_ape'] * 100:.1f}%", "good"),
    ui.kpi("Direction of next change", f"{fc['direction_accuracy'] * 100:.0f}", "%",
           f"{fc['direction_n']:,} moves over 5%"),
    ui.kpi("Fair value R² (no player history)", f"{v_sp['r2_log']:.2f}", "", "performance, club, league, age"),
])

t_player, t_under, t_fees, t_card = st.tabs([":material/person: Player valuation", ":material/savings: Undervalued finder",
                                             ":material/swap_horiz: Transfer fees 2025-26", ":material/description: Model card"])

with t_player:
    top = cur.sort_values("tm_value", ascending=False).reset_index(drop=True)
    default = int(top.index[top.name.eq("Lamine Yamal")][0]) if top.name.eq("Lamine Yamal").any() else 0
    i = st.selectbox("Search player", top.index, index=default, format_func=lambda k: top.at[k, "label"])
    p = top.loc[i]
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f"### {ui.esc(p['name'])}")
            st.markdown(ui.tags([("task", p.sub_position or p.position or "—"), ("", p.club or "—"),
                                 ("", p.league_name or p.league or "—"), ("", p.nationality or "—"),
                                 ("", f"age {p.age:.0f}")]), unsafe_allow_html=True)
            gap_fc = p.forecast_value / p.tm_value - 1
            gap_fv = p.fair_value / p.tm_value - 1
            ui.kpi_row([
                ui.kpi("Transfermarkt value", ui.fmt_eur(p.tm_value), "", f"peak {ui.fmt_eur(p.peak_value)}"),
                ui.kpi("Next update forecast", ui.fmt_eur(p.forecast_value), "", f"{gap_fc * 100:+.0f}%",
                       "good" if gap_fc > 0.05 else "bad" if gap_fc < -0.05 else ""),
                ui.kpi("Performance fair value", ui.fmt_eur(p.fair_value), "", f"{gap_fv * 100:+.0f}% vs TM",
                       "good" if gap_fv > 0.15 else "bad" if gap_fv < -0.15 else ""),
            ])
        h = state.market_value("value_history")
        h = h[h.player_id == p.player_id].sort_values("date")
        fig = go.Figure()
        if len(h):
            fig.add_trace(go.Scatter(x=h.date, y=h.market_value_in_eur / 1e6, mode="lines+markers", name="Transfermarkt",
                                     line=dict(color=ui.ORANGE, width=3, shape="hv"),
                                     customdata=h.current_club_name,
                                     hovertemplate="%{x|%b %Y}<br>€%{y:.1f}M<br>%{customdata}<extra></extra>"))
            last = h.date.max()
            nxt = last + pd.Timedelta(days=180)
            fig.add_trace(go.Scatter(x=[last, nxt], y=[h.market_value_in_eur.iloc[-1] / 1e6, p.forecast_value / 1e6],
                                     mode="lines+markers", name="Forecast", line=dict(color=ui.ORANGE_3, dash="dash")))
        fig.update_layout(title="Market value history (€M)", height=330, legend=dict(orientation="h", y=-0.2))
        ui.plot(fig)
    with right:
        ui.section("Last 365 days", icon="activity")
        share = "—" if pd.isna(p.minutes_share_365) else f"{p.minutes_share_365 * 100:.0f}%"
        st.markdown(ui.kv([("Appearances", f"{p.apps_365:.0f}"), ("Minutes", f"{p.minutes_365:,.0f}"),
                           ("Share of club minutes", share), ("Goals", f"{p.goals_365:.0f}"),
                           ("Assists", f"{p.assists_365:.0f}"),
                           ("Contract until", "—" if pd.isna(p.contract_until) else
                            pd.Timestamp(p.contract_until).strftime("%b %Y")),
                           ("Foot · height", f"{p.foot or '—'} · {'—' if pd.isna(p.height) else f'{p.height:.0f} cm'}")]),
                    unsafe_allow_html=True)
        ui.note("<b>Next update forecast</b> predicts Transfermarkt's next valuation from everything known today. "
                "<b>Performance fair value</b> ignores this player's Transfermarkt history and prices him from "
                "minutes, output, competition level, club and age alone, so a large gap flags a player the market "
                "may be mispricing.")

with t_under:
    c1, c2, c3, c4 = st.columns(4)
    leagues = cur.league_name.dropna().value_counts().index.tolist()
    lg = c1.multiselect("League", leagues, default=[x for x in ["Premier League", "Laliga", "Serie A", "Bundesliga",
                                                               "Ligue 1"] if x in leagues])
    pos = c2.multiselect("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
    max_age = c3.slider("Max age", 17, 38, 25)
    min_share = c4.slider("Min share of club minutes", 0, 100, 40, step=5)
    q = cur[(cur.age <= max_age) & (cur.minutes_share_365.fillna(0) * 100 >= min_share) & (cur.tm_value >= 1e6)]
    if lg:
        q = q[q.league_name.isin(lg)]
    if pos:
        q = q[q.position.isin(pos)]
    q = q.assign(gap=(q.fair_value / q.tm_value - 1) * 100, trend=(q.forecast_value / q.tm_value - 1) * 100)
    q = q.sort_values("gap", ascending=False).head(50)
    st.dataframe(q[["name", "club", "league_name", "sub_position", "age", "tm_value", "fair_value", "gap",
                    "forecast_value", "trend", "minutes_365", "goals_365", "assists_365"]],
                 hide_index=True, width="stretch",
                 column_config={"name": "Player", "club": "Club", "league_name": "League", "sub_position": "Role",
                                "age": st.column_config.NumberColumn("Age", format="%.0f"),
                                "tm_value": st.column_config.NumberColumn("Transfermarkt €", format="compact"),
                                "fair_value": st.column_config.NumberColumn("Fair value €", format="compact"),
                                "gap": st.column_config.NumberColumn("Fair vs TM", format="%+.0f%%"),
                                "forecast_value": st.column_config.NumberColumn("Next update €", format="compact"),
                                "trend": st.column_config.NumberColumn("Expected move", format="%+.0f%%"),
                                "minutes_365": st.column_config.NumberColumn("Min (365d)", format="%d"),
                                "goals_365": st.column_config.NumberColumn("G", format="%d"),
                                "assists_365": st.column_config.NumberColumn("A", format="%d")})

with t_fees:
    tf = state.market_value("test_fees")
    fig = go.Figure()
    lo = min(tf.fee.min(), tf.pred.min(), tf.tm.min()) / 1e6 * 0.8
    lim = [lo, max(tf.fee.max(), tf.pred.max(), tf.tm.max()) / 1e6 * 1.2]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=ui.DIM, dash="dot"), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tf.fee / 1e6, y=tf.tm / 1e6, mode="markers", name="Transfermarkt value",
                             marker=dict(color=ui.DIM, size=6), text=tf.player,
                             hovertemplate="%{text}<br>fee €%{x:.1f}M<br>TM €%{y:.1f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=tf.fee / 1e6, y=tf.pred / 1e6, mode="markers", name="SportPulse",
                             marker=dict(color=ui.ORANGE, size=7), text=tf.player,
                             hovertemplate="%{text}<br>fee €%{x:.1f}M<br>model €%{y:.1f}M<extra></extra>"))
    fig.update_layout(title="Actual fee vs prediction, 400 largest test transfers (€M, log scale)", height=440,
                      xaxis=dict(type="log", title="Actual fee", tickvals=[1, 2, 5, 10, 20, 50, 100, 200], ticktext=["1", "2", "5", "10", "20", "50", "100", "200"], range=[np.log10(lim[0]), np.log10(lim[1])]),
                      yaxis=dict(type="log", title="Predicted", tickvals=[1, 2, 5, 10, 20, 50, 100, 200], ticktext=["1", "2", "5", "10", "20", "50", "100", "200"], range=[np.log10(lim[0]), np.log10(lim[1])]),
                      legend=dict(orientation="h", y=-0.2))
    ui.plot(fig)
    big = fe["big_moves"]
    ui.note(f"Fees of €20M or more ({big['n']} moves): SportPulse median error "
            f"{big['SportPulse']['median_ape'] * 100:.0f}% vs Transfermarkt value "
            f"{big['Transfermarkt value']['median_ape'] * 100:.0f}%. The model still under-prices the few record "
            "transfers driven by bidding wars (for example Isak and Wirtz in 2025).")
    st.dataframe(tf.head(60).assign(err_model=lambda d: (d.pred / d.fee - 1) * 100, err_tm=lambda d: (d.tm / d.fee - 1) * 100)[
        ["date", "player", "from_club", "to_club", "fee", "pred", "err_model", "tm", "err_tm"]],
        hide_index=True, width="stretch",
        column_config={"date": st.column_config.DateColumn("Date", format="MMM YYYY"), "player": "Player",
                       "from_club": "From", "to_club": "To",
                       "fee": st.column_config.NumberColumn("Fee €", format="compact"),
                       "pred": st.column_config.NumberColumn("Model €", format="compact"),
                       "err_model": st.column_config.NumberColumn("Model error", format="%+.0f%%"),
                       "tm": st.column_config.NumberColumn("TM value €", format="compact"),
                       "err_tm": st.column_config.NumberColumn("TM error", format="%+.0f%%")})

with t_card:
    for title, board, note in [
        ("Next Transfermarkt valuation", fc, "test: 2025-07 → 2026-06 valuations"),
        ("Fair value, without the player's valuation history", fv, "test: 2025-07 → 2026-06 valuations"),
        ("Transfer fees", fe, f"test: {f_sp['n']:,} paid transfers, 2025-07 → 2026-07")]:
        ui.section(title, count=f"{board['leaderboard'][0]['n']:,}", icon="scale")
        d = pd.DataFrame(board["leaderboard"])
        st.dataframe(d[["model", "mae_log", "median_ape", "within_25pct", "r2_log", "spearman"]], hide_index=True,
                     width="stretch",
                     column_config={"model": st.column_config.TextColumn("Model", width="large"),
                                    "mae_log": st.column_config.NumberColumn("MAE log ↓", format="%.3f"),
                                    "median_ape": st.column_config.NumberColumn("Median error ↓", format="percent"),
                                    "within_25pct": st.column_config.NumberColumn("Within ±25% ↑", format="percent"),
                                    "r2_log": st.column_config.NumberColumn("R² log ↑", format="%.3f"),
                                    "spearman": st.column_config.NumberColumn("Spearman ↑", format="%.3f")})
    ui.section("Method", icon="layers")
    st.markdown(ui.kv([
        ("Data", f"Transfermarkt via dcaribou/transfermarkt-datasets: {mm['data']['valuations']:,} valuations of "
                 f"{mm['data']['players']:,} players, 1.9M appearances, 175k transfers"),
        ("Leakage control", "every feature uses only appearances, games and valuations dated strictly before the "
                            "prediction date"),
        ("Split", "train 2013-07 → 2024-06 · validation 2024-07 → 2025-06 · test 2025-07 → 2026-07"),
        ("Features", "age, position, foot, height · minutes, goals, assists over 180/365/730 days and career · "
                     "top-5 league, UEFA and national-team minutes · club points and goal difference · club and "
                     "league value level · valuation history (forecast and fee models)"),
        ("Model", "LightGBM, Huber loss on log values; the forecast model learns the change from the last value")]),
        unsafe_allow_html=True)
