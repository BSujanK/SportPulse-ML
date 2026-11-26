import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportpulse import config
from webapp import ui

D = config.ARTIFACTS_DIR / "fpl"


@st.cache_data(show_spinner=False)
def load():
    return (json.loads((D / "metrics.json").read_text()), pd.read_parquet(D / "next_gameweek.parquet"),
            pd.read_parquet(D / "test_predictions.parquet"), pd.read_parquet(D / "weekly_top_xi.parquet"))


m, nxt, tp, wk = load()
nxt = nxt.assign(news=nxt["news"].fillna("").replace({"None": "", "nan": ""}))
d = m["data"]
lb = {r["model"]: r for r in m["leaderboard"]}
prod = lb["SportPulse LightGBM (production)"]
research = lb["Research: SportPulse + archived FPL xP"]
xp = lb["FPL official xP"]

ui.repo_header("sportpulse", "premier-league-player-forecaster", "target", badge="● live",
               tag_items=[("task", "Tabular Regression"), ("task", "Binary Classification"),
                          ("lib", "LightGBM"), ("ds", "fantasy-premier-league 2016–2027"),
                          ("", f"data through GW{d['latest_gameweek']} · {str(d['latest_kickoff'])[:10]}"),
                          ("", f"projecting GW{d['next_gameweek']}")])

ui.kpi_row([
    ui.kpi("Best-XI pts / player", f"{prod['top_xi']:.2f}", "", f"FPL xP {xp['top_xi']:.2f} · form {lb['Form (last 6 mean)']['top_xi']:.2f}", "good"),
    ui.kpi("Points MAE", f"{prod['mae']:.3f}", "", f"FPL xP {xp['mae']:.3f}", "good"),
    ui.kpi("P(60+ min) AUC", f"{m['classifiers']['p_60']['auc']:.3f}"),
    ui.kpi("P(goal) AUC", f"{m['classifiers']['p_goal']['auc']:.3f}"),
    ui.kpi("Test rows", f"{d['test_rows']:,}", "", "2025-26 + 2026-27"),
])

t_proj, t_xi, t_eval, t_card = st.tabs([f":material/bolt: GW{d['next_gameweek']} projections", ":material/groups: Best XI",
                                        ":material/analytics: Evaluation", ":material/description: Model card"])

with t_proj:
    c1, c2, c3, c4 = st.columns(4)
    pos = c1.multiselect("Position", ["GK", "DEF", "MID", "FWD"], ["GK", "DEF", "MID", "FWD"])
    teams = c2.multiselect("Team", sorted(nxt.team.unique()))
    top_price = float(max(15.5, nxt.price.max()))
    max_price = c3.slider("Max price (£m)", 3.5, top_price, top_price, 0.5)
    sort = c4.selectbox("Sort by", ["xpts", "value_per_m", "p_goal", "p_assist", "p_60"])
    v = nxt[nxt.position.isin(pos) & (nxt.price <= max_price)]
    if teams:
        v = v[v.team.isin(teams)]
    v = v.sort_values(sort, ascending=False).head(150).copy()
    v["Fixture"] = v.opponent + v.was_home.map({True: " (H)", False: " (A)"})
    st.dataframe(
        v[["name", "team", "position", "Fixture", "price", "xpts", "p_60", "p_goal", "p_assist",
           "value_per_m", "form6", "xg_r6", "xa_r6", "availability", "news"]],
        hide_index=True, width="stretch", height=600,
        column_config={
            "name": "Player", "team": "Team", "position": "Pos",
            "price": st.column_config.NumberColumn("£m", format="%.1f"),
            "xpts": st.column_config.ProgressColumn("Exp. points", min_value=0,
                                                    max_value=float(nxt.xpts.max()), format="%.2f"),
            "p_60": st.column_config.NumberColumn("P(60′+)", format="percent"),
            "p_goal": st.column_config.NumberColumn("P(goal)", format="percent"),
            "p_assist": st.column_config.NumberColumn("P(assist)", format="percent"),
            "value_per_m": st.column_config.NumberColumn("pts/£m", format="%.2f"),
            "form6": st.column_config.NumberColumn("Form (6)", format="%.1f"),
            "xg_r6": st.column_config.NumberColumn("xG/g (6)", format="%.2f"),
            "xa_r6": st.column_config.NumberColumn("xA/g (6)", format="%.2f"),
            "availability": st.column_config.NumberColumn("Avail.", format="percent"),
            "news": "Team news"})
    st.caption(f"Projections for gameweek {d['next_gameweek']} from the production model refit on every "
               "fixture through the latest gameweek. Availability = FPL 'chance of playing' flag.")

with t_xi:
    budget = st.slider("Squad budget for the XI (£m)", 60.0, 120.0, 85.0, 0.5)
    pool = nxt[nxt.availability > 0.5].sort_values("xpts", ascending=False)
    need = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    # Greedy with formation + budget feasibility (cheapest-fill reserve).
    cheapest = pool.groupby("position").price.min().to_dict()
    xi, spend, cnt = [], 0.0, {k: 0 for k in need}
    for r in pool.itertuples():
        if r.position not in need or cnt[r.position] >= need[r.position][1] or len(xi) == 11:
            continue
        if r.name in {x.name for x in xi}:
            continue
        cnt[r.position] += 1
        mins_left = sum(max(lo - cnt[p], 0) for p, (lo, _) in need.items())
        reserve = sum(max(lo - cnt[p], 0) * cheapest.get(p, 4.0) for p, (lo, _) in need.items())
        if spend + r.price + reserve > budget or 11 - len(xi) - 1 < mins_left:
            cnt[r.position] -= 1
            continue
        xi.append(r)
        spend += r.price
    xi_df = pd.DataFrame(xi)
    if len(xi_df):
        order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        xi_df = xi_df.sort_values(by="position", key=lambda s: s.map(order))
        ui.kpi_row([ui.kpi("Projected XI points", f"{xi_df.xpts.sum():.1f}"),
                    ui.kpi("Cost", f"£{xi_df.price.sum():.1f}", "m"),
                    ui.kpi("Formation", f"{cnt['DEF']}-{cnt['MID']}-{cnt['FWD']}"),
                    ui.kpi("Captain pick", xi_df.loc[xi_df.xpts.idxmax(), "name"])])
        st.dataframe(xi_df[["position", "name", "team", "opponent", "was_home", "price", "xpts", "p_goal"]],
                     hide_index=True, width="stretch",
                     column_config={"price": st.column_config.NumberColumn("£m", format="%.1f"),
                                    "xpts": st.column_config.NumberColumn("Exp. pts", format="%.2f"),
                                    "p_goal": st.column_config.NumberColumn("P(goal)", format="percent")})

with t_eval:
    rows = pd.DataFrame(m["leaderboard"])
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"model": st.column_config.TextColumn("Model", width="large"),
                                "mae": st.column_config.NumberColumn("MAE ↓", format="%.3f"),
                                "rmse": st.column_config.NumberColumn("RMSE ↓", format="%.3f"),
                                "spearman": st.column_config.NumberColumn("Spearman ↑", format="%.3f"),
                                "top_xi": st.column_config.NumberColumn("Best-XI pts/player ↑", format="%.2f")})
    w = wk.copy()
    w["gw"] = w.season + " GW" + w["round"].astype(int).astype(str)
    fig = go.Figure()
    for col, name, color in (("model", "SportPulse (production)", ui.ORANGE), ("model_xp_research", "Research: + archived xP", ui.ORANGE_3),
                             ("xP", "FPL official xP", "#64748b"), ("form", "Form (6)", "#38bdf8")):
        if col in w:
            fig.add_trace(go.Scatter(x=w.gw, y=w[col].rolling(4, min_periods=1).mean(), name=name,
                                     line=dict(color=color, width=3 if col == "model" else 1.6)))
    fig.update_layout(title="Average points of each method's top-11 picks per gameweek (4-GW rolling)",
                      height=420, xaxis=dict(showticklabels=False, title="2025-26 → 2026-27 gameweeks"))
    ui.plot(fig)
    c = pd.DataFrame(m["classifiers"]).T.rename(index={"p_60": "Plays 60+ min", "p_goal": "Scores", "p_assist": "Assists"})
    ui.section("Probability heads", icon="target")
    st.dataframe(c, width="stretch",
                 column_config={k: st.column_config.NumberColumn(lbl, format="%.4f") for k, lbl in
                                {"logloss": "Log-loss ↓", "logloss_prior": "Prior log-loss", "logloss_form": "Form log-loss",
                                 "brier": "Brier ↓", "auc": "AUC ↑", "base_rate": "Base rate"}.items()})
    ui.note("<b>About FPL xP.</b> The benchmark is FPL's own pre-deadline expected points (<code>ep_this</code>, as "
            "archived by vaastav). It reflects injury and team news, which statistics cannot see. Stacking it as a feature "
            f"(research row) lifts backtest best-XI points to {research['top_xi']:.2f}, but it is <b>not deployed</b>: the "
            "live <code>ep_next</code> feed is non-zero for almost every player, while the archive is non-zero for only "
            "15–50% of rows, so a model trained on the archive over-predicts live. Instead, production uses FPL's "
            "chance-of-playing flag as a post-model availability multiplier. Per-gameweek Spearman among active players: "
            + ", ".join(f"{k} {v:.3f}" for k, v in m["gw_spearman_active"].items()) + ".")

with t_card:
    st.markdown(f"""
### Premier League player performance forecaster

Predicts, for every Premier League player and fixture: **FPL points**, **P(plays 60+ min)**, **P(scores)**,
and **P(assists)**. It is trained on **{d['rows']:,} real player-fixture rows** ({d['players']:,} players,
seasons {d['seasons'][0]} → {d['seasons'][-1]}), with data through **gameweek {d['latest_gameweek']}
({str(d['latest_kickoff'])[:10]})**.

**Features (all lagged, i.e. known before the deadline)**
* Rolling form over the last 3 / 6 fixtures and an EWM, for minutes, starts, points, goals, assists, xG, xA,
  xGI, xGC, bonus, BPS, ICT (influence, creativity, threat), clean sheets, saves and cards
* Per-90 rates over the last 10 fixtures; season-to-date averages; days since last match
* Price, ownership and transfer momentum from the previous gameweek
* **Fixture context from the SportPulse match engine:** team and opponent Elo, pi-rating goal difference,
  long-memory goals and shots on target for/against, form points (team perspective, venue-aware)
* (Research variant only) FPL's archived pre-deadline expected points as a prior

**Protocol.** Train 2016-17 → 2023-24 ({d['train_rows']:,} rows), early-stop on 2024-25 ({d['valid_rows']:,}),
test on **2025-26 + 2026-27 to date ({d['test_rows']:,} rows)**, then refit on everything for GW{d['next_gameweek']}.
Fixture-context join coverage: {d['context_join_coverage'] * 100:.1f}%.

**Live adjustments.** Projections are multiplied by FPL's published chance-of-playing flag. Players flagged
injured, suspended or unavailable with no percentage are set to zero.
""")
