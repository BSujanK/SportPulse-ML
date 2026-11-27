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
stats_only = lb.get("Statistics only, no FPL xP (previous production)", {})
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
    for col, name, color in (("model", "SportPulse (production)", ui.ORANGE), ("model_stats", "Statistics only (no FPL xP)", ui.ORANGE_3),
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
    ui.note("<b>About FPL xP.</b> FPL's own pre-deadline expected points carry injury and team news that statistics "
            "cannot see, so production uses them as a model input, alongside the statistics and fixture context. The "
            "archived values match FPL's live feed exactly, so training and serving see the same feature; the archive's "
            "definition changed in 2025-26, and the test window uses the current one. "
            + (f"Without xP, best-XI points are {stats_only['top_xi']:.2f}. " if stats_only else "")
            + "Per-gameweek Spearman among active players: "
            + ", ".join(f"{ {'pred': 'SportPulse', 'pred_stats': 'statistics only', 'xP': 'FPL xP', 'form6': 'recent form'}.get(k, k)} "
                        f"{v:.3f}" for k, v in m["gw_spearman_active"].items()) + ".")

with t_card:
    kick = pd.Timestamp(d["latest_kickoff"])
    st.markdown(f"""
### Premier League player forecaster

For every Premier League player and fixture, the model predicts **FPL points** and the probability that the
player **plays 60+ minutes**, **scores** and **assists**. It is trained on **{d['rows']:,} real player-fixture rows**
({d['players']:,} players, {d['seasons'][0]} to {d['seasons'][-1]}), with data through **gameweek {d['latest_gameweek']}
({kick:%-d %B %Y})**.

#### What the model sees
Only information available before the gameweek deadline:

* **Recent form**: the last 3 and 6 fixtures plus a longer weighted average, covering minutes, starts, points, goals,
  assists, xG, xA, bonus and the ICT index (influence, creativity, threat), clean sheets, saves and cards
* **Rates**: per-90 output over the last 10 fixtures, season-to-date averages and days since the last match
* **Market signals**: price, ownership and transfer momentum from the previous gameweek
* **Fixture difficulty** from the SportPulse match engine: team and opponent Elo, expected goal difference, goals
  and shots on target for and against, and recent points, all from the team's side of the fixture

#### Evaluation
* **Training**: {d['seasons'][0]} to 2023-24 · {d['train_rows']:,} rows
* **Validation**: 2024-25 · {d['valid_rows']:,} rows (early stopping)
* **Test**: 2025-26 and 2026-27 to date · {d['test_rows']:,} rows, never seen in training
* **Serving**: refit on every season, then projected for gameweek {d['next_gameweek']}

#### Availability
Projections are scaled by FPL's published chance-of-playing flag. Players flagged as injured, suspended or
unavailable without a percentage are projected at zero.
""")

