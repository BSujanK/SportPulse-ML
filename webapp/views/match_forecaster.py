import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportpulse import config
from webapp import state, ui

P = state.predictor()
mm = state.match_metrics()
lb = {r["model"]: r for r in mm["leaderboard"]}
prod, mkt, fund = (lb["SportPulse Market-Residual (production)"], lb["Bookmaker market (Shin)"],
                   lb["SportPulse Fundamentals (no odds)"])

ui.repo_header("sportpulse", "match-outcome-market-residual", "zap", badge="● live",
               tag_items=[("task", "Tabular Classification"), ("task", "1X2 · O/U · BTTS · Correct score"),
                          ("lib", "LightGBM"), ("lib", "Optuna"), ("ds", "club-matches-2000-2026"),
                          ("", f"trained through {mm['production']['trained_through']}"), ("lic", "mit")])

left, right = st.columns([1.55, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE WIDGET
# ══════════════════════════════════════════════════════════════════════════════
with right:
    with st.container(border=True):
        st.markdown(ui.widget_head("Inference API", "Tabular Classification"), unsafe_allow_html=True)
        divs = [d for d in config.DIVISIONS if d in set(P.recent.division)]
        div = st.selectbox("League", divs, index=divs.index("E0") if "E0" in divs else 0,
                           format_func=config.division_label)
        teams = P.teams_in(div)
        name = lambda t: t.split("|", 1)[1]
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home", teams, index=0, format_func=name)
        away = c2.selectbox("Away", teams, index=min(1, len(teams) - 1), format_func=name)
        use_odds = st.toggle("Include bookmaker odds", value=False,
                             help="Adds the de-vigged market prices and switches to the market-residual model.")
        odds = None
        if use_odds:
            o1, o2, o3 = st.columns(3)
            odds = (o1.number_input("Home", 1.01, 100.0, 2.10, 0.01),
                    o2.number_input("Draw", 1.01, 100.0, 3.40, 0.01),
                    o3.number_input("Away", 1.01, 100.0, 3.60, 0.01))
        go_btn = st.button("Compute", type="primary", width="stretch")

        if home == away:
            st.warning("Pick two different teams.")
        elif go_btn or "fx" not in st.session_state or st.session_state.get("fx_key") != (home, away, div, odds):
            st.session_state.fx = P.predict(home, away, div, odds=odds)
            st.session_state.fx_key = (home, away, div, odds)

        if home != away and "fx" in st.session_state:
            r = st.session_state.fx
            p = r.p_final if odds else r.p_fundamentals
            model_used = "market-residual" if odds else "fundamentals"
            st.markdown(ui.label_bars([(f"{r.home} win", p[0], ui.WIN), ("Draw", p[1], ui.DRAW),
                                       (f"{r.away} win", p[2], ui.LOSS)]), unsafe_allow_html=True)
            st.caption(f"{'Market-residual model' if odds else 'No-odds model'} · fair odds "
                       f"{1 / p[0]:.2f} / {1 / p[1]:.2f} / {1 / p[2]:.2f}")
            if r.p_market is not None:
                edge = p - r.p_market
                st.caption("Model − market: " + " · ".join(
                    f"{k} {v * 100:+.1f}pp" for k, v in zip(["H", "D", "A"], edge)))
            ui.kpi_grid([ui.kpi(f"{r.home} xG", f"{r.xg_home:.2f}"), ui.kpi(f"{r.away} xG", f"{r.xg_away:.2f}"),
                         ui.kpi("Over 2.5 goals", f"{r.p_over25 * 100:.0f}", "%"),
                         ui.kpi("Both score", f"{r.p_btts * 100:.0f}", "%")], cols=2)
            S = r.scores[:6, :6]
            fig = go.Figure(go.Heatmap(z=S * 100, x=list(range(6)), y=list(range(6)),
                                       colorscale=[[0, "#111111"], [0.5, "#7a3300"], [1, ui.ORANGE]],
                                       text=np.round(S * 100, 1), texttemplate="%{text}",
                                       showscale=False, hovertemplate="%{y}-%{x}: %{z:.1f}%<extra></extra>"))
            fig.update_layout(title="Scoreline probability (%)",
                              xaxis_title=f"{r.away} goals", yaxis_title=f"{r.home} goals",
                              yaxis=dict(autorange="reversed"), height=330)
            ui.plot(fig)
            f = r.features
            ui.section("Model inputs", icon="activity")
            st.markdown(ui.kv([
                ("Elo (home, away)", f"{f['elo_h']:.0f} · {f['elo_a']:.0f}"),
                ("Pi-rating (home at home, away away)", f"{f['pi_h_home']:.2f} · {f['pi_a_away']:.2f}"),
                ("Expected goal difference", f"{f['pi_gd']:+.2f}"),
                ("Recent points per game", f"{f['h_s_pts']:.2f} · {f['a_s_pts']:.2f}"),
                ("Recent goal difference", f"{f['h_s_gf'] - f['h_s_ga']:+.2f} · {f['a_s_gf'] - f['a_s_ga']:+.2f}"),
                ("Rest days", f"{f['rest_h']:.0f} · {f['rest_a']:.0f}")]), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CARD / EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with left:
    t_card, t_eval, t_recent, t_feat = st.tabs([":material/description: Model card", ":material/analytics: Evaluation", ":material/history: Recent predictions",
                                                ":material/data_object: Features"])
    with t_card:
        d = mm["data"]
        st.markdown(f"""
### Match outcome forecaster

A probabilistic model for football match outcomes (home win, draw, away win),
trained on **{d['matches_total']:,} real matches** from **{d['divisions']} leagues** played between
{pd.Timestamp(d['first_date']):%B %Y} and **{pd.Timestamp(d['last_date']):%-d %B %Y}**.

| Component | Method | Purpose |
|---|---|---|
| Fundamentals | LightGBM, Optuna-tuned | Ratings, form and Transfermarkt club squad value (top flights, 2012 on; values frozen at the June 2026 snapshot); used when no odds are available |
| Market residual | LightGBM boosted from the de-vigged market probabilities | Learns systematic corrections to bookmaker prices |
| Calibration | Temperature scaling on the 2023-24 season | Keeps probabilities honest (calibration error {prod['ece'] * 100:.2f}%) |
| Goals | Two Poisson LightGBMs with a Dixon-Coles adjustment (ρ = {mm['goals']['rho']:.3f}) | Scorelines, over/under 2.5, both teams to score, xG |

#### Evaluation protocol
Strictly chronological: no test match is used for training, tuning, early stopping or calibration.

* **Training**: {pd.Timestamp(mm['data']['train'][0]):%b %Y} – {pd.Timestamp(mm['data']['train'][1]) - pd.Timedelta(days=1):%b %Y} · {mm['data']['train'][2]:,} matches
* **Validation**: 2023-24 season · {mm['data']['valid'][2]:,} matches (tuning, early stopping, calibration)
* **Test**: {pd.Timestamp(mm['data']['test'][0]):%b %Y} – {pd.Timestamp(mm['data']['test'][1]):%b %Y} · {mm['data']['test'][2]:,} matches, never seen in training
* **Serving**: refit on all {mm['production']['n_train']:,} matches after evaluation

#### Results on the held-out test window
| Model | RPS ↓ | Log-loss ↓ | Brier ↓ | Accuracy ↑ |
|---|---|---|---|---|
""" + "\n".join(f"| {'**' if r['model'].startswith('SportPulse Market') else ''}{r['model']}"
                f"{'**' if r['model'].startswith('SportPulse Market') else ''} | {r['rps']:.4f} | "
                f"{r['log_loss']:.4f} | {r['brier']:.4f} | {r['accuracy'] * 100:.1f}% |"
                for r in mm["leaderboard"]) + """

#### Intended use & limitations
* Football outcomes are noisy. Even perfect information tops out around **50–55% 1X2 accuracy**. Betting markets
  aggregate a lot of information, and the bookmaker baseline here is a strong benchmark.
  Matching or beating it on a large out-of-sample set is the realistic state of the art.
* The betting simulation on best available prices is a sanity check, not financial advice
  (see Leaderboard → Betting backtest).
* Features cover team-level information only. Injuries, line-ups and transfers are reflected only through the
  market prior (when odds are supplied).
""")

    with t_eval:
        rows = pd.DataFrame(mm["leaderboard"])
        fig = go.Figure(go.Bar(x=rows["rps"], y=rows["model"], orientation="h",
                               marker_color=[ui.ORANGE if m.startswith("SportPulse") else "#3a3a3a"
                                             for m in rows["model"]],
                               text=rows["rps"].map("{:.4f}".format), textposition="outside"))
        fig.update_layout(title="Ranked probability score on test (lower is better)",
                          xaxis=dict(range=[rows.rps.min() - 0.004, rows.rps.max() + 0.003]),
                          yaxis=dict(autorange="reversed"), height=330)
        ui.plot(fig)
        cal = mm["calibration"]["production"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#444", dash="dash"),
                                 name="perfect"))
        for (k, lbl), col in zip({"H": "Home", "D": "Draw", "A": "Away"}.items(), [ui.WIN, ui.DRAW, ui.LOSS]):
            pts = np.array(cal[k])
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers", name=lbl,
                                     line=dict(color=col)))
        fig.update_layout(title="Reliability diagram (test)", xaxis_title="predicted probability",
                          yaxis_title="observed frequency", height=360)
        ui.plot(fig)

    with t_recent:
        tp = state.test_predictions().sort_values("date", ascending=False)
        dv = st.selectbox("League ", ["All"] + list(config.FEATURED_DIVISIONS),
                          format_func=lambda x: x if x == "All" else config.division_label(x))
        if dv != "All":
            tp = tp[tp.division == dv]
        show = tp.head(60).copy()
        show["Match"] = show["home"] + " – " + show["away"]
        show["Score"] = show["hg"].astype(str) + "–" + show["ag"].astype(str)
        show["Pick"] = np.array(["H", "D", "A"])[show[["p_h", "p_d", "p_a"]].to_numpy().argmax(1)]
        show["Correct"] = show["Pick"] == show["result"]
        st.dataframe(
            show[["date", "division", "Match", "Score", "p_h", "p_d", "p_a", "pm_h", "pm_d", "pm_a",
                  "Pick", "Correct", "xg_h", "xg_a"]],
            hide_index=True, width="stretch", height=520,
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "p_h": st.column_config.ProgressColumn("Model H", min_value=0, max_value=1, format="%.2f"),
                "p_d": st.column_config.ProgressColumn("Model D", min_value=0, max_value=1, format="%.2f"),
                "p_a": st.column_config.ProgressColumn("Model A", min_value=0, max_value=1, format="%.2f"),
                "pm_h": st.column_config.NumberColumn("Mkt H", format="%.2f"),
                "pm_d": st.column_config.NumberColumn("Mkt D", format="%.2f"),
                "pm_a": st.column_config.NumberColumn("Mkt A", format="%.2f"),
                "Correct": st.column_config.CheckboxColumn("Correct"),
                "xg_h": st.column_config.NumberColumn("xG H", format="%.2f"),
                "xg_a": st.column_config.NumberColumn("xG A", format="%.2f")})
        st.caption("Out-of-sample predictions: every row was predicted by the model trained before the test window.")

    with t_feat:
        imp = state.feature_importance().head(25)
        fig = go.Figure(go.Bar(x=imp["fundamentals"], y=[ui.readable_feature(f) for f in imp.index], orientation="h",
                               marker_color=ui.ORANGE, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_layout(title="What drives the no-odds model (share of total gain, top 25)", xaxis_tickformat=".0%",
                          yaxis=dict(autorange="reversed"), height=620)
        ui.plot(fig)
        ui.note("<b>Short form</b> covers roughly the last 8–10 matches, <b>long form</b> roughly the last 40–50. "
                "Pi-ratings rate each team separately at home and away, on a goal-difference scale.")
