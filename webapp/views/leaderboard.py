import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportpulse import config
from webapp import state, ui

mm = state.match_metrics()
d = mm["data"]
ui.repo_header("sportpulse", "football-forecasting-leaderboard", "trophy", badge="● running",
               tag_items=[("", "Space"), ("task", "Benchmark"), ("ds", "club-matches-2000-2026"),
                          ("", f"test window {d['test'][0]} → {d['test'][1]}"),
                          ("", f"{d['test_with_odds']:,} matches")])
st.markdown(f'<div class="hf-mono">Evaluated on {d["test_with_odds"]:,} out-of-sample matches across '
            f'{d["divisions"]} leagues. Lower RPS / log-loss / Brier is better.</div>', unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs([":material/leaderboard: 1X2 leaderboard", ":material/flag: By league", ":material/show_chart: Over time", ":material/account_balance: Betting backtest & O/U"])

with t1:
    df = pd.DataFrame(mm["leaderboard"])
    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Type"] = df["model"].map(lambda m: "SportPulse" if m.startswith("SportPulse") or m.startswith("Poisson")
                                 else ("Market" if "market" in m.lower() else "Baseline"))
    df["accuracy"] *= 100
    df["ece"] *= 100
    st.dataframe(
        df[["Rank", "Type", "model", "rps", "log_loss", "brier", "accuracy", "ece", "n"]],
        hide_index=True, width="stretch",
        column_config={
            "model": st.column_config.TextColumn("Model", width="large"),
            "rps": st.column_config.NumberColumn("RPS ↓", format="%.4f"),
            "log_loss": st.column_config.NumberColumn("Log-loss ↓", format="%.4f"),
            "brier": st.column_config.NumberColumn("Brier ↓", format="%.4f"),
            "accuracy": st.column_config.ProgressColumn("Accuracy %", min_value=40, max_value=55, format="%.1f"),
            "ece": st.column_config.NumberColumn("ECE %", format="%.2f"),
            "n": st.column_config.NumberColumn("Matches", format="%d")})
    ui.note("<b>RPS</b> (ranked probability score) is the standard proper scoring rule for ordered football "
            "outcomes. For reference, the winning entry of the 2017 Soccer Prediction Challenge "
            "(no betting data) scored ≈0.206 on its own test set. <b>Accuracy</b> alone is a weak metric: "
            "always picking the favourite already gets ~50%.")
    bench_path = config.MATCH_ARTIFACTS / "benchmark_dixon_coles.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        ui.section("Published models, same matches", count=f"{bench['n_covered']:,}", icon="scale")
        src = {"SportPulse Market-Residual (production)": "this repo",
               "Bookmaker market (Shin)": "Štrumbelj 2014",
               "SportPulse Fundamentals (no odds)": "this repo",
               "Dixon & Coles (1997), monthly refit": "Dixon & Coles 1997"}
        pv = bench["paired_vs_dixon_coles"]
        bdf = pd.DataFrame([{"Model": k, "Source": src.get(k, ""), "RPS ↓": v["rps"], "Log-loss ↓": v["log_loss"],
                             "Δ RPS vs Dixon-Coles": f"{pv[k]['diff']:+.4f}" if k in pv else "reference",
                             "z": f"{pv[k]['z']:.1f}" if k in pv else "reference"}
                            for k, v in bench["models"].items()]).sort_values("RPS ↓")
        st.dataframe(bdf, hide_index=True, width="stretch",
                     column_config={"Model": st.column_config.TextColumn(width="large"),
                                    "RPS ↓": st.column_config.NumberColumn(format="%.4f"),
                                    "Log-loss ↓": st.column_config.NumberColumn(format="%.4f"),
                                    "z": st.column_config.TextColumn("Paired z")})
        pm = bench["paired_production_vs_market"]
        ui.note(f"Dixon-Coles is refit monthly per league on the previous three seasons (time decay ξ = {bench['xi']} "
                f"per day, chosen on 2023-24). Paired per-match test, production vs market: "
                f"ΔRPS {pm['diff']:+.5f}, z = {pm['z']:.1f}. That is a consistent edge, not a statistically "
                "significant one.")
    fp = mm["model"]["fund_params"]
    ui.section("Production configuration", icon="layers")
    st.markdown(ui.kv([("Fundamentals booster", f"{mm['model']['fund_iters']} trees · {fp['num_leaves']} leaves · "
                                                f"lr {fp['learning_rate']}"),
                       ("Regularisation", f"L1 {fp.get('lambda_l1', 0):.2f} · L2 {fp['lambda_l2']:.1f} · "
                                          f"min leaf {fp['min_data_in_leaf']}"),
                       ("Market-residual booster", f"{mm['model']['resid_iters']} trees"),
                       ("Temperature (fund. / final)", f"{mm['model']['t_fund']:.3f} / {mm['model']['t_final']:.3f}"),
                       ("Hyper-parameter search", "Optuna TPE on 2023-24 log-loss during development; fixed defaults since"
                                                  if mm.get("tuning") is None else f"Optuna TPE, {mm['tuning']['trials']} trials, 2023-24 log-loss")]), unsafe_allow_html=True)

with t2:
    pdv = pd.DataFrame(mm["per_division"])
    pdv["Δ RPS vs market"] = pdv["rps_market"] - pdv["rps_model"]
    pdv = pdv.sort_values("Δ RPS vs market", ascending=False)
    fig = go.Figure(go.Bar(x=pdv["Δ RPS vs market"] * 1000, y=pdv["league"], orientation="h",
                           marker_color=[ui.WIN if v > 0 else ui.LOSS for v in pdv["Δ RPS vs market"]]))
    fig.update_layout(title="RPS improvement over bookmaker, ×1000 (positive = model better)",
                      yaxis=dict(autorange="reversed"), height=max(380, 22 * len(pdv)))
    ui.plot(fig)
    st.dataframe(pdv[["league", "n", "rps_model", "rps_market", "rps_fund", "acc_model", "acc_market"]],
                 hide_index=True, width="stretch",
                 column_config={"league": "League", "n": "Matches",
                                "rps_model": st.column_config.NumberColumn("RPS model", format="%.4f"),
                                "rps_market": st.column_config.NumberColumn("RPS market", format="%.4f"),
                                "rps_fund": st.column_config.NumberColumn("RPS fundamentals", format="%.4f"),
                                "acc_model": st.column_config.NumberColumn("Acc model", format="%.3f"),
                                "acc_market": st.column_config.NumberColumn("Acc market", format="%.3f")})

with t3:
    mo = pd.DataFrame(mm["monthly"]).sort_values("month")
    mo = mo[mo.n >= 100]
    fig = go.Figure()
    for col, name, color in [("rps_fund", "Fundamentals (no odds)", ui.ORANGE_3),
                             ("rps_market", "Bookmaker", "#64748b"),
                             ("rps_model", "Production", ui.ORANGE)]:
        fig.add_trace(go.Scatter(x=mo["month"], y=mo[col], name=name, mode="lines+markers",
                                 line=dict(color=color, width=3 if col == "rps_model" else 1.6)))
    fig.update_layout(title="Monthly RPS across the test window", yaxis_title="RPS", height=420)
    ui.plot(fig)
    st.caption("The model was frozen before the first month shown. No retraining inside the window.")

with t4:
    c1, c2 = st.columns(2)
    with c1:
        ui.section("Value-betting simulation")
        bt = pd.DataFrame(mm["betting_backtest"])
        bt["roi"] *= 100
        bt["roi_se"] *= 100
        st.dataframe(bt[["prices", "model", "edge", "bets", "profit_units", "roi", "roi_se", "z"]],
                     hide_index=True, width="stretch",
                     column_config={"edge": st.column_config.NumberColumn("EV edge >", format="percent"),
                                    "roi": st.column_config.NumberColumn("ROI %", format="%+.2f"),
                                    "roi_se": st.column_config.NumberColumn("± s.e. %", format="%.2f"),
                                    "z": st.column_config.NumberColumn("z", format="%+.2f"),
                                    "profit_units": st.column_config.NumberColumn("Profit (units)", format="%+.1f")})
        ui.note("Flat 1-unit stakes whenever expected value exceeds the edge threshold. The <b>market control</b> "
                "bets the de-vigged market probabilities themselves at best prices and loses, so the model's "
                "small positive ROI is not just line-shopping. With |z| &lt; 2, though, <b>no edge is statistically "
                "significant</b>: the honest conclusion is that the model matches, and marginally improves on, an efficient "
                "market. Research only, not betting advice.")
    with c2:
        ui.section("Over / Under 2.5 goals")
        ou = mm["over_under_25"]
        ui.kpi_row([ui.kpi("Model log-loss", f"{ou['logloss_model']:.4f}"),
                    ui.kpi("Market log-loss", f"{ou['logloss_market']:.4f}"),
                    ui.kpi("Base-rate log-loss", f"{ou['logloss_base']:.4f}")])
        g = mm["goals"]
        st.markdown(f"<div class='hf-mono'>n = {ou['n']:,} · accuracy model {ou['acc_model'] * 100:.1f}% vs "
                    f"market {ou['acc_market'] * 100:.1f}% · goal MAE home {g['mae_home']:.3f} / "
                    f"away {g['mae_away']:.3f} · Dixon-Coles ρ = {g['rho']:.3f}</div>", unsafe_allow_html=True)
