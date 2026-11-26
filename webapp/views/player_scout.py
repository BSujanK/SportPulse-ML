import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportpulse.models.players import similar_players
from webapp import state, ui

df = state.players()
pm = state.player_metrics()
ed = int(df.edition.max())

ui.repo_header("sportpulse", "player-scout", "🔎", badge="● live",
               tag_items=[("task", "Tabular Regression"), ("task", "Quantile Regression"),
                          ("task", "Similarity Search"), ("lib", "LightGBM"),
                          ("ds", "ea-fifa-players-15-22"), ("", f"snapshot: FIFA {ed % 100}")])

t_scout, t_under, t_rise, t_card = st.tabs(["🔎 Scout a player", "💎 Undervalued finder",
                                            "🚀 Rising stars", "📄 Model card"])

FACE = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

with t_scout:
    default = int(df.index[df.short_name.str.contains("Pedri", na=False)][0]) \
        if df.short_name.str.contains("Pedri", na=False).any() else 0
    idx = st.selectbox("Search player", df.index, index=int(default),
                       format_func=lambda i: df.at[i, "label"])
    p = df.loc[idx]
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            a, b = st.columns([1, 3])
            if isinstance(p.player_face_url, str):
                a.image(p.player_face_url, width=110)
            b.markdown(f"### {ui.esc(p.short_name)}")
            b.markdown(ui.tags([("task", p.archetype), ("", p.player_positions), ("", p.club_name or "—"),
                                ("", p.league_name or "—"), ("", p.nationality_name),
                                ("", f"age {int(p.age)}")]), unsafe_allow_html=True)
            gap = p.value_gap
            ui.kpi_row([
                ui.kpi("Overall", f"{int(p.overall)}", f"/ pot {int(p.potential)}"),
                ui.kpi("Next season", f"{p.proj_overall:.1f}", "",
                       f"80% PI {p.proj_low:.0f}–{p.proj_high:.0f}",
                       "good" if p.proj_overall > p.overall else "bad" if p.proj_overall < p.overall else ""),
                ui.kpi("Listed value", ui.fmt_eur(p.value_eur)),
                ui.kpi("Model fair value", ui.fmt_eur(p.fair_value_eur), "",
                       "—" if pd.isna(gap) else f"{gap * 100:+.0f}% vs listed",
                       "good" if (gap == gap and gap > 0.1) else "bad" if (gap == gap and gap < -0.1) else ""),
            ])
        hist = state.player_history()
        h = hist[hist.player_id == p.player_id].sort_values("edition")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.edition, y=h.overall, name="overall", mode="lines+markers",
                                 line=dict(color=ui.ORANGE, width=3)))
        fig.add_trace(go.Scatter(x=h.edition, y=h.potential, name="potential", mode="lines",
                                 line=dict(color="#666", dash="dot")))
        fig.add_trace(go.Scatter(x=[ed, ed + 1], y=[p.overall, p.proj_overall], name="forecast",
                                 mode="lines+markers", line=dict(color=ui.ORANGE_3, dash="dash")))
        fig.add_trace(go.Scatter(x=[ed + 1, ed + 1], y=[p.proj_low, p.proj_high], name="80% interval",
                                 mode="lines", line=dict(color=ui.ORANGE_3, width=8), opacity=.35))
        fig.update_layout(title="Rating trajectory + one-season forecast", height=320,
                          xaxis=dict(dtick=1), yaxis_title="rating")
        ui.plot(fig)

    with right:
        with st.container(border=True):
            st.markdown(ui.widget_head("Similar players", "cosine · 40-dim attribute space"),
                        unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            same_pos = f1.toggle("Same position group", True)
            cheaper = f2.toggle("Cheaper only", False)
            max_age = st.slider("Max age", 16, 42, 42)
            emb = state.player_embedding()
            mask = (df.age <= max_age).to_numpy()
            if same_pos:
                mask &= (df.position_group == p.position_group).to_numpy()
            if cheaper and p.value_eur == p.value_eur:
                mask &= (df.value_eur < p.value_eur).to_numpy()
            order, sims = similar_players(emb, df.index.get_loc(idx), top=10, mask=mask)
            sim = df.iloc[order][["short_name", "age", "club_name", "overall", "value_eur", "archetype"]].copy()
            sim.insert(0, "sim", sims)
            st.dataframe(sim, hide_index=True, use_container_width=True, height=390,
                         column_config={
                             "sim": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f"),
                             "short_name": "Player", "club_name": "Club", "overall": "OVR",
                             "value_eur": st.column_config.NumberColumn("Value €", format="compact")})
        cats = FACE
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[p[c] for c in cats] + [p[cats[0]]], theta=cats + [cats[0]],
                                      fill="toself", name=p.short_name, line=dict(color=ui.ORANGE)))
        if len(order):
            q = df.iloc[order[0]]
            fig.add_trace(go.Scatterpolar(r=[q[c] for c in cats] + [q[cats[0]]], theta=cats + [cats[0]],
                                          fill="toself", name=q.short_name, line=dict(color="#38bdf8"),
                                          opacity=.6))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="#222"),
                                     angularaxis=dict(gridcolor="#222")), height=330,
                          title="Attribute profile vs closest match")
        ui.plot(fig)

with t_under:
    c1, c2, c3, c4 = st.columns(4)
    pos = c1.multiselect("Position group", ["GK", "DEF", "MID", "FWD"], ["DEF", "MID", "FWD"])
    age = c2.slider("Age", 16, 40, (18, 27))
    ovr = c3.slider("Min overall", 50, 90, 72)
    budget = c4.select_slider("Max listed value", [1e6, 2e6, 5e6, 10e6, 20e6, 40e6, 80e6, 200e6], 40e6,
                              format_func=ui.fmt_eur)
    u = df[df.position_group.isin(pos) & df.age.between(*age) & (df.overall >= ovr) &
           (df.value_eur <= budget) & df.value_gap.notna()]
    u = u.sort_values("value_gap", ascending=False).head(60)
    st.dataframe(u[["short_name", "age", "club_name", "league_name", "player_positions", "overall",
                    "proj_overall", "value_eur", "fair_value_eur", "value_gap", "archetype"]],
                 hide_index=True, use_container_width=True, height=560,
                 column_config={"short_name": "Player", "club_name": "Club", "league_name": "League",
                                "player_positions": "Pos", "overall": "OVR",
                                "proj_overall": st.column_config.NumberColumn("OVR next", format="%.1f"),
                                "value_eur": st.column_config.NumberColumn("Listed €", format="compact"),
                                "fair_value_eur": st.column_config.NumberColumn("Fair €", format="compact"),
                                "value_gap": st.column_config.NumberColumn("Upside", format="percent")})
    ui.note("Fair value is the out-of-fold prediction of the market-value model: the model never saw this "
            "player in any season. Upside = fair / listed − 1.")

with t_rise:
    c1, c2 = st.columns(2)
    max_age = c1.slider("Max age ", 16, 25, 21)
    min_ovr = c2.slider("Min current overall", 45, 80, 62)
    r = df[(df.age <= max_age) & (df.overall >= min_ovr)].copy()
    r["growth"] = r.proj_overall - r.overall
    r = r.sort_values(["growth", "proj_overall"], ascending=False).head(50)
    fig = go.Figure(go.Bar(x=r.head(20).growth, y=r.head(20).short_name, orientation="h",
                           marker_color=ui.ORANGE, error_x=dict(
                               type="data", symmetric=False, array=(r.head(20).proj_high - r.head(20).proj_overall),
                               arrayminus=(r.head(20).proj_overall - r.head(20).proj_low), color="#666")))
    fig.update_layout(title="Largest projected one-season rating gains (with 80% interval)",
                      yaxis=dict(autorange="reversed"), height=560)
    ui.plot(fig)

with t_card:
    v, dv = pm["value"], pm["development"]
    st.markdown(f"""
### Player models

**Market value** · LightGBM regression on log(€), 60+ features (attributes, age, potential, contract,
reputation, league, position). Evaluated with 5-fold **GroupKFold by player**: a player's every season is held
out together, so the model is always valuing someone it has never seen.

| | R² (log €) | MAE (log) | Median abs. error | within ±10% | within ±25% |
|---|---|---|---|---|---|
| **LightGBM (FIFA {ed % 100} rows)** | **{v['model']['r2']:.4f}** | {v['model']['mae']:.3f} | {v['model']['median_ape'] * 100:.1f}% | {v['model']['within_10pct'] * 100:.0f}% | {v['model']['within_25pct'] * 100:.0f}% |
| Linear baseline (OVR, POT, age + edition) | {v['baseline_linear_log']['r2']:.4f} | {v['baseline_linear_log']['mae']:.3f} | {v['baseline_linear_log']['median_ape'] * 100:.1f}% | {v['baseline_linear_log']['within_10pct'] * 100:.0f}% | {v['baseline_linear_log']['within_25pct'] * 100:.0f}% |
| Stress test: forward edition (train ≤ FIFA {ed % 100 - 2}) | {v['stress_forward_edition']['r2']:.4f} | {v['stress_forward_edition']['mae']:.3f} | {v['stress_forward_edition']['median_ape'] * 100:.1f}% | {v['stress_forward_edition']['within_10pct'] * 100:.0f}% | {v['stress_forward_edition']['within_25pct'] * 100:.0f}% |

The forward-edition stress test is harder because EA re-prices the market every edition
(a 75-rated 25-year-old: €8.0M in FIFA 19/20 → €7.0M in FIFA 21/22).

**Development** · LightGBM (L1) + quantile LightGBM (q10/q50/q90) predicting next-edition overall.
Temporal test: trained on 15→16 … 19→20 pairs, scored on **21→22** ({dv['model']['n']:,} players).

| | MAE | RMSE | R² |
|---|---|---|---|
| **LightGBM** | **{dv['model']['mae']:.3f}** | {dv['model']['rmse']:.3f} | {dv['model']['r2']:.3f} |
| Potential-pull linear baseline | {dv['baseline_potential_pull']['mae']:.3f} | {dv['baseline_potential_pull']['rmse']:.3f} | {dv['baseline_potential_pull']['r2']:.3f} |
| Persistence (no change) | {dv['baseline_persistence']['mae']:.3f} | {dv['baseline_persistence']['rmse']:.3f} | {dv['baseline_persistence']['r2']:.3f} |
| **LightGBM, U23 only** | **{dv['model_u23']['mae']:.3f}** | {dv['model_u23']['rmse']:.3f} | {dv['model_u23']['r2']:.3f} |
| Persistence, U23 only | {dv['baseline_persistence_u23']['mae']:.3f} | {dv['baseline_persistence_u23']['rmse']:.3f} | {dv['baseline_persistence_u23']['r2']:.3f} |

80% prediction-interval empirical coverage: **{dv['model']['interval_80_coverage'] * 100:.1f}%**.

**Data recency.** EA FIFA 22 (2021-22) is the newest player edition available in the public sources this
pipeline uses. Projections are one season ahead of that snapshot. Drop a newer sofifa export with the same schema
into `data/raw/` and re-run `python -m sportpulse.train players` to refresh.
""")
