import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from webapp import pitch, state, ui
from webapp.icons import svg

comps = state.competition_order()
mt = state.analyst("matches")
shots = state.analyst("shots")
tm = state.analyst("team_match")
am = state.analyst_metrics()

c1, c2 = st.columns([1, 2])
comp = c1.selectbox("Competition", comps, index=comps.index("FIFA World Cup 2022") if "FIFA World Cup 2022" in comps else 0)
M = mt[mt.label == comp].sort_values("date", ascending=False)
labels = {r.match_id: f"{r.date:%d %b %Y} · {r.home} {r.home_score}–{r.away_score} {r.away}"
          + (f" · {r.stage}" if r.stage and r.stage != "Regular Season" else "") for r in M.itertuples()}
mid = c2.selectbox("Match", list(labels), format_func=labels.get)
m = M.set_index("match_id").loc[mid]
S = shots[shots.match_id == mid].sort_values(["period", "minute", "second"]).copy()
# Monotone match clock: stoppage time of one period never overlaps the next.
S["t"] = (S.minute + S.second / 60).cummax()
H, A = m.home, m.away

st.markdown(f"""
<div class="hf-card" style="padding:18px 22px;margin:4px 0 12px 0">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap">
    <div style="flex:1;text-align:right"><div style="font-size:1.35rem;font-weight:700">{ui.esc(H)}</div>
      <div class="hf-mono">xG {m.home_xg:.2f} · {ui.esc(m.home_formation or '')}</div></div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:2.1rem;font-weight:600;padding:0 18px;color:{ui.ORANGE}">
      {int(m.home_score)} – {int(m.away_score)}</div>
    <div style="flex:1"><div style="font-size:1.35rem;font-weight:700">{ui.esc(A)}</div>
      <div class="hf-mono">xG {m.away_xg:.2f} · {ui.esc(m.away_formation or '')}</div></div>
  </div>
  <div class="hf-card-meta" style="justify-content:center;margin-top:10px">
    {svg('calendar', 13)} {m.date:%d %B %Y} <span class="hf-dot">•</span> {svg('trophy', 13)} {ui.esc(comp)}
    {('<span class="hf-dot">•</span> ' + ui.esc(m.stage)) if m.stage else ''}
    {('<span class="hf-dot">•</span> ' + svg('map', 13) + ' ' + ui.esc(m.stadium)) if m.stadium else ''}
  </div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs([":material/show_chart: xG race", ":material/sports_soccer: Shot map", ":material/hub: Pass networks",
                ":material/compare_arrows: Team stats", ":material/model_training: xG model"])

with tabs[0]:
    fig = go.Figure()
    for team, color in ((H, ui.ORANGE), (A, "#38bdf8")):
        s = S[S.team == team]
        t = s.t.tolist()
        cum = s.xg.cumsum().tolist()
        end = max(95, (S.minute.max() if len(S) else 90) + 2)
        xs, ys = [0] + t + [end], [0] + cum + [cum[-1] if cum else 0]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(shape="hv", width=2.6, color=color),
                                 name=f"{team} · {s.xg.sum():.2f} xG"))
        g = s[s.goal == 1]
        gi = [t.index(v) for v in g.t]
        fig.add_trace(go.Scatter(x=[t[i] for i in gi], y=[cum[i] for i in gi], mode="markers+text", showlegend=False,
                                 marker=dict(size=12, color=color, line=dict(color="#0a0a0a", width=2)),
                                 text=[p.split()[-1] for p in g.player], textposition="top center",
                                 textfont=dict(size=11, color=color),
                                 hovertemplate="%{text} %{x:.0f}′<extra>goal</extra>"))
    fig.update_layout(title="Cumulative expected goals", xaxis_title="minute", yaxis_title="xG", height=420)
    ui.plot(fig)

with tabs[1]:
    fig = pitch.pitch(height=500, title=f"{H} attacking right · {A} attacking left")
    hs = S[S.team == H]
    aw = S[S.team == A].assign(x=lambda d: 120 - d.x, y=lambda d: 80 - d.y)
    pitch.shot_map(hs, fig=fig, color=ui.ORANGE, name=H, half=False)
    pitch.shot_map(aw, fig=fig, color="#38bdf8", name=A, half=False)
    ui.plot(fig)
    top = S.sort_values("xg", ascending=False).head(10).copy()
    top["play_pattern"] = np.where(top.shot_type == "Penalty", "Penalty", top.play_pattern)
    st.dataframe(top[["minute", "team", "player", "xg", "sb_xg", "body_part", "play_pattern", "outcome"]],
                 hide_index=True, use_container_width=True,
                 column_config={"minute": st.column_config.NumberColumn("Min", format="%d′"), "team": "Team",
                                "player": "Player",
                                "xg": st.column_config.ProgressColumn("SportPulse xG", min_value=0, max_value=1, format="%.2f"),
                                "sb_xg": st.column_config.NumberColumn("StatsBomb xG", format="%.2f"),
                                "body_part": "Body part", "play_pattern": "Phase", "outcome": "Outcome"})

with tabs[2]:
    nodes, edges = state.analyst("net_nodes"), state.analyst("net_edges")
    a, b = st.columns(2, gap="large")
    for col, team, color in ((a, H, ui.ORANGE), (b, A, "#38bdf8")):
        with col:
            n = nodes[(nodes.match_id == mid) & (nodes.team == team)]
            e = edges[(edges.match_id == mid) & (edges.team == team)]
            ui.plot(pitch.pass_network(n, e, f"{team} · starters' average positions", color))
    st.caption("Nodes sit at each starter's average pass location (size = passes made); links join pairs with 3+ "
               "completed passes (width = volume).")

with tabs[3]:
    rows = tm[tm.match_id == mid].set_index("team")
    if H in rows.index and A in rows.index:
        stats = [("xG", "xg", "{:.2f}"), ("Shots", "shots", "{:.0f}"), ("Possession", "possession", "{:.0%}"),
                 ("Field tilt", "field_tilt", "{:.0%}"), ("Passes", "passes", "{:.0f}"),
                 ("Pass completion", "pass_completion", "{:.0%}"), ("Progressive passes", "prog_passes", "{:.0f}"),
                 ("Progressive carries", "prog_carries", "{:.0f}"), ("Expected threat", "xt", "{:.2f}"),
                 ("Pressures", "pressures", "{:.0f}"), ("High regains", "high_regains", "{:.0f}"),
                 ("PPDA (lower = more pressing)", "ppda", "{:.1f}"), ("Set-piece xG", "sp_xg", "{:.2f}")]
        html = ['<div class="hf-cmp">']
        for lbl, k, f in stats:
            h, a_ = float(rows.at[H, k] or 0), float(rows.at[A, k] or 0)
            tot = (h + a_) or 1
            html.append(f'<div class="row"><div class="v l">{f.format(h)}</div><div class="bar"><div class="hb" '
                        f'style="width:{h / tot * 100:.1f}%"></div><div class="ab" style="width:{a_ / tot * 100:.1f}%">'
                        f'</div></div><div class="v r">{f.format(a_)}</div><div class="lbl">{ui.esc(lbl)}</div></div>')
        html.append("</div>")
        st.markdown(f"""<style>
        .hf-cmp .row {{ display:grid; grid-template-columns: 70px 1fr 70px; gap:10px; align-items:center; margin:12px 0 2px 0; }}
        .hf-cmp .lbl {{ grid-column: 1 / span 3; text-align:center; color:{ui.MUTED}; font-size:.76rem; margin-top:-2px; }}
        .hf-cmp .v {{ font-family:'IBM Plex Mono',monospace; font-size:.9rem; }} .hf-cmp .l {{ text-align:right; }}
        .hf-cmp .bar {{ display:flex; height:8px; border-radius:4px; overflow:hidden; background:{ui.CARD_2}; }}
        .hf-cmp .hb {{ background:{ui.ORANGE}; }} .hf-cmp .ab {{ background:#38bdf8; }}
        </style>""" + "".join(html), unsafe_allow_html=True)
        st.markdown(ui.tags([("task", H), ("lib", A)]), unsafe_allow_html=True)

with tabs[4]:
    x = am["xg"]
    cv = pd.DataFrame(x["non_penalty_cv"]).T
    tt = {k: v for k, v in x.get("temporal_recent_test", {}).items() if isinstance(v, dict)}
    st.markdown(f"""
#### Expected-goals model

LightGBM on **{x['non_penalty_shots']:,} non-penalty shots** from **{x['matches']:,} matches** (StatsBomb open data).
Monotone constraints make xG non-increasing in distance and non-decreasing in goal-mouth angle. Features cover
shot geometry, body part and technique, phase of play, the assisting pass, and **freeze-frame geometry**: defenders
and teammates in the shooting cone, the goalkeeper's position, nearest-defender distance and pressure count.
Penalties use their empirical conversion rate ({x['penalty_rate'] * 100:.1f}%).
""")
    ui.section("Out-of-fold, grouped by match", icon="layers")
    st.dataframe(cv[["log_loss", "brier", "auc", "ece", "xg_sum", "goals", "n"]], use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(lbl, format="%.4f") for c, lbl in
                                {"log_loss": "Log-loss ↓", "brier": "Brier ↓", "auc": "AUC ↑", "ece": "ECE ↓"}.items()}
                 | {"xg_sum": st.column_config.NumberColumn("Σ xG", format="%.0f"),
                    "goals": st.column_config.NumberColumn("Goals", format="%.0f"),
                    "n": st.column_config.NumberColumn("Shots", format="%d")})
    if tt:
        ui.section("Temporal test: trained on ≤2021, scored on 2022–24 tournaments", icon="clock")
        st.dataframe(pd.DataFrame(tt).T[["log_loss", "brier", "auc", "ece", "xg_sum", "goals", "n"]],
                     use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(lbl, format="%.4f") for c, lbl in
                                    {"log_loss": "Log-loss ↓", "brier": "Brier ↓", "auc": "AUC ↑", "ece": "ECE ↓"}.items()}
                     | {"xg_sum": st.column_config.NumberColumn("Σ xG", format="%.0f"),
                        "goals": st.column_config.NumberColumn("Goals", format="%.0f"),
                        "n": st.column_config.NumberColumn("Shots", format="%d")})
    a, b = st.columns(2, gap="large")
    with a:
        cal = np.array(x["calibration"])
        fig = go.Figure([go.Scatter(x=[0, .8], y=[0, .8], mode="lines", line=dict(color="#444", dash="dash"), name="ideal"),
                         go.Scatter(x=cal[:, 0], y=cal[:, 1], mode="lines+markers", line=dict(color=ui.ORANGE),
                                    name="SportPulse xG")])
        fig.update_layout(title="Calibration (out-of-fold)", xaxis_title="predicted xG", yaxis_title="goal rate", height=360)
        ui.plot(fig)
    with b:
        imp = pd.Series(x["importance"]).head(12)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index.str.replace("ff_", "frame: ").str.replace("_", " "),
                               orientation="h", marker_color=ui.ORANGE))
        fig.update_layout(title="Feature importance (gain share)", height=360)
        ui.plot(fig)
    ui.note("StatsBomb's commercial xG is trained on far more proprietary data, which makes it a demanding benchmark. "
            "In cross-validation the two models are within 0.001 log-loss; on the unseen 2022–24 tournaments SportPulse "
            "xG is slightly better and better calibrated.")
