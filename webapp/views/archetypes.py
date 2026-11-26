import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from webapp import state, ui

df = state.players()
pm = state.player_metrics()
A = pm["archetypes"]

ui.repo_header("sportpulse", "player-archetypes-gmm", "component",
               tag_items=[("task", "Clustering"), ("lib", "scikit-learn"), ("", "GaussianMixture"),
                          ("", f"k = {A['k']} (BIC)"), ("ds", "ea-fifa-players-15-22")])
ui.note("Style is separated from quality: each player's 29 detailed attributes are centred on their own mean, "
        "standardised, reduced with PCA(12), and clustered with a full-covariance Gaussian mixture. "
        "The number of clusters is chosen by BIC, and each archetype is named from the traits that set it apart from "
        "other players in the same positional line.")

out = df[df.position_group != "GK"]
t1, t2, t3 = st.tabs([":material/scatter_plot: Style map", ":material/badge: Archetype profiles", ":material/tune: Model selection"])

with t1:
    c1, c2 = st.columns([1, 3])
    min_ovr = c1.slider("Min overall", 45, 85, 70)
    hl = c1.text_input("Highlight player", "Messi")
    s = out[out.overall >= min_ovr]
    names = sorted(s.archetype.unique())
    fig = go.Figure()
    for i, n in enumerate(names):
        g = s[s.archetype == n]
        fig.add_trace(go.Scattergl(x=g.style_x, y=g.style_y, mode="markers", name=f"{n} ({len(g)})",
                                   marker=dict(size=5 + (g.overall - min_ovr) / 4, color=ui.SEQ[i % len(ui.SEQ)],
                                               opacity=.75, line=dict(width=0)),
                                   text=g.short_name + " · " + g.club_name.fillna("") + " · " + g.overall.astype(str),
                                   hovertemplate="%{text}<extra>" + n + "</extra>"))
    if hl:
        h = s[s.short_name.str.contains(hl, case=False, na=False)].head(5)
        fig.add_trace(go.Scatter(x=h.style_x, y=h.style_y, mode="markers+text", text=h.short_name,
                                 textposition="top center", name="highlight",
                                 marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#fff", width=2))))
    fig.update_layout(title="Playing-style map (first two principal components of quality-normalised attributes)",
                      height=640, xaxis_title="style PC1", yaxis_title="style PC2",
                      legend=dict(orientation="v", x=1.01, y=1))
    with c2:
        ui.plot(fig)

with t2:
    prof = A["profiles"]
    order = sorted(prof, key=lambda n: -prof[n]["size"])
    for row in range(0, len(order), 3):
        cols = st.columns(3)
        for c, n in zip(cols, order[row:row + 3]):
            p = prof[n]
            top = out[out.archetype == n].nlargest(5, "overall")
            dist = ", ".join(f"{k.replace('_', ' ')} {v:+.2f}" for k, v in p["distinguishing"].items())
            lines = " · ".join(f"{k} {v * 100:.0f}%" for k, v in list(p["line_mix"].items())[:3])
            with c:
                st.markdown(ui.card(n, org="archetype", icon="component", kind="sp",
                                    desc=f"Distinct traits: {dist}",
                                    meta=[f"{p['size']:,} players", lines]), unsafe_allow_html=True)
                st.caption("Top: " + ", ".join(f"{r.short_name} ({r.overall})" for r in top.itertuples()))
    traits = pd.DataFrame({n: prof[n]["traits"] for n in order}).T
    fig = go.Figure(go.Heatmap(z=traits.values, x=[c.replace("_", " ") for c in traits.columns], y=traits.index,
                               colorscale=[[0, "#1e3a8a"], [0.5, "#111111"], [1, ui.ORANGE]], zmid=0,
                               text=traits.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Centroid trait z-scores (style, quality removed)", height=40 * len(traits) + 120)
    ui.plot(fig)

with t3:
    bic = pd.Series({int(k): v for k, v in A["bic"].items()}).sort_index()
    fig = go.Figure(go.Scatter(x=bic.index, y=bic.values, mode="lines+markers", line=dict(color=ui.ORANGE)))
    fig.add_vline(x=A["k"], line_dash="dash", line_color="#888")
    fig.update_layout(title="Bayesian information criterion by number of components (lower is better)",
                      xaxis_title="k", yaxis_title="BIC", height=380)
    ui.plot(fig)
