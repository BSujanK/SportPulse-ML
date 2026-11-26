"""Plotly football pitch (StatsBomb 120 x 80 coordinates) and pitch visualisations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from webapp import ui

LINE = "#3a3a3a"
TURF = "#0f0f0f"


def pitch(height: int = 460, half: bool = False, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    L = dict(color=LINE, width=1.4)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=L, fillcolor=TURF, layer="below"),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=L),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=L),
        dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=L), dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=L),
        dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=L), dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=L),
        dict(type="rect", x0=-2, y0=36, x1=0, y1=44, line=L), dict(type="rect", x0=120, y0=36, x1=122, y1=44, line=L),
    ]
    for cx in (12, 108):
        shapes.append(dict(type="circle", x0=cx - .4, y0=39.6, x1=cx + .4, y1=40.4, line=L, fillcolor=LINE))
    shapes.append(dict(type="circle", x0=59.6, y0=39.6, x1=60.4, y1=40.4, line=L, fillcolor=LINE))
    # penalty arcs (outside the box)
    t = np.linspace(-0.93, 0.93, 30)
    for cx, sgn in ((12, 1), (108, -1)):
        xs, ys = cx + sgn * 10 * np.cos(t), 40 + 10 * np.sin(t)
        keep = xs >= 18 if sgn == 1 else xs <= 102
        fig.add_trace(go.Scatter(x=xs[keep], y=ys[keep], mode="lines", line=L, hoverinfo="skip", showlegend=False))
    fig.update_layout(shapes=shapes, height=height, plot_bgcolor=TURF, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=4, r=4, t=36 if title else 6, b=4), title=title,
                      xaxis=dict(range=[-3, 123], visible=False, fixedrange=True),
                      yaxis=dict(range=[82, -2], visible=False, scaleanchor="x", fixedrange=True),
                      legend=dict(orientation="h", y=-0.02, x=0))
    return fig


def half_pitch(height: int = 470, title: str | None = None) -> go.Figure:
    """Vertical attacking half (goal at the top): horizontal axis = pitch y, vertical axis = pitch x."""
    fig = go.Figure()
    L = dict(color=LINE, width=1.4)
    shapes = [dict(type="rect", x0=0, y0=60, x1=80, y1=120, line=L, fillcolor=TURF, layer="below"),
              dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=L),
              dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=L),
              dict(type="rect", x0=36, y0=120, x1=44, y1=122, line=L),
              dict(type="circle", x0=39.6, y0=107.6, x1=40.4, y1=108.4, line=L, fillcolor=LINE)]
    t = np.linspace(-np.pi, np.pi, 120)
    u, v = 40 + 10 * np.sin(t), 108 - 10 * np.cos(t)
    keep = v <= 102
    fig.add_trace(go.Scatter(x=u[keep], y=v[keep], mode="lines", line=L, hoverinfo="skip", showlegend=False))
    t2 = np.linspace(0, np.pi, 60)
    fig.add_trace(go.Scatter(x=40 + 10 * np.cos(t2), y=60 + 10 * np.sin(t2), mode="lines", line=L,
                             hoverinfo="skip", showlegend=False))
    fig.update_layout(shapes=shapes, height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=4, r=4, t=36 if title else 6, b=4), title=title,
                      xaxis=dict(range=[-2, 82], visible=False, fixedrange=True),
                      yaxis=dict(range=[58, 123], visible=False, scaleanchor="x", fixedrange=True),
                      legend=dict(orientation="h", y=-0.02, x=0))
    return fig


def shot_map(shots: pd.DataFrame, title: str = "", color: str = ui.ORANGE, half: bool = True,
             fig: go.Figure | None = None, name: str = "Shots") -> go.Figure:
    fig = fig or (half_pitch(title=title) if half else pitch(title=title, height=460))
    goal = shots.goal == 1
    for mask, sym, nm, op in ((~goal, "circle-open", f"{name}: no goal", .75), (goal, "circle", f"{name}: goal", .95)):
        s = shots[mask]
        fig.add_trace(go.Scatter(
            x=s.y if half else s.x, y=s.x if half else s.y, mode="markers", name=nm,
            marker=dict(size=6 + 34 * np.sqrt(s.xg.clip(0, 1)), color=color, symbol=sym, opacity=op,
                        line=dict(width=1.4, color=color)),
            text=s.player + " · " + s.minute.astype(str) + "′ · xG " + s.xg.round(2).astype(str),
            hovertemplate="%{text}<extra></extra>"))
    return fig


def heatmap(grid: pd.DataFrame, title: str, nx: int = 12, ny: int = 8, colorscale=None) -> go.Figure:
    z = np.zeros((ny, nx))
    for r in grid.itertuples():
        z[int(r.gy), int(r.gx)] += r.n
    z = z / max(z.sum(), 1) * 100
    fig = pitch(title=title, height=380)
    fig.add_trace(go.Heatmap(z=z, x0=120 / nx / 2, dx=120 / nx, y0=80 / ny / 2, dy=80 / ny, opacity=.78,
                             colorscale=colorscale or [[0, "rgba(15,15,15,0)"], [0.35, "#5a2400"], [1, ui.ORANGE]],
                             showscale=False, hovertemplate="%{z:.1f}% of actions<extra></extra>", zsmooth="best"))
    fig.data = fig.data[-1:] + fig.data[:-1]   # heat below lines
    return fig


def pass_network(nodes: pd.DataFrame, edges: pd.DataFrame, title: str, color: str = ui.ORANGE) -> go.Figure:
    fig = pitch(title=title, height=420)
    pos = nodes.set_index("player")
    mx = max(edges["count"].max() if len(edges) else 1, 1)
    for e in edges.itertuples():
        if e.a in pos.index and e.b in pos.index:
            a, b = pos.loc[e.a], pos.loc[e.b]
            fig.add_trace(go.Scatter(x=[a.x, b.x], y=[a.y, b.y], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=color, width=0.6 + 6 * e.count / mx),
                                     opacity=0.25 + 0.6 * e.count / mx))
    short = nodes.player.str.split().str[-1]
    fig.add_trace(go.Scatter(
        x=nodes.x, y=nodes.y, mode="markers+text", showlegend=False,
        marker=dict(size=12 + 22 * np.sqrt(nodes.passes / max(nodes.passes.max(), 1)), color="#111",
                    line=dict(color=color, width=2)),
        text=nodes.jersey.fillna("").astype(str).str.replace(r"\.0$", "", regex=True), textfont=dict(size=10, color="#fff"),
        customdata=np.c_[nodes.player, nodes.position.fillna(""), nodes.passes],
        hovertemplate="%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]} passes<extra></extra>"))
    del short
    return fig
