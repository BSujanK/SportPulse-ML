import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sportpulse.analysis.analyst import PROFILE_METRICS
from webapp import pitch, state, ui

comps = state.competition_order()
prof = state.analyst("team_profiles")
tm = state.analyst("team_match")
shots = state.analyst("shots")
players = state.analyst("players")
mt = state.analyst("matches")

c1, c2 = st.columns([1, 1])
comp = c1.selectbox("Competition", comps, index=comps.index("FIFA World Cup 2022") if "FIFA World Cup 2022" in comps else 0)
pc = prof[prof.label == comp].sort_values(["matches", "points_pg"], ascending=False)
team = c2.selectbox("Team", pc.team.tolist(), format_func=lambda t: f"{t}  ·  {int(pc.set_index('team').at[t, 'matches'])} matches")
P = pc.set_index("team").loc[team]
T = tm[(tm.label == comp) & (tm.team == team)].sort_values("date")
form = mt[mt.match_id.isin(T.match_id)]
forms = pd.concat([form.loc[form.home == team, "home_formation"], form.loc[form.away == team, "away_formation"]])
top_form = forms.dropna().value_counts()
w, d_, l_ = int((T.points == 3).sum()), int((T.points == 1).sum()), int((T.points == 0).sum())

ui.repo_header("opponent-report", team, "shield",
               tag_items=[("task", comp), ("", f"{len(T)} matches · {w}W {d_}D {l_}L"),
                          ("", f"usual shape {top_form.index[0]}" if len(top_form) else "shape n/a"),
                          ("ds", "StatsBomb open data")])
if P.matches < 0.6 * pc.matches.max():
    ui.note(f"Open data for {comp} covers {int(P.matches)} of this team's matches, so percentiles below compare "
            "against teams with similarly partial coverage. Read them as indicative.")

ui.kpi_row([ui.kpi("xG / match", f"{P.xg_pg:.2f}"), ui.kpi("xG against / match", f"{P.xga_pg:.2f}"),
            ui.kpi("Possession", f"{P.possession * 100:.0f}", "%"), ui.kpi("PPDA", f"{P.ppda:.1f}", "",
                                                                             "lower = more pressing"),
            ui.kpi("Field tilt", f"{P.field_tilt * 100:.0f}", "%", "final-third pass share"),
            ui.kpi("Points / match", f"{P.points_pg:.2f}")])

tabs = st.tabs([":material/insights: Overview", ":material/sports_soccer: Attack",
                ":material/shield: Defence & pressing", ":material/groups: Players", ":material/event_note: Match log"])

# ── Overview ──────────────────────────────────────────────────────────────────
with tabs[0]:
    left, right = st.columns([1.1, 1], gap="large")
    keys = list(PROFILE_METRICS)
    with left:
        vals = [float(P[f"pct_{k}"]) * 100 for k in keys]
        labels = [PROFILE_METRICS[k][0] for k in keys]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[50] * len(keys) + [50], theta=labels + [labels[0]], mode="lines",
                                      line=dict(color="#444", dash="dot"), name="competition median"))
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
                                      fillcolor="rgba(255,107,0,.18)", line=dict(color=ui.ORANGE, width=2), name=team))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="#222",
                                                                                tickfont=dict(size=9, color="#666")),
                                     angularaxis=dict(gridcolor="#222", tickfont=dict(size=11))),
                          height=480, title="Style profile · percentile within competition", showlegend=False,
                          margin=dict(l=70, r=70, t=50, b=30))
        ui.plot(fig)
    with right:
        ui.section("Scouting notes", icon="file")
        pct = {k: float(P[f"pct_{k}"]) for k in keys}
        QUALITY = ["npxg", "npxg_against", "xt", "prog_passes", "prog_carries", "high_regains", "sp_xg",
                   "pass_completion"]
        STYLE = {"possession": ("Dominates the ball", "Comfortable without the ball"),
                 "field_tilt": ("Pins opponents back (high field tilt)", "Spends long spells in their own half"),
                 "ppda": ("Presses aggressively", "Sits off and protects space"),
                 "long_ball_share": ("Direct, long build-up", "Short, patient build-up"),
                 "def_line_height": ("Defends high up the pitch", "Defends deep"),
                 "crosses": ("Cross-heavy wide attacks", "Rarely crosses")}
        fmt = {"possession": lambda v: f"{v * 100:.0f}%", "field_tilt": lambda v: f"{v * 100:.0f}%",
               "pass_completion": lambda v: f"{v * 100:.0f}%", "long_ball_share": lambda v: f"{v * 100:.0f}%"}
        val = lambda k: fmt.get(k, lambda v: f"{v:.2f}")(float(P[k]))  # noqa: E731
        strong = [k for k in sorted(QUALITY, key=pct.get, reverse=True) if pct[k] >= .6][:3]
        weak = [k for k in sorted(QUALITY, key=pct.get) if pct[k] <= .4][:3]
        rows = [("good", f"<b>{PROFILE_METRICS[k][0]}</b>: {val(k)}, better than {pct[k] * 100:.0f}% of the competition")
                for k in strong]
        rows += [("bad", f"<b>{PROFILE_METRICS[k][0]}</b>: {val(k)}, bottom {max(pct[k] * 100, 1):.0f}%")
                 for k in weak]
        st.markdown(ui.bullets(rows or [("info", "No metric stands out from the competition average.")]),
                    unsafe_allow_html=True)
        ui.section("Style", icon="activity")
        style = [("info", f"{STYLE[k][0] if pct[k] >= .5 else STYLE[k][1]}: <b>{PROFILE_METRICS[k][0]}</b> {val(k)} "
                          f"({pct[k] * 100:.0f}th pct)") for k in sorted(STYLE, key=lambda k: -abs(pct[k] - .5))[:3]]
        st.markdown(ui.bullets(style), unsafe_allow_html=True)
        ui.section("Game-plan cues", icon="target")
        cues = []
        if pct["ppda"] >= .7:
            cues.append(("info", f"Aggressive press (PPDA {P.ppda:.1f}, pressures at {P.press_height:.0f} m). Plan a "
                                 "secure first pass and a long outlet. Their high regains average "
                                 f"{P.high_regains:.1f} per match."))
        elif pct["ppda"] <= .3:
            cues.append(("info", f"Passive without the ball (PPDA {P.ppda:.1f}). Expect time in build-up; the danger is "
                                 "patient circulation into a set mid/low block."))
        if pct["long_ball_share"] >= .7:
            cues.append(("info", f"Direct build-up ({P.long_ball_share * 100:.0f}% long balls from the first third). "
                                 "Prepare for aerial duels and second balls."))
        if pct["possession"] <= .3 and pct["npxg"] >= .5:
            cues.append(("info", "Effective without the ball: low possession but above-median chance creation. "
                                 "Guard against transitions after losing the ball."))
        if pct["sp_xg"] >= .75:
            cues.append(("bad", f"Set-piece threat: {P.sp_xg:.2f} xG per match from corners, free kicks and throw-ins."))
        if pct["npxg_against"] <= .3:
            cues.append(("good", f"Defensively exposed: {P.npxg_against:.2f} npxG conceded per match."))
        if pct["def_line_height"] >= .75:
            cues.append(("info", f"High defensive line (actions at {P.def_line_height:.0f} m avg.). Runs in behind "
                                 "and early balls over the top are the obvious lever."))
        if not cues:
            cues.append(("info", "Balanced profile with no extreme tendency. Match-specific line-ups and form will dominate."))
        st.markdown(ui.bullets(cues), unsafe_allow_html=True)
        st.caption("Cues are rule-based readings of the percentiles, not model predictions.")

# ── Attack ────────────────────────────────────────────────────────────────────
with tabs[1]:
    S = shots[(shots.label == comp) & (shots.team == team) & (shots.shot_type != "Penalty")]
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        ui.plot(pitch.shot_map(S, title=f"Non-penalty shots · {len(S)} shots · {S.xg.sum():.1f} xG · {int(S.goal.sum())} goals"))
    with b:
        g = S.groupby("play_pattern").agg(xg=("xg", "sum"), shots=("xg", "size")).sort_values("xg")
        fig = go.Figure(go.Bar(x=g.xg, y=g.index, orientation="h", marker_color=ui.ORANGE, cliponaxis=False,
                               text=g.shots.astype(str) + " shots", textposition="outside"))
        fig.update_layout(title="xG by phase of play", height=280, margin=dict(l=10, r=70, t=40, b=10),
                          xaxis=dict(range=[0, float(g.xg.max()) * 1.25 if len(g) else 1]))
        ui.plot(fig)
        at = S.assist_type.fillna("unassisted").map(
            {"pass": "Pass", "cross": "Cross", "through_ball": "Through ball", "cut_back": "Cut-back",
             "unassisted": "Unassisted / individual"}).value_counts()
        fig = go.Figure(go.Bar(x=at.values, y=at.index, orientation="h", marker_color=ui.ORANGE_3))
        fig.update_layout(title="Shot creation: assist type", height=240, margin=dict(l=10, r=10, t=40, b=10))
        ui.plot(fig)
    ui.plot(pitch.heatmap(state.analyst("heat_touch").query("label == @comp and team == @team"),
                          "Where they play: passes and carries"))

# ── Defence ───────────────────────────────────────────────────────────────────
with tabs[2]:
    opp_ids = T[["match_id", "opponent"]]
    SA = shots.merge(opp_ids, left_on=["match_id", "team"], right_on=["match_id", "opponent"])
    SA = SA[SA.shot_type != "Penalty"]
    a, b = st.columns(2, gap="large")
    with a:
        ui.plot(pitch.shot_map(SA, title=f"Shots conceded · {len(SA)} shots · {SA.xg.sum():.1f} xG", color="#38bdf8",
                               name="Conceded"))
    with b:
        hp = state.analyst("heat_pressure").query("label == @comp and team == @team")
        ui.plot(pitch.heatmap(hp, "Pressing map: where they apply pressure"))
    ui.plot(pitch.heatmap(state.analyst("heat_defense").query("label == @comp and team == @team"),
                          "Ball-winning map: tackles, interceptions, recoveries, blocks",
                          colorscale=[[0, "rgba(15,15,15,0)"], [0.35, "#0c3b52"], [1, "#38bdf8"]]))

# ── Players ───────────────────────────────────────────────────────────────────
with tabs[3]:
    pl = players[(players.label == comp) & (players.team == team)].sort_values("minutes", ascending=False)
    mn = st.slider("Minimum minutes", 0, int(max(pl.minutes.max(), 90)), min(90, int(pl.minutes.max())), 15)
    pl = pl[pl.minutes >= mn]
    st.dataframe(
        pl[["player", "position", "minutes", "goals", "npxg", "xa", "xt", "npxg_p90", "xa_p90", "xt_p90",
            "prog_passes_p90", "prog_carries_p90", "pressures_p90", "tackles_int_p90"]],
        hide_index=True, width="stretch", height=min(40 * (len(pl) + 1), 640),
        column_config={
            "player": st.column_config.TextColumn("Player", width="medium"), "position": "Position",
            "minutes": st.column_config.NumberColumn("Min", format="%d"),
            "goals": st.column_config.NumberColumn("G", format="%d"),
            "npxg": st.column_config.NumberColumn("npxG", format="%.2f"),
            "xa": st.column_config.NumberColumn("xA", format="%.2f"),
            "xt": st.column_config.NumberColumn("xT", format="%.2f"),
            "npxg_p90": st.column_config.ProgressColumn("npxG/90", min_value=0, max_value=1, format="%.2f"),
            "xa_p90": st.column_config.ProgressColumn("xA/90", min_value=0, max_value=0.6, format="%.2f"),
            "xt_p90": st.column_config.ProgressColumn("xT/90", min_value=0, max_value=0.6, format="%.2f"),
            "prog_passes_p90": st.column_config.NumberColumn("Prog. passes/90", format="%.1f"),
            "prog_carries_p90": st.column_config.NumberColumn("Prog. carries/90", format="%.1f"),
            "pressures_p90": st.column_config.NumberColumn("Pressures/90", format="%.1f"),
            "tackles_int_p90": st.column_config.NumberColumn("Tkl+Int/90", format="%.1f")})
    st.caption("xA = xG of the shots a player's passes created · xT = expected-threat added by passes and carries.")

# ── Match log ─────────────────────────────────────────────────────────────────
with tabs[4]:
    L = T.copy()
    L["Result"] = np.select([L.points == 3, L.points == 1], ["W", "D"], "L")
    L["Score"] = L.goals_for.astype(int).astype(str) + "–" + L.goals_against.astype(int).astype(str)
    st.dataframe(L[["date", "opponent", "venue", "Result", "Score", "xg", "xg_against", "possession", "ppda", "field_tilt"]],
                 hide_index=True, width="stretch",
                 column_config={"date": st.column_config.DateColumn("Date"), "opponent": "Opponent", "venue": "Venue",
                                "xg": st.column_config.NumberColumn("xG", format="%.2f"),
                                "xg_against": st.column_config.NumberColumn("xGA", format="%.2f"),
                                "possession": st.column_config.ProgressColumn("Possession", min_value=0, max_value=1,
                                                                              format="percent"),
                                "ppda": st.column_config.NumberColumn("PPDA", format="%.1f"),
                                "field_tilt": st.column_config.NumberColumn("Field tilt", format="percent")})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=L.date.dt.strftime("%d %b") + " " + L.opponent.str[:12], y=L.xg, name="xG", marker_color=ui.ORANGE))
    fig.add_trace(go.Bar(x=L.date.dt.strftime("%d %b") + " " + L.opponent.str[:12], y=-L.xg_against, name="xG against",
                         marker_color="#38bdf8"))
    fig.update_layout(barmode="relative", title="xG for (up) and against (down) by match", height=340)
    ui.plot(fig)
