"""Analyst layer: expected threat, team style profiles, player output, match reports.

All tables are aggregated from StatsBomb event data (120 x 80 pitch, every
team attacking towards x = 120) and written as small parquet artifacts, so
the app never needs raw events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NX, NY = 16, 12          # xT grid
HX, HY = 12, 8           # heat-map grid


# ══════════════════════════════════════════════════════════════════════════════
#  Expected threat (Singh 2018): value of possessing the ball in each zone
# ══════════════════════════════════════════════════════════════════════════════
def _cell(x, y, nx=NX, ny=NY):
    cx = np.clip((np.asarray(x) / 120 * nx).astype(int), 0, nx - 1)
    cy = np.clip((np.asarray(y) / 80 * ny).astype(int), 0, ny - 1)
    return cx * ny + cy


def expected_threat(passes: pd.DataFrame, carries: pd.DataFrame, shots: pd.DataFrame,
                    iters: int = 30) -> np.ndarray:
    n = NX * NY
    mv = pd.concat([passes[["x", "y", "end_x", "end_y", "complete"]],
                    carries[["x", "y", "end_x", "end_y"]].assign(complete=True)], ignore_index=True)
    mv = mv.dropna(subset=["x", "y"])
    s_cell = _cell(shots.x, shots.y)
    m_cell = _cell(mv.x, mv.y)
    shots_n = np.bincount(s_cell, minlength=n).astype(float)
    goals_n = np.bincount(s_cell, weights=shots["goal"].to_numpy(float), minlength=n)
    moves_n = np.bincount(m_cell, minlength=n).astype(float)
    total = shots_n + moves_n
    with np.errstate(invalid="ignore", divide="ignore"):
        p_shot = np.nan_to_num(shots_n / total)
        p_move = np.nan_to_num(moves_n / total)
        p_goal = np.nan_to_num(goals_n / shots_n)
    ok = mv["complete"].to_numpy(bool) & mv["end_x"].notna().to_numpy()
    T = np.zeros((n, n))
    np.add.at(T, (m_cell[ok], _cell(mv.end_x[ok], mv.end_y[ok])), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.nan_to_num(T / moves_n[:, None])     # failed moves leak probability (possession lost)
    xt = np.zeros(n)
    for _ in range(iters):
        xt = p_shot * p_goal + p_move * (T @ xt)
    return xt.reshape(NX, NY)


def xt_added(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    flat = grid.ravel()
    ok = df["end_x"].notna().to_numpy() & df.get("complete", pd.Series(True, index=df.index)).to_numpy(bool)
    out = np.zeros(len(df))
    out[ok] = flat[_cell(df.end_x[ok], df.end_y[ok])] - flat[_cell(df.x[ok], df.y[ok])]
    return out


def progressive(df: pd.DataFrame) -> np.ndarray:
    """Wyscout-style: moves the ball >= 25% closer to goal (from outside own 40%), or into the box."""
    d0 = np.hypot(120 - df.x, 40 - df.y)
    d1 = np.hypot(120 - df.end_x, 40 - df.end_y)
    into_box = (df.end_x >= 102) & df.end_y.between(18, 62) & ~((df.x >= 102) & df.y.between(18, 62))
    return (((d1 <= 0.75 * d0) & (df.x >= 48)) | into_box).fillna(False).to_numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Team & player aggregates
# ══════════════════════════════════════════════════════════════════════════════
def _opponent(matches: pd.DataFrame) -> pd.DataFrame:
    h = matches[["match_id", "home", "away"]].rename(columns={"home": "team", "away": "opponent"})
    a = matches[["match_id", "away", "home"]].rename(columns={"away": "team", "home": "opponent"})
    return pd.concat([h, a], ignore_index=True)


def team_match_stats(matches, shots, passes, defense, carries) -> pd.DataFrame:
    opp = _opponent(matches)
    base = opp.merge(matches[["match_id", "label", "date", "home", "home_score", "away_score"]], on="match_id")
    base["venue"] = np.where(base.team == base.home, "home", "away")
    base["goals_for"] = np.where(base.venue == "home", base.home_score, base.away_score)
    base["goals_against"] = np.where(base.venue == "home", base.away_score, base.home_score)
    key = ["match_id", "team"]
    np_shots = shots[shots.shot_type != "Penalty"]
    agg = {
        "xg": shots.groupby(key).xg.sum(), "npxg": np_shots.groupby(key).xg.sum(),
        "shots": shots.groupby(key).size(),
        "sp_xg": shots[shots.play_pattern.isin(["From Corner", "From Free Kick", "From Throw In"])].groupby(key).xg.sum(),
        "passes": passes.groupby(key).size(),
        "pass_completion": passes.groupby(key).complete.mean(),
        "final_third_passes": passes[passes.x >= 80].groupby(key).size(),
        "own_half_passes": passes[passes.x < 72].groupby(key).size(),
        "prog_passes": passes[passes.progressive & passes.complete].groupby(key).size(),
        "prog_carries": carries[carries.progressive].groupby(key).size(),
        "crosses": passes[passes.cross].groupby(key).size(),
        "long_balls_own_third": passes[(passes.x < 40) & (passes.length > 32)].groupby(key).size(),
        "passes_own_third": passes[passes.x < 40].groupby(key).size(),
        "xt": pd.concat([passes[["match_id", "team", "xt_added"]], carries[["match_id", "team", "xt_added"]]])
                .groupby(key).xt_added.sum(),
        "pressures": defense[defense.type == "Pressure"].groupby(key).size(),
        "press_height": defense[defense.type == "Pressure"].groupby(key).x.mean(),
        "def_actions_high": defense[defense.type.isin(["Tackle", "Interception", "Foul Committed", "Dribbled Past"])
                                    & (defense.x > 48)].groupby(key).size(),
        "high_regains": defense[defense.type.isin(["Ball Recovery", "Interception"]) & (defense.x >= 80)].groupby(key).size(),
        "def_line_height": defense[defense.type.isin(["Tackle", "Interception", "Ball Recovery", "Block",
                                                      "Clearance"])].groupby(key).x.mean(),
    }
    t = base.set_index(key)
    for k, s in agg.items():
        t[k] = s
    t = t.reset_index()
    fill0 = [c for c in agg if c not in ("pass_completion", "press_height", "def_line_height")]
    t[fill0] = t[fill0].fillna(0)
    # Opponent-relative metrics
    o = t[["match_id", "team", "xg", "npxg", "shots", "passes", "final_third_passes", "own_half_passes",
           "xt"]].rename(columns=lambda c: c if c in ("match_id",) else f"opp_{c}")
    o = o.rename(columns={"opp_team": "opponent"})
    t = t.merge(o, on=["match_id", "opponent"], how="left")
    t["possession"] = t.passes / (t.passes + t.opp_passes)
    t["field_tilt"] = t.final_third_passes / (t.final_third_passes + t.opp_final_third_passes)
    t["ppda"] = t.opp_own_half_passes / t.def_actions_high.replace(0, np.nan)
    t["xg_against"] = t.opp_xg
    t["npxg_against"] = t.opp_npxg
    t["long_ball_share"] = t.long_balls_own_third / t.passes_own_third.replace(0, np.nan)
    t["points"] = np.select([t.goals_for > t.goals_against, t.goals_for == t.goals_against], [3, 1], 0)
    return t


PROFILE_METRICS = {
    "npxg": ("Non-penalty xG", True), "npxg_against": ("npxG conceded", False),
    "possession": ("Possession", True), "field_tilt": ("Field tilt", True),
    "ppda": ("PPDA (pressing intensity)", False), "press_height": ("Pressure height (m)", True),
    "high_regains": ("High regains", True), "prog_passes": ("Progressive passes", True),
    "prog_carries": ("Progressive carries", True), "xt": ("Expected threat created", True),
    "crosses": ("Crosses", True), "long_ball_share": ("Long-ball share (build-up)", True),
    "sp_xg": ("Set-piece xG", True), "pass_completion": ("Pass completion", True),
    "def_line_height": ("Defensive action height (m)", True),
}


def team_profiles(tm: pd.DataFrame) -> pd.DataFrame:
    g = tm.groupby(["label", "team"])
    prof = g[list(PROFILE_METRICS)].mean()
    prof["matches"] = g.size()
    prof["points_pg"] = g.points.mean()
    prof["goals_pg"], prof["conceded_pg"] = g.goals_for.mean(), g.goals_against.mean()
    prof["xg_pg"], prof["xga_pg"] = g.xg.mean(), g.xg_against.mean()
    prof = prof.reset_index()
    # Percentile within competition (higher = more of the trait; for "lower is better" metrics invert)
    for m, (_, higher) in PROFILE_METRICS.items():
        r = prof.groupby("label")[m].rank(pct=True)
        pct = r if higher else 1 - r + 1 / prof.groupby("label")[m].transform("count")
        prof[f"pct_{m}"] = pct.fillna(0.5)   # metric unavailable for this team: neutral percentile
    return prof


def player_minutes(lineups, passes, shots, defense, carries) -> pd.DataFrame:
    ev = pd.concat([d[["match_id", "team", "player", "minute"]] for d in (passes, shots, defense, carries)])
    span = ev.groupby(["match_id", "team", "player"]).minute.agg(["min", "max"]).reset_index()
    starters = lineups[["match_id", "player"]].assign(starter=True)
    span = span.merge(starters, on=["match_id", "player"], how="left")
    span["starter"] = span["starter"].fillna(False).astype(bool)
    start = np.where(span.starter, 0, span["min"])
    end = np.where(span["max"] >= 80, np.maximum(span["max"], 90), span["max"])
    span["minutes"] = np.clip(end - start, 1, 125)
    return span[["match_id", "team", "player", "minutes", "starter"]]


def player_stats(matches, lineups, shots, passes, defense, carries) -> pd.DataFrame:
    mins = player_minutes(lineups, passes, shots, defense, carries)
    lab = matches[["match_id", "label"]]
    mins = mins.merge(lab, on="match_id")
    key = ["label", "team", "player"]
    S = shots.drop(columns="label", errors="ignore").merge(lab, on="match_id")
    P = passes.merge(lab, on="match_id")
    C = carries.merge(lab, on="match_id")
    D = defense.merge(lab, on="match_id")
    out = mins.groupby(key).agg(minutes=("minutes", "sum"), matches=("match_id", "nunique"),
                                starts=("starter", "sum"))
    np_s = S[S.shot_type != "Penalty"]
    out["goals"] = S.groupby(key).goal.sum()
    out["npxg"] = np_s.groupby(key).xg.sum()
    out["shots"] = S.groupby(key).size()
    out["xa"] = S.dropna(subset=["assist_player"]).groupby(["label", "team", "assist_player"]).xg.sum() \
        .rename_axis(key)
    out["key_passes"] = P[P.shot_assist | P.goal_assist].groupby(key).size()
    out["assists"] = P[P.goal_assist].groupby(key).size()
    out["prog_passes"] = P[P.progressive & P.complete].groupby(key).size()
    out["prog_carries"] = C[C.progressive].groupby(key).size()
    out["xt_pass"] = P.groupby(key).xt_added.sum()
    out["xt_carry"] = C.groupby(key).xt_added.sum()
    out["pressures"] = D[D.type == "Pressure"].groupby(key).size()
    out["tackles_int"] = D[D.type.isin(["Tackle", "Interception"])].groupby(key).size()
    out["passes"] = P.groupby(key).size()
    out["pass_completion"] = P.groupby(key).complete.mean()
    out = out.fillna({c: 0 for c in out.columns if c != "pass_completion"}).reset_index()
    pos = lineups.merge(lab, on="match_id").groupby(key).position.agg(lambda s: s.mode().iat[0])
    out = out.merge(pos.rename("position").reset_index(), on=key, how="left")
    out["xt"] = out.xt_pass + out.xt_carry
    p90 = out.minutes / 90
    for c in ["goals", "npxg", "shots", "xa", "key_passes", "prog_passes", "prog_carries", "xt",
              "pressures", "tackles_int"]:
        out[f"{c}_p90"] = out[c] / p90.where(out.minutes >= 1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Match-level report data
# ══════════════════════════════════════════════════════════════════════════════
def pass_networks(passes: pd.DataFrame, lineups: pd.DataFrame, min_passes: int = 3):
    """Average positions (of pass origins) and completed-pass links among starters."""
    st = lineups[["match_id", "team", "player", "position", "jersey"]]
    ok = passes[passes.complete & passes.recipient.notna()]
    ok = ok.merge(st[["match_id", "player"]], on=["match_id", "player"])
    ok = ok.merge(st[["match_id", "player"]].rename(columns={"player": "recipient"}), on=["match_id", "recipient"])
    nodes = ok.groupby(["match_id", "team", "player"]).agg(x=("x", "mean"), y=("y", "mean"),
                                                          passes=("x", "size")).reset_index()
    nodes = nodes.merge(st, on=["match_id", "team", "player"], how="left")
    a = np.where(ok.player < ok.recipient, ok.player, ok.recipient)
    b = np.where(ok.player < ok.recipient, ok.recipient, ok.player)
    edges = ok.assign(a=a, b=b).groupby(["match_id", "team", "a", "b"]).size().rename("count").reset_index()
    edges = edges[edges["count"] >= min_passes]
    return nodes, edges


def heat_grid(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = df.dropna(subset=["x", "y"]).copy()
    d["gx"] = np.clip((d.x / 120 * HX).astype(int), 0, HX - 1)
    d["gy"] = np.clip((d.y / 80 * HY).astype(int), 0, HY - 1)
    return d.groupby(by + ["gx", "gy"]).size().rename("n").reset_index()
