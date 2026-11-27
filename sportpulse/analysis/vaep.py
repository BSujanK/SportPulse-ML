"""VAEP action values (Decroos, Bransen, Van Haaren & Davis, KDD 2019) on StatsBomb open data.

Every on-ball action gets P(scores) and P(concedes): the probability that the acting team
scores / concedes within the next 10 actions, estimated by LightGBM from the action and the
two before it (all in the acting team's frame of reference). The value of an action is

    VAEP(a_i) = [P_s(i) - P_s(i-1)] - [P_c(i) - P_c(i-1)]

where the previous state's probabilities are swapped when possession changed hands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from sportpulse.config import SEED

HORIZON = 10
TYPES = ["pass", "cross", "throw_in", "freekick_pass", "corner", "goalkick", "kickoff", "carry", "shot",
         "shot_penalty", "shot_freekick", "tackle", "interception", "recovery", "clearance", "block", "foul"]
DEF_MAP = {"Tackle": "tackle", "Interception": "interception", "Ball Recovery": "recovery",
           "Clearance": "clearance", "Block": "block", "Foul Committed": "foul"}
PASS_MAP = {"Throw-in": "throw_in", "Free Kick": "freekick_pass", "Corner": "corner", "Goal Kick": "goalkick",
            "Kick Off": "kickoff"}
PRIORITY = {"defense": 0, "pass": 1, "carry": 2, "shot": 3}
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=63, verbose=-1, seed=SEED,
              num_threads=0)


def build_actions(matches, passes, carries, shots, defense) -> pd.DataFrame:
    """One time-ordered SPADL-like action table for all matches."""
    keep = ["match_id", "team", "player", "period", "minute", "second", "x", "y", "possession"]
    p = passes[keep + ["end_x", "end_y", "complete", "pass_type", "cross", "under_pressure", "play_pattern"]].copy()
    p["type"] = np.where(p.cross.fillna(False), "cross", p.pass_type.map(PASS_MAP).fillna("pass"))
    p["success"] = p.complete.fillna(False).astype(float)
    p["src"] = "pass"
    c = carries[keep + ["end_x", "end_y", "under_pressure", "play_pattern"]].copy()
    c["type"], c["success"], c["src"] = "carry", 1.0, "carry"
    s = shots[keep + ["end_x", "end_y", "shot_type", "outcome", "under_pressure", "play_pattern"]].copy()
    s["type"] = s.shot_type.map({"Penalty": "shot_penalty", "Free Kick": "shot_freekick"}).fillna("shot")
    s["success"] = s.outcome.eq("Goal").astype(float)
    s["end_x"], s["end_y"] = 120.0, 40.0
    s["src"] = "shot"
    d = defense[defense.type.isin(DEF_MAP)][keep + ["type", "outcome", "under_pressure", "play_pattern"]].copy()
    d["type"] = d.type.map(DEF_MAP)
    d["success"] = (~d.outcome.fillna("").isin(["Lost In Play", "Lost Out"])).astype(float)
    d["end_x"], d["end_y"], d["src"] = d.x, d.y, "defense"
    cols = keep + ["end_x", "end_y", "type", "success", "under_pressure", "play_pattern", "src"]
    a = pd.concat([t[cols] for t in (p, c, s, d)], ignore_index=True)
    a["t"] = a.minute * 60 + a.second
    a["prio"] = a.src.map(PRIORITY)
    a = a[a.period < 5].sort_values(["match_id", "period", "t", "prio"], kind="stable").reset_index(drop=True)
    a["under_pressure"] = a.under_pressure.fillna(False).astype(float)
    home = matches.set_index("match_id")["home"]
    a["is_home"] = (a.team == a.match_id.map(home)).astype(float)
    return a


def _goal_angle(x, y):
    dx, dy = 120 - x, np.abs(40 - y)
    return np.hypot(dx, dy), np.arctan2(7.32 * dx, dx ** 2 + dy ** 2 - (7.32 / 2) ** 2)


def features_labels(a: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n = len(a)
    match, period, team = a.match_id.to_numpy(), a.period.to_numpy(), a.team.to_numpy()
    goal = ((a.src == "shot") & (a.success == 1)).to_numpy()
    # running score from the acting team's point of view (before the action)
    home_goal = goal & (a.is_home.to_numpy() == 1)
    away_goal = goal & (a.is_home.to_numpy() == 0)
    g = a.groupby("match_id", sort=False)
    hg = g[[]].ngroup()  # placeholder for alignment
    del hg
    cum_h = pd.Series(home_goal.astype(int)).groupby(match).cumsum().to_numpy() - home_goal
    cum_a = pd.Series(away_goal.astype(int)).groupby(match).cumsum().to_numpy() - away_goal
    gd = np.where(a.is_home.to_numpy() == 1, cum_h - cum_a, cum_a - cum_h)

    X = pd.DataFrame(index=a.index)
    tcat = pd.Categorical(a.type, categories=TYPES)
    X["type"] = tcat
    X["success"] = a.success.to_numpy()
    for c in ("x", "y", "end_x", "end_y"):
        X[c] = a[c].to_numpy(float)
    X["dx"], X["dy"] = X.end_x - X.x, X.end_y - X.y
    X["move"] = np.hypot(X.dx, X.dy)
    X["dist"], X["angle"] = _goal_angle(X.x, X.y)
    X["end_dist"], X["end_angle"] = _goal_angle(X.end_x, X.end_y)
    X["under_pressure"] = a.under_pressure.to_numpy()
    X["play_pattern"] = pd.Categorical(a.play_pattern)
    X["goal_diff"] = np.clip(gd, -3, 3)
    X["period"] = period
    X["minute"] = a.minute.to_numpy(float)
    t = a.t.to_numpy(float)
    for k in (1, 2):
        ok = np.r_[np.zeros(k, bool), (match[k:] == match[:-k]) & (period[k:] == period[:-k])]
        idx = np.maximum(np.arange(n) - k, 0)
        same = ok & (team[idx] == team)
        flip = ok & ~same
        px, py = a.x.to_numpy(float)[idx], a.y.to_numpy(float)[idx]
        pex, pey = a.end_x.to_numpy(float)[idx], a.end_y.to_numpy(float)[idx]
        px, pex = np.where(flip, 120 - px, px), np.where(flip, 120 - pex, pex)
        py, pey = np.where(flip, 80 - py, py), np.where(flip, 80 - pey, pey)
        nan = ~ok
        X[f"type_{k}"] = pd.Categorical(np.where(nan, None, a.type.to_numpy()[idx]), categories=TYPES)
        X[f"success_{k}"] = np.where(nan, np.nan, a.success.to_numpy()[idx])
        X[f"same_team_{k}"] = np.where(nan, np.nan, same.astype(float))
        X[f"x_{k}"], X[f"y_{k}"] = np.where(nan, np.nan, px), np.where(nan, np.nan, py)
        X[f"end_x_{k}"], X[f"end_y_{k}"] = np.where(nan, np.nan, pex), np.where(nan, np.nan, pey)
        X[f"dt_{k}"] = np.where(nan, np.nan, t - t[idx])
        X[f"gap_{k}"] = np.where(nan, np.nan, np.hypot(X.x - pex, X.y - pey))

    # third previous action (type, team, time gap)
    ok3 = np.r_[np.zeros(3, bool), (match[3:] == match[:-3]) & (period[3:] == period[:-3])]
    i3 = np.maximum(np.arange(n) - 3, 0)
    X["type_3"] = pd.Categorical(np.where(ok3, a.type.to_numpy()[i3], None), categories=TYPES)
    X["same_team_3"] = np.where(ok3, (team[i3] == team).astype(float), np.nan)
    X["dt_3"] = np.where(ok3, t - t[i3], np.nan)
    # possession context (StatsBomb possession id): actions and seconds since it began, whether the actor's team
    # started it, and how far the actor is from where it started (in the actor's frame)
    if "possession" in a:
        pid = match.astype(np.int64) * 10000 + a.possession.fillna(-1).to_numpy().astype(np.int64)
        first = pd.Series(np.arange(n)).groupby(pid, sort=False).transform("first").to_numpy()
        own = team == team[first]
        sx = a.x.to_numpy(float)[first]
        X["poss_n"] = pd.Series(np.arange(n)).groupby(pid, sort=False).cumcount().to_numpy().astype(float)
        X["poss_t"] = t - t[first]
        X["poss_own"] = own.astype(float)
        X["poss_progress"] = X.x.to_numpy() - np.where(own, sx, 120 - sx)

    # labels: goal for / against the actor within the next HORIZON actions of the same match and period
    scores, concedes = np.zeros(n, bool), np.zeros(n, bool)
    for k in range(HORIZON):
        j = np.minimum(np.arange(n) + k, n - 1)
        ok = (np.arange(n) + k < n) & (match[j] == match) & (period[j] == period) & goal[j]
        scores |= ok & (team[j] == team)
        concedes |= ok & (team[j] != team)
    return X, scores.astype(float), concedes.astype(float)


def fit(X, y, idx_train, idx_valid=None, rounds=400) -> lgb.Booster:
    dtr = lgb.Dataset(X.iloc[idx_train], y[idx_train], params={"max_bin": 63})
    if idx_valid is None:
        return lgb.train(PARAMS, dtr, rounds)
    dva = lgb.Dataset(X.iloc[idx_valid], y[idx_valid], reference=dtr)
    return lgb.train(PARAMS, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])


def values(a: pd.DataFrame, ps: np.ndarray, pc: np.ndarray) -> pd.DataFrame:
    """Offensive, defensive and total VAEP per action."""
    match, period, team = a.match_id.to_numpy(), a.period.to_numpy(), a.team.to_numpy()
    n = len(a)
    idx = np.maximum(np.arange(n) - 1, 0)
    ok = np.r_[False, (match[1:] == match[:-1]) & (period[1:] == period[:-1])]
    same = team[idx] == team
    prev_s = np.where(same, ps[idx], pc[idx])
    prev_c = np.where(same, pc[idx], ps[idx])
    goal_prev = ((a.src.to_numpy() == "shot") & (a.success.to_numpy() == 1))[idx]
    reset = ~ok | goal_prev                                   # new period / after a goal: kickoff state
    prev_s, prev_c = np.where(reset, 0.0, prev_s), np.where(reset, 0.0, prev_c)
    off, dfn = ps - prev_s, -(pc - prev_c)
    return pd.DataFrame({"p_scores": ps, "p_concedes": pc, "vaep_off": off, "vaep_def": dfn, "vaep": off + dfn},
                        index=a.index)
