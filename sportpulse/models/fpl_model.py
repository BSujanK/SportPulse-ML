"""Premier League player performance forecaster (per player, per fixture).

Targets: FPL points (regression), P(60+ minutes), P(goal), P(assist).
Features are strictly lagged: a row only sees the player's previous fixtures,
plus pre-match team context from the leak-free match feature engine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sportpulse.config import SEED

STATS = ["minutes", "total_points", "goals_scored", "assists", "bonus", "bps", "ict_index",
         "influence", "creativity", "threat", "clean_sheets", "goals_conceded", "saves",
         "expected_goals", "expected_assists", "expected_goal_involvements",
         "expected_goals_conceded", "starts", "yellow_cards", "played", "played60"]
RATE_STATS = ["total_points", "goals_scored", "assists", "expected_goals", "expected_assists",
              "bonus", "threat", "creativity", "saves"]
CONTEXT = ["ctx_elo_for", "ctx_elo_against", "ctx_elo_diff", "ctx_pi_gd", "ctx_xg_for",
           "ctx_xg_against", "ctx_team_gf", "ctx_team_ga", "ctx_opp_gf", "ctx_opp_ga",
           "ctx_team_sot", "ctx_opp_sot", "ctx_team_pts", "ctx_opp_pts", "ctx_win_p"]

TARGETS = {
    "points": ("total_points", "regression"),
    "p_60": ("played60", "binary"),
    "p_goal": ("scored", "binary"),
    "p_assist": ("assisted", "binary"),
}
PARAMS = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED,
              num_threads=0)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["played"] = (df["minutes"] > 0).astype(float)
    df["played60"] = (df["minutes"] >= 60).astype(float)
    df["scored"] = (df["goals_scored"] > 0).astype(float)
    df["assisted"] = (df["assists"] > 0).astype(float)
    return df


def player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged rolling features. `df` must be sorted by (player_code, kickoff)."""
    g = df.groupby("player_code", sort=False)
    out = pd.DataFrame(index=df.index)
    prev = g[STATS].shift(1)
    pg = prev.groupby(df["player_code"], sort=False)
    for w in (3, 6):
        r = pg.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
        out[[f"{s}_r{w}" for s in STATS]] = r[STATS].reindex(df.index).to_numpy()
    e = pg.transform(lambda x: x.ewm(halflife=6, ignore_na=True).mean())
    out[[f"{s}_ewm" for s in STATS]] = e[STATS].reindex(df.index).to_numpy()
    # per-90 rates over the last 10 fixtures (minutes-weighted)
    s10 = pg.rolling(10, min_periods=1).sum().reset_index(level=0, drop=True).sort_index().reindex(df.index)
    mins = s10["minutes"].replace(0, np.nan)
    for s in RATE_STATS:
        out[f"{s}_p90_10"] = (s10[s] / mins * 90).clip(upper=30)
    out["minutes_sum_10"] = s10["minutes"]
    out["apps_total"] = g.cumcount()
    out["last_minutes"] = prev["minutes"]
    out["days_since"] = (df["kickoff"] - g["kickoff"].shift(1)).dt.days.clip(upper=120)
    # season-to-date
    gs = df.groupby(["player_code", "season"], sort=False)
    out["season_apps"] = gs.cumcount()
    out["season_pts_pg"] = gs["total_points"].transform(lambda x: x.shift(1).expanding().mean())
    out["season_min_pg"] = gs["minutes"].transform(lambda x: x.shift(1).expanding().mean())
    # market signals from the previous gameweek (known before the deadline)
    out["price"] = g["value"].shift(1) / 10
    out["log_selected"] = np.log1p(g["selected"].shift(1))
    out["transfer_rate"] = g["transfers_balance"].shift(1) / (g["selected"].shift(1) + 1000)
    out["price_change_5"] = out["price"] - g["value"].shift(6) / 10
    out["was_home"] = df["was_home"].astype(float)
    out["position"] = pd.Categorical(df["position"], categories=["GK", "DEF", "MID", "FWD"])
    return out


@dataclass
class FPLForecaster:
    features: list[str]
    iters: dict = field(default_factory=dict)
    boosters: dict = field(default_factory=dict)
    targets: tuple = tuple(TARGETS)

    def _X(self, F: pd.DataFrame) -> pd.DataFrame:
        X = F.reindex(columns=self.features).copy()
        X["position"] = pd.Categorical(F["position"], categories=["GK", "DEF", "MID", "FWD"])
        return X

    def fit(self, F, Y, train, valid):
        X = self._X(F)
        for name in self.targets:
            col, obj = TARGETS[name]
            p = dict(PARAMS, objective=obj)
            y = Y[col].to_numpy(float)
            dtr = lgb.Dataset(X[train], y[train])
            dva = lgb.Dataset(X[valid], y[valid], reference=dtr)
            b = lgb.train(p, dtr, 6000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            self.boosters[name], self.iters[name] = b, b.best_iteration
        return self

    def refit(self, F, Y, mask, n_ref):
        X = self._X(F)
        scale = mask.sum() / max(n_ref, 1)
        for name in self.targets:
            col, obj = TARGETS[name]
            self.iters[name] = max(int(self.iters[name] * scale), 1)
            self.boosters[name] = lgb.train(dict(PARAMS, objective=obj),
                                            lgb.Dataset(X[mask], Y[col].to_numpy(float)[mask]),
                                            self.iters[name])
        return self

    def predict(self, F) -> pd.DataFrame:
        X = self._X(F)
        return pd.DataFrame({n: b.predict(X, num_iteration=self.iters[n])
                             for n, b in self.boosters.items()}, index=F.index)

    def importance(self, name="points") -> pd.Series:
        b = self.boosters[name]
        s = pd.Series(b.feature_importance("gain"), self.features)
        return (s / s.sum()).sort_values(ascending=False)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for n, b in self.boosters.items():
            b.save_model(str(path / f"fpl_{n}.lgb"))
        (path / "fpl_meta.json").write_text(json.dumps({"features": self.features, "iters": self.iters}))

    @classmethod
    def load(cls, path: Path):
        meta = json.loads((path / "fpl_meta.json").read_text())
        m = cls(meta["features"], meta["iters"])
        m.boosters = {n: lgb.Booster(model_file=str(path / f"fpl_{n}.lgb")) for n in TARGETS}
        return m
