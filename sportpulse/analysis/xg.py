"""Expected-goals model on StatsBomb open event data.

LightGBM binary classifier on non-penalty shots with monotone constraints
(xG cannot increase with distance or decrease with goal-mouth angle), using
shot geometry, technique/context, the assisting pass and freeze-frame
geometry. Penalties get their empirical conversion rate.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from sportpulse.config import SEED
from sportpulse.evaluation.metrics import binary_log_loss

CATS = {"body_part": ["Right Foot", "Left Foot", "Head", "Other"],
        "shot_type": ["Open Play", "Free Kick", "Corner", "Kick Off"],
        "technique": ["Normal", "Volley", "Half Volley", "Lob", "Backheel", "Overhead Kick", "Diving Header"],
        "play_pattern": ["Regular Play", "From Counter", "From Corner", "From Free Kick", "From Throw In",
                         "From Goal Kick", "From Keeper", "From Kick Off", "Other"],
        "assist_type": ["pass", "cross", "through_ball", "cut_back"],
        "assist_height": ["Ground Pass", "Low Pass", "High Pass"]}
NUM = ["distance", "angle", "x", "y", "dy", "first_time", "one_on_one", "aerial_won", "open_goal",
       "follows_dribble", "under_pressure", "assisted", "assist_x",
       "ff_available", "ff_def_in_cone", "ff_mates_in_cone", "ff_def_within_2", "ff_def_within_5",
       "ff_nearest_def", "ff_gk_dist_goal", "ff_gk_dist_shooter", "ff_gk_in_cone", "ff_gk_off_line"]
FEATURES = NUM + list(CATS)
MONO = {"distance": -1, "angle": 1}
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=80,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=SEED, num_threads=0,
              monotone_constraints=[MONO.get(f, 0) for f in FEATURES], monotone_constraints_method="advanced")


def features(shots: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=shots.index)
    for c in NUM:
        if c == "dy":
            X[c] = (shots["y"] - 40).abs()
        elif c == "assisted":
            X[c] = shots["assist_type"].notna().astype(float)
        else:
            X[c] = pd.to_numeric(shots.get(c), errors="coerce").astype(float)
    for c, levels in CATS.items():
        X[c] = pd.Categorical(shots[c], categories=levels)
    return X[FEATURES]


def _metrics(y, p) -> dict:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    from sportpulse.evaluation.metrics import ece
    pp = np.c_[1 - p, p]
    return {"log_loss": binary_log_loss(p, y), "brier": float(((p - y) ** 2).mean()),
            "auc": float(roc_auc_score(y, p)), "ece": ece(pp, y.astype(int)),
            "xg_sum": float(p.sum()), "goals": float(y.sum()), "n": int(len(y))}


def train_and_evaluate(shots: pd.DataFrame, recent_labels: set[str]) -> tuple[lgb.Booster, pd.Series, dict]:
    """Returns (production booster, out-of-fold xG for every shot, metrics)."""
    shots = shots.copy()
    pen = shots["shot_type"].eq("Penalty").to_numpy()
    y = shots["goal"].to_numpy(float)
    X = features(shots)
    np_mask = ~pen
    Xn, yn, groups = X[np_mask], y[np_mask], shots.loc[np_mask, "match_id"].to_numpy()

    oof = np.zeros(np_mask.sum())
    iters = []
    for tr, te in GroupKFold(5).split(Xn, yn, groups):
        # inner early-stopping split by match
        g = np.unique(groups[tr])
        rng = np.random.default_rng(SEED)
        inner = np.isin(groups[tr], rng.choice(g, len(g) // 8, replace=False))
        a, v = tr[~inner], tr[inner]
        dtr = lgb.Dataset(Xn.iloc[a], yn[a])
        b = lgb.train(PARAMS, dtr, 5000, valid_sets=[lgb.Dataset(Xn.iloc[v], yn[v], reference=dtr)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[te] = b.predict(Xn.iloc[te], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
    n_iter = int(np.mean(iters) * 1.1)

    pen_rate = float(y[pen].mean()) if pen.any() else 0.76
    xg = np.full(len(shots), pen_rate)
    xg[np_mask] = oof

    # Baselines on the same folds
    geo = np.c_[shots["distance"], shots["angle"], shots["distance"] ** 2][np_mask]
    lr_oof = np.zeros(np_mask.sum())
    for tr, te in GroupKFold(5).split(geo, yn, groups):
        lr_oof[te] = LogisticRegression(max_iter=2000).fit(geo[tr], yn[tr]).predict_proba(geo[te])[:, 1]
    sb = shots.loc[np_mask, "sb_xg"].to_numpy(float)
    ok = np.isfinite(sb)

    metrics = {"non_penalty_cv": {
        "SportPulse xG (LightGBM)": _metrics(yn[ok], oof[ok]),
        "StatsBomb xG (commercial)": _metrics(yn[ok], sb[ok]),
        "Distance + angle logistic": _metrics(yn[ok], lr_oof[ok]),
    }, "penalty_rate": pen_rate, "iters": n_iter,
        "shots": int(len(shots)), "non_penalty_shots": int(np_mask.sum()), "matches": int(shots.match_id.nunique())}

    # Temporal generalisation: train on older competitions, test on 2022-2024 tournaments
    recent = shots.loc[np_mask, "label"].isin(recent_labels).to_numpy()
    if recent.any() and (~recent).any():
        dtr = lgb.Dataset(Xn[~recent], yn[~recent])
        b = lgb.train(PARAMS, dtr, n_iter)
        pr = b.predict(Xn[recent])
        okr = recent & ok
        metrics["temporal_recent_test"] = {
            "SportPulse xG (LightGBM)": _metrics(yn[okr], b.predict(Xn[okr])),
            "StatsBomb xG (commercial)": _metrics(yn[okr], sb[okr]),
            "n": int(okr.sum()), "competitions": sorted(recent_labels)}
        del pr

    booster = lgb.train(PARAMS, lgb.Dataset(Xn, yn), n_iter)
    imp = pd.Series(booster.feature_importance("gain"), FEATURES)
    metrics["importance"] = (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()
    # Calibration curve (OOF)
    bins = np.quantile(oof, np.linspace(0, 1, 13))
    idx = np.clip(np.digitize(oof, bins[1:-1]), 0, 11)
    metrics["calibration"] = [(float(oof[idx == k].mean()), float(yn[idx == k].mean()), int((idx == k).sum()))
                              for k in range(12) if (idx == k).any()]
    return booster, pd.Series(xg, index=shots.index), metrics
