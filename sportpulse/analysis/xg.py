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
INTERNATIONAL = ("World Cup", "Euro", "Copa America", "African Cup", "U20")
NUM = ["female", "distance", "angle", "x", "y", "dy", "first_time", "one_on_one", "aerial_won", "open_goal",
       "follows_dribble", "under_pressure", "assisted", "assist_x",
       "ff_available", "ff_def_in_cone", "ff_mates_in_cone", "ff_def_within_2", "ff_def_within_5",
       "ff_nearest_def", "ff_gk_dist_goal", "ff_gk_dist_shooter", "ff_gk_in_cone", "ff_gk_off_line",
       "ff_lane_blockers", "ff_pressure_w", "ff_goal_visible", "ff_gk_lane_offset"]
FEATURES = NUM + list(CATS)
MONO = {"distance": -1, "angle": 1}
# Optuna TPE (25 trials, re-run after the freeze-frame geometry features were added) on a held-out
# fifth of pre-2022 matches; the 2022-24 test set is not used.
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=10, min_data_in_leaf=74,
              feature_fraction=0.66, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.49,
              min_sum_hessian_in_leaf=0.0043, verbose=-1, seed=SEED, num_threads=0,
              monotone_constraints=[MONO.get(f, 0) for f in FEATURES], monotone_constraints_method="advanced")


# Recency weighting: tested at 15 / 8 / 5 / 3-year half-lives; none held up in the full pipeline, so it is off.
HALF_LIFE_YEARS = None


def recency_weights(shots: pd.DataFrame, ref=None) -> np.ndarray:
    if HALF_LIFE_YEARS is None or "date" not in shots:
        return np.ones(len(shots))
    dates = pd.to_datetime(shots["date"])
    ref = pd.Timestamp(ref) if ref is not None else dates.max()
    age = ((ref - dates).dt.days / 365.25).clip(lower=0).to_numpy()
    return 0.5 ** (age / HALF_LIFE_YEARS)


def features(shots: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=shots.index)
    for c in NUM:
        if c == "dy":
            X[c] = (shots["y"] - 40).abs()
        elif c == "international":
            comp = shots.get("competition", pd.Series("", index=shots.index)).astype(str)
            X[c] = comp.str.contains("|".join(INTERNATIONAL)).astype(float)
        elif c == "female":
            X[c] = shots.get("gender", pd.Series("male", index=shots.index)).eq("female").astype(float)
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
    wn = recency_weights(shots.loc[np_mask])

    oof = np.zeros(np_mask.sum())
    iters = []
    for tr, te in GroupKFold(5).split(Xn, yn, groups):
        # inner early-stopping split by match
        g = np.unique(groups[tr])
        rng = np.random.default_rng(SEED)
        inner = np.isin(groups[tr], rng.choice(g, len(g) // 8, replace=False))
        a, v = tr[~inner], tr[inner]
        dtr = lgb.Dataset(Xn.iloc[a], yn[a], weight=wn[a])
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

    # Temporal generalisation: train on matches before July 2022, test on the 2022-2024 men's tournaments
    recent = shots.loc[np_mask, "label"].isin(recent_labels).to_numpy()
    older = (shots.loc[np_mask, "date"] < "2022-07-01").to_numpy() if "date" in shots else ~recent
    if recent.any() and older.any():
        # The round count must not come from the CV above, whose folds include the hold-out shots:
        # early-stop on a held-back eighth of the pre-July-2022 matches instead.
        og = groups[older]
        rng = np.random.default_rng(SEED)
        held = np.isin(og, rng.choice(np.unique(og), len(np.unique(og)) // 8, replace=False))
        w_old = recency_weights(shots.loc[np_mask][older], ref="2022-07-01")
        Xo, yo = Xn[older], yn[older]
        dtr = lgb.Dataset(Xo[~held], yo[~held], weight=w_old[~held])
        b = lgb.train(PARAMS, dtr, 5000, valid_sets=[lgb.Dataset(Xo[held], yo[held], reference=dtr)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pr = b.predict(Xn[recent], num_iteration=b.best_iteration)
        okr = recent & ok
        metrics["temporal_recent_test"] = {
            "SportPulse xG (LightGBM)": _metrics(yn[okr], b.predict(Xn[okr], num_iteration=b.best_iteration)),
            "StatsBomb xG (commercial)": _metrics(yn[okr], sb[okr]),
            "n": int(okr.sum()), "competitions": sorted(recent_labels), "trained_before": "2022-07-01"}
        del pr

    booster = lgb.train(PARAMS, lgb.Dataset(Xn, yn, weight=wn), n_iter)
    imp = pd.Series(booster.feature_importance("gain"), FEATURES)
    metrics["importance"] = (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict()
    # Calibration curve (OOF)
    bins = np.quantile(oof, np.linspace(0, 1, 13))
    idx = np.clip(np.digitize(oof, bins[1:-1]), 0, 11)
    metrics["calibration"] = [(float(oof[idx == k].mean()), float(yn[idx == k].mean()), int((idx == k).sum()))
                              for k in range(12) if (idx == k).any()]
    return booster, pd.Series(xg, index=shots.index), metrics
