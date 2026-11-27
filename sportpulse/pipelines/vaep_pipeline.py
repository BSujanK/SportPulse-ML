"""VAEP pipeline: action values and player ratings on StatsBomb open data.

    python -m sportpulse.train vaep

Evaluation: P(scores) / P(concedes) models trained on matches before July 2022 and tested on the
2022-24 men's tournaments, against calibrated xT and a location-only model. Production action
values are out-of-fold (2 folds by match), so no action is valued by a model that saw it.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from sportpulse import config
from sportpulse.analysis import vaep as VA
from sportpulse.data import statsbomb
from sportpulse.evaluation.metrics import binary_log_loss
from sportpulse.pipelines.analyst_pipeline import RECENT

log = logging.getLogger(__name__)
OUT = config.ARTIFACTS_DIR / "analyst"
LOC = ["type", "x", "y", "end_x", "end_y", "success"]


def _met(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": binary_log_loss(p, y), "brier": float(((p - y) ** 2).mean()), "auc": float(roc_auc_score(y, p))}


def _xt_features(X: pd.DataFrame) -> np.ndarray:
    grid = np.load(OUT / "xt_grid.npy")
    nx, ny = grid.shape

    def xt(x, y):
        return grid[np.clip((x / 120 * nx).astype(int), 0, nx - 1), np.clip((y / 80 * ny).astype(int), 0, ny - 1)]
    xs = xt(X.x.to_numpy(), X.y.to_numpy())
    xe = xt(X.end_x.fillna(X.x).to_numpy(), X.end_y.fillna(X.y).to_numpy())
    t = X.type.astype(str)
    return np.c_[xs, xe * X.success.to_numpy(), t.eq("shot").to_numpy(), t.eq("shot_penalty").to_numpy()]


def player_minutes(a: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Estimated minutes per player-match: starters from kick-off, substitutes from their first action."""
    g = a.groupby(["match_id", "team", "player"]).minute.agg(["min", "max"]).reset_index()
    end = a.groupby("match_id").minute.max().rename("end")
    g = g.merge(end, on="match_id")
    starters = set(zip(lineups.match_id, lineups.player))
    is_start = np.array([(m, p) in starters for m, p in zip(g.match_id, g.player)])
    start = np.where(is_start, 0, np.maximum(g["min"] - 2, 0))
    stop = np.where(g["end"] - g["max"] <= 8, g["end"], g["max"] + 2)   # subbed off: last action + 2
    g["minutes"] = np.clip(stop - start, 1, None)
    return g[["match_id", "team", "player", "minutes"]]


def run() -> dict:
    t0 = time.time()
    d = statsbomb.ingest()
    m = d["matches"]
    a = VA.build_actions(m, d["passes"], d["carries"], d["shots"], d["defense"])
    X, ys, yc = VA.features_labels(a)
    log.info("VAEP: %d actions", len(a))
    md = a.match_id.map(m.set_index("match_id").date)
    lab = a.match_id.map(m.set_index("match_id").label)
    test = lab.isin(RECENT).to_numpy()
    train = (pd.to_datetime(md) < pd.Timestamp("2022-07-01")).to_numpy() & ~test
    mids = np.unique(a.match_id[train])
    rng = np.random.default_rng(config.SEED)
    inner = np.isin(a.match_id.to_numpy(), rng.choice(mids, len(mids) // 10, replace=False))
    tr, va, te = np.where(train & ~inner)[0], np.where(train & inner)[0], np.where(test)[0]

    Z = _xt_features(X)
    evaluation, iters = {}, {}
    for name, y in (("scores", ys), ("concedes", yc)):
        r = {"Base rate": _met(y[te], np.full(len(te), y[tr].mean()))}
        lr = LogisticRegression(max_iter=500).fit(Z[tr][::5], y[tr][::5])
        r["xT (calibrated)"] = _met(y[te], lr.predict_proba(Z[te])[:, 1])
        b = VA.fit(X[LOC], y, tr, va)
        r["Location-only LightGBM"] = _met(y[te], b.predict(X[LOC].iloc[te], num_iteration=b.best_iteration))
        b = VA.fit(X, y, tr, va)
        r["SportPulse VAEP (LightGBM)"] = _met(y[te], b.predict(X.iloc[te], num_iteration=b.best_iteration))
        iters[name] = int(b.best_iteration * 1.1)
        evaluation[name] = r
        log.info("VAEP %s: %s", name, r["SportPulse VAEP (LightGBM)"])

    # Out-of-fold probabilities for every action
    ps, pc = np.zeros(len(a)), np.zeros(len(a))
    for f_tr, f_te in GroupKFold(2).split(X, ys, a.match_id):
        ps[f_te] = VA.fit(X, ys, f_tr, rounds=iters["scores"]).predict(X.iloc[f_te])
        pc[f_te] = VA.fit(X, yc, f_tr, rounds=iters["concedes"]).predict(X.iloc[f_te])
    v = VA.values(a, ps, pc)
    a = pd.concat([a, v], axis=1)

    # Player ratings per competition-season
    mins = player_minutes(a, d["lineups"])
    names = statsbomb.display_names(m.match_id)
    agg = a.groupby(["match_id", "team", "player"])[["vaep", "vaep_off", "vaep_def"]].sum().reset_index()
    agg["actions"] = a.groupby(["match_id", "team", "player"]).size().to_numpy()
    agg = agg.merge(mins, on=["match_id", "team", "player"])
    agg = agg.merge(m[["match_id", "label", "gender"]], on="match_id")
    pl = agg.groupby(["label", "gender", "player", "team"]).agg(
        matches=("match_id", "nunique"), minutes=("minutes", "sum"), actions=("actions", "sum"),
        vaep=("vaep", "sum"), vaep_off=("vaep_off", "sum"), vaep_def=("vaep_def", "sum")).reset_index()
    for c in ("vaep", "vaep_off", "vaep_def"):
        pl[f"{c}_p90"] = pl[c] / pl.minutes * 90
    pl["player"] = pl.player.map(lambda p: names.get(p, p))
    pl.to_parquet(OUT / "vaep_players.parquet", index=False)
    agg["player"] = agg.player.map(lambda p: names.get(p, p))
    agg.to_parquet(OUT / "vaep_player_matches.parquet", index=False)
    by_type = a.groupby("type").vaep.agg(["mean", "sum", "size"]).reset_index()
    by_type.to_parquet(OUT / "vaep_by_type.parquet", index=False)

    metrics = {"evaluation": evaluation, "iters": iters, "actions": int(len(a)),
               "test_actions": int(len(te)), "train_actions": int(len(tr) + len(va)),
               "test_competitions": sorted(RECENT), "trained_before": "2022-07-01",
               "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "vaep_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
