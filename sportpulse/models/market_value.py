"""Real-world player valuation on Transfermarkt data.

Three LightGBM models, each with a chronological train / validation / test split:

* forecast   next Transfermarkt valuation, given everything known on that date
             (trained on the change in log value; baseline = persistence)
* fair_value value from performance, age, club and league, without the player's own valuation history
             (club and league value levels still come from Transfermarkt valuations)
             (for scouting players without a reliable market value)
* fee        actual transfer fee paid (baseline = the Transfermarkt value before the move)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sportpulse.config import SEED

CATS = {"position": ["Goalkeeper", "Defender", "Midfield", "Attack"],
        "foot": ["right", "left", "both"]}
PARAMS = dict(objective="huber", alpha=0.9, learning_rate=0.03, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              cat_smooth=20, verbose=-1, seed=SEED, num_threads=0)


def metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    y_log, p_log = np.asarray(y_log, float), np.asarray(p_log, float)
    e = p_log - y_log
    ape = np.abs(np.exp(e) - 1)
    ss = ((y_log - y_log.mean()) ** 2).sum()
    return {"mae_log": float(np.abs(e).mean()), "median_ape": float(np.median(ape)),
            "within_25pct": float((ape <= 0.25).mean()), "r2_log": float(1 - (e ** 2).sum() / ss),
            "spearman": float(spearmanr(y_log, p_log).statistic), "n": int(len(y_log))}


@dataclass
class ValueBooster:
    features: list[str]
    cats: list[str] = field(default_factory=list)
    levels: dict = field(default_factory=dict)
    params: dict = field(default_factory=lambda: dict(PARAMS))
    booster: lgb.Booster | None = None
    iters: int = 0
    offset_col: str | None = None        # predict y - offset (e.g. previous log value)

    def X(self, F: pd.DataFrame) -> pd.DataFrame:
        X = F.reindex(columns=self.columns).copy()
        for c in self.cats:
            X[c] = pd.Categorical(F[c].astype("string"), categories=self.levels[c])
        return X

    @property
    def columns(self) -> list[str]:
        return self.features + [c for c in self.cats if c not in self.features]

    def _off(self, F):
        return F[self.offset_col].to_numpy(float) if self.offset_col else 0.0

    def fit(self, F, y, train, valid):
        for c in self.cats:
            if c not in self.levels:
                vc = F.loc[train, c].astype("string").value_counts()
                self.levels[c] = sorted(vc[vc >= 30].index.tolist())
        X, t = self.X(F), np.asarray(y, float) - self._off(F)
        dtr = lgb.Dataset(X[train], t[train], categorical_feature=self.cats or "auto")
        dva = lgb.Dataset(X[valid], t[valid], reference=dtr)
        self.booster = lgb.train(self.params, dtr, 8000, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(200, verbose=False)])
        self.iters = self.booster.best_iteration
        return self

    def refit(self, F, y, mask, n_ref):
        X, t = self.X(F), np.asarray(y, float) - self._off(F)
        self.iters = max(int(self.iters * mask.sum() / max(n_ref, 1)), 50)
        self.booster = lgb.train(self.params, lgb.Dataset(X[mask], t[mask], categorical_feature=self.cats or "auto"),
                                 self.iters)
        return self

    def predict(self, F) -> np.ndarray:
        return self.booster.predict(self.X(F), num_iteration=self.iters) + self._off(F)

    def importance(self) -> dict:
        s = pd.Series(self.booster.feature_importance("gain"), self.columns)
        return (s / s.sum()).sort_values(ascending=False).round(4).to_dict()

    def save(self, path: Path, name: str):
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / f"{name}.lgb"))
        (path / f"{name}_meta.json").write_text(json.dumps(
            {"features": self.features, "cats": self.cats, "levels": self.levels, "iters": self.iters,
             "offset_col": self.offset_col}))

    @classmethod
    def load(cls, path: Path, name: str) -> "ValueBooster":
        m = json.loads((path / f"{name}_meta.json").read_text())
        b = cls(m["features"], m["cats"], m["levels"], iters=m["iters"], offset_col=m["offset_col"])
        b.booster = lgb.Booster(model_file=str(path / f"{name}.lgb"))
        return b
