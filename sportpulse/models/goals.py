"""Scoreline model: LightGBM Poisson rates + Dixon-Coles dependence.

Home and away goal rates (lambda, mu) are predicted by two LightGBM
regressors with a Poisson objective. The joint scoreline distribution uses
the Dixon & Coles (1997) correction for the under-dispersion of low scores
(0-0, 1-0, 0-1, 1-1), with the dependence parameter rho fitted by maximum
likelihood on the validation season.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from sportpulse.config import SEED

MAX_GOALS = 10

GOAL_PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=31,
                   min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, num_threads=0)


def dc_tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def markets_from_rates(lam: np.ndarray, mu: np.ndarray, rho: float) -> pd.DataFrame:
    """Vectorised derived markets for many fixtures."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], lam[:, None])           # (n, G)
    pa = poisson.pmf(g[None, :], mu[:, None])
    M = ph[:, :, None] * pa[:, None, :]                  # (n, G, G)
    M[:, 0, 0] *= 1 - lam * mu * rho
    M[:, 0, 1] *= 1 + lam * rho
    M[:, 1, 0] *= 1 + mu * rho
    M[:, 1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    M /= M.sum(axis=(1, 2), keepdims=True)
    i, j = np.meshgrid(g, g, indexing="ij")
    out = pd.DataFrame({
        "p_h": (M * (i > j)).sum((1, 2)), "p_d": (M * (i == j)).sum((1, 2)),
        "p_a": (M * (i < j)).sum((1, 2)),
        "p_over25": (M * (i + j > 2.5)).sum((1, 2)),
        "p_btts": (M * ((i > 0) & (j > 0))).sum((1, 2)),
        "xg_h": lam, "xg_a": mu,
    })
    flat = M.reshape(len(lam), -1).argmax(1)
    out["top_score"] = [f"{k // (MAX_GOALS + 1)}-{k % (MAX_GOALS + 1)}" for k in flat]
    return out


@dataclass
class GoalsModel:
    features: list[str]
    divisions: list[str]
    params: dict = field(default_factory=lambda: dict(GOAL_PARAMS))
    iters_h: int = 0
    iters_a: int = 0
    rho: float = -0.05
    booster_h: lgb.Booster | None = None
    booster_a: lgb.Booster | None = None

    def _frame(self, feats):
        X = feats.reindex(columns=[c for c in self.features if c != "division"]).astype(float)
        if "division" in self.features:
            X["division"] = pd.Categorical(feats["division"], categories=self.divisions)
        return X[self.features]

    def fit(self, feats, hg, ag, train, valid):
        X = self._frame(feats)
        boosters = []
        for target in (hg, ag):
            dtr = lgb.Dataset(X[train], target[train])
            dva = lgb.Dataset(X[valid], target[valid], reference=dtr)
            b = lgb.train(self.params, dtr, 5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            boosters.append(b)
        self.booster_h, self.booster_a = boosters
        self.iters_h, self.iters_a = (b.best_iteration for b in boosters)
        lam, mu = self.rates(feats[valid])
        x, y = hg[valid], ag[valid]

        def nll(rho):
            base = poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
            return -np.mean(base + np.log(np.clip(dc_tau(x, y, lam, mu, rho), 1e-9, None)))
        self.rho = float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x)
        return self

    def refit_full(self, feats, hg, ag, mask, n_train_ref):
        scale = mask.sum() / max(n_train_ref, 1)
        X = self._frame(feats)
        self.iters_h = max(int(self.iters_h * scale), 1)
        self.iters_a = max(int(self.iters_a * scale), 1)
        self.booster_h = lgb.train(self.params, lgb.Dataset(X[mask], hg[mask]), self.iters_h)
        self.booster_a = lgb.train(self.params, lgb.Dataset(X[mask], ag[mask]), self.iters_a)
        return self

    def rates(self, feats):
        X = self._frame(feats)
        return (self.booster_h.predict(X, num_iteration=self.iters_h),
                self.booster_a.predict(X, num_iteration=self.iters_a))

    def predict_markets(self, feats) -> pd.DataFrame:
        lam, mu = self.rates(feats)
        return markets_from_rates(lam, mu, self.rho)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.booster_h.save_model(str(path / "goals_home.lgb"))
        self.booster_a.save_model(str(path / "goals_away.lgb"))
        meta = {k: getattr(self, k) for k in
                ["features", "divisions", "params", "iters_h", "iters_a", "rho"]}
        (path / "goals_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path):
        meta = json.loads((path / "goals_meta.json").read_text())
        m = cls(**meta)
        m.booster_h = lgb.Booster(model_file=str(path / "goals_home.lgb"))
        m.booster_a = lgb.Booster(model_file=str(path / "goals_away.lgb"))
        return m
