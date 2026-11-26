"""Hierarchical match-outcome model.

Stage 1 — *Fundamentals* (no betting data)
    LightGBM multiclass on ratings + form + league context. Used for fixtures
    where no odds are available, and as the prior when odds are missing.

Stage 2 — *Market-residual booster*
    LightGBM boosted from the market prior: the de-vigged (Shin) bookmaker
    probabilities enter as `init_score = log p_market`, so trees only learn
    *corrections* to an already very strong forecast. This is the most
    reliable way to add value on top of an efficient market and beats
    training from scratch on the same features.

Stage 3 — *Temperature calibration* fitted on the validation season.

All splits are chronological; the test window is never seen during
training, tuning, early stopping or calibration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from sportpulse.config import SEED
from sportpulse.evaluation.metrics import log_loss

MKT_COLS = ["mkt_p_h", "mkt_p_d", "mkt_p_a"]

# Optuna TPE (30 trials, 2023-24 validation log-loss) best configuration.
DEFAULT_FUND_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.02, num_leaves=33,
    min_data_in_leaf=106, feature_fraction=0.98, bagging_fraction=0.89, bagging_freq=1,
    lambda_l2=57.3, lambda_l1=3.8, verbose=-1, seed=SEED, num_threads=0,
)
DEFAULT_RESID_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.01, num_leaves=15,
    min_data_in_leaf=2000, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=20.0, verbose=-1, seed=SEED, num_threads=0,
)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda t: log_loss(softmax(logits / t), y),
                          bounds=(0.5, 2.0), method="bounded")
    return float(res.x)


@dataclass
class MatchOutcomeModel:
    fund_features: list[str]
    all_features: list[str]
    divisions: list[str]
    fund_params: dict = field(default_factory=lambda: dict(DEFAULT_FUND_PARAMS))
    resid_params: dict = field(default_factory=lambda: dict(DEFAULT_RESID_PARAMS))
    fund_iters: int = 0
    resid_iters: int = 0
    t_fund: float = 1.0
    t_final: float = 1.0
    fund_booster: lgb.Booster | None = None
    resid_booster: lgb.Booster | None = None

    # ── data helpers ────────────────────────────────────────────────────────
    def _frame(self, feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        X = feats.reindex(columns=[c for c in cols if c != "division"]).astype(float)
        if "division" in cols:
            X["division"] = pd.Categorical(feats["division"], categories=self.divisions)
        return X[cols]

    @staticmethod
    def _market(feats: pd.DataFrame) -> np.ndarray:
        return feats.reindex(columns=MKT_COLS).to_numpy(dtype=float)

    def _prior_logits(self, feats: pd.DataFrame, fund_logits: np.ndarray) -> np.ndarray:
        P = self._market(feats)
        has = np.isfinite(P).all(axis=1)
        prior = np.where(has[:, None], np.log(np.clip(P, 1e-6, 1)), fund_logits)
        return prior

    # ── training ────────────────────────────────────────────────────────────
    def fit(self, feats: pd.DataFrame, y: np.ndarray, train: np.ndarray, valid: np.ndarray,
            early_stopping: int = 150) -> "MatchOutcomeModel":
        Xf = self._frame(feats, self.fund_features)
        dtr = lgb.Dataset(Xf[train], y[train], free_raw_data=False)
        dva = lgb.Dataset(Xf[valid], y[valid], reference=dtr)
        self.fund_booster = lgb.train(
            self.fund_params, dtr, 5000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        self.fund_iters = self.fund_booster.best_iteration
        fund_logits = self.fund_booster.predict(Xf, num_iteration=self.fund_iters, raw_score=True)
        self.t_fund = fit_temperature(fund_logits[valid], y[valid])

        prior = self._prior_logits(feats, fund_logits)
        Xa = self._frame(feats, self.all_features)
        dtr = lgb.Dataset(Xa[train], y[train], init_score=prior[train], free_raw_data=False)
        dva = lgb.Dataset(Xa[valid], y[valid], init_score=prior[valid], reference=dtr)
        self.resid_booster = lgb.train(
            self.resid_params, dtr, 5000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        self.resid_iters = max(self.resid_booster.best_iteration, 1)
        final_logits = prior + self.resid_booster.predict(
            Xa, num_iteration=self.resid_iters, raw_score=True)
        self.t_final = fit_temperature(final_logits[valid], y[valid])
        return self

    def refit_full(self, feats: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
                   n_train_ref: int) -> "MatchOutcomeModel":
        """Production refit on all labelled data with the tuned iteration counts
        (scaled for the larger sample). Calibration temperatures are kept."""
        scale = mask.sum() / max(n_train_ref, 1)
        fi = max(int(self.fund_iters * scale), 1)
        ri = max(int(self.resid_iters * scale), 1)
        Xf = self._frame(feats, self.fund_features)
        self.fund_booster = lgb.train(self.fund_params, lgb.Dataset(Xf[mask], y[mask]), fi)
        self.fund_iters = fi
        fund_logits = self.fund_booster.predict(Xf, raw_score=True)
        prior = self._prior_logits(feats, fund_logits)
        Xa = self._frame(feats, self.all_features)
        self.resid_booster = lgb.train(
            self.resid_params, lgb.Dataset(Xa[mask], y[mask], init_score=prior[mask]), ri)
        self.resid_iters = ri
        return self

    # ── inference ───────────────────────────────────────────────────────────
    def predict(self, feats: pd.DataFrame) -> dict[str, np.ndarray]:
        Xf = self._frame(feats, self.fund_features)
        fund_logits = self.fund_booster.predict(Xf, num_iteration=self.fund_iters, raw_score=True)
        prior = self._prior_logits(feats, fund_logits)
        Xa = self._frame(feats, self.all_features)
        final_logits = prior + self.resid_booster.predict(
            Xa, num_iteration=self.resid_iters, raw_score=True)
        return {"fundamentals": softmax(fund_logits / self.t_fund),
                "final": softmax(final_logits / self.t_final),
                "has_market": np.isfinite(self._market(feats)).all(axis=1)}

    def feature_importance(self) -> pd.DataFrame:
        f = pd.Series(self.fund_booster.feature_importance("gain"), self.fund_features)
        r = pd.Series(self.resid_booster.feature_importance("gain"), self.all_features)
        out = pd.DataFrame({"fundamentals": f / f.sum(), "market_residual": r / max(r.sum(), 1e-9)})
        return out.fillna(0).sort_values("fundamentals", ascending=False)

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.fund_booster.save_model(str(path / "outcome_fundamentals.lgb"))
        self.resid_booster.save_model(str(path / "outcome_market_residual.lgb"))
        meta = {k: getattr(self, k) for k in
                ["fund_features", "all_features", "divisions", "fund_params", "resid_params",
                 "fund_iters", "resid_iters", "t_fund", "t_final"]}
        (path / "outcome_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "MatchOutcomeModel":
        meta = json.loads((path / "outcome_meta.json").read_text())
        m = cls(**meta)
        m.fund_booster = lgb.Booster(model_file=str(path / "outcome_fundamentals.lgb"))
        m.resid_booster = lgb.Booster(model_file=str(path / "outcome_market_residual.lgb"))
        return m
