"""Player models trained on EA Sports FIFA 15-22 attribute data.

* ValueModel        — LightGBM regression of log market value.
* DevelopmentModel  — next-season overall rating with quantile intervals.
* ArchetypeModel    — Gaussian mixture on quality-normalised playing style,
                      k chosen by BIC, archetype names derived from centroids.
* similar_players   — cosine similarity in standardised attribute space.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sportpulse.config import SEED
from sportpulse.data.loaders import DETAILED_ATTRS, FACE_ATTRS, GK_ATTRS

CAT_COLS = ["position_group", "main_position", "league_name"]
NUM_BASE = ["age", "height_cm", "weight_kg", "bmi", "overall", "potential", "pot_gap",
            "international_reputation", "weak_foot", "skill_moves", "foot_left",
            "contract_years_left", "league_level", "n_positions", "edition"]

LGB_REG = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               verbose=-1, seed=SEED, num_threads=0)


def _prep(df: pd.DataFrame, cats: dict[str, list]) -> pd.DataFrame:
    X = df.copy()
    X["pot_gap"] = X["potential"] - X["overall"]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def category_levels(df: pd.DataFrame) -> dict[str, list]:
    out = {}
    for c in CAT_COLS:
        vc = df[c].value_counts()
        out[c] = sorted(vc[vc >= 30].index.astype(str).tolist())
    return out


# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ValueModel:
    cats: dict
    features: list[str] = field(default_factory=lambda: NUM_BASE + FACE_ATTRS + DETAILED_ATTRS
                                + GK_ATTRS + CAT_COLS)
    iters: int = 0
    booster: lgb.Booster | None = None

    def fit(self, df, train, valid):
        X = _prep(df, self.cats)[self.features]
        y = np.log(df["value_eur"].to_numpy())
        dtr = lgb.Dataset(X[train], y[train])
        dva = lgb.Dataset(X[valid], y[valid], reference=dtr)
        self.booster = lgb.train(LGB_REG, dtr, 10000, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(200, verbose=False)])
        self.iters = self.booster.best_iteration
        return self

    def refit(self, df, mask):
        X = _prep(df, self.cats)[self.features]
        y = np.log(df["value_eur"].to_numpy())
        self.booster = lgb.train(LGB_REG, lgb.Dataset(X[mask], y[mask]), self.iters)
        return self

    def oof_predict(self, df, n_splits: int = 5) -> np.ndarray:
        """Group K-fold by player: every prediction is for a player whose *other*
        seasons were also excluded from training. Sets `iters` for the final refit."""
        from sklearn.model_selection import GroupKFold
        X = _prep(df, self.cats)[self.features]
        y = np.log(df["value_eur"].to_numpy())
        groups = df["player_id"].to_numpy()
        oof, iters = np.zeros(len(df)), []
        rng = np.random.default_rng(SEED)
        for tr_idx, te_idx in GroupKFold(n_splits).split(X, y, groups):
            g = np.unique(groups[tr_idx])
            inner = np.isin(groups[tr_idx], rng.choice(g, len(g) // 10, replace=False))
            a, v = tr_idx[~inner], tr_idx[inner]
            dtr = lgb.Dataset(X.iloc[a], y[a])
            b = lgb.train(LGB_REG, dtr, 20000,
                          valid_sets=[lgb.Dataset(X.iloc[v], y[v], reference=dtr)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[te_idx] = b.predict(X.iloc[te_idx], num_iteration=b.best_iteration)
            iters.append(b.best_iteration)
        self.iters = int(np.mean(iters) * 1.1)
        return np.exp(oof)

    def predict(self, df) -> np.ndarray:
        X = _prep(df, self.cats)[self.features]
        return np.exp(self.booster.predict(X, num_iteration=self.iters))


# ══════════════════════════════════════════════════════════════════════════════
def development_pairs(players: pd.DataFrame) -> pd.DataFrame:
    """Join each player-edition with the same player's next edition."""
    nxt = players[["player_id", "edition", "overall"]].copy()
    nxt["edition"] -= 1
    pairs = players.merge(nxt.rename(columns={"overall": "overall_next"}),
                          on=["player_id", "edition"], how="inner")
    pairs["delta"] = pairs["overall_next"] - pairs["overall"]
    return pairs


@dataclass
class DevelopmentModel:
    cats: dict
    features: list[str] = field(default_factory=lambda: NUM_BASE + FACE_ATTRS + DETAILED_ATTRS
                                + GK_ATTRS + CAT_COLS + ["log_value"])
    quantiles: tuple = (0.1, 0.5, 0.9)
    iters: dict = field(default_factory=dict)
    boosters: dict = field(default_factory=dict)

    def _X(self, df):
        X = _prep(df, self.cats)
        X["log_value"] = np.log(df["value_eur"])
        return X[self.features]

    def _params(self, q):
        if q is None:
            return dict(LGB_REG, objective="regression_l1")
        return dict(LGB_REG, objective="quantile", alpha=q)

    def fit(self, pairs, train, valid):
        X, y = self._X(pairs), pairs["delta"].to_numpy(float)
        for q in (None,) + self.quantiles:
            key = "mean" if q is None else f"q{int(q * 100)}"
            dtr = lgb.Dataset(X[train], y[train])
            dva = lgb.Dataset(X[valid], y[valid], reference=dtr)
            b = lgb.train(self._params(q), dtr, 6000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            self.boosters[key], self.iters[key] = b, b.best_iteration
        return self

    def refit(self, pairs, mask):
        X, y = self._X(pairs), pairs["delta"].to_numpy(float)
        for q in (None,) + self.quantiles:
            key = "mean" if q is None else f"q{int(q * 100)}"
            self.iters[key] = int(self.iters[key] * 1.1) or 1
            self.boosters[key] = lgb.train(self._params(q), lgb.Dataset(X[mask], y[mask]),
                                           self.iters[key])
        return self

    def predict(self, df) -> pd.DataFrame:
        X = self._X(df)
        out = pd.DataFrame(index=df.index)
        for key, b in self.boosters.items():
            out[key] = df["overall"].to_numpy() + b.predict(X, num_iteration=self.iters[key])
        # enforce monotone quantiles
        qs = sorted(k for k in out.columns if k.startswith("q"))
        out[qs] = np.sort(out[qs].to_numpy(), axis=1)
        return out


# ══════════════════════════════════════════════════════════════════════════════
ATTR_GROUPS = {
    "finishing": ["attacking_finishing", "mentality_positioning", "attacking_volleys"],
    "aerial": ["attacking_heading_accuracy", "power_jumping", "power_strength"],
    "pace": ["movement_acceleration", "movement_sprint_speed"],
    "dribbling": ["skill_dribbling", "movement_agility", "movement_balance", "skill_ball_control"],
    "creation": ["mentality_vision", "attacking_short_passing", "skill_long_passing", "skill_curve"],
    "crossing": ["attacking_crossing"],
    "defending": ["defending_marking_awareness", "defending_standing_tackle",
                  "defending_sliding_tackle", "mentality_interceptions"],
    "engine": ["power_stamina", "mentality_aggression"],
    "long_shots": ["power_long_shots", "power_shot_power"],
}
LINES = {"CB": "CB", "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB", "CDM": "DM", "CM": "CM",
         "CAM": "AM", "LM": "WM", "RM": "WM", "LW": "W", "RW": "W", "LF": "W", "RF": "W",
         "ST": "ST", "CF": "ST"}
# (line) -> ordered {distinguishing trait: role name}; first entry is the default role.
NAMES = {
    "CB": {"_": "Centre-Back", "aerial": "Aerial Centre-Back", "defending": "Stopper Centre-Back",
           "creation": "Ball-Playing Centre-Back", "pace": "Recovery Centre-Back"},
    "FB": {"_": "Full-Back", "crossing": "Overlapping Full-Back", "pace": "Attacking Wing-Back",
           "defending": "Defensive Full-Back", "creation": "Inverted Full-Back",
           "dribbling": "Inverted Full-Back", "engine": "Wing-Back"},
    "DM": {"_": "Holding Midfielder", "defending": "Ball-Winning Midfielder",
           "creation": "Deep-Lying Playmaker", "aerial": "Anchor Midfielder"},
    "CM": {"_": "Central Midfielder", "engine": "Box-to-Box Midfielder",
           "creation": "Deep-Lying Playmaker", "long_shots": "Goal-Threat Midfielder",
           "defending": "Ball-Winning Midfielder", "dribbling": "Advanced Playmaker"},
    "AM": {"_": "Attacking Midfielder", "creation": "Advanced Playmaker",
           "dribbling": "Creative Dribbler", "finishing": "Shadow Striker",
           "long_shots": "Shadow Striker"},
    "WM": {"_": "Wide Midfielder", "crossing": "Wide Midfielder", "pace": "Wide Runner",
           "dribbling": "Wide Playmaker", "creation": "Wide Playmaker"},
    "W": {"_": "Winger", "dribbling": "Inside Forward", "pace": "Pace Winger",
          "crossing": "Classic Winger", "finishing": "Inside Forward", "long_shots": "Inside Forward"},
    "ST": {"_": "Striker", "aerial": "Target Forward", "finishing": "Poacher",
           "pace": "Channel Runner", "creation": "False Nine", "dribbling": "False Nine",
           "long_shots": "Complete Forward"},
}


@dataclass
class ArchetypeModel:
    k: int = 0
    names: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    bic: dict = field(default_factory=dict)
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    gmm: GaussianMixture | None = None

    @staticmethod
    def _style(df):
        A = df[DETAILED_ATTRS].to_numpy(float)
        return A - A.mean(axis=1, keepdims=True)   # remove overall quality -> pure style

    def fit(self, outfield: pd.DataFrame, k_range=range(6, 15)):
        S = self._style(outfield)
        self.scaler = StandardScaler().fit(S)
        Z = self.scaler.transform(S)
        self.pca = PCA(n_components=12, random_state=SEED).fit(Z)
        E = self.pca.transform(Z)
        best = None
        for k in k_range:
            g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(E)
            self.bic[int(k)] = float(g.bic(E))
            if best is None or self.bic[int(k)] < best[0]:
                best = (self.bic[int(k)], g)
        self.gmm, self.k = best[1], best[1].n_components
        labels = self.gmm.predict(E)
        self._name_clusters(outfield, Z, labels)
        return self

    def _name_clusters(self, df, Z, labels):
        """Name each cluster by its dominant positional line and the trait that most
        distinguishes it from *other players in that same line* (so the label reflects
        style, not just position)."""
        zdf = pd.DataFrame(Z, columns=DETAILED_ATTRS)
        traits = pd.DataFrame({g: zdf[cols].mean(axis=1) for g, cols in ATTR_GROUPS.items()})
        line = df["main_position"].map(LINES).fillna("CM").to_numpy()
        line_mean = traits.groupby(line).mean()
        used: set[str] = set()
        order = sorted(range(self.k), key=lambda c: -(labels == c).sum())  # big clusters name first
        for c in order:
            m = labels == c
            lines = pd.Series(line[m]).value_counts(normalize=True)
            dom = lines.index[0]
            distinct = (traits[m].mean() - line_mean.loc[dom]).sort_values(ascending=False)
            roles = NAMES[dom]
            name = None
            for tr in distinct.index:
                cand = roles.get(tr)
                if cand and cand not in used:
                    name = cand
                    break
            if name is None:
                name = roles["_"] if roles["_"] not in used else f"{roles['_']} · {distinct.index[0].replace('_', ' ')}"
            n = 2
            base = name
            while name in used:
                name = f"{base} {n}"
                n += 1
            used.add(name)
            self.names[int(c)] = name
            self.profiles[int(c)] = {
                "traits": {k: round(float(v), 3) for k, v in traits[m].mean().items()},
                "distinguishing": {k: round(float(v), 3) for k, v in distinct.head(3).items()},
                "position_mix": df.loc[m, "position_group"].value_counts(normalize=True).round(3).to_dict(),
                "line_mix": lines.round(3).to_dict(), "size": int(m.sum())}

    def predict(self, df) -> tuple[np.ndarray, np.ndarray]:
        E = self.pca.transform(self.scaler.transform(self._style(df)))
        return self.gmm.predict(E), self.gmm.predict_proba(E).max(1)

    def embed2d(self, df) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(self._style(df)))[:, :2]


# ══════════════════════════════════════════════════════════════════════════════
def similarity_space(df: pd.DataFrame) -> np.ndarray:
    """Unit-norm standardised attribute vectors (quality + style)."""
    cols = FACE_ATTRS + DETAILED_ATTRS + GK_ATTRS
    A = df[cols].astype(float)
    A = A.fillna(A.median())
    Z = StandardScaler().fit_transform(A)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def similar_players(emb: np.ndarray, idx: int, top: int = 10, mask: np.ndarray | None = None):
    sims = emb @ emb[idx]
    sims[idx] = -np.inf
    if mask is not None:
        sims = np.where(mask, sims, -np.inf)
    order = np.argsort(-sims)[:top]
    return order, sims[order]


def save_player_models(path: Path, value: ValueModel, dev: DevelopmentModel,
                       arche: ArchetypeModel, cats: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    value.booster.save_model(str(path / "value.lgb"))
    for k, b in dev.boosters.items():
        b.save_model(str(path / f"development_{k}.lgb"))
    joblib.dump(arche, path / "archetypes.joblib")
    (path / "player_meta.json").write_text(json.dumps(
        {"cats": cats, "value_iters": value.iters, "value_features": value.features,
         "dev_iters": dev.iters, "dev_features": dev.features}, indent=2))
