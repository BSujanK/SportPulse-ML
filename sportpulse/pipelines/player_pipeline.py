"""Player-model pipeline (EA FIFA 15-22, ~142k player-seasons).

All evaluations are *temporal*: models are trained on earlier editions and
scored on a later, unseen edition.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sportpulse import config
from sportpulse.data.loaders import load_players
from sportpulse.models.players import (ArchetypeModel, DevelopmentModel, ValueModel,
                                       category_levels, development_pairs, save_player_models)

log = logging.getLogger(__name__)


def _reg_metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    ss = ((y_true - y_true.mean()) ** 2).sum()
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - (err ** 2).sum() / ss), "n": int(len(y_true))}


def run(refresh: bool = False) -> dict:
    t0 = time.time()
    out = config.PLAYER_ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)
    players = load_players(force_download=refresh)
    cats = category_levels(players)
    latest = players["edition"].max()
    metrics: dict = {"data": {"player_seasons": int(len(players)),
                              "players": int(players.player_id.nunique()),
                              "editions": sorted(players.edition.unique().tolist())}}

    # ── 1. Market value ──────────────────────────────────────────────────────
    # Primary protocol: 5-fold GroupKFold by player over all editions (each
    # prediction is for a player never seen in training, in any season).
    # Stress test: forward-edition extrapolation (train <=2020, test 2022).
    pv = players[players["value_eur"].notna()].reset_index(drop=True)
    vals = pv["value_eur"].to_numpy()
    ly = np.log(vals)
    e = pv["edition"].to_numpy()
    te = e == latest

    def value_scores(pred, mask):
        ape = np.abs(pred[mask] - vals[mask]) / vals[mask]
        return {**_reg_metrics(ly[mask], np.log(pred[mask])), "median_ape": float(np.median(ape)),
                "within_10pct": float((ape <= .10).mean()), "within_25pct": float((ape <= .25).mean())}

    vm = ValueModel(cats)
    oof = vm.oof_predict(pv)
    base_cols = ["overall", "potential", "age"]
    Xb = np.c_[pv[base_cols], pv.age ** 2, pv.overall ** 2, pd.get_dummies(pv.edition).to_numpy()]
    lin_oof = np.zeros(len(pv))
    from sklearn.model_selection import GroupKFold
    for a, b in GroupKFold(5).split(Xb, ly, pv["player_id"]):
        lin_oof[b] = LinearRegression().fit(Xb[a], ly[a]).predict(Xb[b])

    fwd = ValueModel(cats).fit(pv, e <= latest - 2, e == latest - 1)
    metrics["value"] = {
        "model": {**value_scores(oof, te), "iters": vm.iters},
        "model_all_editions": value_scores(oof, np.ones(len(pv), bool)),
        "baseline_linear_log": value_scores(np.exp(lin_oof), te),
        "stress_forward_edition": value_scores(fwd.predict(pv), te),
        "split": "5-fold GroupKFold by player (all editions); scores on FIFA22 rows",
    }
    vm.refit(pv, np.ones(len(pv), bool))
    imp = pd.Series(vm.booster.feature_importance("gain"), vm.features)
    metrics["value"]["top_features"] = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
    # Out-of-fold "fair value" for the latest edition (never trained on that player).
    fair_latest = pd.Series(oof[te], index=pv.loc[te, "player_id"].to_numpy())

    # ── 2. Development: overall rating next edition ────────────────────────
    pairs = development_pairs(players[players["value_eur"].notna()]).reset_index(drop=True)
    pe = pairs["edition"].to_numpy()
    tr, va, te = pe <= latest - 3, pe == latest - 2, pe == latest - 1
    dm = DevelopmentModel(cats).fit(pairs, tr, va)
    dp = dm.predict(pairs)
    yt = pairs["overall_next"].to_numpy(float)
    persist = pairs["overall"].to_numpy(float)
    gap = (pairs["potential"] - pairs["overall"]).to_numpy(float)
    Xl = np.c_[gap, pairs["age"], pairs["age"] ** 2]
    pull = LinearRegression().fit(Xl[tr], pairs["delta"].to_numpy()[tr])
    cover = ((yt[te] >= dp["q10"].to_numpy()[te] - .5) & (yt[te] <= dp["q90"].to_numpy()[te] + .5)).mean()
    young = te & (pairs["age"].to_numpy() <= 23)
    metrics["development"] = {
        "model": {**_reg_metrics(yt[te], dp["mean"].to_numpy()[te]),
                  "interval_80_coverage": float(cover)},
        "model_u23": _reg_metrics(yt[young], dp["mean"].to_numpy()[young]),
        "baseline_persistence": _reg_metrics(yt[te], persist[te]),
        "baseline_persistence_u23": _reg_metrics(yt[young], persist[young]),
        "baseline_potential_pull": _reg_metrics(yt[te], persist[te] + pull.predict(Xl[te])),
        "split": "train pairs 15->16..19->20, early-stop 20->21, test 21->22",
    }
    # Production refit on every pair, then project the latest edition one season ahead.
    dm.refit(pairs, np.ones(len(pairs), bool))

    # ── 3. Archetypes on the latest edition (outfield) ─────────────────────
    snap = players[players["edition"] == latest].reset_index(drop=True)
    outfield = snap[snap["position_group"] != "GK"].reset_index(drop=True)
    am = ArchetypeModel().fit(outfield)
    lab, conf = am.predict(outfield)
    emb = am.embed2d(outfield)
    arche = pd.DataFrame({"player_id": outfield.player_id, "archetype": [am.names[int(l)] for l in lab],
                          "archetype_conf": conf, "style_x": emb[:, 0], "style_y": emb[:, 1]})
    metrics["archetypes"] = {"k": am.k, "bic": am.bic, "names": am.names,
                             "sizes": {am.names[c]: p["size"] for c, p in am.profiles.items()},
                             "profiles": {am.names[c]: p for c, p in am.profiles.items()}}

    # ── 4. Serving snapshot ────────────────────────────────────────────────
    snap = snap.merge(arche, on="player_id", how="left")
    snap["archetype"] = snap["archetype"].fillna("Goalkeeper")
    snap["fair_value_eur"] = snap["player_id"].map(fair_latest)
    snap["value_gap"] = snap["fair_value_eur"] / snap["value_eur"] - 1
    proj = dm.predict(snap.assign(value_eur=snap["value_eur"].fillna(snap["value_eur"].median())))
    snap["proj_overall"] = proj["mean"].round(1)
    snap["proj_low"], snap["proj_high"] = proj["q10"].round(1), proj["q90"].round(1)
    snap.to_parquet(out / "players_latest.parquet", index=False)

    # Rating history for player pages
    hist = players[["player_id", "edition", "overall", "potential", "value_eur", "club_name"]]
    hist = hist[hist.player_id.isin(snap.player_id)]
    hist.to_parquet(out / "player_history.parquet", index=False)

    save_player_models(out, vm, dm, am, cats)
    metrics["runtime_sec"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Player pipeline done in %.0fs", metrics["runtime_sec"])
    return metrics
