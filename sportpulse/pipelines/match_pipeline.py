"""End-to-end training pipeline for the match models.

    python -m sportpulse.train matches [--trials 30]

Steps
  1. load + clean 239k real matches (2000 -> latest)
  2. leak-free sequential feature engine
  3. chronological split  train < 2023-07 <= valid < 2024-07 <= test
  4. (optional) Optuna search for the fundamentals booster on valid log-loss
  5. fit hierarchical outcome model + Poisson/Dixon-Coles goals model
  6. evaluate on the untouched test window against strong baselines
  7. refit on *all* labelled data (up to the latest match) for serving
  8. write artifacts: models, metrics, test predictions, team/league states
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sportpulse import config
from sportpulse.data.loaders import load_matches
from sportpulse.evaluation import metrics as M
from sportpulse.features import club_value
from sportpulse.features.engine import build_features, fundamental_features
from sportpulse.models.goals import GoalsModel
from sportpulse.models.match_outcome import DEFAULT_FUND_PARAMS, MatchOutcomeModel

log = logging.getLogger(__name__)
BURN_IN = "2002-07-01"   # ratings need ~2 seasons to converge


def _tune_fundamentals(feats, y, train, valid, fund_cols, divisions, trials):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tmp = MatchOutcomeModel(fund_cols, fund_cols, divisions)
    X = tmp._frame(feats, fund_cols)
    ds_params = {"feature_pre_filter": False, "max_bin": 255}
    dtr = lgb.Dataset(X[train], y[train], free_raw_data=False, params=ds_params)
    dva = lgb.Dataset(X[valid], y[valid], reference=dtr, free_raw_data=False, params=ds_params)

    def objective(trial):
        p = dict(DEFAULT_FUND_PARAMS,
                 learning_rate=0.05,
                 num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
                 min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 3000, log=True),
                 feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
                 bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                 lambda_l2=trial.suggest_float("lambda_l2", 1e-2, 100, log=True),
                 lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
                 feature_pre_filter=False)
        b = lgb.train(p, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        return b.best_score["valid_0"]["multi_logloss"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(DEFAULT_FUND_PARAMS, **study.best_params, learning_rate=0.02)
    log.info("Optuna best valid logloss %.5f params %s", study.best_value, study.best_params)
    return best, {"best_valid_logloss_lr05": study.best_value, "trials": trials,
                  "best_params": study.best_params}


def run(trials: int = 0, refresh: bool = False) -> dict:
    t0 = time.time()
    out_dir = config.MATCH_ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading matches")
    matches = load_matches(force_download=refresh)
    log.info("Building features for %d matches", len(matches))
    feats, teams, leagues = build_features(matches)
    feats["division"] = matches["division"].values
    feats = club_value.add_features(feats, matches, club_value.team_values(matches))

    y = matches["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    hg, ag = matches["hg"].to_numpy(float), matches["ag"].to_numpy(float)
    d = matches["date"]
    train = ((d >= BURN_IN) & (d < config.TRAIN_END)).to_numpy()
    valid = ((d >= config.TRAIN_END) & (d < config.VALID_END)).to_numpy()
    test = (d >= config.VALID_END).to_numpy()
    divisions = sorted(matches["division"].unique())

    # Club squad values help the fundamentals model on validation and test, but not the market-residual
    # model (as a feature or through its prior). They feed a second fundamentals booster only.
    all_cols = [c for c in feats.columns if c not in club_value.COLS]
    fund_cols = fundamental_features(all_cols)

    tuning = None
    model = MatchOutcomeModel(fund_cols, all_cols, divisions, extra_fund_features=list(club_value.COLS))
    if trials:
        model.fund_params, tuning = _tune_fundamentals(feats, y, train, valid, fund_cols,
                                                       divisions, trials)
    log.info("Fitting outcome model")
    model.fit(feats, y, train, valid)
    log.info("Fitting goals model")
    # club values were validated for the outcome model only; the goals model keeps its feature set
    goals = GoalsModel(all_cols, divisions).fit(feats, hg, ag, train, valid)

    # ── Evaluation on the held-out test window ──────────────────────────────
    pred = model.predict(feats)
    gm = goals.predict_markets(feats)
    P_mkt = feats[["mkt_p_h", "mkt_p_d", "mkt_p_a"]].to_numpy()
    has_mkt = np.isfinite(P_mkt).all(1)

    # Baselines fitted on train only
    base_rates = np.bincount(y[train], minlength=3) / train.sum()
    P_base = np.tile(base_rates, (len(y), 1))
    elo_lr = LogisticRegression(max_iter=1000).fit(
        feats.loc[train, ["elo_diff"]].to_numpy() / 400, y[train])
    P_elo = elo_lr.predict_proba(feats[["elo_diff"]].to_numpy() / 400)
    pi_x = feats[["pi_gd", "elo_diff"]].to_numpy() / [1, 400]
    pi_lr = LogisticRegression(max_iter=1000).fit(pi_x[train], y[train])
    P_pi = pi_lr.predict_proba(pi_x)
    P_goals = gm[["p_h", "p_d", "p_a"]].to_numpy()

    tm = test & has_mkt
    candidates = {
        "Base rates (train prior)": P_base,
        "Elo logistic": P_elo,
        "Pi-rating + Elo logistic": P_pi,
        "SportPulse Fundamentals (no odds)": pred["fundamentals"],
        "Poisson-DC goals model": P_goals,
        "Bookmaker market (Shin)": P_mkt,
        "SportPulse Market-Residual (production)": pred["final"],
    }
    leaderboard = []
    for name, P in candidates.items():
        s = M.summary(P[tm], y[tm])
        s["model"] = name
        leaderboard.append(s)
    leaderboard = sorted(leaderboard, key=lambda r: r["rps"])

    # Per-league breakdown for the production model and the market
    per_div = []
    for div in divisions:
        mm = tm & (matches["division"].to_numpy() == div)
        if mm.sum() < 150:
            continue
        a, b = M.summary(pred["final"][mm], y[mm]), M.summary(P_mkt[mm], y[mm])
        f = M.summary(pred["fundamentals"][mm], y[mm])
        per_div.append({"division": div, "league": config.division_label(div), "n": a["n"],
                        "rps_model": a["rps"], "rps_market": b["rps"], "rps_fund": f["rps"],
                        "acc_model": a["accuracy"], "acc_market": b["accuracy"],
                        "logloss_model": a["log_loss"], "logloss_market": b["log_loss"]})

    # Over/Under 2.5
    over = (hg + ag > 2.5).astype(float)
    p_o_mkt = feats["mkt_p_o25"].to_numpy()
    om = test & np.isfinite(p_o_mkt)
    ou = {"n": int(om.sum()),
          "logloss_model": M.binary_log_loss(gm["p_over25"].to_numpy()[om], over[om]),
          "logloss_market": M.binary_log_loss(p_o_mkt[om], over[om]),
          "logloss_base": M.binary_log_loss(np.full(om.sum(), over[train].mean()), over[om]),
          "acc_model": float(((gm["p_over25"].to_numpy()[om] > .5) == over[om]).mean()),
          "acc_market": float(((p_o_mkt[om] > .5) == over[om]).mean())}
    goal_mae = {"mae_home": float(np.abs(gm["xg_h"].to_numpy()[test] - hg[test]).mean()),
                "mae_away": float(np.abs(gm["xg_a"].to_numpy()[test] - ag[test]).mean()),
                "rho": goals.rho}

    calib = {name: {c: M.calibration_curve(P[tm], y[tm], k) for k, c in enumerate("HDA")}
             for name, P in [("production", pred["final"]), ("fundamentals", pred["fundamentals"]),
                             ("market", P_mkt)]}

    # Rolling (monthly) RPS across the test window
    tdf = pd.DataFrame({"month": d[tm].dt.to_period("M").astype(str).to_numpy(), "y": y[tm]})
    monthly = []
    for mth, idx in tdf.groupby("month").indices.items():
        ii = np.where(tm)[0][idx]
        monthly.append({"month": mth, "n": len(ii),
                        "rps_model": M.rps(pred["final"][ii], y[ii]),
                        "rps_market": M.rps(P_mkt[ii], y[ii]),
                        "rps_fund": M.rps(pred["fundamentals"][ii], y[ii])})

    importance = model.feature_importance()
    importance.to_csv(out_dir / "feature_importance.csv")

    test_pred = matches.loc[test, ["date", "division", "home", "away", "hg", "ag", "result",
                                   "odds_h", "odds_d", "odds_a", "max_h", "max_d", "max_a"]].copy()
    for k, c in enumerate("hda"):
        test_pred[f"p_{c}"] = pred["final"][test, k]
        test_pred[f"pf_{c}"] = pred["fundamentals"][test, k]
        test_pred[f"pm_{c}"] = P_mkt[test, k]
    test_pred[["xg_h", "xg_a", "p_over25", "p_btts"]] = gm.loc[test, ["xg_h", "xg_a", "p_over25",
                                                                       "p_btts"]].to_numpy()
    test_pred.to_parquet(out_dir / "test_predictions.parquet", index=False)
    betting = M.betting_report(test_pred)

    metrics = {
        "data": {"matches_total": int(len(matches)), "teams": len(teams),
                 "divisions": len(divisions),
                 "first_date": str(d.min().date()), "last_date": str(d.max().date()),
                 "train": [BURN_IN, config.TRAIN_END, int(train.sum())],
                 "valid": [config.TRAIN_END, config.VALID_END, int(valid.sum())],
                 "test": [config.VALID_END, str(d.max().date()), int(test.sum())],
                 "test_with_odds": int(tm.sum())},
        "leaderboard": leaderboard, "per_division": per_div, "over_under_25": ou,
        "validation": {"fundamentals_rps": M.rps(pred["fundamentals"][valid & has_mkt], y[valid & has_mkt]),
                       "final_rps": M.rps(pred["final"][valid & has_mkt], y[valid & has_mkt])},
        "goals": goal_mae, "betting_backtest": betting, "calibration": calib,
        "monthly": monthly, "tuning": tuning,
        "model": {"fund_iters": model.fund_iters, "resid_iters": model.resid_iters,
                  "t_fund": model.t_fund, "t_final": model.t_final,
                  "fund_params": model.fund_params},
    }

    # ── Production refit on every labelled match up to the latest date ─────
    log.info("Refitting on all data for production")
    full = (d >= BURN_IN).to_numpy()
    model.refit_full(feats, y, full, int(train.sum()))
    goals.refit_full(feats, hg, ag, full, int(train.sum()))
    model.save(out_dir)
    goals.save(out_dir)
    metrics["production"] = {"trained_through": str(d.max().date()), "n_train": int(full.sum())}

    # ── Serving state ───────────────────────────────────────────────────────
    snap = []
    for tid, st in teams.items():
        s = st.snapshot()
        s["team_id"], s["team"] = tid, tid.split("|", 1)[1]
        s["country"] = tid.split("|", 1)[0]
        snap.append(s)
    pd.DataFrame(snap).to_parquet(out_dir / "team_state.parquet", index=False)
    pd.DataFrame([{"division": k, **v.snapshot()} for k, v in leagues.items()]).to_parquet(
        out_dir / "league_state.parquet", index=False)
    recent = matches[d >= "2023-07-01"][["date", "division", "home", "away", "home_id", "away_id",
                                         "hg", "ag", "result", "hs", "as_", "hst", "ast",
                                         "hc", "ac", "odds_h", "odds_d", "odds_a"]]
    recent.to_parquet(out_dir / "recent_matches.parquet", index=False)
    # Elo history for featured teams (charts)
    elo_hist = pd.DataFrame({"date": d, "division": matches["division"],
                             "home_id": matches["home_id"], "away_id": matches["away_id"],
                             "elo_h": feats["elo_h"], "elo_a": feats["elo_a"]})
    elo_hist = elo_hist[elo_hist.division.isin(config.FEATURED_DIVISIONS) & (d >= "2015-07-01")]
    elo_hist.to_parquet(out_dir / "elo_history.parquet", index=False)

    metrics["runtime_sec"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Match pipeline done in %.0fs", metrics["runtime_sec"])
    return metrics
