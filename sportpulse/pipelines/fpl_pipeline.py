"""Premier League player forecaster pipeline (FPL data 2016-17 -> current gameweek).

    python -m sportpulse.train fpl

Split by season: train <= 2023-24 · valid 2024-25 · test 2025-26 + 2026-27 (to date).
Benchmarks: FPL's own expected points (xP) and rolling-form heuristics.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sportpulse import config
from sportpulse.data import fpl
from sportpulse.data.loaders import load_matches
from sportpulse.evaluation import metrics as M
from sportpulse.features.engine import build_features, pre_match_row
from sportpulse.models.fpl_model import CONTEXT, FPLForecaster, add_targets, player_features

log = logging.getLogger(__name__)
OUT = config.ARTIFACTS_DIR / "fpl"
TEST_SEASONS = ("2025-26", "2026-27")
VALID_SEASON = "2024-25"


def season_of(d: pd.Series) -> pd.Series:
    y = d.dt.year - (d.dt.month < 7)
    return y.astype(str) + "-" + (y + 1).astype(str).str[-2:]


def _context_rows(f: pd.DataFrame) -> tuple[dict, dict]:
    """Team-perspective context for the home and the away side of a feature row."""
    home = {"ctx_elo_for": f["elo_h"], "ctx_elo_against": f["elo_a"], "ctx_elo_diff": f["elo_diff"],
            "ctx_pi_gd": f["pi_gd"], "ctx_xg_for": f["xh_l"], "ctx_xg_against": f["xa_l"],
            "ctx_team_gf": f["h_l_gf"], "ctx_team_ga": f["h_l_ga"], "ctx_opp_gf": f["a_l_gf"],
            "ctx_opp_ga": f["a_l_ga"], "ctx_team_sot": f["h_l_stf"], "ctx_opp_sot": f["a_l_stf"],
            "ctx_team_pts": f["h_s_pts"], "ctx_opp_pts": f["a_s_pts"], "ctx_win_p": f["elo_p_home"]}
    away = {"ctx_elo_for": f["elo_a"], "ctx_elo_against": f["elo_h"], "ctx_elo_diff": -f["elo_diff"],
            "ctx_pi_gd": -f["pi_gd"], "ctx_xg_for": f["xa_l"], "ctx_xg_against": f["xh_l"],
            "ctx_team_gf": f["a_l_gf"], "ctx_team_ga": f["a_l_ga"], "ctx_opp_gf": f["h_l_gf"],
            "ctx_opp_ga": f["h_l_ga"], "ctx_team_sot": f["a_l_stf"], "ctx_opp_sot": f["h_l_stf"],
            "ctx_team_pts": f["a_s_pts"], "ctx_opp_pts": f["h_s_pts"], "ctx_win_p": 1 - f["elo_p_home"]}
    return home, away


def build_context(matches, feats) -> pd.DataFrame:
    e0 = matches["division"].eq("E0").to_numpy()
    m, f = matches[e0], feats[e0]
    season = season_of(m["date"]).to_numpy()
    h, a = _context_rows(f)
    H = pd.DataFrame(h).assign(season=season, team=m["home"].to_numpy(), opponent=m["away"].to_numpy(),
                               was_home=True)
    A = pd.DataFrame(a).assign(season=season, team=m["away"].to_numpy(), opponent=m["home"].to_numpy(),
                               was_home=False)
    ctx = pd.concat([H, A], ignore_index=True)
    return ctx.drop_duplicates(["season", "team", "opponent", "was_home"], keep="last")


def infer_teams(df: pd.DataFrame) -> pd.Series:
    """A player's team in a fixture = the opponent of the players on the other side."""
    opp = (df.groupby(["season", "fixture", "was_home"])["opponent_team"]
             .agg(lambda x: x.mode().iat[0]))
    key = pd.MultiIndex.from_arrays([df["season"], df["fixture"], ~df["was_home"]])
    tid = opp.reindex(key).to_numpy()
    names = {s: fpl.team_names(s) for s in df["season"].unique()}
    return pd.Series([names[s].get(t) for s, t in zip(df["season"], tid)], index=df.index)


def _future_rows(teams, leagues, players_raw: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """One synthetic row per current-squad player for their next fixture(s)."""
    if upcoming.empty:
        return pd.DataFrame()
    nxt = upcoming[upcoming["event"] == upcoming["event"].min()]
    names = fpl.team_names(fpl.current_season())
    ctx_rows = []
    for fx in nxt.itertuples():
        hid, aid = f"England|{fx.home}", f"England|{fx.away}"
        if hid not in teams or aid not in teams:
            continue
        r = pre_match_row(teams[hid].snapshot(), teams[aid].snapshot(),
                          leagues["E0"].snapshot(), pd.Timestamp(fx.date))
        h, a = _context_rows(pd.Series(r))
        for side, team, opp, c in ((True, fx.home, fx.away, h), (False, fx.away, fx.home, a)):
            ctx_rows.append({"team": team, "opponent": opp, "was_home": side, "kickoff": fx.kickoff_time,
                             "event": fx.event, **{k: float(v) for k, v in c.items()}})
    ctx = pd.DataFrame(ctx_rows)
    pr = players_raw.assign(team=players_raw["team"].map(names))
    fut = pr.merge(ctx, on="team", how="inner")
    out = pd.DataFrame({
        "player_code": fut["code"].astype("int64"), "name": fut["web_name"], "team": fut["team"],
        "opponent": fut["opponent"], "was_home": fut["was_home"],
        "kickoff": pd.to_datetime(fut["kickoff"], utc=True).dt.tz_localize(None),
        "position": fut["element_type"].map(fpl.POS), "season": fpl.current_season(),
        "round": fut["event"], "status": fut.get("status"), "news": fut.get("news"),
        "chance_next": pd.to_numeric(fut.get("chance_of_playing_next_round"), errors="coerce"),
        "now_cost": fut.get("now_cost"), "is_future": True,
        "xP": pd.to_numeric(fut.get("ep_next"), errors="coerce"),
        **{c: fut[c] for c in CONTEXT}})
    return out


TEAM_XG = ["tm_xg6", "tm_xga6", "opp_xg6", "opp_xga6", "tm_goals6", "opp_goals_against6"]


def add_team_xg_features(full: pd.DataFrame) -> pd.DataFrame:
    """Team- and opponent-level rolling xG for/against from the players' own FPL xG (lagged).

    Team xG in a fixture = sum of its players' xG; team xGA = max of its players' xG conceded
    (the goalkeeper plays the full match). Rolling means over the previous 6 fixtures.
    """
    g = full.groupby(["team", "kickoff"], sort=False)
    tf = pd.DataFrame({
        "xg": g["expected_goals"].sum(min_count=1), "xga": g["expected_goals_conceded"].max(),
        "goals": g["goals_scored"].sum(min_count=1), "conceded": g["goals_conceded"].max(),
        "opponent": g["opponent"].first()}).reset_index().sort_values(["team", "kickoff"])
    roll = lambda c: tf.groupby("team")[c].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())  # noqa: E731
    tf["tm_xg6"], tf["tm_xga6"] = roll("xg"), roll("xga")
    tf["tm_goals6"], tf["tm_conceded6"] = roll("goals"), roll("conceded")
    opp = tf[["team", "kickoff", "tm_xg6", "tm_xga6", "tm_conceded6"]].rename(
        columns={"team": "opponent", "tm_xg6": "opp_xg6", "tm_xga6": "opp_xga6", "tm_conceded6": "opp_goals_against6"})
    tf = tf.merge(opp, on=["opponent", "kickoff"], how="left")
    out = full[["team", "kickoff"]].merge(tf[["team", "kickoff"] + TEAM_XG], on=["team", "kickoff"], how="left")
    return out[TEAM_XG].set_axis(full.index)


def _top_xi(df: pd.DataFrame, col: str) -> float:
    """Mean actual points of the 11 highest-ranked players per gameweek."""
    return float(df.groupby(["season", "round"]).apply(
        lambda g: g.nlargest(11, col)["total_points"].mean(), include_groups=False).mean())


def run(refresh: bool = False, trials: int = 0) -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log.info("Match engine for fixture context")
    matches = load_matches(force_download=refresh)
    feats, teams, leagues = build_features(matches)
    ctx = build_context(matches, feats)

    log.info("Loading FPL player data")
    df = fpl.load_player_gameweeks(force=refresh)
    df["team"] = infer_teams(df)
    df["is_future"] = False

    players_raw = pd.read_csv(fpl.FPL_DIR / fpl.current_season() / "players_raw.csv")
    fut = _future_rows(teams, leagues, players_raw, fpl.upcoming_fixtures(refresh))

    hist = df.merge(ctx, on=["season", "team", "opponent", "was_home"], how="left")
    ctx_cov = float(hist["ctx_elo_for"].notna().mean())
    log.info("Context join coverage %.3f", ctx_cov)
    full = pd.concat([hist, fut], ignore_index=True, sort=False)
    full = full.sort_values(["player_code", "kickoff"], kind="mergesort").reset_index(drop=True)
    full = add_targets(full)
    for c in ["minutes", "total_points", "goals_scored", "assists"]:
        full.loc[full["is_future"].astype(bool), c] = np.nan
    full.loc[full["is_future"].astype(bool), ["played", "played60", "scored", "assisted"]] = np.nan

    log.info("Player features for %d rows", len(full))
    F = player_features(full)
    F[CONTEXT] = full[CONTEXT].to_numpy(float)
    F[TEAM_XG] = add_team_xg_features(full).to_numpy(float)
    F["xg_share6"] = (F["expected_goals_r6"] / F["tm_xg6"].replace(0, np.nan)).clip(upper=1)
    F["xa_share6"] = (F["expected_assists_r6"] / F["tm_xg6"].replace(0, np.nan)).clip(upper=1)
    # FPL's own pre-deadline expected points: carries injury / team news, used like a market prior
    F["fpl_xp"] = full["xP"].to_numpy(float)
    feature_cols = [c for c in F.columns]
    F["position"] = full["position"]

    is_fut = full["is_future"].astype(bool).to_numpy()
    s = full["season"].to_numpy()
    lab = ~is_fut
    train = lab & (s < VALID_SEASON) & (s >= "2016-17")
    valid = lab & (s == VALID_SEASON)
    test = lab & np.isin(s, TEST_SEASONS)

    # Production model: statistics + fixture context + FPL's own pre-deadline expected points (which carry
    # injury and team news). The archived xP equals the live `ep_this` snapshot exactly, and the live `ep_next`
    # has the same distribution, so the feature is served consistently. The archive's definition changed in
    # 2025-26 (many more zeros); the test window uses the current definition and still improves.
    # A statistics-only model is kept as the comparison row.
    stats_cols = [c for c in feature_cols if c != "fpl_xp"]
    from sportpulse.models.fpl_model import tune_head
    head_params = {}
    if trials:
        for head in ("p_play", "pts_if_played", "points"):
            head_params[head] = tune_head(F, full, train, valid, feature_cols, head, trials)
            log.info("Tuned %s: %s", head, head_params[head])
    model = FPLForecaster(feature_cols, head_params=head_params).fit(F, full, train, valid)
    w = model.fit_blend(F, full, valid)
    log.info("Direct/hurdle blend weight on direct head: %.2f", w)
    P = model.predict(F)
    stats_model = FPLForecaster(stats_cols).fit(F, full, train, valid)
    stats_model.fit_blend(F, full, valid)
    P_stats = stats_model.predict(F)["points"]

    # ── evaluation ──────────────────────────────────────────────────────────
    T = full[test].copy()
    T["pred"] = P.loc[test, "points"].to_numpy()
    T["pred_direct"] = P.loc[test, "direct"].to_numpy()
    T["pred_hurdle"] = P.loc[test, "hurdle"].to_numpy()
    T["pred_stats"] = P_stats[test].to_numpy()
    T["form6"] = F.loc[test, "total_points_r6"].fillna(0).to_numpy()
    T["form_ewm"] = F.loc[test, "total_points_ewm"].fillna(0).to_numpy()
    for k in ("p_60", "p_goal", "p_assist"):
        T[k] = P.loc[test, k].to_numpy()
    has_xp = T["xP"].notna()

    def reg(pred, mask=slice(None)):
        y, p = T["total_points"].to_numpy()[mask], pred[mask]
        return {"mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
                "spearman": float(spearmanr(y, p).statistic)}

    xm = has_xp.to_numpy()
    leaderboard = [
        {"model": "SportPulse LightGBM (production)", **reg(T["pred"].to_numpy(), xm),
         "top_xi": _top_xi(T[xm], "pred")},
        {"model": "Direct points head only", **reg(T["pred_direct"].to_numpy(), xm),
         "top_xi": _top_xi(T[xm], "pred_direct")},
        {"model": "Hurdle head only (P(plays) × points if plays)", **reg(T["pred_hurdle"].to_numpy(), xm),
         "top_xi": _top_xi(T[xm], "pred_hurdle")},
        {"model": "Statistics only, no FPL xP (previous production)", **reg(T["pred_stats"].to_numpy(), xm),
         "top_xi": _top_xi(T[xm], "pred_stats")},
        {"model": "FPL official xP", **reg(T["xP"].to_numpy(), xm), "top_xi": _top_xi(T[xm], "xP")},
        {"model": "Form (last 6 mean)", **reg(T["form6"].to_numpy(), xm), "top_xi": _top_xi(T[xm], "form6")},
        {"model": "Form (EWM)", **reg(T["form_ewm"].to_numpy(), xm), "top_xi": _top_xi(T[xm], "form_ewm")},
        {"model": "Oracle (actual points)", "mae": 0.0, "rmse": 0.0, "spearman": 1.0,
         "top_xi": _top_xi(T[xm], "total_points")},
    ]
    # Gameweek-level ranking quality among players who have been playing
    active = T[xm & (F.loc[test, "minutes_r6"].to_numpy() > 0)]
    gw_rho = {}
    for col in ("pred", "pred_stats", "xP", "form6"):
        gw_rho[col] = float(active.groupby(["season", "round"]).apply(
            lambda g: spearmanr(g["total_points"], g[col]).statistic, include_groups=False).mean())

    cls = {}
    for k, col, base_col in (("p_60", "played60", "played60_r6"), ("p_goal", "scored", None),
                             ("p_assist", "assisted", None)):
        yk = T[col].to_numpy()
        pk = np.clip(T[k].to_numpy(), 1e-4, 1 - 1e-4)
        prior = np.full(len(yk), full.loc[train, col].mean())
        r = {"logloss": M.binary_log_loss(pk, yk), "logloss_prior": M.binary_log_loss(prior, yk),
             "brier": float(((pk - yk) ** 2).mean()), "base_rate": float(yk.mean())}
        if base_col:
            b = np.clip(F.loc[test, base_col].fillna(0).to_numpy(), 0.02, 0.98)
            r["logloss_form"] = M.binary_log_loss(b, yk)
        from sklearn.metrics import roc_auc_score
        r["auc"] = float(roc_auc_score(yk, pk))
        cls[k] = r

    by_season = {}
    for sn in TEST_SEASONS:
        mm = (T["season"] == sn).to_numpy() & xm
        if mm.sum():
            by_season[sn] = {"rows": int(mm.sum()), "gameweeks": int(T.loc[mm, "round"].nunique()),
                             "model": reg(T["pred"].to_numpy(), mm), "xP": reg(T["xP"].to_numpy(), mm),
                             "top_xi_model": _top_xi(T[mm], "pred"), "top_xi_xP": _top_xi(T[mm], "xP")}

    weekly = (T[xm].groupby(["season", "round"]).apply(lambda g: pd.Series({
        "model": g.nlargest(11, "pred")["total_points"].mean(),
        "model_stats": g.nlargest(11, "pred_stats")["total_points"].mean(),
        "xP": g.nlargest(11, "xP")["total_points"].mean(),
        "form": g.nlargest(11, "form6")["total_points"].mean()}), include_groups=False).reset_index())

    # ── production refit + next-gameweek projections ───────────────────────
    model.refit(F, full, lab & (s >= "2016-17"), int(train.sum()))
    model.save(OUT)
    Pf = model.predict(F[is_fut]) if is_fut.any() else pd.DataFrame()
    nxt = full[is_fut].copy()
    if len(nxt):
        avail = (nxt["chance_next"].fillna(100) / 100).to_numpy()
        avail = np.where(nxt["status"].isin(["i", "s", "u", "n"]) & nxt["chance_next"].isna(), 0, avail)
        nxt["xpts_raw"] = Pf["points"].to_numpy()
        nxt["availability"] = avail
        # FPL's ep_next (a model input) already prices in availability, so the flag is shown, not applied again
        nxt["xpts"] = nxt["xpts_raw"]
        for k in ("p_60", "p_goal", "p_assist"):
            nxt[k] = Pf[k].to_numpy()
        nxt["price"] = nxt["now_cost"] / 10
        nxt["form6"] = F.loc[is_fut, "total_points_r6"].to_numpy()
        nxt["xg_r6"] = F.loc[is_fut, "expected_goals_r6"].to_numpy()
        nxt["xa_r6"] = F.loc[is_fut, "expected_assists_r6"].to_numpy()
        nxt["minutes_r6"] = F.loc[is_fut, "minutes_r6"].to_numpy()
        nxt["value_per_m"] = nxt["xpts"] / nxt["price"]
        keep = ["player_code", "name", "team", "opponent", "was_home", "kickoff", "round", "position",
                "price", "xpts", "xpts_raw", "availability", "p_60", "p_goal", "p_assist", "form6",
                "xg_r6", "xa_r6", "minutes_r6", "value_per_m", "status", "news", "ctx_win_p",
                "ctx_xg_for", "ctx_xg_against"]
        nxt[keep].sort_values("xpts", ascending=False).to_parquet(OUT / "next_gameweek.parquet", index=False)

    tcols = ["season", "round", "kickoff", "name", "team", "opponent", "position", "total_points",
             "pred", "pred_stats", "xP", "form6", "minutes", "p_60", "p_goal", "p_assist"]
    T[tcols].to_parquet(OUT / "test_predictions.parquet", index=False)
    weekly.to_parquet(OUT / "weekly_top_xi.parquet", index=False)
    imp = model.importance().head(30)

    last = full.loc[lab & (s == fpl.current_season()), ["round", "kickoff"]]
    metrics = {
        "data": {"rows": int(lab.sum()), "players": int(full.loc[lab, "player_code"].nunique()),
                 "seasons": sorted(full.loc[lab, "season"].unique().tolist()),
                 "latest_gameweek": int(last["round"].max()) if len(last) else None,
                 "latest_kickoff": str(last["kickoff"].max()) if len(last) else None,
                 "next_gameweek": int(nxt["round"].min()) if len(nxt) else None,
                 "context_join_coverage": ctx_cov,
                 "train_rows": int(train.sum()), "valid_rows": int(valid.sum()), "test_rows": int(test.sum())},
        "leaderboard": leaderboard, "gw_spearman_active": gw_rho, "classifiers": cls,
        "by_season": by_season, "iters": model.iters, "blend_w_direct": model.blend_w,
        "head_params": model.head_params,
        "importance": imp.round(4).to_dict(), "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("FPL pipeline done in %.0fs", metrics["runtime_sec"])
    return metrics
