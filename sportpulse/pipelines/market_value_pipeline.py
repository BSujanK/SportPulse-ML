"""Transfermarkt valuation pipeline: next-valuation forecast, fair value, transfer fees.

    python -m sportpulse.train value
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from sportpulse import config
from sportpulse.data import transfermarkt as T
from sportpulse.models.market_value import PARAMS, ValueBooster, metrics

log = logging.getLogger(__name__)
OUT = config.ARTIFACTS_DIR / "market_value"
START, TRAIN_END, VALID_END = "2013-07-01", "2024-07-01", "2025-07-01"
LAGS = ["prev_log_value", "days_since_value", "prev_change", "change_1y", "peak_log_value", "below_peak",
        "n_valuations"]
CAT_COLS = ["position", "sub_position", "foot", "league"]
# Optuna TPE (20 trials, validation 2024-25) for the next-valuation model: validation and test both improved.
# The same search did not help the fair-value or fee models, which keep the defaults.
FORECAST_PARAMS = dict(num_leaves=120, min_data_in_leaf=56, feature_fraction=0.735, lambda_l2=26.1, alpha=0.341)


# Career and 730-day top-5-league, UEFA and national-team minutes and goal involvements. Validated for the
# fair-value model only (validation 0.5474 -> 0.5343, test 0.5874 -> 0.5821 MAE log), so only it uses them.
HISTORY = [f"{c}_{t}" for t in ("career", "730") for c in ("min_top5", "min_uefa", "min_national", "ga_top5", "ga_uefa")]


def competition_history(A: T.AsOf, pid, dates) -> pd.DataFrame:
    day = T._day(pd.Series(pd.to_datetime(dates)))
    out = {}
    for tag, w in (("career", None), ("730", 730)):
        sx = A.apps.sums(np.asarray(pid, np.int64), day, w)
        for c in ("min_top5", "min_uefa", "min_national", "ga_top5", "ga_uefa"):
            out[f"{c}_{tag}"] = sx[c].to_numpy()
    return pd.DataFrame(out)


def _splits(dates: pd.Series):
    d = pd.to_datetime(dates)
    return (((d >= START) & (d < TRAIN_END)).to_numpy(), ((d >= TRAIN_END) & (d < VALID_END)).to_numpy(),
            (d >= VALID_END).to_numpy())


def _age_band(age):
    return pd.cut(age, [0, 20, 23, 26, 29, 32, 60], labels=False).fillna(-1).astype(int)


def valuation_frame(A: T.AsOf) -> pd.DataFrame:
    v = T.table("player_valuations")
    v = v[v.date >= START].reset_index(drop=True)
    F = A.build(v.player_id.to_numpy(), v.date, club=v.current_club_id.fillna(-1).to_numpy(np.int64),
                comp=v.player_club_domestic_competition_id)
    F["league"] = v.player_club_domestic_competition_id.astype("string").to_numpy()
    F[HISTORY] = competition_history(A, v.player_id.to_numpy(), v.date).to_numpy()
    F["y"] = np.log(v.market_value_in_eur.clip(lower=10_000).to_numpy(float))
    F["date"], F["player_id"] = v.date.to_numpy(), v.player_id.to_numpy()
    return F


def fee_frame(A: T.AsOf) -> pd.DataFrame:
    t = T.table("transfers")
    end = T.table("player_valuations").date.max() + pd.Timedelta(days=30)
    t = t[(t.transfer_fee > 0) & (t.transfer_date >= START) & (t.transfer_date <= end)]
    t = t.drop_duplicates(["player_id", "transfer_date"]).reset_index(drop=True)
    clubs = T.table("clubs").set_index("club_id")["domestic_competition_id"]
    from_lg, to_lg = t.from_club_id.map(clubs), t.to_club_id.map(clubs)
    F = A.build(t.player_id.to_numpy(), t.transfer_date, club=t.from_club_id.to_numpy(np.int64), comp=from_lg)
    day = T._day(t.transfer_date)
    to = A.club_strength(t.to_club_id.to_numpy(np.int64), day, "to_").reset_index(drop=True)
    F = pd.concat([F, to], axis=1)
    F["to_league_value_index"] = A.league_index(to_lg, day)
    F["league"], F["to_league"] = from_lg.astype("string").to_numpy(), to_lg.astype("string").to_numpy()
    F["y"] = np.log(t.transfer_fee.to_numpy(float))
    F["date"], F["player_id"] = t.transfer_date.to_numpy(), t.player_id.to_numpy()
    F["from_club"], F["to_club"] = t.from_club_name.to_numpy(), t.to_club_name.to_numpy()
    return F


def run() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    A = T.AsOf()
    log.info("Valuation frame")
    V = valuation_frame(A)
    num = [c for c in V.columns if c not in CAT_COLS + HISTORY + ["y", "date", "player_id"]]
    tr, va, te = _splits(V["date"])
    res = {"data": {"valuations": int(len(V)), "players": int(V.player_id.nunique()),
                    "first": str(pd.Timestamp(V.date.min()).date()), "last": str(pd.Timestamp(V.date.max()).date()),
                    "train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())}}

    # ── 1. Next-valuation forecast ──────────────────────────────────────────────────────
    has = V["prev_log_value"].notna().to_numpy()
    fc = ValueBooster(num, CAT_COLS, params=dict(PARAMS, **FORECAST_PARAMS),
                      offset_col="prev_log_value").fit(V, V.y, tr & has, va & has)
    m = te & has
    y = V.y.to_numpy()
    pers = V["prev_log_value"].to_numpy()
    band = _age_band(V["age"])
    trend = pd.Series(y - pers)[tr & has].groupby(band[tr & has]).mean()
    p_trend = pers + band.map(trend).fillna(0).to_numpy()
    p_fc = fc.predict(V)
    chg = np.abs(y - pers) > np.log(1.05)
    res["forecast"] = {
        "leaderboard": [dict(model="SportPulse forecast (LightGBM)", **metrics(y[m], p_fc[m])),
                        dict(model="Persistence + age trend", **metrics(y[m], p_trend[m])),
                        dict(model="Persistence (last Transfermarkt value)", **metrics(y[m], pers[m]))],
        "direction_accuracy": float((np.sign(p_fc - pers) == np.sign(y - pers))[m & chg].mean()),
        "direction_n": int((m & chg).sum()), "iters": fc.iters, "importance": dict(list(fc.importance().items())[:15])}
    log.info("forecast %s", res["forecast"]["leaderboard"][0])

    # ── 2. Fair value, no Transfermarkt input for the player ───────────────────────────
    perf = [c for c in num if c not in LAGS] + HISTORY
    fv = ValueBooster(perf, CAT_COLS).fit(V, V.y, tr, va)
    p_fv = fv.predict(V)
    key = pd.DataFrame({"l": V.league.fillna("NA").astype(str), "p": V.position.fillna("NA").astype(str), "a": band})
    med = pd.Series(y[tr]).groupby([key.l[tr], key.p[tr], key.a[tr]]).median()
    p_grp = med.reindex(pd.MultiIndex.from_frame(key)).to_numpy()
    p_grp = np.where(np.isfinite(p_grp), p_grp, np.median(y[tr]))
    res["fair_value"] = {
        "leaderboard": [dict(model="SportPulse fair value (LightGBM)", **metrics(y[te], p_fv[te])),
                        dict(model="League × position × age median", **metrics(y[te], p_grp[te]))],
        "iters": fv.iters, "importance": dict(list(fv.importance().items())[:15])}
    log.info("fair value %s", res["fair_value"]["leaderboard"][0])

    # ── 3. Transfer fees ───────────────────────────────────────────────────────────────
    log.info("Fee frame")
    Fe = fee_frame(A)
    fnum = [c for c in Fe.columns if c not in CAT_COLS + ["to_league", "y", "date", "player_id", "from_club", "to_club"]]
    ftr, fva, fte = _splits(Fe["date"])
    fh = Fe["prev_log_value"].notna().to_numpy()
    fee = ValueBooster(fnum, CAT_COLS + ["to_league"]).fit(Fe, Fe.y, ftr, fva)
    p_fee = fee.predict(Fe)
    yf, tmv = Fe.y.to_numpy(), Fe["prev_log_value"].to_numpy()
    a, b = np.polyfit(tmv[ftr & fh], yf[ftr & fh], 1)[::-1]
    mf = fte & fh
    res["fees"] = {
        "leaderboard": [dict(model="SportPulse fee model (LightGBM)", **metrics(yf[mf], p_fee[mf])),
                        dict(model="Transfermarkt value, calibrated", **metrics(yf[mf], a + b * tmv[mf])),
                        dict(model="Transfermarkt value", **metrics(yf[mf], tmv[mf]))],
        "n_train": int(ftr.sum()), "n_valid": int(fva.sum()), "iters": fee.iters,
        "importance": dict(list(fee.importance().items())[:15])}
    log.info("fees %s", res["fees"]["leaderboard"][0])
    big = mf & (yf >= np.log(20e6))
    res["fees"]["big_moves"] = {"n": int(big.sum()),
                                "SportPulse": metrics(yf[big], p_fee[big]),
                                "Transfermarkt value": metrics(yf[big], tmv[big])}
    tt = Fe[mf].assign(fee=np.exp(yf[mf]), pred=np.exp(p_fee[mf]), tm=np.exp(tmv[mf]))
    names = T.table("players").set_index("player_id")["name"]
    tt["player"] = tt.player_id.map(names)
    tt.sort_values("fee", ascending=False).head(400)[
        ["date", "player", "from_club", "to_club", "age", "fee", "pred", "tm"]].to_parquet(OUT / "test_fees.parquet",
                                                                                        index=False)

    # ── Production refit on all data and current-player outputs ────────────────────────
    allv = (tr | va | te)
    fc.refit(V, V.y, allv & has, (tr & has).sum())
    fv.refit(V, V.y, allv, tr.sum())
    fee.refit(Fe, Fe.y, ftr | fva | fte, ftr.sum())
    for mdl, n in ((fc, "forecast"), (fv, "fair_value"), (fee, "fee")):
        mdl.save(OUT, n)

    P = T.table("players")
    asof = pd.Timestamp(V.date.max()) + pd.Timedelta(days=1)
    cur = P[(P.last_season >= asof.year - 1) & P.market_value_in_eur.notna()].reset_index(drop=True)
    # club and league from the latest valuation (loans are recorded there; the players table keeps the parent club)
    lv = T.table("player_valuations").sort_values("date").groupby("player_id").last()
    lv = lv[lv.date >= asof - pd.Timedelta(days=400)]
    cur = cur[cur.player_id.isin(lv.index)].reset_index(drop=True)
    cur["current_club_id"] = cur.player_id.map(lv.current_club_id).astype(np.int64)
    cur["current_club_name"] = cur.player_id.map(lv.current_club_name)
    cur["current_club_domestic_competition_id"] = cur.player_id.map(lv.player_club_domestic_competition_id)
    C = A.build(cur.player_id.to_numpy(), [asof] * len(cur), club=cur.current_club_id.to_numpy(np.int64),
                comp=cur.current_club_domestic_competition_id)
    C["league"] = cur.current_club_domestic_competition_id.astype("string").to_numpy()
    C[HISTORY] = competition_history(A, cur.player_id.to_numpy(), [asof] * len(cur)).to_numpy()
    covered = (C["club_games_365"] >= 15).to_numpy()      # clubs whose games are in the dataset
    cur, C = cur[covered].reset_index(drop=True), C[covered].reset_index(drop=True)
    out = pd.DataFrame({
        "player_id": cur.player_id, "name": cur.name, "club": cur.current_club_name,
        "league": cur.current_club_domestic_competition_id, "position": cur.position,
        "sub_position": cur.sub_position, "nationality": cur.country_of_citizenship, "foot": cur.foot,
        "height": cur.height_in_cm, "age": C.age.round(1), "contract_until": cur.contract_expiration_date,
        "image_url": cur.image_url, "tm_value": cur.market_value_in_eur,
        "peak_value": cur.highest_market_value_in_eur,
        "forecast_value": np.exp(fc.predict(C)), "fair_value": np.exp(fv.predict(C)),
        "apps_365": C.apps_365, "minutes_365": C.minutes_365, "goals_365": C.goals_365,
        "assists_365": C.assists_365, "minutes_share_365": C.minutes_share_365.clip(0, 1),
        "club_games_365": C.club_games_365})
    comp_names = T.table("competitions").set_index("competition_id")["name"].str.replace("-", " ").str.title()
    out["league_name"] = out.league.map(comp_names)
    out.to_parquet(OUT / "current_players.parquet", index=False)
    hist = T.table("player_valuations")
    hist = hist[hist.player_id.isin(out.player_id[out.tm_value >= 1e6]) & (hist.date >= "2010-01-01")]
    hist[["player_id", "date", "market_value_in_eur", "current_club_name"]].to_parquet(OUT / "value_history.parquet",
                                                                                      index=False)
    res["asof"] = str(asof.date())
    res["current_players"] = int(len(out))
    res["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    return res
