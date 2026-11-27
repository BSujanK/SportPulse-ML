"""National-team forecaster (martj42/international_results, 1872 -> latest).

Ratings follow the World Football Elo methodology (eloratings.net):
tournament-weighted K, goal-difference multiplier and home advantage only
at genuinely non-neutral venues. On top of ratings + exponentially weighted
form, a LightGBM multiclass model produces calibrated 1X2 probabilities.

    python -m sportpulse.train international
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sportpulse import config
from sportpulse.data.loaders import _download
from sportpulse.evaluation import metrics as M
from sportpulse.features.engine import EWM
from sportpulse.models.match_outcome import fit_temperature, softmax

log = logging.getLogger(__name__)
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
OUT = config.ARTIFACTS_DIR / "international"
# Grid-searched on the 2023 -> mid-2024 validation window (eloratings.net standard: HFA 100, K x1).
HFA = 80.0
K_SCALE = 1.25     # multiplier on the eloratings.net K-factors
TRAIN_START, TRAIN_END, VALID_END = "1994-01-01", "2023-01-01", "2024-07-01"


def tournament_weight(t: str) -> tuple[float, int]:
    """(Elo K, importance level 0-4)."""
    t = str(t)
    if t == "FIFA World Cup":
        return 60.0, 4
    if t in {"UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
             "Confederations Cup", "CONCACAF Championship", "Oceania Nations Cup"}:
        return 50.0, 3
    if "qualification" in t or "Nations League" in t:
        return 40.0, 2
    if t == "Friendly":
        return 20.0, 0
    return 30.0, 1


@dataclass
class NationState:
    elo: float = 1500.0
    games: int = 0
    last: pd.Timestamp | None = None
    s: dict = field(default_factory=lambda: {k: EWM(0.25) for k in ("gf", "ga", "pts")})
    l: dict = field(default_factory=lambda: {k: EWM(0.06) for k in ("gf", "ga", "pts")})
    comp: dict = field(default_factory=lambda: {k: EWM(0.15) for k in ("gf", "ga")})  # competitive only
    surprise_s: EWM = field(default_factory=lambda: EWM(0.25))   # result minus Elo expectation
    surprise_l: EWM = field(default_factory=lambda: EWM(0.06))
    opp_elo: EWM = field(default_factory=lambda: EWM(0.15))      # schedule strength
    momentum: EWM = field(default_factory=lambda: EWM(0.2))      # recent Elo changes


def _row(H: NationState, A: NationState, neutral: bool, level: int, date) -> dict:
    hfa = 0.0 if neutral else HFA
    r = {"elo_h": H.elo, "elo_a": A.elo, "elo_diff": H.elo - A.elo + hfa,
         "elo_p": 1 / (1 + 10 ** (-(H.elo - A.elo + hfa) / 400)), "neutral": float(neutral),
         "level": level, "games_h": min(H.games, 300), "games_a": min(A.games, 300)}
    for side, T in (("h", H), ("a", A)):
        r[f"rest_{side}"] = min((date - T.last).days, 400) if T.last is not None else np.nan
        for k in ("gf", "ga", "pts"):
            r[f"{side}_s_{k}"], r[f"{side}_l_{k}"] = T.s[k].value, T.l[k].value
        r[f"{side}_c_gf"], r[f"{side}_c_ga"] = T.comp["gf"].value, T.comp["ga"].value
    for side, T in (("h", H), ("a", A)):
        r[f"{side}_surp_s"], r[f"{side}_surp_l"] = T.surprise_s.value, T.surprise_l.value
        r[f"{side}_opp_elo"], r[f"{side}_mom"] = T.opp_elo.value, T.momentum.value
    r["d_surp_l"] = r["h_surp_l"] - r["a_surp_l"]
    r["d_opp_elo"] = r["h_opp_elo"] - r["a_opp_elo"]
    r["d_l_gd"] = (r["h_l_gf"] - r["h_l_ga"]) - (r["a_l_gf"] - r["a_l_ga"])
    r["d_s_pts"] = r["h_s_pts"] - r["a_s_pts"]
    return r


def load_results(force: bool = False) -> pd.DataFrame:
    p = _download(URL, config.RAW_DIR / "international_results.csv", force)
    d = pd.read_csv(p)
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["home_score", "away_score"]).sort_values("date", kind="mergesort")
    d["neutral"] = d["neutral"].astype(str).str.upper().eq("TRUE")
    # A "neutral" fixture played in one team's own country is effectively home.
    d.loc[d["neutral"] & (d["country"] == d["home_team"]), "neutral"] = False
    d["hg"], d["ag"] = d["home_score"].astype(int), d["away_score"].astype(int)
    d["result"] = np.select([d.hg > d.ag, d.hg < d.ag], ["H", "A"], "D")
    return d.reset_index(drop=True)


def build(d: pd.DataFrame):
    teams: dict[str, NationState] = {}
    rows = []
    for r in d.itertuples(index=False):
        H = teams.setdefault(r.home_team, NationState())
        A = teams.setdefault(r.away_team, NationState())
        k, level = tournament_weight(r.tournament)
        rows.append(_row(H, A, r.neutral, level, r.date))
        gd = r.hg - r.ag
        hfa = 0.0 if r.neutral else HFA
        exp = 1 / (1 + 10 ** (-(H.elo - A.elo + hfa) / 400))
        res = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        mult = 1.0 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
        delta = K_SCALE * k * mult * (res - exp)
        for T, opp_elo, surprise, d_ in ((H, A.elo, res - exp, delta), (A, H.elo, exp - res, -delta)):
            T.surprise_s.update(surprise); T.surprise_l.update(surprise)
            T.opp_elo.update(opp_elo); T.momentum.update(d_)
        H.elo += delta
        A.elo -= delta
        for T, gf, ga in ((H, r.hg, r.ag), (A, r.ag, r.hg)):
            pts = 3 if gf > ga else 1 if gf == ga else 0
            for k2, v in (("gf", gf), ("ga", ga), ("pts", pts)):
                T.s[k2].update(float(v)); T.l[k2].update(float(v))
            if level >= 1:
                T.comp["gf"].update(float(gf)); T.comp["ga"].update(float(ga))
            T.games += 1
            T.last = r.date
    return pd.DataFrame(rows), teams


def squad_snapshots(end: str | None = None) -> pd.DataFrame:
    """Monthly Transfermarkt squad values per nation (cached artifact; rebuilt when the source is present)."""
    path = OUT / "squad_values.parquet"
    try:
        from sportpulse.data import transfermarkt
        sq = transfermarkt.squad_values(end=end)
        OUT.mkdir(parents=True, exist_ok=True)
        sq.to_parquet(path, index=False)
        return sq
    except Exception as e:  # noqa: BLE001 - offline: fall back to the committed snapshot
        log.warning("Transfermarkt squad values unavailable (%s); using %s", e, path)
        return pd.read_parquet(path)


def squad_row(sq: pd.DataFrame, home, away, date) -> dict:
    """Squad-value differences (home minus away) from the snapshot of the match month."""
    m = pd.Timestamp(date).to_period("M").to_timestamp()
    S = sq[sq.month == min(m, sq.month.max())].set_index("country")
    h, a = S.reindex([home]).iloc[0], S.reindex([away]).iloc[0]
    return {"sv26_diff": h.sv26 - a.sv26, "sv11_diff": h.sv11 - a.sv11}


def add_squad_features(F: pd.DataFrame, d: pd.DataFrame, sq: pd.DataFrame) -> pd.DataFrame:
    """Peeters (2018): crowd-sourced Transfermarkt squad values forecast international results.
    The snapshot for a match is taken on the 1st of its month, from valuations strictly before."""
    S = sq.set_index(["month", "country"])
    mon = d["date"].dt.to_period("M").dt.to_timestamp()
    sides = {}
    for side, col in (("h", "home_team"), ("a", "away_team")):
        j = S.reindex(pd.MultiIndex.from_arrays([mon, d[col]]))
        sides[side] = j
    F = F.copy()
    F["sv26_diff"] = sides["h"].sv26.to_numpy() - sides["a"].sv26.to_numpy()
    F["sv11_diff"] = sides["h"].sv11.to_numpy() - sides["a"].sv11.to_numpy()
    return F


SEEDS = (42, 7, 13, 21, 99)
# Optuna TPE (30 trials) on validation log-loss.
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.02, num_leaves=5,
              min_data_in_leaf=104, feature_fraction=0.65, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=0.17, verbose=-1, seed=config.SEED, num_threads=0)


def run(refresh: bool = False) -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_results(refresh)
    F, teams = build(d)
    sq = squad_snapshots(end=(d["date"].max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"))
    F = add_squad_features(F, d, sq)
    y = d["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    dt = d["date"]
    train = ((dt >= TRAIN_START) & (dt < TRAIN_END)).to_numpy()
    valid = ((dt >= TRAIN_END) & (dt < VALID_END)).to_numpy()
    test = (dt >= VALID_END).to_numpy()
    cols = list(F.columns)

    # 5-seed bagged ensemble (validation and test both improve over a single model)
    iters_by_seed, zs = {}, []
    for seed in SEEDS:
        b = lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(F[train], y[train]), 5000,
                      valid_sets=[lgb.Dataset(F[valid], y[valid])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        iters_by_seed[seed] = b.best_iteration
        zs.append(b.predict(F, num_iteration=b.best_iteration, raw_score=True))
    logits = np.mean(zs, axis=0)
    iters = int(np.mean(list(iters_by_seed.values())))
    temp = fit_temperature(logits[valid], y[valid])
    P = softmax(logits / temp)

    # Baseline: World Football Elo with the standard eloratings.net settings (not our tuned ones)
    global HFA, K_SCALE
    tuned = (HFA, K_SCALE)
    HFA, K_SCALE = 100.0, 1.0
    F_std, _ = build(d)
    HFA, K_SCALE = tuned
    elo_lr = LogisticRegression(max_iter=1000).fit(F_std.loc[train, ["elo_diff"]] / 400, y[train])
    P_elo = elo_lr.predict_proba(F_std[["elo_diff"]] / 400)
    base = np.tile(np.bincount(y[train], minlength=3) / train.sum(), (len(y), 1))
    # Peeters (2018) replication: Transfermarkt squad value + home advantage, multinomial logistic
    sv = F["sv26_diff"].to_numpy()
    Zp = np.c_[np.nan_to_num(sv), np.isnan(sv), (~d["neutral"].astype(bool)).to_numpy()]
    P_sv = LogisticRegression(max_iter=1000).fit(Zp[train], y[train]).predict_proba(Zp)

    def board(mask):
        out = []
        for name, PP in (("SportPulse International (LightGBM)", P), ("World Football Elo (standard, logistic)", P_elo),
                         ("Squad value logistic (Peeters 2018)", P_sv), ("Base rates", base)):
            s = M.summary(PP[mask], y[mask]); s["model"] = name; out.append(s)
        return sorted(out, key=lambda r: r["rps"])

    wc = test & (d["tournament"] == "FIFA World Cup").to_numpy()
    comp = test & (d["tournament"] != "Friendly").to_numpy()
    metrics = {
        "data": {"matches": int(len(d)), "teams": len(teams), "first": str(dt.min().date()),
                 "last": str(dt.max().date()), "train": int(train.sum()), "valid": int(valid.sum()),
                 "test": int(test.sum()), "test_window": [VALID_END, str(dt.max().date())],
                 "world_cup_2026_matches": int(wc.sum())},
        "leaderboard": board(test), "competitive": board(comp),
        "world_cup_2026": board(wc) if wc.sum() else [],
        "iters": iters, "temperature": temp,
    }
    # Production refit on everything from TRAIN_START
    full = (dt >= TRAIN_START).to_numpy()
    for f_ in OUT.glob("international*.lgb"):
        f_.unlink()
    for seed, it in iters_by_seed.items():
        b = lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(F[full], y[full]), int(it * full.sum() / train.sum()))
        b.save_model(str(OUT / f"international_s{seed}.lgb"))
    meta = {"features": cols, "temperature": temp, "hfa": HFA, "k_scale": K_SCALE, "seeds": list(SEEDS)}
    (OUT / "international_meta.json").write_text(json.dumps(meta))

    snap = []
    for name, T in teams.items():
        if T.last is not None and T.last >= pd.Timestamp("2022-01-01"):
            snap.append({"team": name, "elo": T.elo, "games": T.games, "last": T.last,
                         "form_pts": T.s["pts"].value, "gf": T.l["gf"].value, "ga": T.l["ga"].value,
                         **{f"state_{k}": v for k, v in _state_dict(T).items()}})
    ranking = pd.DataFrame(snap).sort_values("elo", ascending=False).reset_index(drop=True)
    ranking.to_parquet(OUT / "ranking.parquet", index=False)
    tp = d.loc[test, ["date", "home_team", "away_team", "hg", "ag", "result", "tournament", "neutral"]].copy()
    tp[["p_h", "p_d", "p_a"]] = P[test]
    tp.to_parquet(OUT / "test_predictions.parquet", index=False)
    metrics["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def _state_dict(T: NationState) -> dict:
    d = {"elo": T.elo, "games": T.games}
    for k in ("gf", "ga", "pts"):
        d[f"s_{k}_num"], d[f"s_{k}_den"] = T.s[k].num, T.s[k].den
        d[f"l_{k}_num"], d[f"l_{k}_den"] = T.l[k].num, T.l[k].den
    for k in ("gf", "ga"):
        d[f"c_{k}_num"], d[f"c_{k}_den"] = T.comp[k].num, T.comp[k].den
    for k in ("surprise_s", "surprise_l", "opp_elo", "momentum"):
        e = getattr(T, k)
        d[f"{k}_num"], d[f"{k}_den"] = e.num, e.den
    return d


class InternationalPredictor:
    def __init__(self, path=OUT):
        self.meta = json.loads((path / "international_meta.json").read_text())
        self.boosters = [lgb.Booster(model_file=str(path / f"international_s{s}.lgb"))
                         for s in self.meta.get("seeds", [])] or [lgb.Booster(model_file=str(path / "international.lgb"))]
        self.ranking = pd.read_parquet(path / "ranking.parquet")
        self.by_team = self.ranking.set_index("team")
        sp = path / "squad_values.parquet"
        self.squads = pd.read_parquet(sp) if sp.exists() else None

    def _state(self, team) -> NationState:
        r = self.by_team.loc[team]
        T = NationState(elo=r["state_elo"], games=int(r["state_games"]), last=pd.Timestamp(r["last"]))
        for k in ("gf", "ga", "pts"):
            T.s[k].num, T.s[k].den = r[f"state_s_{k}_num"], r[f"state_s_{k}_den"]
            T.l[k].num, T.l[k].den = r[f"state_l_{k}_num"], r[f"state_l_{k}_den"]
        for k in ("gf", "ga"):
            T.comp[k].num, T.comp[k].den = r[f"state_c_{k}_num"], r[f"state_c_{k}_den"]
        for k in ("surprise_s", "surprise_l", "opp_elo", "momentum"):
            e = getattr(T, k)
            if f"state_{k}_num" in r:
                e.num, e.den = r[f"state_{k}_num"], r[f"state_{k}_den"]
        return T

    def predict(self, home, away, neutral=True, tournament="FIFA World Cup", date=None):
        date = pd.Timestamp(date or pd.Timestamp.today().normalize())
        _, level = tournament_weight(tournament)
        row = pd.DataFrame([_row(self._state(home), self._state(away), neutral, level, date)])
        if self.squads is not None:
            for k, val in squad_row(self.squads, home, away, date).items():
                row[k] = val
        X = row.reindex(columns=self.meta["features"])          # missing squad values -> NaN
        z = np.mean([b.predict(X, raw_score=True) for b in self.boosters], axis=0)
        return softmax(z / self.meta["temperature"])[0], row.iloc[0].to_dict()
