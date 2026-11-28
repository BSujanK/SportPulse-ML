"""Published-model benchmark: Dixon & Coles (1997) on the SportPulse test window.

The classic time-weighted bivariate-Poisson model (attack / defence per team, home advantage,
low-score correction rho, exponential time decay xi), refit at the start of every month for
every league on the previous three seasons, and evaluated on exactly the rows of the club
leaderboard (test window, market odds available). xi is chosen on the 2023-24 validation season.

    python scripts/benchmark_dixon_coles.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sportpulse import config  # noqa: E402
from sportpulse.data.loaders import load_matches  # noqa: E402
from sportpulse.evaluation import metrics as M  # noqa: E402
from sportpulse.features.market import shin_probabilities  # noqa: E402

MAXG = 10
WINDOW_DAYS = 3 * 365


def _tau(x, y, lh, la, rho):
    t = np.ones_like(lh)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-6, None)


def fit_dc(hi, ai, hg, ag, w, n):
    """Weighted MLE. Poisson part with analytic gradient, then rho by 1-D search (two-stage)."""
    def nll(theta):
        mu, h, att, dfn = theta[0], theta[1], theta[2:2 + n], theta[2 + n:]
        eh = mu + h + att[hi] - dfn[ai]
        ea = mu + att[ai] - dfn[hi]
        lh, la = np.exp(eh), np.exp(ea)
        f = np.sum(w * (lh - hg * eh)) + np.sum(w * (la - ag * ea))
        rh, ra = w * (lh - hg), w * (la - ag)
        g = np.zeros_like(theta)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g[2:2 + n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
        g[2 + n:] = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n)
        lam = 1e-3                                  # tiny ridge for identifiability
        f += lam * (att @ att + dfn @ dfn)
        g[2:2 + n] += 2 * lam * att
        g[2 + n:] += 2 * lam * dfn
        return f, g

    theta0 = np.zeros(2 + 2 * n)
    theta0[0] = np.log(max(np.average((hg + ag) / 2, weights=w), 0.1))
    res = minimize(nll, theta0, jac=True, method="L-BFGS-B")
    th = res.x
    mu, h, att, dfn = th[0], th[1], th[2:2 + n], th[2 + n:]
    lh = np.exp(mu + h + att[hi] - dfn[ai])
    la = np.exp(mu + att[ai] - dfn[hi])
    r = minimize_scalar(lambda rho: -np.sum(w * np.log(_tau(hg, ag, lh, la, rho))),
                        bounds=(-0.2, 0.2), method="bounded")
    return mu, h, att, dfn, r.x


def outcome_probs(lh, la, rho):
    g = np.arange(MAXG + 1)
    ph, pa = poisson.pmf(g[None, :], lh[:, None]), poisson.pmf(g[None, :], la[:, None])
    S = ph[:, :, None] * pa[:, None, :]
    S[:, 0, 0] *= 1 - lh * la * rho
    S[:, 0, 1] *= 1 + lh * rho
    S[:, 1, 0] *= 1 + la * rho
    S[:, 1, 1] *= 1 - rho
    S = np.clip(S, 0, None)
    S /= S.sum((1, 2), keepdims=True)
    i, j = np.meshgrid(g, g, indexing="ij")
    return np.c_[(S * (i > j)).sum((1, 2)), (S * (i == j)).sum((1, 2)), (S * (i < j)).sum((1, 2))]


def rolling_predictions(m: pd.DataFrame, start, end, xi, promoted: float = 0.25) -> np.ndarray:
    P = np.full((len(m), 3), np.nan)
    months = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthBegin(1), freq="MS")
    for div, g in m.groupby("division", sort=False):
        for t0, t1 in zip(months[:-1], months[1:]):
            tgt = g[(g.date >= t0) & (g.date < t1)]
            if tgt.empty:
                continue
            hist = g[(g.date < t0) & (g.date >= t0 - pd.Timedelta(days=WINDOW_DAYS))]
            if len(hist) < 200:
                continue
            teams = pd.Index(pd.unique(np.r_[hist.home.to_numpy(), hist.away.to_numpy()]))
            n = len(teams)
            w = np.exp(-xi * (t0 - hist.date).dt.days.to_numpy(float))
            mu, h, att, dfn, rho = fit_dc(teams.get_indexer(hist.home), teams.get_indexer(hist.away),
                                          hist.hg.to_numpy(float), hist.ag.to_numpy(float), w, n)
            # Unseen (promoted) teams: the average promoted side is weaker than average, so use the mean
            # rating of the bottom quarter. Strength is att + dfn: dfn lowers the opponent's scoring rate.
            q = np.argsort(att + dfn)[: max(int(n * promoted), 1)]
            a_new, d_new = att[q].mean(), dfn[q].mean()
            hi, ai = teams.get_indexer(tgt.home), teams.get_indexer(tgt.away)
            ah = np.where(hi >= 0, att[hi], a_new); dh = np.where(hi >= 0, dfn[hi], d_new)
            aa = np.where(ai >= 0, att[ai], a_new); da = np.where(ai >= 0, dfn[ai], d_new)
            lh, la = np.exp(mu + h + ah - da), np.exp(mu + aa - dh)
            P[m.index.get_indexer(tgt.index)] = outcome_probs(lh, la, rho)
    return P


def _rps_rows(p, y):
    c = np.cumsum(p, 1)[:, :2] - np.cumsum(np.eye(3)[y], 1)[:, :2]
    return (c ** 2).sum(1) / 2


def _paired(a, b):
    """Mean per-match RPS difference a - b, its standard error and z (negative = a is better)."""
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": float(d.mean()), "se": float(se), "z": float(d.mean() / se)}


def main():
    m = load_matches().reset_index(drop=True)
    y = m.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    odds = m[["odds_h", "odds_d", "odds_a"]].to_numpy(float)
    ok = np.isfinite(odds).all(1) & (odds > 1).all(1)
    P_mkt = np.full((len(m), 3), np.nan)
    P_mkt[ok] = shin_probabilities(odds[ok])
    d = m.date
    valid = ((d >= config.TRAIN_END) & (d < config.VALID_END)).to_numpy()
    test = (d >= config.VALID_END).to_numpy()

    # Give the published model its best shot: the time decay and the promoted-team prior (mean rating of the
    # weakest quarter, half, or the whole league) are both chosen on the validation season.
    best = None
    for xi in (0.0010, 0.0019, 0.0030, 0.0045):        # Dixon & Coles' own choice is ~0.0019 per day
        for promoted in (0.25, 0.5, 1.0):
            P = rolling_predictions(m, config.TRAIN_END, pd.Timestamp(config.VALID_END) - pd.Timedelta(days=1), xi, promoted)
            mk = valid & ok & np.isfinite(P).all(1)
            sc = M.rps(P[mk], y[mk])
            print(f"xi={xi:.4f} promoted={promoted:.2f}  validation RPS {sc:.5f}", flush=True)
            if best is None or sc < best[2]:
                best = (xi, promoted, sc)
    xi, promoted = best[0], best[1]
    P = rolling_predictions(m, config.VALID_END, d.max(), xi, promoted)
    tm = test & ok
    cover = tm & np.isfinite(P).all(1)
    # SportPulse out-of-sample test predictions (written by the match pipeline before the production refit)
    key = ["date", "division", "home", "away"]
    tp = pd.read_parquet(config.MATCH_ARTIFACTS / "test_predictions.parquet").drop_duplicates(key)
    j = m[key].merge(tp, on=key, how="left")
    P_sp = j[["p_h", "p_d", "p_a"]].to_numpy(float)
    P_fund = j[["pf_h", "pf_d", "pf_a"]].to_numpy(float)
    cover &= np.isfinite(P_sp).all(1)
    yc = y[cover]
    models = {"SportPulse Market-Residual (production)": P_sp, "Bookmaker market (Shin)": P_mkt,
              "SportPulse Fundamentals (no odds)": P_fund, "Dixon & Coles (1997), monthly refit": P}
    per_match = {k: _rps_rows(v[cover], yc) for k, v in models.items()}
    ref = per_match["Dixon & Coles (1997), monthly refit"]
    res = {"xi": xi, "promoted_prior": promoted, "n_test": int(tm.sum()), "n_covered": int(cover.sum()),
           "models": {k: M.summary(v[cover], yc) for k, v in models.items()},
           "paired_vs_dixon_coles": {k: _paired(v, ref) for k, v in per_match.items() if v is not ref},
           "paired_production_vs_market": _paired(per_match["SportPulse Market-Residual (production)"],
                                                  per_match["Bookmaker market (Shin)"])}
    print(json.dumps(res, indent=2, default=float))
    out = config.MATCH_ARTIFACTS / "benchmark_dixon_coles.json"
    out.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
