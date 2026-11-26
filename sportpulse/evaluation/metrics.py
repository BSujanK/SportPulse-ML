"""Probabilistic forecast metrics.

For football 1X2 forecasts the *Ranked Probability Score* (Epstein 1969;
Constantinou & Fenton 2012) is the accepted primary metric because outcomes
are ordered (home win > draw > away win). We also report log-loss, Brier,
accuracy and expected calibration error.
"""
from __future__ import annotations

import numpy as np


def one_hot(y: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Mean ranked probability score. p: (n,k) in ordered classes, y: (n,) int."""
    o = one_hot(y, p.shape[1])
    cp, co = np.cumsum(p, axis=1), np.cumsum(o, axis=1)
    return float(np.mean(np.sum((cp[:, :-1] - co[:, :-1]) ** 2, axis=1) / (p.shape[1] - 1)))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((p - one_hot(y, p.shape[1])) ** 2, axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(p.argmax(1) == y))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Classwise expected calibration error (averaged over classes)."""
    errs = []
    for k in range(p.shape[1]):
        pk, yk = p[:, k], (y == k).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(pk, edges) - 1, 0, bins - 1)
        e = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                e += m.mean() * abs(pk[m].mean() - yk[m].mean())
        errs.append(e)
    return float(np.mean(errs))


def binary_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    return {"rps": rps(p, y), "log_loss": log_loss(p, y), "brier": brier(p, y),
            "accuracy": accuracy(p, y), "ece": ece(p, y), "n": int(len(y))}


def calibration_curve(p: np.ndarray, y: np.ndarray, k: int, bins: int = 12):
    """Return (mean predicted, observed frequency, count) per bin for class k."""
    pk, yk = p[:, k], (y == k).astype(float)
    edges = np.quantile(pk, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0, 1
    idx = np.clip(np.digitize(pk, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() > 0:
            out.append((float(pk[m].mean()), float(yk[m].mean()), int(m.sum())))
    return out


def betting_backtest(p_model: np.ndarray, odds: np.ndarray, y: np.ndarray,
                     edge: float = 0.03, kelly_frac: float = 0.0) -> dict:
    """Flat-stake value-betting simulation: bet 1 unit when model EV > edge.

    This is a sanity check of whether the model carries information the
    market price does not. A positive ROI on best available odds over a
    large out-of-sample period is extremely rare and should be reported, not
    assumed.
    """
    ev = p_model * odds - 1
    ok = np.isfinite(ev)
    ev = np.where(ok, ev, -np.inf)
    best = ev.argmax(1)
    take = ev[np.arange(len(y)), best] > edge
    stake = int(take.sum())
    r = np.where(best[take] == y[take], odds[take, best[take]] - 1, -1.0)
    se = float(r.std(ddof=1) / np.sqrt(stake)) if stake > 1 else float("nan")
    return {"bets": stake, "profit_units": float(r.sum()),
            "roi": float(r.mean()) if stake else 0.0, "roi_se": se,
            "z": float(r.mean() / se) if stake > 1 and se > 0 else float("nan")}


def betting_report(df) -> list[dict]:
    """Backtest on saved out-of-sample predictions: model vs a market-self control,
    at average and at best available prices, for several edge thresholds."""
    y = df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    rows = []
    for price, cols in [("average odds", ["odds_h", "odds_d", "odds_a"]),
                        ("best odds", ["max_h", "max_d", "max_a"])]:
        O = df[cols].to_numpy(float)
        ok = np.isfinite(O).all(1)
        for who, pc in [("SportPulse", ["p_h", "p_d", "p_a"]), ("Market (control)", ["pm_h", "pm_d", "pm_a"])]:
            P = df[pc].to_numpy(float)
            for e in (0.02, 0.05, 0.10):
                rows.append({"prices": price, "model": who, "edge": e,
                             **betting_backtest(P[ok], O[ok], y[ok], e)})
    return rows
