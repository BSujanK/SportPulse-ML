import numpy as np
import pandas as pd
import pytest

from sportpulse import config
from sportpulse.evaluation import metrics as M
from sportpulse.features.engine import build_features, fundamental_features
from sportpulse.features.market import shin_probabilities, two_way_probabilities
from sportpulse.models.goals import markets_from_rates, score_matrix


# ── metrics ──────────────────────────────────────────────────────────────────
def test_rps_perfect_and_worst():
    y = np.array([0, 1, 2])
    assert M.rps(np.eye(3), y) == pytest.approx(0.0)
    worst = np.array([[0, 0, 1.0], [1, 0, 0], [1.0, 0, 0]])
    assert M.rps(worst[[0]], np.array([0])) == pytest.approx(1.0)


def test_rps_respects_ordering():
    # Predicting a draw when home wins is better than predicting away.
    y = np.array([0])
    assert M.rps(np.array([[0, 1.0, 0]]), y) < M.rps(np.array([[0, 0, 1.0]]), y)


def test_log_loss_and_brier_uniform():
    p = np.full((4, 3), 1 / 3)
    y = np.array([0, 1, 2, 0])
    assert M.log_loss(p, y) == pytest.approx(np.log(3))
    assert M.brier(p, y) == pytest.approx(2 / 3)


# ── market ───────────────────────────────────────────────────────────────────
def test_shin_sums_to_one_and_keeps_order():
    odds = np.array([[1.5, 4.2, 6.5], [2.9, 3.3, 2.5], [np.nan, 3.0, 3.0]])
    p = shin_probabilities(odds)
    assert np.allclose(p[:2].sum(1), 1)
    assert np.isnan(p[2]).all()
    assert p[0, 0] > p[0, 1] > p[0, 2]
    # Shin shifts mass toward the favourite relative to proportional de-vig
    prop = (1 / odds[0]) / (1 / odds[0]).sum()
    assert p[0, 0] > prop[0] and p[0, 2] < prop[2]


def test_two_way():
    p = two_way_probabilities(np.array([[1.9, 1.9]]))
    assert np.allclose(p, 0.5)


# ── goals ────────────────────────────────────────────────────────────────────
def test_score_matrix_and_markets():
    S = score_matrix(1.6, 1.1, -0.08)
    assert S.sum() == pytest.approx(1)
    mk = markets_from_rates(np.array([1.6]), np.array([1.1]), -0.08)
    assert mk[["p_h", "p_d", "p_a"]].sum(axis=1).iloc[0] == pytest.approx(1)
    i, j = np.indices(S.shape)
    assert mk["p_h"].iloc[0] == pytest.approx(S[i > j].sum())
    assert 0 < mk["p_over25"].iloc[0] < 1


# ── feature engine: leakage & parity ─────────────────────────────────────────
def _toy_matches(n=60, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"England|T{i}" for i in range(6)]
    rows = []
    d0 = pd.Timestamp("2020-08-01")
    for k in range(n):
        h, a = rng.choice(len(teams), 2, replace=False)
        hg, ag = rng.poisson(1.5), rng.poisson(1.1)
        rows.append({"date": d0 + pd.Timedelta(days=3 * k), "division": "E0",
                     "home_id": teams[h], "away_id": teams[a], "hg": hg, "ag": ag, "tier": 1,
                     "hs": 12.0, "as_": 9.0, "hst": 4.0, "ast": 3.0, "hc": 5.0, "ac": 4.0,
                     "odds_h": 2.0, "odds_d": 3.4, "odds_a": 3.8, "max_h": 2.1, "max_d": 3.5,
                     "max_a": 4.0, "odds_o25": 1.9, "odds_u25": 1.9, "ah_line": -0.25})
    return pd.DataFrame(rows)


def test_features_do_not_see_own_result():
    m = _toy_matches()
    f1, _, _ = build_features(m)
    m2 = m.copy()
    m2.loc[len(m2) - 1, ["hg", "ag"]] = [9, 0]   # change only the last result
    f2, _, _ = build_features(m2)
    pd.testing.assert_frame_equal(f1, f2)          # last row's features unchanged


def test_future_results_do_not_change_past_features():
    m = _toy_matches()
    f1, _, _ = build_features(m)
    m2 = m.copy()
    m2.loc[40:, ["hg", "ag"]] = [[7, 0]] * (len(m2) - 40)
    f2, _, _ = build_features(m2)
    pd.testing.assert_frame_equal(f1.iloc[:41], f2.iloc[:41])
    assert not f1.iloc[45:].equals(f2.iloc[45:])


def test_fundamental_features_exclude_market():
    m = _toy_matches()
    f, _, _ = build_features(m)
    fc = fundamental_features(f.columns)
    assert not any(c.startswith("mkt_") or c.startswith("ah_") for c in fc)
    assert "elo_diff" in fc and "pi_gd" in fc


def test_division_metadata_consistent():
    for code, (country, name, tier) in config.DIVISIONS.items():
        assert tier >= 1 and country and name


# ── analyst layer ────────────────────────────────────────────────────────────
def test_freeze_frame_geometry():
    from sportpulse.data.statsbomb import freeze_frame_features
    ff = [{"location": [118, 40], "teammate": False, "position": {"name": "Goalkeeper"}},
          {"location": [110, 40], "teammate": False, "position": {"name": "Center Back"}},
          {"location": [100, 10], "teammate": False, "position": {"name": "Left Back"}}]
    f = freeze_frame_features([100, 40], ff)
    assert f["ff_def_in_cone"] == 1 and f["ff_gk_in_cone"] == 1
    assert f["ff_nearest_def"] == pytest.approx(10.0)
    assert f["ff_gk_dist_goal"] == pytest.approx(2.0)


def test_expected_threat_increases_towards_goal():
    from sportpulse.analysis.analyst import expected_threat
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 120, n); y = rng.uniform(0, 80, n)
    passes = pd.DataFrame({"x": x, "y": y, "end_x": np.clip(x + rng.normal(8, 10, n), 0, 119.9),
                           "end_y": np.clip(y + rng.normal(0, 10, n), 0, 79.9), "complete": rng.random(n) < .8})
    carries = passes.iloc[:0][["x", "y", "end_x", "end_y"]]
    sx = rng.uniform(95, 120, 600); sy = rng.uniform(25, 55, 600)
    shots = pd.DataFrame({"x": sx, "y": sy, "goal": (rng.random(600) < (sx - 90) / 60).astype(float)})
    g = expected_threat(passes, carries, shots)
    assert g[-1].mean() > g[0].mean() and (g >= 0).all()


def test_progressive_pass_rule():
    from sportpulse.analysis.analyst import progressive
    df = pd.DataFrame({"x": [60, 60, 20], "y": [40, 40, 40], "end_x": [90, 62, 60], "end_y": [40, 40, 40]})
    assert progressive(df).tolist() == [True, False, False]
