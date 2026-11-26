"""Integration tests against the trained artifacts (skipped if not trained)."""
import json

import pandas as pd
import pytest

from sportpulse import config

pytestmark = pytest.mark.skipif(not (config.MATCH_ARTIFACTS / "metrics.json").exists(),
                                reason="artifacts not trained")


@pytest.fixture(scope="module")
def predictor():
    from sportpulse.inference import MatchPredictor
    return MatchPredictor()


def test_predict_fixture_probabilities(predictor):
    teams = predictor.teams_in("E0")
    r = predictor.predict(teams[0], teams[1], "E0")
    assert r.p_final.sum() == pytest.approx(1, abs=1e-6)
    assert r.p_fundamentals.sum() == pytest.approx(1, abs=1e-6)
    assert r.scores.sum() == pytest.approx(1, abs=1e-6)
    assert 0.1 < r.xg_home < 5 and 0.1 < r.xg_away < 5


def test_market_input_moves_prediction(predictor):
    teams = predictor.teams_in("E0")
    fav = predictor.predict(teams[0], teams[1], "E0", odds=(1.25, 6.5, 12.0))
    dog = predictor.predict(teams[0], teams[1], "E0", odds=(9.0, 5.0, 1.33))
    assert fav.p_final[0] > 0.6 > dog.p_final[0]


def test_home_advantage_symmetry(predictor):
    """Same pair, venues swapped: each side should be stronger at home."""
    t = predictor.teams_in("E0")
    a, b = t[0], t[1]
    ab = predictor.predict(a, b, "E0").p_fundamentals
    ba = predictor.predict(b, a, "E0").p_fundamentals
    assert ab[0] > ba[2] - 0.02  # a at home >= a away (tolerance)


def test_reported_metrics_beat_baselines():
    m = json.loads((config.MATCH_ARTIFACTS / "metrics.json").read_text())
    lb = {r["model"]: r for r in m["leaderboard"]}
    prod = lb["SportPulse Market-Residual (production)"]
    assert prod["rps"] <= lb["Bookmaker market (Shin)"]["rps"] + 5e-4
    assert lb["SportPulse Fundamentals (no odds)"]["rps"] < lb["Elo logistic"]["rps"]
    assert prod["ece"] < 0.03
    assert m["data"]["last_date"] >= "2025-01-01"   # trained on recent data


def test_player_snapshot():
    p = pd.read_parquet(config.PLAYER_ARTIFACTS / "players_latest.parquet")
    assert p["player_id"].is_unique
    assert p["archetype"].notna().all()
    assert (p["proj_low"] <= p["proj_high"]).all()
    m = json.loads((config.PLAYER_ARTIFACTS / "metrics.json").read_text())
    assert m["development"]["model"]["mae"] < m["development"]["baseline_persistence"]["mae"]
    assert m["value"]["model"]["r2"] > m["value"]["baseline_linear_log"]["r2"]
    assert len(set(m["archetypes"]["names"].values())) == m["archetypes"]["k"]
