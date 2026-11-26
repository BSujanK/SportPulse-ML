"""Serving layer: load artifacts once, predict any fixture.

Feature parity with training is guaranteed because live rows are produced by
the very same `pre_match_row` function used inside the training pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from sportpulse import config
from sportpulse.features.engine import pre_match_row
from sportpulse.features.market import shin_probabilities
from sportpulse.models.goals import GoalsModel, score_matrix
from sportpulse.models.match_outcome import MatchOutcomeModel

CALENDAR_YEAR_LEAGUES = {"BRA", "ARG", "USA", "JAP", "SWE", "NOR", "FIN", "IRL", "CHN"}


@dataclass
class FixturePrediction:
    home: str
    away: str
    division: str
    p_final: np.ndarray          # H, D, A (market-residual model when odds given)
    p_fundamentals: np.ndarray   # H, D, A (no betting information)
    p_market: np.ndarray | None
    xg_home: float
    xg_away: float
    p_over25: float
    p_btts: float
    scores: np.ndarray           # (11, 11) scoreline matrix
    features: dict

    @property
    def fair_odds(self) -> np.ndarray:
        return 1.0 / np.clip(self.p_final, 1e-6, 1)


class MatchPredictor:
    def __init__(self, path=config.MATCH_ARTIFACTS):
        self.path = path
        self.outcome = MatchOutcomeModel.load(path)
        self.goals = GoalsModel.load(path)
        self.teams = pd.read_parquet(path / "team_state.parquet").set_index("team_id")
        self.leagues = pd.read_parquet(path / "league_state.parquet").set_index("division")
        self.recent = pd.read_parquet(path / "recent_matches.parquet")

    @cached_property
    def metrics(self) -> dict:
        return json.loads((self.path / "metrics.json").read_text())

    def teams_in(self, division: str) -> list[str]:
        """Teams of the division's current season (falls back to the previous season
        while the new one is still only a few rounds old)."""
        r = self.recent[self.recent.division == division]
        if r.empty:
            return []
        last = r.date.max()
        if division in CALENDAR_YEAR_LEAGUES:
            start = pd.Timestamp(year=last.year, month=1, day=1)
            prev_start = pd.Timestamp(year=last.year - 1, month=1, day=1)
        else:
            y = last.year if last.month >= 7 else last.year - 1
            start, prev_start = pd.Timestamp(year=y, month=7, day=1), pd.Timestamp(year=y - 1, month=7, day=1)
        cur = r[r.date >= start]
        prev = r[(r.date >= prev_start) & (r.date < start)]
        cur_t = set(cur.home_id) | set(cur.away_id)
        prev_t = set(prev.home_id) | set(prev.away_id)
        teams = cur_t if len(cur_t) >= 0.8 * max(len(prev_t), 1) else prev_t | cur_t
        return sorted(teams)

    def latest_season_divisions(self) -> list[str]:
        return [d for d in config.DIVISIONS if d in set(self.recent.division)]

    def _snap(self, team_id: str) -> dict:
        s = self.teams.loc[team_id].to_dict()
        s["last_date"] = pd.Timestamp(s["last_date"]) if pd.notna(s["last_date"]) else None
        return s

    def predict(self, home_id: str, away_id: str, division: str,
                date: pd.Timestamp | None = None, odds: tuple | None = None) -> FixturePrediction:
        date = pd.Timestamp(date or pd.Timestamp.today().normalize())
        lg = self.leagues.loc[division].to_dict()
        row = pre_match_row(self._snap(home_id), self._snap(away_id), lg, date)
        p_mkt = None
        if odds and all(o and o > 1 for o in odds):
            p_mkt = shin_probabilities(np.array([odds], float))[0]
            row.update({"mkt_p_h": p_mkt[0], "mkt_p_d": p_mkt[1], "mkt_p_a": p_mkt[2],
                        "mkt_max_p_h": p_mkt[0], "mkt_max_p_d": p_mkt[1], "mkt_max_p_a": p_mkt[2],
                        "mkt_overround": float(sum(1 / o for o in odds))})
        row["tier"] = config.DIVISIONS[division][2]
        row["division"] = division
        X = pd.DataFrame([row])
        out = self.outcome.predict(X)
        lam, mu = self.goals.rates(X)
        S = score_matrix(float(lam[0]), float(mu[0]), self.goals.rho)
        i, j = np.indices(S.shape)
        return FixturePrediction(
            home=home_id.split("|", 1)[1], away=away_id.split("|", 1)[1], division=division,
            p_final=out["final"][0], p_fundamentals=out["fundamentals"][0], p_market=p_mkt,
            xg_home=float(lam[0]), xg_away=float(mu[0]),
            p_over25=float(S[(i + j) > 2.5].sum()), p_btts=float(S[(i > 0) & (j > 0)].sum()),
            scores=S, features=row)

    def power_rankings(self, division: str) -> pd.DataFrame:
        ids = self.teams_in(division)
        t = self.teams.loc[ids].copy()
        t["team"] = [i.split("|", 1)[1] for i in ids]
        t["pi_overall"] = (t["pi_home"] + t["pi_away"]) / 2
        t["form_gd"] = t["s_gf"] - t["s_ga"]
        t["form_ppg"] = t["s_pts"]
        cols = ["team", "elo", "pi_overall", "pi_home", "pi_away", "form_ppg", "form_gd",
                "l_gf", "l_ga", "s_stf", "s_sta", "mkt_strength", "games", "last_date"]
        return t[cols].sort_values("elo", ascending=False).reset_index(drop=True)


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for r in matches.itertuples():
        for team, gf, ga, home in ((r.home, r.hg, r.ag, True), (r.away, r.ag, r.hg, False)):
            t = rows.setdefault(team, {"Team": team, "P": 0, "W": 0, "D": 0, "L": 0,
                                       "GF": 0, "GA": 0, "form": []})
            t["P"] += 1; t["GF"] += gf; t["GA"] += ga
            res = "W" if gf > ga else "D" if gf == ga else "L"
            t[res] += 1
            t["form"].append(res)
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df["GD"] = df.GF - df.GA
    df["Pts"] = 3 * df.W + df.D
    df["Form"] = df["form"].map(lambda f: "".join(f[-5:]))
    df = df.drop(columns="form").sort_values(["Pts", "GD", "GF"], ascending=False)
    df.insert(0, "#", range(1, len(df) + 1))
    return df.reset_index(drop=True)
