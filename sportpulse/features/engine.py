"""Sequential, leak-free pre-match feature engine.

A single chronological pass over every match. For each match we first read
the *current* state of both teams (everything known before kick-off) to
build the feature row, and only then update the state with the result.

The same `TeamState` / `pre_match_row` code is used for live inference on
upcoming fixtures, which guarantees train/serve feature parity.

Ratings implemented
-------------------
* Elo with goal-difference multiplier and auto-correlation correction
  (FiveThirtyEight / eloratings.net style), per-country pools.
* pi-ratings (Constantinou & Fenton, 2013): separate home/away ratings
  expressed on a goal-difference scale. Top performer in the 2017 Soccer
  Prediction Challenge.
* Exponentially weighted form (short & long memory) for goals, shots,
  shots on target, corners and points, robust to missing statistics.
* League context: running home-goal / away-goal / draw rates per division.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sportpulse.features.market import shin_probabilities, two_way_probabilities

# ── Hyper-parameters (grid-searched on the 2023-24 validation season) ──
ELO_K = 20.0
ELO_HFA = 65.0
ELO_INIT_TOP = 1500.0
ELO_TIER_STEP = 90.0
PI_LAMBDA = 0.035
PI_GAMMA = 0.7
PI_C = 3.0
EWM_SHORT = 0.18   # ~ last 8-10 matches (tuned on valid)
EWM_LONG = 0.04    # ~ last 40-50 matches (tuned on valid)
LEAGUE_EWM = 0.01

FORM_STATS = ["gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "pts"]


@dataclass
class EWM:
    """Missing-aware exponentially weighted mean."""
    alpha: float
    num: float = 0.0
    den: float = 0.0

    def update(self, x: float) -> None:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        self.num = (1 - self.alpha) * self.num + self.alpha * x
        self.den = (1 - self.alpha) * self.den + self.alpha

    @property
    def value(self) -> float:
        return self.num / self.den if self.den > 1e-9 else float("nan")


@dataclass
class TeamState:
    elo: float
    pi_home: float = 0.0
    pi_away: float = 0.0
    games: int = 0
    last_date: pd.Timestamp | None = None
    division: str = ""
    short: dict = field(default_factory=lambda: {s: EWM(EWM_SHORT) for s in FORM_STATS})
    long: dict = field(default_factory=lambda: {s: EWM(EWM_LONG) for s in FORM_STATS})
    mkt_strength: EWM = field(default_factory=lambda: EWM(0.15))  # market-implied exp. points

    def snapshot(self) -> dict:
        d = {"elo": self.elo, "pi_home": self.pi_home, "pi_away": self.pi_away,
             "games": self.games, "last_date": self.last_date, "division": self.division,
             "mkt_strength": self.mkt_strength.value}
        for s in FORM_STATS:
            d[f"s_{s}"] = self.short[s].value
            d[f"l_{s}"] = self.long[s].value
        return d


@dataclass
class LeagueState:
    hg: EWM = field(default_factory=lambda: EWM(LEAGUE_EWM))
    ag: EWM = field(default_factory=lambda: EWM(LEAGUE_EWM))
    draw: EWM = field(default_factory=lambda: EWM(LEAGUE_EWM))
    home_win: EWM = field(default_factory=lambda: EWM(LEAGUE_EWM))
    n: int = 0

    def snapshot(self) -> dict:
        return {"lg_hg": self.hg.value, "lg_ag": self.ag.value,
                "lg_draw": self.draw.value, "lg_hw": self.home_win.value, "lg_n": self.n}


def _pi_expected(r: float) -> float:
    return math.copysign(10 ** (abs(r) / PI_C) - 1, r)


def _elo_expect(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


# ── Feature row (shared by training and inference) ───────────────────────────
def pre_match_row(h: dict, a: dict, lg: dict, date: pd.Timestamp) -> dict:
    """Build the feature dictionary from two team snapshots + league snapshot."""
    row = {}
    row["elo_h"], row["elo_a"] = h["elo"], a["elo"]
    row["elo_diff"] = h["elo"] - a["elo"] + ELO_HFA
    row["elo_p_home"] = _elo_expect(row["elo_diff"])
    row["pi_h_home"], row["pi_a_away"] = h["pi_home"], a["pi_away"]
    row["pi_h_away"], row["pi_a_home"] = h["pi_away"], a["pi_home"]
    row["pi_gd"] = _pi_expected(h["pi_home"]) - _pi_expected(a["pi_away"])
    row["games_h"], row["games_a"] = min(h["games"], 200), min(a["games"], 200)
    for side, st in (("h", h), ("a", a)):
        rest = (date - st["last_date"]).days if st["last_date"] is not None else np.nan
        row[f"rest_{side}"] = min(rest, 60) if rest == rest else np.nan
        for s in FORM_STATS:
            row[f"{side}_s_{s}"] = st[f"s_{s}"]
            row[f"{side}_l_{s}"] = st[f"l_{s}"]
        row[f"mkt_str_{side}"] = st["mkt_strength"]
    # Differential / interaction features (what actually drives the outcome)
    for w in ("s", "l"):
        row[f"d_{w}_gd"] = (h[f"{w}_gf"] - h[f"{w}_ga"]) - (a[f"{w}_gf"] - a[f"{w}_ga"])
        row[f"d_{w}_sot"] = (h[f"{w}_stf"] - h[f"{w}_sta"]) - (a[f"{w}_stf"] - a[f"{w}_sta"])
        row[f"d_{w}_shots"] = (h[f"{w}_sf"] - h[f"{w}_sa"]) - (a[f"{w}_sf"] - a[f"{w}_sa"])
        row[f"d_{w}_pts"] = h[f"{w}_pts"] - a[f"{w}_pts"]
        row[f"xh_{w}"] = (h[f"{w}_gf"] + a[f"{w}_ga"]) / 2   # naive expected home goals
        row[f"xa_{w}"] = (a[f"{w}_gf"] + h[f"{w}_ga"]) / 2
    row["d_mkt_str"] = h["mkt_strength"] - a["mkt_strength"]
    row.update(lg)
    row["month"] = date.month
    return row


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Run the chronological pass.

    Returns (feature table aligned with `matches`, final team states, final league states).
    """
    teams: dict[str, TeamState] = {}
    leagues: dict[str, LeagueState] = {}

    # Pre-compute fair market probabilities (they're match-level, not state).
    p1x2 = shin_probabilities(matches[["odds_h", "odds_d", "odds_a"]].to_numpy())
    pmax = shin_probabilities(matches[["max_h", "max_d", "max_a"]].to_numpy())
    pou = two_way_probabilities(matches[["odds_o25", "odds_u25"]].to_numpy())
    overround = (1 / matches[["odds_h", "odds_d", "odds_a"]]).sum(axis=1, min_count=3).to_numpy()
    exp_pts_h = 3 * p1x2[:, 0] + p1x2[:, 1]
    exp_pts_a = 3 * p1x2[:, 2] + p1x2[:, 1]

    cols = {c: matches[c].to_numpy() for c in
            ["date", "division", "home_id", "away_id", "hg", "ag", "tier",
             "hs", "as_", "hst", "ast", "hc", "ac"]}
    rows = []
    n = len(matches)
    for i in range(n):
        date = pd.Timestamp(cols["date"][i])
        div, hid, aid = cols["division"][i], cols["home_id"][i], cols["away_id"][i]
        tier = int(cols["tier"][i])
        for tid in (hid, aid):
            if tid not in teams:
                teams[tid] = TeamState(elo=ELO_INIT_TOP - ELO_TIER_STEP * (tier - 1))
        lg = leagues.setdefault(div, LeagueState())
        H, A = teams[hid], teams[aid]
        H.division = A.division = div

        rows.append(pre_match_row(H.snapshot(), A.snapshot(), lg.snapshot(), date))

        # ── update with the result ─────────────────────────────────────────
        hg, ag = int(cols["hg"][i]), int(cols["ag"][i])
        gd = hg - ag
        # Elo
        diff = H.elo - A.elo + ELO_HFA
        exp_h = _elo_expect(diff)
        score_h = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        mult = math.log(abs(gd) + 1) + 1.0
        mult *= 2.2 / ((diff if gd > 0 else -diff) * 0.001 + 2.2) if gd != 0 else 1.0
        delta = ELO_K * mult * (score_h - exp_h)
        H.elo += delta
        A.elo -= delta
        # pi-ratings
        pred_gd = _pi_expected(H.pi_home) - _pi_expected(A.pi_away)
        err = abs(gd - pred_gd)
        psi = PI_C * math.log10(1 + err)
        psi_h = psi if gd > pred_gd else -psi
        d_hh = psi_h * PI_LAMBDA
        d_aa = -psi_h * PI_LAMBDA
        H.pi_home += d_hh
        H.pi_away += d_hh * PI_GAMMA
        A.pi_away += d_aa
        A.pi_home += d_aa * PI_GAMMA
        # Form
        pts_h = 3 if gd > 0 else 1 if gd == 0 else 0
        pts_a = 3 if gd < 0 else 1 if gd == 0 else 0
        hs, as_, hst, ast, hc, ac = (float(cols[c][i]) for c in ("hs", "as_", "hst", "ast", "hc", "ac"))
        upd_h = {"gf": hg, "ga": ag, "sf": hs, "sa": as_, "stf": hst, "sta": ast,
                 "cf": hc, "ca": ac, "pts": pts_h}
        upd_a = {"gf": ag, "ga": hg, "sf": as_, "sa": hs, "stf": ast, "sta": hst,
                 "cf": ac, "ca": hc, "pts": pts_a}
        for T, upd in ((H, upd_h), (A, upd_a)):
            for s, v in upd.items():
                T.short[s].update(float(v))
                T.long[s].update(float(v))
            T.games += 1
            T.last_date = date
        H.mkt_strength.update(float(exp_pts_h[i]))
        A.mkt_strength.update(float(exp_pts_a[i]))
        # League context
        lg.hg.update(hg); lg.ag.update(ag)
        lg.draw.update(float(gd == 0)); lg.home_win.update(float(gd > 0))
        lg.n += 1

    feats = pd.DataFrame(rows, index=matches.index)
    # Match-level market features (known pre-match)
    feats["mkt_p_h"], feats["mkt_p_d"], feats["mkt_p_a"] = p1x2.T
    feats["mkt_max_p_h"], feats["mkt_max_p_d"], feats["mkt_max_p_a"] = pmax.T
    feats["mkt_overround"] = overround
    feats["mkt_p_o25"] = pou[:, 0]
    feats["ah_line"] = matches["ah_line"].to_numpy()
    feats["tier"] = matches["tier"].to_numpy()
    return feats, teams, leagues


# Feature groups ----------------------------------------------------------------
def fundamental_features(cols) -> list[str]:
    """Everything except the current match's market prices / market-derived strength."""
    return [c for c in cols if not c.startswith(("mkt_", "ah_")) and c != "d_mkt_str"]


def market_features(cols) -> list[str]:
    return list(cols)
