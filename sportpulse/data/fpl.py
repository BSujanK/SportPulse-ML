"""Fantasy Premier League data (vaastav/Fantasy-Premier-League, 2016-17 -> current season).

Two uses:
1. Player-level, per-fixture performance data (minutes, goals, assists, xG, xA,
   bonus, FPL points) for the player performance forecaster.
2. A *live results feed*: finished Premier League fixtures of the current season
   that are newer than the club-match dataset are appended to it, so the
   Premier League ratings are current to the latest gameweek.
"""
from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd

from sportpulse import config
from sportpulse.data.loaders import _download

log = logging.getLogger(__name__)

FPL_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FPL_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
               "2023-24", "2024-25", "2025-26", "2026-27"]
FPL_DIR = config.RAW_DIR / "fpl"

# FPL team name -> football-data.co.uk name
FPL_TO_FD = {
    "Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United",
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester", "Luton Town": "Luton", "Norwich City": "Norwich",
    "Cardiff City": "Cardiff", "Swansea City": "Swansea", "Stoke City": "Stoke",
    "Huddersfield Town": "Huddersfield", "West Brom": "West Brom", "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest", "Wolverhampton": "Wolves",
}
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fd_name(n: str) -> str:
    return FPL_TO_FD.get(n, n)


def _get(season: str, name: str, force: bool = False):
    sub = "gws/merged_gw.csv" if name == "merged_gw.csv" else name
    path = FPL_DIR / season / name
    try:
        _download(f"{FPL_BASE}/{season}/{sub}", path, force)
    except Exception as e:  # noqa: BLE001 - some early seasons lack teams/fixtures
        log.debug("FPL %s/%s unavailable: %s", season, name, e)
        return None
    return path


@functools.lru_cache(maxsize=None)
def team_names(season: str, force: bool = False) -> dict[int, str]:
    p = _get(season, "teams.csv", force)
    if p is not None:
        t = pd.read_csv(p)
        return dict(zip(t["id"], t["name"].map(fd_name)))
    master = FPL_DIR / "master_team_list.csv"
    _download(f"{FPL_BASE}/master_team_list.csv", master)
    t = pd.read_csv(master)
    t = t[t["season"] == season]
    return dict(zip(t["team"], t["team_name"].map(fd_name)))


def current_season() -> str:
    return FPL_SEASONS[-1]


def fixtures(season: str, force: bool = False) -> pd.DataFrame:
    p = _get(season, "fixtures.csv", force)
    f = pd.read_csv(p)
    names = team_names(season, force)
    f["home"], f["away"] = f["team_h"].map(names), f["team_a"].map(names)
    f["date"] = pd.to_datetime(f["kickoff_time"], utc=True).dt.tz_localize(None).dt.normalize()
    return f


def latest_premier_league_results(after: pd.Timestamp, force: bool = False) -> pd.DataFrame:
    """Finished fixtures of the current FPL season played after `after`."""
    f = fixtures(current_season(), force)
    f = f[f["finished"].astype(bool) & (f["date"] > after)]
    out = pd.DataFrame({"division": "E0", "date": f["date"], "home": f["home"], "away": f["away"],
                        "hg": f["team_h_score"].astype(int), "ag": f["team_a_score"].astype(int)})
    out["source"] = "fpl"
    return out.reset_index(drop=True)


def upcoming_fixtures(force: bool = False) -> pd.DataFrame:
    f = fixtures(current_season(), force)
    f = f[~f["finished"].astype(bool)].sort_values("kickoff_time")
    return f[["event", "kickoff_time", "date", "home", "away", "team_h", "team_a",
              "team_h_difficulty", "team_a_difficulty"]].reset_index(drop=True)


def load_player_gameweeks(force: bool = False) -> pd.DataFrame:
    """Every player appearance-row (one per player per fixture), all seasons."""
    frames = []
    for s in FPL_SEASONS:
        p = _get(s, "merged_gw.csv", force)
        raw = _get(s, "players_raw.csv", force)
        if p is None or raw is None:
            continue
        g = pd.read_csv(p, low_memory=False, encoding_errors="replace")
        pr = pd.read_csv(raw, low_memory=False, encoding_errors="replace")
        names = team_names(s, force)
        pr = pr.rename(columns={"id": "element"})
        g = g.merge(pr[["element", "code", "element_type", "team", "web_name"]].rename(
            columns={"team": "team_id_raw"}), on="element", how="left", suffixes=("", "_pr"))
        g["season"] = s
        g["player_code"] = g["code"]
        g["position"] = g["element_type"].map(POS) if "position" not in g else \
            g["position"].replace({"GKP": "GK"}).fillna(g["element_type"].map(POS))
        # Team names: present from 2020-21; otherwise from players_raw team id (last club of season)
        team_from_id = g["team_id_raw"].map(names)
        g["team"] = g["team"].map(fd_name) if "team" in g else team_from_id
        g["opponent"] = g["opponent_team"].map(names)
        g["kickoff"] = pd.to_datetime(g["kickoff_time"], utc=True).dt.tz_localize(None)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.dropna(subset=["player_code", "kickoff"])
    df["player_code"] = df["player_code"].astype("int64")
    num = ["minutes", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded",
           "bonus", "bps", "saves", "yellow_cards", "red_cards", "own_goals", "penalties_missed",
           "penalties_saved", "influence", "creativity", "threat", "ict_index", "value", "selected",
           "transfers_balance", "expected_goals", "expected_assists", "expected_goal_involvements",
           "expected_goals_conceded", "xP", "starts", "team_h_score", "team_a_score"]
    for c in num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["was_home"] = df["was_home"].astype(str).str.lower().isin(["true", "1"])
    df["name"] = df["name"].astype(str).str.replace(r"_\d+$", "", regex=True).str.replace("_", " ")
    df = df.sort_values(["player_code", "kickoff", "fixture"]).reset_index(drop=True)
    return df


def _ensure(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df
