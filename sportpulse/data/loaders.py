"""Download (with on-disk caching) and clean the raw datasets.

Nothing in this module fabricates values: missing statistics stay missing
(NaN) and are handled natively by the models downstream.
"""
from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sportpulse import config

log = logging.getLogger(__name__)


def _download(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "sportpulse-ml/2.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


# ══════════════════════════════════════════════════════════════════════════════
#  MATCHES
# ══════════════════════════════════════════════════════════════════════════════
_MATCH_RENAME = {
    "Division": "division", "MatchDate": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHome": "hg", "FTAway": "ag", "FTResult": "result",
    "HomeShots": "hs", "AwayShots": "as_", "HomeTarget": "hst", "AwayTarget": "ast",
    "HomeCorners": "hc", "AwayCorners": "ac", "HomeFouls": "hf", "AwayFouls": "af",
    "HomeYellow": "hy", "AwayYellow": "ay", "HomeRed": "hr", "AwayRed": "ar",
    "OddHome": "odds_h", "OddDraw": "odds_d", "OddAway": "odds_a",
    "MaxHome": "max_h", "MaxDraw": "max_d", "MaxAway": "max_a",
    "Over25": "odds_o25", "Under25": "odds_u25",
    "HandiSize": "ah_line", "HandiHome": "ah_h", "HandiAway": "ah_a",
}


def load_matches(force_download: bool = False) -> pd.DataFrame:
    """Return the cleaned, chronologically sorted match table."""
    path = _download(config.MATCHES_URL, config.RAW_DIR / "matches.csv", force_download)
    raw = pd.read_csv(path, low_memory=False)
    df = raw[list(_MATCH_RENAME)].rename(columns=_MATCH_RENAME)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "hg", "ag"])
    df = df[df["division"].isin(config.DIVISIONS)]
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    # Recompute result from the score (source column has a handful of NaNs).
    df["result"] = np.select([df.hg > df.ag, df.hg < df.ag], ["H", "A"], "D")

    # Team identity is (country, name): avoids collisions across countries.
    country = df["division"].map(lambda c: config.DIVISIONS[c][0])
    df["home_id"] = country + "|" + df["home"].str.strip()
    df["away_id"] = country + "|" + df["away"].str.strip()
    df["country"] = country
    df["tier"] = df["division"].map(lambda c: config.DIVISIONS[c][2]).astype(int)

    # ── Odds sanity: decimal odds must be in (1, 100] and the book overround
    #    must be plausible; anything else is a data-entry error -> NaN.
    for trio in (["odds_h", "odds_d", "odds_a"], ["max_h", "max_d", "max_a"]):
        o = df[trio].where((df[trio] > 1.0) & (df[trio] <= 100))
        over = (1.0 / o).sum(axis=1, min_count=3)
        bad = ~over.between(0.98, 1.35)
        o.loc[bad] = np.nan
        df[trio] = o
    for pair in (["odds_o25", "odds_u25"], ["ah_h", "ah_a"]):
        o = df[pair].where((df[pair] > 1.0) & (df[pair] <= 50))
        over = (1.0 / o).sum(axis=1, min_count=2)
        o.loc[~over.between(0.98, 1.25)] = np.nan
        df[pair] = o

    # Stats sanity: negative/absurd counts -> NaN.
    for c in ["hs", "as_", "hst", "ast", "hc", "ac", "hf", "af", "hy", "ay", "hr", "ar"]:
        df[c] = df[c].where(df[c].between(0, 60))
    # Shots on target cannot exceed shots.
    df.loc[df.hst > df.hs, ["hs", "hst"]] = np.nan
    df.loc[df.ast > df.as_, ["as_", "ast"]] = np.nan

    df = df.drop_duplicates(subset=["date", "home_id", "away_id"])
    df = df.sort_values(["date", "division", "home"], kind="mergesort").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYERS
# ══════════════════════════════════════════════════════════════════════════════
DETAILED_ATTRS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
    "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
    "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking_awareness",
    "defending_standing_tackle", "defending_sliding_tackle",
]
GK_ATTRS = ["goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
            "goalkeeping_positioning", "goalkeeping_reflexes"]
FACE_ATTRS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

_POS_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "LW": "FWD", "RW": "FWD", "ST": "FWD", "CF": "FWD", "LF": "FWD", "RF": "FWD",
}
_ID_RE = re.compile(r"/players/(\d+)/(\d+)/")


def _player_id(url: str) -> float:
    m = _ID_RE.search(str(url))
    return float(m.group(1) + m.group(2)) if m else np.nan


def load_players(force_download: bool = False) -> pd.DataFrame:
    """All FIFA editions stacked, one row per (player, edition)."""
    frames = []
    for v in config.FIFA_EDITIONS:
        path = _download(config.FIFA_URL_TEMPLATE.format(v=v),
                         config.RAW_DIR / f"fifa{v}.csv", force_download)
        d = pd.read_csv(path, low_memory=False)
        d["edition"] = 2000 + v
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["player_id"] = df["player_face_url"].map(_player_id)
    df = df.dropna(subset=["player_id", "overall"])
    df["player_id"] = df["player_id"].astype("int64")
    df = df.drop_duplicates(subset=["player_id", "edition"])

    df["main_position"] = df["player_positions"].astype(str).str.split(",").str[0].str.strip()
    df["position_group"] = df["main_position"].map(_POS_GROUP).fillna("MID")
    df["n_positions"] = df["player_positions"].astype(str).str.count(",") + 1

    contract = pd.to_numeric(df["club_contract_valid_until"], errors="coerce")
    df["contract_years_left"] = (contract - df["edition"] + 1).clip(0, 8)
    df["foot_left"] = (df["preferred_foot"] == "Left").astype(int)
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    df["value_eur"] = df["value_eur"].where(df["value_eur"] > 0)
    df["wage_eur"] = df["wage_eur"].where(df["wage_eur"] > 0)
    df["league_level"] = pd.to_numeric(df["league_level"], errors="coerce")

    keep = (["player_id", "edition", "short_name", "age", "dob", "height_cm", "weight_kg",
             "bmi", "club_name", "league_name", "league_level", "nationality_name",
             "player_positions", "main_position", "position_group", "n_positions",
             "overall", "potential", "value_eur", "wage_eur", "release_clause_eur",
             "international_reputation", "weak_foot", "skill_moves", "foot_left",
             "contract_years_left", "player_face_url"]
            + FACE_ATTRS + DETAILED_ATTRS + GK_ATTRS)
    df = df[keep].sort_values(["edition", "overall"], ascending=[True, False])
    return df.reset_index(drop=True)
