"""Club squad value from Transfermarkt, as a pre-match feature for club matches.

For each football-data team (division|team) mapped to a Transfermarkt club by fixture join, the
monthly squad value (cv18: log summed value of the 18 most valuable players, cv11: mean log value
of the top 11) is taken on the 1st of the match month, from valuations strictly before it and only
at clubs covered that season. Covered: the 24 top flights both sources share, from 2012-13.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sportpulse import config

log = logging.getLogger(__name__)
PATH = config.MATCH_ARTIFACTS / "club_values.parquet"
COLS = ["cv18_diff", "cv11_diff"]


def team_values(matches: pd.DataFrame) -> pd.DataFrame:
    """Monthly values per team_id. Rebuilt from Transfermarkt when available, else the committed snapshot."""
    try:
        from sportpulse.data import transfermarkt as T
        cmap = T.club_id_map(matches)
        cv = T.club_values(end=(matches.date.max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"))
        # map keys are division|team; team ids are country|team (stable across divisions)
        ids = pd.DataFrame({"team_id": [f"{config.DIVISIONS[k.split('|', 1)[0]][0]}|{k.split('|', 1)[1]}" for k in cmap],
                            "club_id": list(cmap.values())}).drop_duplicates("team_id")
        tv = ids.merge(cv, on="club_id")[["team_id", "month", "cv18", "cv11"]]
        PATH.parent.mkdir(parents=True, exist_ok=True)
        tv.to_parquet(PATH, index=False)
        log.info("Club values: %d teams mapped, %d team-months", len(cmap), len(tv))
        return tv
    except Exception as e:  # noqa: BLE001 - offline: use the committed snapshot
        log.warning("Transfermarkt club values unavailable (%s); using %s", e, PATH)
        return pd.read_parquet(PATH)


def add_features(feats: pd.DataFrame, matches: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    S = tv.set_index(["month", "team_id"])
    mon = matches["date"].dt.to_period("M").dt.to_timestamp()
    h = S.reindex(pd.MultiIndex.from_arrays([mon, matches["home_id"]]))
    a = S.reindex(pd.MultiIndex.from_arrays([mon, matches["away_id"]]))
    feats["cv18_diff"] = h.cv18.to_numpy() - a.cv18.to_numpy()
    feats["cv11_diff"] = h.cv11.to_numpy() - a.cv11.to_numpy()
    return feats


class ClubValueLookup:
    """Live lookup: the snapshot of the fixture's month, or the latest one available."""

    def __init__(self, path=PATH):
        self.tv = pd.read_parquet(path) if path.exists() else None

    def row(self, home_id: str, away_id: str, date) -> dict:
        if self.tv is None:
            return {c: np.nan for c in COLS}
        m = min(pd.Timestamp(date).to_period("M").to_timestamp(), self.tv.month.max())
        S = self.tv[self.tv.month == m].set_index("team_id")
        h, a = S.reindex([home_id]).iloc[0], S.reindex([away_id]).iloc[0]
        return {"cv18_diff": h.cv18 - a.cv18, "cv11_diff": h.cv11 - a.cv11}
