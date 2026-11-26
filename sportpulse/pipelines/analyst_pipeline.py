"""Analyst pipeline: StatsBomb events -> xG, xT, team styles, player output, match reports.

    python -m sportpulse.train analyst
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from sportpulse import config
from sportpulse.analysis import analyst as A
from sportpulse.analysis import xg as XG
from sportpulse.data import statsbomb

log = logging.getLogger(__name__)
OUT = config.ARTIFACTS_DIR / "analyst"
RECENT = {"FIFA World Cup 2022", "UEFA Euro 2024", "Copa America 2024", "African Cup of Nations 2023",
          "1. Bundesliga 2023/2024", "Ligue 1 2022/2023", "Major League Soccer 2023"}


def run(refresh: bool = False) -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    d = statsbomb.ingest(force=refresh)
    matches, shots, passes, carries, defense, lineups = (d[k] for k in
                                                         ("matches", "shots", "passes", "carries", "defense", "lineups"))
    # Penalty shoot-outs (period 5) are not part of the match: drop them everywhere.
    shots, passes, carries, defense = (t[t.period < 5].reset_index(drop=True)
                                       for t in (shots, passes, carries, defense))
    shots = shots.merge(matches[["match_id", "label"]], on="match_id")
    log.info("xG model on %d shots", len(shots))
    booster, xg, xg_metrics = XG.train_and_evaluate(shots, RECENT)
    shots["xg"] = xg.to_numpy()
    booster.save_model(str(OUT / "xg.lgb"))

    log.info("Expected threat")
    grid = A.expected_threat(passes, carries, shots)
    passes["xt_added"] = A.xt_added(passes, grid)
    carries["xt_added"] = A.xt_added(carries.assign(complete=True), grid)
    passes["progressive"] = A.progressive(passes)
    carries["progressive"] = A.progressive(carries) & (np.hypot(carries.end_x - carries.x, carries.end_y - carries.y) >= 5)

    log.info("Team / player / match aggregates")
    tm = A.team_match_stats(matches, shots, passes, defense, carries)
    prof = A.team_profiles(tm)
    players = A.player_stats(matches, lineups, shots, passes, defense, carries)
    nodes, edges = A.pass_networks(passes, lineups)
    lab = matches[["match_id", "label"]]
    heat_press = A.heat_grid(defense[defense.type == "Pressure"].merge(lab, on="match_id"), ["label", "team"])
    heat_def = A.heat_grid(defense[defense.type.isin(["Tackle", "Interception", "Ball Recovery", "Block"])]
                           .merge(lab, on="match_id"), ["label", "team"])
    heat_touch = A.heat_grid(pd.concat([passes[["match_id", "team", "x", "y"]], carries[["match_id", "team", "x", "y"]]])
                             .merge(lab, on="match_id"), ["label", "team"])
    formations = lineups.groupby(["match_id", "team"]).formation.first().reset_index()

    mx = tm[["match_id", "team", "xg"]]
    mt = matches.merge(mx.rename(columns={"team": "home", "xg": "home_xg"}), on=["match_id", "home"], how="left") \
                .merge(mx.rename(columns={"team": "away", "xg": "away_xg"}), on=["match_id", "away"], how="left")
    mt = mt.merge(formations.rename(columns={"team": "home", "formation": "home_formation"}), on=["match_id", "home"], how="left")
    mt = mt.merge(formations.rename(columns={"team": "away", "formation": "away_formation"}), on=["match_id", "away"], how="left")

    # Display names (nicknames) and clean formation strings for the app
    names = statsbomb.display_names(matches.match_id)
    nm = lambda col: col.map(lambda v: names.get(v, v) if isinstance(v, str) else v)  # noqa: E731
    for df_, cols in ((shots, ["player", "assist_player"]), (players, ["player"]), (nodes, ["player"]),
                      (edges, ["a", "b"])):
        for c in cols:
            df_[c] = nm(df_[c])
    for c in ("home_formation", "away_formation"):
        mt[c] = mt[c].map(lambda v: "-".join(str(int(float(v)))) if pd.notna(v) else None)

    shot_cols = ["match_id", "label", "team", "player", "period", "minute", "second", "x", "y", "end_x", "end_y",
                 "body_part", "shot_type", "play_pattern", "outcome", "goal", "xg", "sb_xg", "assist_player",
                 "assist_type", "distance"]
    shots[shot_cols].to_parquet(OUT / "shots.parquet", index=False)
    mt.to_parquet(OUT / "matches.parquet", index=False)
    tm.to_parquet(OUT / "team_match.parquet", index=False)
    prof.to_parquet(OUT / "team_profiles.parquet", index=False)
    players.to_parquet(OUT / "players.parquet", index=False)
    nodes.to_parquet(OUT / "net_nodes.parquet", index=False)
    edges.to_parquet(OUT / "net_edges.parquet", index=False)
    heat_press.to_parquet(OUT / "heat_pressure.parquet", index=False)
    heat_def.to_parquet(OUT / "heat_defense.parquet", index=False)
    heat_touch.to_parquet(OUT / "heat_touch.parquet", index=False)
    np.save(OUT / "xt_grid.npy", grid)

    metrics = {"xg": xg_metrics,
               "data": {"matches": int(len(matches)), "competitions": sorted(matches.label.unique().tolist()),
                        "recent_competitions": sorted(RECENT), "shots": int(len(shots)),
                        "passes": int(len(passes)), "carries": int(len(carries)),
                        "defensive_actions": int(len(defense)), "players": int(players.player.nunique()),
                        "last_match": str(matches.date.max().date())},
               "xt": {"grid": [NX for NX in grid.shape], "max": float(grid.max())},
               "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Analyst pipeline done in %.0fs", metrics["runtime_sec"])
    return metrics
