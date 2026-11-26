"""StatsBomb open event data -> compact analysis tables.

Source: https://github.com/statsbomb/open-data (free, CC BY-NC-SA, attribution
"Data provided by StatsBomb"). Coordinates: 120 x 80 pitch, every team's
events oriented attacking towards x = 120.

Events are streamed match by match and reduced to:
  matches · shots (+ freeze-frame geometry) · passes · carries · defensive actions · lineups
Raw JSON is not kept on disk.
"""
from __future__ import annotations

import json
import logging
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sportpulse import config

log = logging.getLogger(__name__)
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = config.PROCESSED_DIR / "statsbomb"

# (competition_id, season_id). Recent competitions form the analysis set; the
# 2015/16 big-five seasons and older tournaments add shot volume for the xG model.
COMPETITIONS = [
    (43, 106), (55, 282), (223, 282), (1267, 107), (9, 281), (7, 235), (44, 107),   # 2022-2024
    (55, 43), (7, 108), (11, 90), (43, 3),                                          # 2018-2022
    (2, 27), (11, 27), (12, 27), (9, 27), (7, 27), (16, 4), (16, 1),                # 2015-2019
]
GOAL_X, POST_L, POST_R = 120.0, 36.0, 44.0
DEF_TYPES = {"Pressure", "Duel", "Interception", "Ball Recovery", "Block", "Clearance",
             "Foul Committed", "Dribbled Past"}


def _get_json(url: str, retries: int = 4):
    req = urllib.request.Request(url, headers={"User-Agent": "sportpulse-ml/2.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001 - transient network errors
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def load_match_list() -> pd.DataFrame:
    rows = []
    for c, s in COMPETITIONS:
        for m in _get_json(f"{BASE}/matches/{c}/{s}.json"):
            rows.append({
                "match_id": m["match_id"], "competition_id": c, "season_id": s,
                "competition": m["competition"]["competition_name"], "season": m["season"]["season_name"],
                "date": m["match_date"], "kickoff": m.get("kick_off"),
                "home": m["home_team"]["home_team_name"], "away": m["away_team"]["away_team_name"],
                "home_score": m["home_score"], "away_score": m["away_score"],
                "stage": (m.get("competition_stage") or {}).get("name"),
                "stadium": (m.get("stadium") or {}).get("name"),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["label"] = df["competition"] + " " + df["season"]
    return df.drop_duplicates("match_id").reset_index(drop=True)


# ── freeze-frame geometry ─────────────────────────────────────────────────────
def _in_triangle(p, a, b, c) -> bool:
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def freeze_frame_features(loc, ff) -> dict:
    out = {"ff_available": 0.0}
    if not ff:
        return out
    a, b = (GOAL_X, POST_L), (GOAL_X, POST_R)
    opp = [f for f in ff if not f.get("teammate")]
    mates = [f for f in ff if f.get("teammate") and not f.get("actor")]
    gk = [f for f in opp if (f.get("position") or {}).get("name") == "Goalkeeper"]
    outfield = [f for f in opp if f not in gk]
    dist = lambda f: math.dist(loc, f["location"])  # noqa: E731
    out.update({
        "ff_available": 1.0,
        "ff_def_in_cone": float(sum(_in_triangle(f["location"], loc, a, b) for f in outfield)),
        "ff_mates_in_cone": float(sum(_in_triangle(f["location"], loc, a, b) for f in mates)),
        "ff_def_within_2": float(sum(dist(f) < 2 for f in outfield)),
        "ff_def_within_5": float(sum(dist(f) < 5 for f in outfield)),
        "ff_nearest_def": min((dist(f) for f in outfield), default=np.nan),
    })
    if gk:
        g = gk[0]["location"]
        out.update({"ff_gk_dist_goal": math.dist(g, (GOAL_X, 40.0)), "ff_gk_dist_shooter": math.dist(g, loc),
                    "ff_gk_in_cone": float(_in_triangle(g, loc, a, b)), "ff_gk_off_line": GOAL_X - g[0]})
    return out


def _name(d, k="name"):
    return (d or {}).get(k)


def parse_match(match_id: int) -> dict[str, list]:
    ev = _get_json(f"{BASE}/events/{match_id}.json")
    by_id = {e["id"]: e for e in ev}
    shots, passes, carries, defense, lineups = [], [], [], [], []
    for e in ev:
        t = e["type"]["name"]
        team, player = _name(e.get("team")), _name(e.get("player"))
        loc = e.get("location")
        base = {"match_id": match_id, "team": team, "player": player, "period": e["period"],
                "minute": e["minute"], "second": e["second"], "possession": e.get("possession"),
                "play_pattern": _name(e.get("play_pattern")), "under_pressure": bool(e.get("under_pressure"))}
        if t == "Starting XI":
            tac = e.get("tactics", {})
            for p in tac.get("lineup", []):
                lineups.append({"match_id": match_id, "team": team, "formation": tac.get("formation"),
                                "player": p["player"]["name"], "position": p["position"]["name"],
                                "jersey": p.get("jersey_number")})
        elif t == "Shot" and loc:
            s = e["shot"]
            kp = by_id.get(s.get("key_pass_id"))
            kpp = (kp or {}).get("pass", {})
            dx, dy = GOAL_X - loc[0], 40.0 - loc[1]
            ang = math.atan2(8 * dx, dx ** 2 + dy ** 2 - 16) if dx > 0 else 0.0
            shots.append({**base, "id": e["id"], "x": loc[0], "y": loc[1],
                          "end_x": (s.get("end_location") or [np.nan] * 2)[0],
                          "end_y": (s.get("end_location") or [np.nan] * 2)[1],
                          "distance": math.hypot(dx, dy), "angle": ang if ang >= 0 else ang + math.pi,
                          "body_part": _name(s.get("body_part")), "shot_type": _name(s.get("type")),
                          "technique": _name(s.get("technique")), "outcome": _name(s.get("outcome")),
                          "first_time": bool(s.get("first_time")), "one_on_one": bool(s.get("one_on_one")),
                          "aerial_won": bool(s.get("aerial_won")), "open_goal": bool(s.get("open_goal")),
                          "follows_dribble": bool(s.get("follows_dribble")), "deflected": bool(s.get("deflected")),
                          "assist_type": ("through_ball" if kpp.get("through_ball") or _name(kpp.get("technique")) == "Through Ball"
                                          else "cut_back" if kpp.get("cut_back") else "cross" if kpp.get("cross")
                                          else None if kp is None else "pass"),
                          "assist_height": _name(kpp.get("height")), "assist_player": _name((kp or {}).get("player")),
                          "assist_x": ((kp or {}).get("location") or [np.nan])[0],
                          "goal": float(_name(s.get("outcome")) == "Goal"), "sb_xg": s.get("statsbomb_xg"),
                          **freeze_frame_features(loc, s.get("freeze_frame"))})
        elif t == "Pass" and loc:
            p = e["pass"]
            end = p.get("end_location") or [np.nan, np.nan]
            passes.append({**base, "x": loc[0], "y": loc[1], "end_x": end[0], "end_y": end[1],
                           "recipient": _name(p.get("recipient")), "complete": p.get("outcome") is None,
                           "height": _name(p.get("height")), "pass_type": _name(p.get("type")),
                           "cross": bool(p.get("cross")), "switch": bool(p.get("switch")),
                           "through_ball": bool(p.get("through_ball")) or _name(p.get("technique")) == "Through Ball",
                           "shot_assist": bool(p.get("shot_assist")), "goal_assist": bool(p.get("goal_assist")),
                           "length": p.get("length")})
        elif t == "Carry" and loc:
            end = e["carry"].get("end_location") or [np.nan, np.nan]
            carries.append({**base, "x": loc[0], "y": loc[1], "end_x": end[0], "end_y": end[1]})
        elif t in DEF_TYPES and loc:
            sub = None
            if t == "Duel":
                sub = _name(e["duel"].get("type"))
                if sub != "Tackle":
                    continue
            defense.append({**base, "type": t if t != "Duel" else "Tackle", "x": loc[0], "y": loc[1],
                            "outcome": _name((e.get(t.lower().replace(" ", "_")) or {}).get("outcome"))})
    return {"shots": shots, "passes": passes, "carries": carries, "defense": defense, "lineups": lineups}


def ingest(force: bool = False, workers: int = 16) -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["shots", "passes", "carries", "defense", "lineups"]
    if not force and all((OUT / f"{n}.parquet").exists() for n in names + ["matches"]):
        return {n: pd.read_parquet(OUT / f"{n}.parquet") for n in names + ["matches"]}
    matches = load_match_list()
    acc = {n: [] for n in names}
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(parse_match, int(m)): m for m in matches["match_id"]}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                for n in names:
                    acc[n].extend(r[n])
            except Exception as e:  # noqa: BLE001 - skip broken matches, report count
                failed.append((futs[f], str(e)[:80]))
            if i % 200 == 0:
                log.info("parsed %d / %d matches", i, len(futs))
    if failed:
        log.warning("%d matches failed: %s", len(failed), failed[:3])
        matches = matches[~matches.match_id.isin([m for m, _ in failed])]
    out = {n: pd.DataFrame(acc[n]) for n in names}
    out["matches"] = matches.reset_index(drop=True)
    for n, d in out.items():
        d.to_parquet(OUT / f"{n}.parquet", index=False)
    return out


def display_names(match_ids, workers: int = 16) -> dict[str, str]:
    """Full StatsBomb names -> commonly used names ("Lionel Andrés Messi Cuccittini" -> "Lionel Messi"),
    from the per-match line-up files (cached)."""
    path = OUT / "display_names.parquet"
    if path.exists():
        d = pd.read_parquet(path)
        return dict(zip(d.player, d.display))
    rows = []

    def one(mid):
        out = []
        for team in _get_json(f"{BASE}/lineups/{mid}.json"):
            for p in team.get("lineup", []):
                out.append((p["player_name"], p.get("player_nickname") or p["player_name"]))
        return out

    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, int(m)) for m in match_ids]):
            try:
                rows.extend(f.result())
            except Exception:  # noqa: BLE001 - names are cosmetic; fall back to full names
                pass
    d = pd.DataFrame(rows, columns=["player", "display"]).drop_duplicates("player")
    d.to_parquet(path, index=False)
    return dict(zip(d.player, d.display))
