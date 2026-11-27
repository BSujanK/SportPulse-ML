"""Transfermarkt data (dcaribou/transfermarkt-datasets snapshot, up to July 2026).

Market valuations, appearances, games and transfers with real fees. Every feature built
here for a (player, date) query uses only rows strictly before that date.
"""
from __future__ import annotations

import logging
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd

from sportpulse import config

log = logging.getLogger(__name__)
BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{}.csv.gz"
DIR = config.RAW_DIR / "transfermarkt"
TABLES = ("players", "player_valuations", "appearances", "games", "clubs", "competitions", "transfers")
TOP5 = {"GB1", "ES1", "IT1", "L1", "FR1"}
UEFA = {"CL", "EL", "UCOL", "ECLQ", "CLQ", "ELQ", "USC"}
DAY0 = pd.Timestamp("2000-01-01")
SPAN = 100_000                      # days per entity in the combined (entity, day) key


def download(force: bool = False) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        p = DIR / f"{t}.csv.gz"
        if force or not p.exists():
            log.info("Downloading %s", t)
            urllib.request.urlretrieve(BASE.format(t), p)


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    download()
    df = pd.read_csv(DIR / f"{name}.csv.gz", low_memory=False)
    for c in ("date", "date_of_birth", "transfer_date", "contract_expiration_date"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _day(s) -> np.ndarray:
    return ((pd.to_datetime(s) - DAY0).dt.days if isinstance(s, pd.Series)
            else (pd.DatetimeIndex(s) - DAY0).days).to_numpy(np.int64)


class WindowIndex:
    """Sums of per-event stats over [t - window, t) for an entity, via one global cumsum."""

    def __init__(self, entity: np.ndarray, day: np.ndarray, stats: pd.DataFrame):
        key = entity.astype(np.int64) * SPAN + day
        o = np.argsort(key, kind="stable")
        self.key = key[o]
        self.names = list(stats.columns)
        vals = stats.to_numpy(float)[o]
        self.cum = np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(vals, 0)])

    def sums(self, entity, day, window: int | None) -> pd.DataFrame:
        e = np.asarray(entity, np.int64)
        hi = np.searchsorted(self.key, e * SPAN + np.asarray(day), "left")
        lo_day = np.zeros_like(day) if window is None else np.asarray(day) - window
        lo = np.searchsorted(self.key, e * SPAN + np.maximum(lo_day, 0), "left")
        out = self.cum[hi] - self.cum[lo]
        n = (hi - lo).astype(float)
        return pd.DataFrame(np.c_[n, out], columns=["n"] + self.names)


class AsOf:
    """Leak-free feature builder for (player, date[, club]) queries."""

    def __init__(self):
        apps = table("appearances")
        comps = table("competitions").set_index("competition_id")
        games = table("games")
        a = pd.DataFrame({
            "minutes": apps.minutes_played.fillna(0), "goals": apps.goals.fillna(0),
            "assists": apps.assists.fillna(0), "yellow": apps.yellow_cards.fillna(0),
            "red": apps.red_cards.fillna(0), "full90": (apps.minutes_played >= 85).astype(float),
        })
        comp = apps.competition_id.fillna("NA").astype(str)
        typ = comp.map(comps["type"]).fillna("")
        a["min_top5"] = a.minutes * comp.isin(TOP5)
        a["min_uefa"] = a.minutes * comp.isin(UEFA)
        a["min_national"] = a.minutes * typ.eq("national_team_competition")
        a["ga_top5"] = (a.goals + a.assists) * comp.isin(TOP5)
        a["ga_uefa"] = (a.goals + a.assists) * comp.isin(UEFA)
        self.apps = WindowIndex(apps.player_id.to_numpy(), _day(apps.date), a)
        # last club before t: store per-appearance club id in a sorted key array
        k = apps.player_id.to_numpy(np.int64) * SPAN + _day(apps.date)
        o = np.argsort(k, kind="stable")
        self._app_key, self._app_club = k[o], apps.player_club_id.to_numpy(np.int64)[o]

        # club results (all competitions) for strength over the previous year
        g = games.dropna(subset=["home_club_goals", "away_club_goals"])
        hg, ag = g.home_club_goals.to_numpy(float), g.away_club_goals.to_numpy(float)
        pts_h = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts_a = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        dom = g.competition_type.eq("domestic_league").to_numpy(float)
        uefa = g.competition_id.isin(UEFA).to_numpy(float)
        club = np.r_[g.home_club_id.to_numpy(), g.away_club_id.to_numpy()]
        day = np.r_[_day(g.date), _day(g.date)]
        st = pd.DataFrame({"pts": np.r_[pts_h, pts_a], "gd": np.r_[hg - ag, ag - hg],
                           "league_games": np.r_[dom, dom], "league_pts": np.r_[pts_h * dom, pts_a * dom],
                           "uefa_games": np.r_[uefa, uefa]})
        self.club = WindowIndex(club, day, st)

        # valuation history (strictly before t)
        v = table("player_valuations").sort_values(["player_id", "date"])
        self._v_key = v.player_id.to_numpy(np.int64) * SPAN + _day(v.date)
        self._v_val = np.log(v.market_value_in_eur.clip(lower=10_000).to_numpy(float))
        self._v_pid = v.player_id.to_numpy(np.int64)
        self._v_cummax = v.assign(l=self._v_val).groupby("player_id").l.cummax().to_numpy()
        self._v_start = np.searchsorted(self._v_key, self._v_pid * SPAN, "left")

        # league and club value indices: mean log valuation over the previous 365 days
        comp_v = v.player_club_domestic_competition_id.fillna("NA").astype(str)
        self._comp_codes = {c: i for i, c in enumerate(sorted(comp_v.unique()))}
        self.league_val = WindowIndex(comp_v.map(self._comp_codes).to_numpy(), _day(v.date),
                                      pd.DataFrame({"lv": self._v_val}))
        self.club_val = WindowIndex(v.current_club_id.fillna(-1).to_numpy(np.int64) + 1, _day(v.date),
                                    pd.DataFrame({"cv": self._v_val}))
        self.players = table("players").set_index("player_id")

    # -- pieces ---------------------------------------------------------------------------
    def last_club(self, pid, day) -> np.ndarray:
        i = np.searchsorted(self._app_key, np.asarray(pid, np.int64) * SPAN + day, "left") - 1
        ok = (i >= 0) & (self._app_key[np.maximum(i, 0)] // SPAN == pid)
        return np.where(ok, self._app_club[np.maximum(i, 0)], -1)

    def valuation_lags(self, pid, day) -> pd.DataFrame:
        pid = np.asarray(pid, np.int64)
        i = np.searchsorted(self._v_key, pid * SPAN + day, "left") - 1
        ok = (i >= 0) & (self._v_pid[np.maximum(i, 0)] == pid)
        j = np.maximum(i, 0)
        prev = np.where(ok, self._v_val[j], np.nan)
        prev_day = np.where(ok, self._v_key[j] % SPAN, np.nan)
        ok2 = ok & (i - 1 >= 0) & (self._v_pid[np.maximum(i - 1, 0)] == pid)
        prev2 = np.where(ok2, self._v_val[np.maximum(i - 1, 0)], np.nan)
        y1 = np.searchsorted(self._v_key, pid * SPAN + day - 365, "left") - 1
        ok1 = (y1 >= 0) & (self._v_pid[np.maximum(y1, 0)] == pid)
        val_1y = np.where(ok1, self._v_val[np.maximum(y1, 0)], np.nan)
        n_val = np.where(ok, i - self._v_start[j] + 1, 0)
        return pd.DataFrame({"prev_log_value": prev, "days_since_value": day - prev_day,
                             "prev_change": prev - prev2, "change_1y": prev - val_1y,
                             "peak_log_value": np.where(ok, self._v_cummax[j], np.nan),
                             "below_peak": np.where(ok, self._v_cummax[j], np.nan) - prev,
                             "n_valuations": n_val})

    def performance(self, pid, day) -> pd.DataFrame:
        out = {}
        for tag, w in (("365", 365), ("180", 180), ("730", 730), ("career", None)):
            s = self.apps.sums(pid, day, w)
            cols = ["n", "minutes", "goals", "assists"] + (
                ["yellow", "red", "full90", "min_top5", "min_uefa", "min_national", "ga_top5", "ga_uefa"]
                if tag == "365" else [])
            for c in cols:
                out[f"{'apps' if c == 'n' else c}_{tag}"] = s[c].to_numpy()
            m = s["minutes"].replace(0, np.nan).to_numpy()
            out[f"goals_p90_{tag}"] = np.clip(s["goals"].to_numpy() / m * 90, 0, 3)
            out[f"assists_p90_{tag}"] = np.clip(s["assists"].to_numpy() / m * 90, 0, 3)
        return pd.DataFrame(out)

    def club_strength(self, club, day, prefix: str) -> pd.DataFrame:
        s = self.club.sums(club, day, 365)
        n = s["n"].replace(0, np.nan)
        lg = s["league_games"].replace(0, np.nan)
        cv = self.club_val.sums(np.asarray(club, np.int64) + 1, day, 365)
        return pd.DataFrame({f"{prefix}games_365": s["n"], f"{prefix}ppg_365": s["pts"] / n,
                             f"{prefix}gd_pg_365": s["gd"] / n, f"{prefix}league_ppg_365": s["league_pts"] / lg,
                             f"{prefix}uefa_games_365": s["uefa_games"],
                             f"{prefix}value_index": cv["cv"] / cv["n"].replace(0, np.nan)})

    def league_index(self, comp, day) -> np.ndarray:
        code = pd.Series(comp).fillna("NA").astype(str).map(self._comp_codes).fillna(-1).to_numpy(np.int64)
        s = self.league_val.sums(np.maximum(code, 0), day, 365)
        return np.where(code >= 0, s["lv"] / s["n"].replace(0, np.nan), np.nan)

    def player_static(self, pid, day) -> pd.DataFrame:
        P = self.players.reindex(pid)
        dob = P["date_of_birth"]
        age = ((DAY0 + pd.to_timedelta(day, "D")) - dob.to_numpy()).days / 365.25
        return pd.DataFrame({"age": np.asarray(age, float), "height": P["height_in_cm"].to_numpy(float),
                             "position": P["position"].to_numpy(), "sub_position": P["sub_position"].to_numpy(),
                             "foot": P["foot"].to_numpy()})

    def build(self, pid, dates, club=None, comp=None, lags: bool = True) -> pd.DataFrame:
        pid = np.asarray(pid, np.int64)
        day = _day(pd.Series(pd.to_datetime(dates)))
        club = self.last_club(pid, day) if club is None else np.asarray(club, np.int64)
        parts = [self.player_static(pid, day), self.performance(pid, day), self.club_strength(club, day, "club_")]
        if lags:
            parts.append(self.valuation_lags(pid, day))
        F = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
        if comp is not None:
            F["league_value_index"] = self.league_index(comp, day)
        # club coverage is thin outside the dataset's leagues: no share when the club has < 15 known games
        F["minutes_share_365"] = (F["minutes_365"] / (F["club_games_365"] * 90)).where(F["club_games_365"] >= 15)
        F["age2"] = F["age"] ** 2
        return F
