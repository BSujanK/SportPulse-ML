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


# Transfermarkt citizenship -> martj42 international-results team names
COUNTRY_ALIAS = {"Türkiye": "Turkey", "Korea, South": "South Korea", "Cote d'Ivoire": "Ivory Coast",
                 "Ireland": "Republic of Ireland", "Bosnia-Herzegovina": "Bosnia and Herzegovina", "The Gambia": "Gambia",
                 "Curacao": "Curaçao", "Chinese Taipei": "Taiwan", "Brunei Darussalam": "Brunei", "Hongkong": "Hong Kong",
                 "Sao Tome and Principe": "São Tomé and Príncipe", "St. Kitts & Nevis": "Saint Kitts and Nevis",
                 "St. Lucia": "Saint Lucia", "Korea, North": "North Korea", "Southern Sudan": "South Sudan",
                 "Macao": "Macau", "St. Vincent & Grenadinen": "Saint Vincent and the Grenadines"}


def _season(d: pd.Series) -> np.ndarray:
    return (d.dt.year - (d.dt.month < 7)).to_numpy()


def squad_values(start: str = "2005-01-01", end: str | None = None, covered_only: bool = True) -> pd.DataFrame:
    """Monthly national squad strength: on the 1st of each month, every player's latest valuation from the
    previous 365 days (strictly before that date), grouped by citizenship. sv26 = log of the summed value
    of the 26 most valuable players, sv11 = mean log value of the top 11, svn = players found.

    covered_only keeps a valuation only if the player's club at that date played games in the dataset that
    season. The source lists players who reached a covered league at any time up to 2026, so without this
    an early snapshot would include players selected by their later careers (look-ahead)."""
    P = table("players").set_index("player_id")
    cit = P.country_of_citizenship.map(lambda c: COUNTRY_ALIAS.get(c, c))
    v = table("player_valuations")
    if covered_only:
        g = table("games")
        seasons = pd.concat([pd.DataFrame({"club": g.home_club_id, "season": _season(g.date)}),
                             pd.DataFrame({"club": g.away_club_id, "season": _season(g.date)})])
        covered = set(zip(seasons.club, seasons.season))
        keep = [(c, s) in covered for c, s in zip(v.current_club_id, _season(v.date))]
        v = v[np.asarray(keep)]
    v = v.assign(country=v.player_id.map(cit), lv=np.log(v.market_value_in_eur.clip(lower=10_000)))
    v = v.dropna(subset=["country"]).sort_values("date")
    end = end or (v.date.max() + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d")
    out = []
    for m in pd.date_range(start, end, freq="MS"):
        w = v[(v.date < m) & (v.date >= m - pd.Timedelta(days=365))].drop_duplicates("player_id", keep="last")
        w = w.sort_values("lv", ascending=False)
        w["rank"] = w.groupby("country").cumcount()
        top26 = w[w["rank"] < 26].groupby("country").lv.apply(lambda s: np.log(np.exp(s).sum()))
        top11 = w[w["rank"] < 11].groupby("country").lv.mean()
        out.append(pd.DataFrame({"month": m, "country": top26.index, "sv26": top26.values,
                                 "sv11": top11.reindex(top26.index).values,
                                 "svn": w.groupby("country").size().reindex(top26.index).values}))
    return pd.concat(out, ignore_index=True)


FD_TO_TM = {"E0": "GB1", "SP1": "ES1", "I1": "IT1", "F1": "FR1", "T1": "TR1", "D1": "L1", "N1": "NL1", "P1": "PO1",
            "B1": "BE1", "RUS": "RU1", "G1": "GR1", "SC0": "SC1", "DEN": "DK1", "USA": "MLS1", "POL": "PL1",
            "BRA": "BRA1", "ROM": "RO1", "SUI": "C1", "ARG": "ARG1", "AUT": "A1", "NOR": "NO1", "SWE": "SE1",
            "JAP": "JAP1", "MEX": "MEX1"}


def club_league(club_ids, dates) -> np.ndarray:
    """Each club's domestic league in the season of each date, from that season's league games (point in time).
    Seasons after the Transfermarkt snapshot fall back to the snapshot's league; uncovered clubs get NaN."""
    g = table("games")
    g = g[g.competition_type.eq("domestic_league")]
    rows = pd.concat([pd.DataFrame({"club": g.home_club_id, "season": _season(g.date), "lg": g.competition_id}),
                      pd.DataFrame({"club": g.away_club_id, "season": _season(g.date), "lg": g.competition_id})])
    lookup = rows.groupby(["club", "season"]).lg.agg(lambda s: s.mode().iloc[0])
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    key = pd.MultiIndex.from_arrays([pd.Series(club_ids).reset_index(drop=True), _season(d)])
    out = lookup.reindex(key).to_numpy(object)
    snapshot = table("clubs").set_index("club_id")["domestic_competition_id"]
    late = (d >= g.date.max() - pd.Timedelta(days=30)).to_numpy()
    fallback = pd.Series(club_ids).reset_index(drop=True).map(snapshot).to_numpy(object)
    return np.where(pd.isna(out) & late, fallback, out)


def club_id_map(matches: pd.DataFrame) -> dict:
    """football-data (division|team) -> Transfermarkt club_id, by joining fixtures on date and score
    within each league and taking each name's majority club (at least 5 joined fixtures, 80% agreement)."""
    g = table("games")
    g = g[g.competition_type.eq("domestic_league")].dropna(subset=["home_club_goals"])
    out = {}
    for div, comp in FD_TO_TM.items():
        a = matches[matches.division == div][["date", "home", "away", "hg", "ag"]]
        b = g[g.competition_id == comp][["date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]]
        b = b.rename(columns={"home_club_goals": "hg", "away_club_goals": "ag"})
        # several fixtures on one day often share a score: vote only on (date, score) keys unique in both sources
        key = ["date", "hg", "ag"]
        a = a[~a.duplicated(key, keep=False)]
        b = b[~b.duplicated(key, keep=False)]
        j = a.merge(b, on=key)
        votes = pd.concat([j[["home", "home_club_id"]].set_axis(["team", "club"], axis=1),
                           j[["away", "away_club_id"]].set_axis(["team", "club"], axis=1)])
        for team, s in votes.groupby("team").club:
            vc = s.value_counts()
            if vc.iloc[0] >= 5 and vc.iloc[0] / vc.sum() >= 0.8:
                out[f"{div}|{team}"] = int(vc.index[0])
    return out


def club_values(start: str = "2012-07-01", end: str | None = None) -> pd.DataFrame:
    """Monthly club squad strength from players' latest valuations (previous 365 days, strictly before the
    1st of the month) at clubs covered that season. cv18 = log summed value of the 18 most valuable,
    cv11 = mean log value of the top 11."""
    v = table("player_valuations")
    g = table("games")
    seasons = pd.concat([pd.DataFrame({"club": g.home_club_id, "season": _season(g.date)}),
                         pd.DataFrame({"club": g.away_club_id, "season": _season(g.date)})])
    covered = set(zip(seasons.club, seasons.season))
    v = v[np.asarray([(c, s) in covered for c, s in zip(v.current_club_id, _season(v.date))])]
    v = v.assign(lv=np.log(v.market_value_in_eur.clip(lower=10_000))).sort_values("date")
    end = end or (v.date.max() + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d")
    out = []
    for m in pd.date_range(start, end, freq="MS"):
        w = v[(v.date < m) & (v.date >= m - pd.Timedelta(days=365))].drop_duplicates("player_id", keep="last")
        w = w.sort_values("lv", ascending=False)
        w["rank"] = w.groupby("current_club_id").cumcount()
        top18 = w[w["rank"] < 18].groupby("current_club_id").lv.apply(lambda s: np.log(np.exp(s).sum()))
        top11 = w[w["rank"] < 11].groupby("current_club_id").lv.mean()
        out.append(pd.DataFrame({"month": m, "club_id": top18.index.astype(int), "cv18": top18.values,
                                 "cv11": top11.reindex(top18.index).values}))
    return pd.concat(out, ignore_index=True)
