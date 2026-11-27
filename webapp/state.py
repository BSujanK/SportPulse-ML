"""Cached artifact access for the Streamlit app."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from sportpulse import config
from sportpulse.inference import MatchPredictor
from sportpulse.models.players import similarity_space


def artifacts_ready() -> bool:
    """Every metrics file a page reads unconditionally must exist, or the app shows the 'train first' message."""
    need = [config.MATCH_ARTIFACTS, config.PLAYER_ARTIFACTS, config.ARTIFACTS_DIR / "fpl",
            config.ARTIFACTS_DIR / "international", config.ARTIFACTS_DIR / "analyst", config.ARTIFACTS_DIR / "market_value"]
    return all((d / "metrics.json").exists() for d in need)


@st.cache_resource(show_spinner="Loading match models…")
def predictor() -> MatchPredictor:
    return MatchPredictor()


@st.cache_data(show_spinner=False)
def match_metrics() -> dict:
    return json.loads((config.MATCH_ARTIFACTS / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def player_metrics() -> dict:
    return json.loads((config.PLAYER_ARTIFACTS / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def test_predictions() -> pd.DataFrame:
    return pd.read_parquet(config.MATCH_ARTIFACTS / "test_predictions.parquet")


@st.cache_data(show_spinner=False)
def feature_importance() -> pd.DataFrame:
    return pd.read_csv(config.MATCH_ARTIFACTS / "feature_importance.csv", index_col=0)


@st.cache_data(show_spinner=False)
def elo_history() -> pd.DataFrame:
    return pd.read_parquet(config.MATCH_ARTIFACTS / "elo_history.parquet")


@st.cache_data(show_spinner=False)
def recent_matches() -> pd.DataFrame:
    return pd.read_parquet(config.MATCH_ARTIFACTS / "recent_matches.parquet")


@st.cache_data(show_spinner="Loading player database…")
def players() -> pd.DataFrame:
    df = pd.read_parquet(config.PLAYER_ARTIFACTS / "players_latest.parquet")
    df["label"] = df["short_name"] + " · " + df["club_name"].fillna("Free agent") + \
        " · " + df["overall"].astype(int).astype(str)
    return df


@st.cache_data(show_spinner=False)
def player_history() -> pd.DataFrame:
    return pd.read_parquet(config.PLAYER_ARTIFACTS / "player_history.parquet")


@st.cache_resource(show_spinner=False)
def player_embedding() -> np.ndarray:
    return similarity_space(players())


def season_start(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Start of the most recent season present in the data (July boundary)."""
    last = recent_matches()["date"].max()
    year = last.year if last.month >= 7 else last.year - 1
    return pd.Timestamp(f"{year}-07-01")


# ── Analyst layer (StatsBomb) ───────────────────────────────────────────────
AN = config.ARTIFACTS_DIR / "analyst"


@st.cache_data(show_spinner=False)
def analyst(name: str) -> pd.DataFrame:
    return pd.read_parquet(AN / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def analyst_metrics() -> dict:
    return json.loads((AN / "metrics.json").read_text())


def competition_order() -> list[str]:
    """Competitions, most recent first."""
    m = analyst("matches")
    return m.groupby("label").date.max().sort_values(ascending=False).index.tolist()


MV = config.ARTIFACTS_DIR / "market_value"


@st.cache_data(show_spinner=False)
def market_value(name: str) -> pd.DataFrame:
    return pd.read_parquet(MV / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def market_value_metrics() -> dict:
    return json.loads((MV / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def vaep_metrics() -> dict | None:
    p = AN / "vaep_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None
