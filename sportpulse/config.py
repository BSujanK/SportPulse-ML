"""Central configuration: paths, data sources, league metadata, split dates."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
MATCH_ARTIFACTS = ARTIFACTS_DIR / "match"
PLAYER_ARTIFACTS = ARTIFACTS_DIR / "players"

SEED = 42

# ── Data sources (public, GitHub-hosted mirrors) ─────────────────────────────
# Club matches 2000-2026, 38 divisions, results + match stats + pre-match odds.
# Originally compiled from football-data.co.uk and clubelo.com by xgabora.
MATCHES_URL = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
)
# EA Sports FIFA 15-22 player attribute tables (sofifa.com scrape).
FIFA_URL_TEMPLATE = "https://raw.githubusercontent.com/305kishan/FIFA/main/data/FIFA{v}.csv"
FIFA_EDITIONS = list(range(15, 23))

# ── League metadata ──────────────────────────────────────────────────────────
# code -> (country, display name, tier)
DIVISIONS: dict[str, tuple[str, str, int]] = {
    "E0": ("England", "Premier League", 1),
    "E1": ("England", "Championship", 2),
    "E2": ("England", "League One", 3),
    "E3": ("England", "League Two", 4),
    "EC": ("England", "National League", 5),
    "SC0": ("Scotland", "Premiership", 1),
    "SC1": ("Scotland", "Championship", 2),
    "SC2": ("Scotland", "League One", 3),
    "SC3": ("Scotland", "League Two", 4),
    "D1": ("Germany", "Bundesliga", 1),
    "D2": ("Germany", "2. Bundesliga", 2),
    "SP1": ("Spain", "La Liga", 1),
    "SP2": ("Spain", "Segunda División", 2),
    "I1": ("Italy", "Serie A", 1),
    "I2": ("Italy", "Serie B", 2),
    "F1": ("France", "Ligue 1", 1),
    "F2": ("France", "Ligue 2", 2),
    "N1": ("Netherlands", "Eredivisie", 1),
    "B1": ("Belgium", "Pro League", 1),
    "P1": ("Portugal", "Primeira Liga", 1),
    "T1": ("Turkey", "Süper Lig", 1),
    "G1": ("Greece", "Super League", 1),
    "ARG": ("Argentina", "Primera División", 1),
    "AUT": ("Austria", "Bundesliga", 1),
    "BRA": ("Brazil", "Série A", 1),
    "CHN": ("China", "Super League", 1),
    "DEN": ("Denmark", "Superliga", 1),
    "FIN": ("Finland", "Veikkausliiga", 1),
    "IRL": ("Ireland", "Premier Division", 1),
    "JAP": ("Japan", "J1 League", 1),
    "MEX": ("Mexico", "Liga MX", 1),
    "NOR": ("Norway", "Eliteserien", 1),
    "POL": ("Poland", "Ekstraklasa", 1),
    "ROM": ("Romania", "Liga I", 1),
    "RUS": ("Russia", "Premier League", 1),
    "SUI": ("Switzerland", "Super League", 1),
    "SWE": ("Sweden", "Allsvenskan", 1),
    "USA": ("USA", "MLS", 1),
}

FEATURED_DIVISIONS = ["E0", "SP1", "D1", "I1", "F1", "E1", "N1", "P1"]

# ── Temporal split (strictly chronological, no shuffling) ────────────────────
TRAIN_END = "2023-07-01"   # train:  2000-07 .. 2023-06
VALID_END = "2024-07-01"   # valid:  2023-07 .. 2024-06 (early stopping, calibration, blending)
#                            test:   2024-07 .. latest (2024-25, 2025-26, 2026-27 so far)

RESULT_CLASSES = ["H", "D", "A"]  # canonical order used everywhere


def division_label(code: str) -> str:
    country, name, _ = DIVISIONS.get(code, (code, code, 1))
    return f"{country} · {name}"
