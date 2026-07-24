"""
╔══════════════════════════════════════════════════════════════╗
║     FOOTBALL ANALYTICS — STREAMLIT INTERACTIVE WEB APP       ║
║  Run: streamlit run football_app.py                         ║
╚══════════════════════════════════════════════════════════════╝

Author: B Sujan Kumar
GitHub:  https://github.com/BSujanK
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
import warnings
import time
import random
from io import StringIO
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import Ridge, LinearRegression
from sklearn.cluster          import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics          import (accuracy_score, confusion_matrix,
                                      r2_score, mean_squared_error)
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Football Analytics ML",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS — dark football theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Core palette ──────────────────────────────────────────
   Core Black   : #0a0a0a / #111111 / #1a1a1a / #222222
   Plasma Orange: #FF6B00 / #FF8C00 / #FFA500 (accent glow)
──────────────────────────────────────────────────────── */

:root {
    --black-deep:   #0a0a0a;
    --black-card:   #111111;
    --black-border: #222222;
    --black-hover:  #1a1a1a;
    --orange:       #FF6B00;
    --orange-glow:  #FF8C00;
    --orange-soft:  rgba(255,107,0,0.12);
    --orange-mid:   rgba(255,107,0,0.25);
    --text-primary: #F0F0F0;
    --text-muted:   #888888;
    --text-dim:     #555555;
    --win-green:    #22c55e;
    --draw-amber:   #f59e0b;
    --loss-red:     #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    color: var(--text-primary);
}

/* App background */
.stApp {
    background: var(--black-deep) !important;
    color: var(--text-primary);
}
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid var(--black-border) !important;
}
/* sidebar text handled per-element */
section[data-testid="stSidebar"] .stRadio > div > label {
    background: #151515 !important;
    border: 1px solid var(--black-border) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    transition: all .18s !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    border-color: var(--orange) !important;
    background: var(--orange-soft) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.3rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: var(--orange) !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--black-border) !important; }

/* ── Streamlit metric overrides ──────────────────────── */
[data-testid="stMetric"] {
    background: var(--black-card) !important;
    border: 1px solid var(--black-border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2rem !important;
    color: var(--orange) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
[data-testid="stMetricDelta"] { color: var(--text-muted) !important; }

/* ── Expanders ───────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--black-card) !important;
    border: 1px solid var(--black-border) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
}
[data-testid="stExpander"]:hover { border-color: var(--orange) !important; }
[data-testid="stExpander"] summary {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 1.5px !important;
    color: var(--text-primary) !important;
}

/* ── Buttons (main content only, not sidebar) ────────── */
.main .stButton > button,
section[data-testid="stMain"] .stButton > button {
    background: linear-gradient(135deg, #FF6B00, #c83200) !important;
    color: #000 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all .18s !important;
}
.main .stButton > button:hover,
section[data-testid="stMain"] .stButton > button:hover {
    background: linear-gradient(135deg, #FF8C00, #FF4500) !important;
    box-shadow: 0 0 18px rgba(255,107,0,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: var(--black-card) !important; border-radius: 8px !important; padding: 4px !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; border-radius: 6px !important; }
.stTabs [aria-selected="true"] { background: var(--orange-soft) !important; color: var(--orange) !important; border-bottom: 2px solid var(--orange) !important; }

/* ── Info / warning boxes ────────────────────────────── */
.stAlert { background: var(--black-card) !important; border-left: 3px solid var(--orange) !important; color: var(--text-primary) !important; border-radius: 8px !important; }

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] { background: var(--black-card) !important; border: 1px solid var(--black-border) !important; border-radius: 8px !important; }

/* ── Sliders ─────────────────────────────────────────── */
/* All text in sliders — visible light grey */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p,
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { 
    color: #F0F0F0 !important; font-weight: 600 !important; 
}
/* The current value shown as text (e.g. "78") */
[data-testid="stSlider"] > label + div > div:last-child,
[data-testid="stSlider"] input[type="range"] + div,
[data-testid="stSlider"] div[data-testid] > span {
    color: #FF8C00 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}
/* Thumb handle */
[data-testid="stSlider"] [role="slider"] {
    background: #FF6B00 !important;
    border: 2px solid #FFB347 !important;
    box-shadow: 0 0 10px rgba(255,107,0,0.55) !important;
}
/* Floating value on thumb during drag */
[data-testid="stThumbValue"],
[data-testid="stSliderThumbValue"] {
    background: #FF6B00 !important;
    color: #000 !important;
    font-weight: 900 !important;
    font-size: 12px !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}
/* Tick bar min/max labels */
[data-testid="stTickBar"] > div { color: #555 !important; font-size: 10px !important; }

/* ── Select boxes ────────────────────────────────────── */
[data-baseweb="select"] { background: var(--black-card) !important; border-color: var(--black-border) !important; }
[data-baseweb="select"] * { color: var(--text-primary) !important; }

/* ── Hero banner ─────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #111111 0%, #0a0a0a 60%, #1a0800 100%);
    border: 1px solid var(--black-border);
    border-left: 4px solid var(--orange);
    border-radius: 14px;
    padding: 34px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '⚽';
    position: absolute;
    right: 40px; top: 50%;
    transform: translateY(-50%);
    font-size: 90px;
    opacity: 0.05;
}
.hero::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(255,107,0,0.06), transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    line-height: 1.1;
}
.hero h1 span { color: var(--orange); }
.hero p { color: var(--text-muted); font-size: 13px; margin: 0; letter-spacing: 0.5px; }

/* ── Section titles ──────────────────────────────────── */
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--orange);
    border-bottom: 1px solid var(--black-border);
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 4px; height: 16px;
    background: var(--orange);
    border-radius: 2px;
}

/* ── Result boxes ────────────────────────────────────── */
.result-box {
    background: var(--black-card);
    border: 1px solid var(--black-border);
    border-radius: 12px;
    padding: 22px 28px;
    margin: 12px 0;
}
.result-box.win  { border-color: var(--win-green);  background: rgba(34,197,94,.05); }
.result-box.draw { border-color: var(--draw-amber); background: rgba(245,158,11,.05); }
.result-box.loss { border-color: var(--loss-red);   background: rgba(239,68,68,.05); }

/* ── Player cards ────────────────────────────────────── */
.player-card {
    background: var(--black-card);
    border: 1px solid var(--black-border);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    transition: border-color .18s, background .18s;
}
.player-card:hover {
    border-color: var(--orange);
    background: var(--orange-soft);
}

/* ── Tag badges ──────────────────────────────────────── */
.tag {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .8px;
    text-transform: uppercase;
    margin: 2px;
}
.tag-fw { background: rgba(255,107,0,.15); color: var(--orange); }
.tag-mf { background: rgba(240,200,60,.12); color: #f0c83c; }
.tag-df { background: rgba(34,197,94,.12);  color: var(--win-green); }
.tag-gk { background: rgba(139,92,246,.15); color: #a78bfa; }

/* ── Spinner / caption / markdown text ──────────────── */
.stCaption { color: var(--text-muted) !important; }
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 1.5px !important;
}
[data-testid="stMarkdownContainer"] h2 { color: var(--text-primary) !important; }
[data-testid="stMarkdownContainer"] h3 { color: var(--orange) !important; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--black-deep); }
::-webkit-scrollbar-thumb { background: var(--orange); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB GLOBAL DARK THEME — plasma orange palette
# ══════════════════════════════════════════════════════════════
_rc = {
    "figure.facecolor":  "#0a0a0a",
    "axes.facecolor":    "#111111",
    "axes.edgecolor":    "#222222",
    "axes.labelcolor":   "#888888",
    "axes.titlecolor":   "#F0F0F0",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "text.color":        "#F0F0F0",
    "grid.color":        "#222222",
    "legend.facecolor":  "#111111",
    "legend.edgecolor":  "#222222",
    "legend.labelcolor": "#F0F0F0",
    "figure.titlecolor": "#F0F0F0",
    "savefig.facecolor": "#0a0a0a",
}
for _k, _v in _rc.items():
    try:
        plt.rcParams[_k] = _v
    except KeyError:
        pass  # skip keys not supported in this matplotlib version

# ══════════════════════════════════════════════════════════════
#  DATA LOADING — cached so it only runs once
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    """Download or generate football data — same logic as sports_ml_real.py"""

    def fetch(url):
        try:
            r = requests.get(url, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text))
        except:
            return None

    FIFA_URLS = [
        "https://raw.githubusercontent.com/305kishan/FIFA/main/data/FIFA20.csv",
        "https://raw.githubusercontent.com/ddebonis47/classwork/refs/heads/main/fifa21%20raw%20data%20v2.csv",
        "https://raw.githubusercontent.com/apoorva-21/fifa-analysis/master/data/players_20.csv",
    ]
    EPL_URLS = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    ]

    # ── Try to fetch FIFA data ─────────────────────────────────
    fifa_raw = None
    for url in FIFA_URLS:
        fifa_raw = fetch(url)
        if fifa_raw is not None:
            break

    # ── Try to fetch EPL data ──────────────────────────────────
    epl_raw = None
    for url in EPL_URLS:
        epl_raw = fetch(url)
        if epl_raw is not None:
            break

    # ── Process player data ────────────────────────────────────
    if fifa_raw is not None:
        p = fifa_raw.copy()
        col_map = {}
        for c in p.columns:
            cl = c.lower().strip()
            if cl in ("short_name","name","player_name"): col_map[c] = col_map.get("Name", "Name"); col_map[c] = "Name"
            if cl in ("nationality","nationality_name"):  col_map[c] = "Nationality"
            if cl in ("player_positions","position","pos"): col_map[c] = "Position"
            if cl == "age":                               col_map[c] = "Age"
            if cl == "overall":                           col_map[c] = "Overall"
            if cl in ("pace","pac"):                      col_map[c] = "Pace"
            if cl in ("shooting","sho"):                  col_map[c] = "Shooting"
            if cl in ("passing","pas"):                   col_map[c] = "Passing"
            if cl in ("dribbling","dri"):                 col_map[c] = "Dribbling"
            if cl in ("defending","def"):                 col_map[c] = "Defending"
            if cl in ("physic","physical","phy"):         col_map[c] = "Physical"
            if cl in ("value_eur","value"):               col_map[c] = "Value_M"
        p.rename(columns=col_map, inplace=True)
        for col in ["Age","Overall","Pace","Shooting","Passing","Dribbling","Defending","Physical"]:
            if col in p.columns:
                p[col] = pd.to_numeric(p[col], errors="coerce")
        if "Position" in p.columns:
            def clean_pos(v):
                if pd.isna(v): return "Midfielder"
                v = str(v).split(",")[0].strip().upper()
                if v in ("ST","CF","LW","RW","LF","RF"): return "Forward"
                if v in ("CAM","CM","CDM","LM","RM"):    return "Midfielder"
                if v in ("CB","LB","RB","LWB","RWB"):    return "Defender"
                if v == "GK":                            return "Goalkeeper"
                return "Midfielder"
            p["Position"] = p["Position"].apply(clean_pos)
        n = len(p)
        for col, fn in [("Goals",    lambda: np.random.poisson(5,n)),
                        ("Assists",  lambda: np.random.poisson(3,n)),
                        ("Pace",     lambda: np.random.randint(50,98,n)),
                        ("Shooting", lambda: np.random.randint(45,95,n)),
                        ("Passing",  lambda: np.random.randint(50,95,n)),
                        ("Dribbling",lambda: np.random.randint(45,97,n)),
                        ("Defending",lambda: np.random.randint(30,90,n)),
                        ("Physical", lambda: np.random.randint(50,95,n)),
                        ("Value_M",  lambda: np.round(np.random.exponential(20,n).clip(1,200),1)),
                        ("Position", lambda: np.random.choice(["Forward","Midfielder","Defender","Goalkeeper"],n))]:
            if col not in p.columns: p[col] = fn()
        overall = pd.to_numeric(p.get("Overall", 70), errors="coerce").fillna(70)
        p["Performance"] = (
            p["Goals"]*6 + p["Assists"]*4 + overall*0.8 +
            np.random.normal(0,8,n)
        ).round(1)
        p.dropna(subset=["Overall"], inplace=True)
        p = p.sample(min(500,len(p)), random_state=42).reset_index(drop=True)
        data_source = "FIFA (real)"
    else:
        # Fallback
        n = 300
        POSITIONS = ["Forward","Midfielder","Defender","Goalkeeper"]
        p = pd.DataFrame({
            "Name":        [f"Player_{i:03d}" for i in range(1,n+1)],
            "Nationality": np.random.choice(["England","Spain","France","Germany","Brazil"],n),
            "Position":    np.random.choice(POSITIONS,n,p=[0.28,0.35,0.27,0.10]),
            "Age":         np.random.randint(17,37,n),
            "Overall":     np.random.randint(60,95,n),
            "Goals":       np.random.poisson(7,n),
            "Assists":     np.random.poisson(4,n),
            "Pace":        np.random.randint(50,98,n),
            "Shooting":    np.random.randint(45,95,n),
            "Passing":     np.random.randint(50,95,n),
            "Dribbling":   np.random.randint(45,97,n),
            "Defending":   np.random.randint(30,90,n),
            "Physical":    np.random.randint(50,95,n),
            "Value_M":     np.round(np.random.exponential(20,n).clip(1,200),1),
        })
        p["Performance"] = (
            p["Goals"]*6 + p["Assists"]*4 + p["Overall"]*0.8 + np.random.normal(0,8,n)
        ).round(1)
        data_source = "Generated"

    # ── Process match data ─────────────────────────────────────
    if epl_raw is not None:
        m = epl_raw.copy()
        for col in ["FTHG","FTAG","HS","AS","HST","AST"]:
            if col in m.columns:
                m[col] = pd.to_numeric(m[col], errors="coerce")
        m.dropna(subset=["FTHG","FTAG","FTR"], inplace=True)
        nm = len(m)
        for col, fn in [("HC", lambda: np.random.randint(2,12,nm)),
                        ("AC", lambda: np.random.randint(1,11,nm)),
                        ("HF", lambda: np.random.randint(5,18,nm)),
                        ("AF", lambda: np.random.randint(5,18,nm)),
                        ("HY", lambda: np.random.randint(0, 4,nm)),
                        ("AY", lambda: np.random.randint(0, 4,nm))]:
            if col not in m.columns: m[col] = fn()
        match_source = "EPL (real)"
    else:
        teams = ["Arsenal","Chelsea","Liverpool","Man City","Man Utd","Tottenham",
                 "Newcastle","Aston Villa","West Ham","Brighton","Brentford",
                 "Fulham","Crystal Palace","Wolves","Everton","Nottm Forest",
                 "Bournemouth","Luton","Sheffield Utd","Burnley"]
        nm = 380
        hg = np.random.poisson(1.5,nm); ag = np.random.poisson(1.2,nm)
        m = pd.DataFrame({
            "HomeTeam": np.random.choice(teams,nm),
            "AwayTeam": np.random.choice(teams,nm),
            "FTHG":hg,"FTAG":ag,
            "FTR": np.where(hg>ag,"H",np.where(hg<ag,"A","D")),
            "HS": np.random.randint(5,22,nm), "AS": np.random.randint(3,20,nm),
            "HST":np.random.randint(2,10,nm), "AST":np.random.randint(1,9,nm),
            "HC": np.random.randint(2,12,nm), "AC": np.random.randint(1,11,nm),
            "HF": np.random.randint(5,18,nm), "AF": np.random.randint(5,18,nm),
            "HY": np.random.randint(0,4,nm),  "AY": np.random.randint(0,4,nm),
        })
        match_source = "Generated"

    return p, m, data_source, match_source

@st.cache_resource(show_spinner=False)
def train_models(players_df, matches_df):
    """Train all ML models — cached so they only train once"""
    models = {}

    # ── Match predictor ────────────────────────────────────────
    feat = [c for c in ["HS","AS","HST","AST","HC","AC","HF","AF","HY","AY"]
            if c in matches_df.columns]
    X = matches_df[feat].dropna()
    le = LabelEncoder()
    y  = le.fit_transform(matches_df.loc[X.index,"FTR"])
    Xt,Xe,yt,ye = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=200,max_depth=6,
                                  random_state=42,class_weight="balanced")
    clf.fit(Xt,yt)
    acc = accuracy_score(ye, clf.predict(Xe))
    models["match_clf"]   = clf
    models["match_le"]    = le
    models["match_feats"] = feat
    models["match_acc"]   = acc

    # ── Performance scorer ─────────────────────────────────────
    pf = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
                       "Dribbling","Defending","Physical"]
          if c in players_df.columns]
    pdf = players_df[pf+["Performance"]].dropna()
    sc  = StandardScaler()
    Xp  = sc.fit_transform(pdf[pf])
    yp  = pdf["Performance"]
    Xpt,Xpe,ypt,ype = train_test_split(Xp,yp,test_size=0.25,random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(Xpt,ypt)
    r2  = r2_score(ype, reg.predict(Xpe))
    models["perf_reg"]    = reg
    models["perf_scaler"] = sc
    models["perf_feats"]  = pf
    models["perf_r2"]     = r2

    # ── Clustering ─────────────────────────────────────────────
    cf = [c for c in ["Goals","Assists","Passing","Dribbling",
                       "Defending","Physical","Pace"]
          if c in players_df.columns]
    cd   = players_df[cf].dropna()
    scc  = StandardScaler()
    cds  = scc.fit_transform(cd)
    km   = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(cds)
    cnames = {0:"🛡️ Defensive Wall",1:"⚡ Speed Merchant",
               2:"🎯 Goal Machine",3:"🎪 Playmaker"}
    players_df = players_df.copy()
    players_df.loc[cd.index,"Cluster"]      = labels
    players_df.loc[cd.index,"Cluster_Name"] = [cnames[l] for l in labels]
    models["cluster_km"]     = km
    models["cluster_scaler"] = scc
    models["cluster_feats"]  = cf
    models["cluster_names"]  = cnames
    models["players_df"]     = players_df

    # ── Recommender ────────────────────────────────────────────
    rf = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
                       "Dribbling","Defending","Physical"]
          if c in players_df.columns]
    rd   = players_df[rf+["Performance"]+(["Name"] if "Name" in players_df.columns else [])].dropna().reset_index(drop=True)
    scr  = MinMaxScaler()
    rds  = scr.fit_transform(rd[rf])
    sim  = cosine_similarity(rds)
    models["rec_data"]   = rd
    models["rec_sim"]    = sim
    models["rec_feats"]  = rf
    models["rec_scaler"] = scr

    return models

# ══════════════════════════════════════════════════════════════
#  LOAD DATA + TRAIN  — styled splash screen
# ══════════════════════════════════════════════════════════════
_data_loaded = ("_players_df" in st.session_state)

if not _data_loaded:
    splash = st.empty()
    splash.markdown("""
    <style>
    .splash-wrap {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 72vh; gap: 28px;
    }
    .splash-ball { font-size: 4rem; animation: spin 2s linear infinite; }
    @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
    .splash-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.4rem; font-weight: 800;
        letter-spacing: 4px; text-transform: uppercase;
        color: #F0F0F0;
    }
    .splash-title span { color: #FF6B00; }
    .splash-steps { display: flex; flex-direction: column; gap: 10px; width: 380px; }
    .splash-step {
        display: flex; align-items: center; gap: 14px;
        background: #111; border: 1px solid #222;
        border-radius: 10px; padding: 12px 18px;
    }
    .splash-step-icon {
        width: 36px; height: 36px; border-radius: 8px;
        background: rgba(255,107,0,0.12); border: 1px solid #FF6B00;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; flex-shrink: 0;
    }
    .splash-step-label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.95rem; font-weight: 600;
        letter-spacing: 1px; text-transform: uppercase; color: #888;
    }
    .splash-step-sub { font-size: 10px; color: #444; margin-top: 2px; }
    .splash-bar-wrap {
        width: 380px; background: #111; border: 1px solid #222;
        border-radius: 99px; height: 4px; overflow: hidden;
    }
    .splash-bar {
        height: 100%; width: 30%;
        background: linear-gradient(90deg, #FF6B00, #FFB347);
        border-radius: 99px;
        animation: progress 2.5s ease-in-out infinite alternate;
    }
    @keyframes progress { 0%{width:15%;margin-left:0} 100%{width:60%;margin-left:40%} }
    </style>
    <div class="splash-wrap">
      <div class="splash-ball">⚽</div>
      <div class="splash-title">Football <span>Analytics ML</span></div>
      <div class="splash-steps">
        <div class="splash-step">
          <div class="splash-step-icon">🌐</div>
          <div>
            <div class="splash-step-label">Fetching Data</div>
            <div class="splash-step-sub">FIFA players · EPL match results</div>
          </div>
        </div>
        <div class="splash-step">
          <div class="splash-step-icon">🧹</div>
          <div>
            <div class="splash-step-label">Processing Dataset</div>
            <div class="splash-step-sub">Cleaning · normalising · feature engineering</div>
          </div>
        </div>
        <div class="splash-step">
          <div class="splash-step-icon">🤖</div>
          <div>
            <div class="splash-step-label">Training ML Models</div>
            <div class="splash-step-sub">Random Forest · Ridge · K-Means · Cosine Sim</div>
          </div>
        </div>
        <div class="splash-step">
          <div class="splash-step-icon">📊</div>
          <div>
            <div class="splash-step-label">Building Analytics</div>
            <div class="splash-step-sub">Clusters · recommender · league table</div>
          </div>
        </div>
      </div>
      <div class="splash-bar-wrap"><div class="splash-bar"></div></div>
    </div>
    """, unsafe_allow_html=True)

players_df, matches_df, data_src, match_src = load_data()
models = train_models(players_df, matches_df)
players_df = models["players_df"]

if not _data_loaded:
    st.session_state["_players_df"] = True
    splash.empty()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAV — interactive custom navigation
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Hide default radio widget */
section[data-testid="stSidebar"] .stRadio { display: none !important; }
section[data-testid="stSidebar"] { padding-top: 0 !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.sb-logo {
    background: linear-gradient(135deg, #111 0%, #1a0800 100%);
    border-bottom: 2px solid #FF6B00;
    padding: 22px 20px 16px 20px;
    margin-bottom: 6px;
}
.sb-logo-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.25rem; font-weight: 800;
    letter-spacing: 3px; text-transform: uppercase;
    color: #F0F0F0; line-height: 1.1;
}
.sb-logo-title span { color: #FF6B00; }
.sb-logo-sub { font-size: 10px; color: #555; letter-spacing: 2px;
               text-transform: uppercase; margin-top: 4px; }
.sb-section-label { font-size: 9px; font-weight: 700; letter-spacing: 2.5px;
                    text-transform: uppercase; color: #444;
                    padding: 14px 20px 6px 20px; }
.sb-nav-btn {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; margin: 2px 10px;
    border-radius: 9px; cursor: pointer;
    transition: all 0.18s ease;
    border: 1px solid transparent;
    position: relative; overflow: hidden;
}
.sb-nav-btn::before {
    content: ''; position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px;
    background: transparent; border-radius: 0 2px 2px 0;
    transition: background 0.18s;
}
.sb-nav-btn:hover { background: rgba(255,107,0,0.08) !important;
                    border-color: rgba(255,107,0,0.2) !important; }
.sb-nav-btn:hover::before { background: #FF6B00; }
.sb-nav-btn.active { background: rgba(255,107,0,0.14) !important;
                     border-color: rgba(255,107,0,0.35) !important; }
.sb-nav-btn.active::before { background: #FF6B00; }
.sb-nav-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    background: #1a1a1a; border: 1px solid #222;
    transition: all 0.18s;
}
.sb-nav-btn.active .sb-nav-icon,
.sb-nav-btn:hover  .sb-nav-icon { background: rgba(255,107,0,0.15); border-color: #FF6B00; }
.sb-nav-text { flex: 1; min-width: 0; }
.sb-nav-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.88rem; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase;
    color: #888; transition: color 0.18s; white-space: nowrap;
}
.sb-nav-btn.active .sb-nav-label { color: #FF6B00 !important; }
.sb-nav-btn:hover  .sb-nav-label { color: #F0F0F0 !important; }
.sb-nav-desc { font-size: 10px; color: #444; margin-top: 1px;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sb-nav-btn.active .sb-nav-desc { color: #888 !important; }
.sb-nav-arrow { font-size: 10px; color: transparent; transition: color 0.18s; }
.sb-nav-btn.active .sb-nav-arrow { color: #FF6B00; }
.sb-stats {
    margin: 14px 10px 0 10px;
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 10px; padding: 12px 14px;
}
.sb-stats-title { font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
                  color: #444; margin-bottom: 10px; font-weight: 700; }
.sb-stat-row { display: flex; justify-content: space-between; align-items: center;
               padding: 5px 0; border-bottom: 1px solid #1a1a1a; }
.sb-stat-row:last-child { border-bottom: none; }
.sb-stat-key { font-size: 10px; color: #555; letter-spacing: 0.5px; }
.sb-stat-val { font-size: 11px; font-weight: 700; color: #FF6B00;
               font-family: 'Barlow Condensed', sans-serif; letter-spacing: 1px; }
.sb-footer { margin: 14px 10px 10px 10px; padding: 10px 14px;
             background: #0d0d0d; border-radius: 8px; border: 1px solid #1a1a1a;
             font-size: 9px; color: #333; letter-spacing: 1.5px;
             text-transform: uppercase; text-align: center; }
.sb-footer span { color: #FF6B00; }
/* nav buttons styled per-item */
</style>
""", unsafe_allow_html=True)

_PAGES = [
    ("🏠 Dashboard",         "🏠", "Dashboard",         "Overview & KPIs"),
    ("🏆 Match Predictor",   "🏆", "Match Predictor",   "Predict outcomes"),
    ("📈 Performance Scorer","📈", "Performance Scorer","Score any player"),
    ("🗂️ Player Clusters",  "🗂️", "Player Clusters",   "K-Means archetypes"),
    ("🤖 Player Recommender","🤖", "Player Recommender","Find similar players"),
    ("🌍 EDA Explorer",      "🌍", "EDA Explorer",       "Explore raw data"),
    ("📊 League Table",      "📊", "League Table",       "Standings & H2H"),
    ("🖼️ ML Chart Gallery",  "🖼️", "ML Chart Gallery",   "All 7 ML charts"),
]

if "page" not in st.session_state:
    st.session_state.page = "🏠 Dashboard"

with st.sidebar:
    acc_pct = f"{models['match_acc']*100:.1f}%"
    r2_val  = f"{models['perf_r2']:.3f}"

    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-title">⚽ Football<br><span>Analytics ML</span></div>
      <div class="sb-logo-sub">Powered by scikit-learn</div>
    </div>
    <div class="sb-section-label">Navigation</div>
    """, unsafe_allow_html=True)

    # ── Nav buttons: base styles for ALL sidebar buttons ─────────
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #1c1c1c 0%, #121212 100%) !important;
        color: #888 !important;
        border: 1px solid #2a2a2a !important;
        border-left: 3px solid transparent !important;
        border-radius: 10px !important;
        min-height: 50px !important;
        padding: 10px 14px !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        text-align: left !important;
        width: 100% !important;
        margin-bottom: 3px !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #2d1200 0%, #1a0a00 100%) !important;
        color: #F0F0F0 !important;
        border-color: #FF6B00 !important;
        border-left-color: #FF6B00 !important;
        box-shadow: 0 0 14px rgba(255,107,0,0.2) !important;
        transform: translateX(3px) !important;
    }
    section[data-testid="stSidebar"] .stButton > button p {
        color: inherit !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for i, (page_key, icon, label, desc) in enumerate(_PAGES):
        is_active = st.session_state.page == page_key
        safe_key  = page_key.replace(" ","_").replace("/","").replace("️","")
        # Inject active gradient for current page button
        if is_active:
            st.markdown(f"""<style>
            section[data-testid="stSidebar"] div:nth-child({i*2+1}) .stButton > button,
            section[data-testid="stSidebar"] .stButton:nth-of-type({i+1}) > button,
            section[data-testid="stSidebar"] button[kind="secondary"][data-testid*="{safe_key[:8]}"] {{
                background: linear-gradient(135deg, #FF6B00 0%, #E85D04 50%, #9D0208 100%) !important;
                color: #000 !important;
                border-left: 3px solid #FFD580 !important;
                border-color: rgba(255,107,0,0.4) !important;
                box-shadow: 0 0 18px rgba(255,107,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08) !important;
                transform: translateX(3px) !important;
            }}
            section[data-testid="stSidebar"] div:nth-child({i*2+1}) .stButton > button p {{
                color: #000 !important;
            }}
            </style>""", unsafe_allow_html=True)
        btn_label = f"{icon}  {label}"
        if st.button(btn_label, key=f"nav_{safe_key}",
                     use_container_width=True, help=desc):
            st.session_state.page = page_key
            st.rerun()

    st.markdown(f"""
    <div class="sb-stats">
      <div class="sb-stats-title">Live Model Stats</div>
      <div class="sb-stat-row">
        <span class="sb-stat-key">Match Accuracy</span>
        <span class="sb-stat-val">{acc_pct}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-key">Player R&#178;</span>
        <span class="sb-stat-val">{r2_val}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-key">Players</span>
        <span class="sb-stat-val">{len(players_df):,}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-key">Matches</span>
        <span class="sb-stat-val">{len(matches_df):,}</span>
      </div>
      <div class="sb-stat-row">
        <span class="sb-stat-key">Data Source</span>
        <span class="sb-stat-val">{data_src}</span>
      </div>
    </div>
    <div class="sb-footer">Built with Python &#183; <span>sklearn</span> &#183; Streamlit</div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  HELPER — matplotlib fig → streamlit
# ══════════════════════════════════════════════════════════════
def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def role_tag(role):
    tags = {"Forward":"tag-fw","Midfielder":"tag-mf",
            "Defender":"tag-df","Goalkeeper":"tag-gk"}
    cls  = tags.get(role,"tag-mf")
    return f'<span class="tag {cls}">{role}</span>'

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero">
      <h1>Football Analytics <span>ML System</span></h1>
      <p>Real-time predictions &nbsp;·&nbsp; Player scoring &nbsp;·&nbsp; Clustering &nbsp;·&nbsp; Recommendations</p>
    </div>""", unsafe_allow_html=True)

    # KPI strip
    acc  = models["match_acc"]
    r2   = models["perf_r2"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Match Accuracy",  f"{acc*100:.1f}%", "Random Forest")
    c2.metric("Player R²",       f"{r2:.3f}",       "Ridge Regression")
    c3.metric("Players",         f"{len(players_df):,}")
    c4.metric("Matches",         f"{len(matches_df):,}")
    c5.metric("Clusters",        "4",               "K-Means")

    st.markdown('<div class="section-title">Top 10 Players by Performance</div>',
                unsafe_allow_html=True)
    top10 = players_df.nlargest(10,"Performance")
    fig,ax = plt.subplots(figsize=(10,4))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111111")
    colors = plt.colormaps["plasma"](np.linspace(0.2,0.85,10))
    bars   = ax.barh(top10["Name"] if "Name" in top10 else top10.index.astype(str),
                     top10["Performance"], color=colors, edgecolor="none")
    ax.invert_yaxis()
    ax.tick_params(colors="#F0F0F0", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Performance Score", color="#888888")
    for bar,val in zip(bars,top10["Performance"]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.0f}", va="center", color="#F0F0F0", fontsize=8)
    show_fig(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Match Results Distribution</div>',
                    unsafe_allow_html=True)
        res_c = matches_df["FTR"].map({"H":"Home Win","D":"Draw","A":"Away Win"}).value_counts()
        fig,ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
        ax.bar(res_c.index, res_c.values,
               color=["#FF6B00","#f59e0b","#ef4444"], edgecolor="none", width=0.5)
        ax.tick_params(colors="#F0F0F0"); ax.spines[:].set_visible(False)
        for i,(idx,v) in enumerate(res_c.items()):
            ax.text(i, v+1, str(v), ha="center", color="#F0F0F0", fontsize=9)
        show_fig(fig)

    with col2:
        st.markdown('<div class="section-title">Position Distribution</div>',
                    unsafe_allow_html=True)
        if "Position" in players_df.columns:
            pos_c = players_df["Position"].value_counts()
            fig,ax = plt.subplots(figsize=(5,3))
            fig.patch.set_facecolor("#0a0a0a")
            pie_colors = ["#FF6B00","#FF8C00","#FFA040","#FFC080"]
            wedges, texts, autotexts = ax.pie(
                pos_c, labels=pos_c.index, autopct="%1.0f%%",
                colors=pie_colors[:len(pos_c)],
                startangle=90, wedgeprops=dict(width=0.6))
            for t in texts: t.set_color("#F0F0F0"); t.set_fontsize(9)
            for at in autotexts: at.set_color("#0a0a0a"); at.set_fontsize(8); at.set_fontweight("bold")
            show_fig(fig)

# ══════════════════════════════════════════════════════════════
#  PAGE 2 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Match Predictor":
    st.markdown("## 🏆 Match Outcome Predictor")
    st.markdown("Adjust the match statistics below to predict the result.")

    clf   = models["match_clf"]
    le    = models["match_le"]
    feats = models["match_feats"]
    acc   = models["match_acc"]

    st.info(f"Model: **Random Forest**  |  Accuracy: **{acc*100:.1f}%**  |  "
            f"Features: {', '.join(feats)}")

    st.markdown("### ⚙️ Enter Match Statistics")
    col1, col2 = st.columns(2)

    inputs = {}
    feat_labels = {
        "HS":"Home Shots","AS":"Away Shots",
        "HST":"Home Shots on Target","AST":"Away Shots on Target",
        "HC":"Home Corners","AC":"Away Corners",
        "HF":"Home Fouls","AF":"Away Fouls",
        "HY":"Home Yellow Cards","AY":"Away Yellow Cards",
    }
    defaults = {"HS":13,"AS":10,"HST":5,"AST":4,"HC":6,"AC":4,
                "HF":11,"AF":12,"HY":2,"AY":2}

    for i,feat in enumerate(feats):
        col = col1 if i%2==0 else col2
        label = feat_labels.get(feat, feat)
        mx    = 25 if "S" in feat else 15
        inputs[feat] = col.slider(label, 0, mx, defaults.get(feat,5))

    if st.button("🔮 Predict Match Outcome", use_container_width=True):
        X_input = pd.DataFrame([inputs])[feats]
        pred    = clf.predict(X_input)[0]
        proba   = clf.predict_proba(X_input)[0]
        classes = le.classes_   # A=Away Win, D=Draw, H=Home Win
        label_map = {"H":"🏠 Home Win","D":"⚖️ Draw","A":"✈️ Away Win"}
        color_map = {"H":"win","D":"draw","A":"loss"}
        result  = label_map[classes[pred]]
        box_cls = color_map[classes[pred]]

        st.markdown(f"""
        <div class="result-box {box_cls}">
          <h2 style="margin:0;color:white">Predicted Result: {result}</h2>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Win Probabilities")
        prob_df = pd.DataFrame({
            "Result": [label_map.get(c,c) for c in classes],
            "Probability (%)": [round(p*100,1) for p in proba]
        })
        col1,col2 = st.columns([2,1])
        with col1:
            fig,ax = plt.subplots(figsize=(6,2.5))
            fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
            colors = ["#22c55e","#f59e0b","#ef4444"]
            bars   = ax.barh(prob_df["Result"], prob_df["Probability (%)"],
                             color=colors, edgecolor="none")
            ax.set_xlim(0,100); ax.tick_params(colors="#F0F0F0"); ax.spines[:].set_visible(False)
            for bar,val in zip(bars, prob_df["Probability (%)"]):
                ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                        f"{val}%", va="center", color="#F0F0F0", fontsize=10)
            show_fig(fig)
        with col2:
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown("### 🔍 Feature Importances")
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values()
    fig,ax = plt.subplots(figsize=(8,3.5))
    fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
    imp.plot(kind="barh", ax=ax,
             color=plt.colormaps["plasma"](np.linspace(0.2,0.9,len(imp))))
    ax.tick_params(colors="#F0F0F0"); ax.spines[:].set_visible(False)
    ax.set_xlabel("Importance Score", color="#888888")
    ax.xaxis.label.set_color("#888888")
    for label in ax.get_yticklabels(): label.set_color("#F0F0F0")
    show_fig(fig)

# ══════════════════════════════════════════════════════════════
#  PAGE 3 — PERFORMANCE SCORER
# ══════════════════════════════════════════════════════════════
elif page == "📈 Performance Scorer":
    st.markdown("## 📈 Player Performance Scorer")

    reg   = models["perf_reg"]
    sc    = models["perf_scaler"]
    feats = models["perf_feats"]
    r2    = models["perf_r2"]

    st.info(f"Model: **Ridge Regression**  |  R²: **{r2:.3f}**")

    st.markdown("### 🎮 Score a Custom Player")
    cols = st.columns(4)
    p_inputs = {}
    feat_defaults = {"Goals":10,"Assists":7,"Pace":78,"Shooting":75,
                     "Passing":72,"Dribbling":74,"Defending":45,"Physical":70}
    for i,feat in enumerate(feats):
        c = cols[i%4]
        mx = 200 if feat in ("Goals","Assists") else 99
        p_inputs[feat] = c.slider(feat, 0, mx, feat_defaults.get(feat, 50))

    if st.button("🧮 Calculate Performance Score", use_container_width=True):
        row  = pd.DataFrame([p_inputs])[feats]
        rows = sc.transform(row)
        pred = reg.predict(rows)[0]
        st.markdown(f"""
        <div class="result-box win" style="text-align:center">
          <div style="font-size:3rem;font-weight:700;color:#4fffc8">{pred:.1f}</div>
          <div style="color:#6b7a8d">Predicted Performance Score</div>
        </div>""", unsafe_allow_html=True)

        # Compare to dataset
        pct = (players_df["Performance"] < pred).mean() * 100
        st.success(f"This player ranks better than **{pct:.1f}%** of all players in the dataset!")

    # Top performers table
    st.markdown("### 🏅 Top 20 Players by Performance Score")
    top20 = players_df.nlargest(20,"Performance")
    show_cols = [c for c in ["Name","Position","Overall","Performance","Goals","Assists"]
                 if c in top20.columns]
    st.dataframe(top20[show_cols].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 4 — CLUSTERS
# ══════════════════════════════════════════════════════════════
elif page == "🗂️ Player Clusters":
    st.markdown("## 🗂️ Player Clustering (K-Means)")
    st.markdown("Players automatically grouped into **4 archetypes** based on their stats.")

    cnames = models["cluster_names"]
    feats  = models["cluster_feats"]

    if "Cluster_Name" in players_df.columns:
        # Cluster summary
        col1,col2,col3,col4 = st.columns(4)
        for i,(cid,cname) in enumerate(cnames.items()):
            ct = players_df[players_df["Cluster"]==cid]
            [col1,col2,col3,col4][i].metric(cname, f"{len(ct)} players")

        # Scatter
        st.markdown("### 🔵 Cluster Scatter Plot")
        cx = st.selectbox("X axis", feats, index=0)
        cy = st.selectbox("Y axis", feats, index=1)

        fig,ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
        colors_c = ["#FF6B00","#FF8C00","#FFA040","#FFC080"]
        for cid,cname in cnames.items():
            m = players_df["Cluster"]==cid
            ax.scatter(players_df.loc[m,cx], players_df.loc[m,cy],
                       color=colors_c[cid], label=cname, s=60, alpha=0.8, edgecolors="none")
        ax.tick_params(colors="#F0F0F0"); ax.spines[:].set_visible(False)
        ax.set_xlabel(cx, color="#888888"); ax.set_ylabel(cy, color="#888888")
        ax.legend(facecolor="#111111", labelcolor="#F0F0F0", fontsize=9, edgecolor="#222222")
        show_fig(fig)

        # Cluster table
        st.markdown("### 📋 Players by Cluster")
        selected_cluster = st.selectbox("Filter by cluster",
                                         ["All"] + list(cnames.values()))
        df_show = players_df if selected_cluster=="All" \
                  else players_df[players_df["Cluster_Name"]==selected_cluster]
        show_cols = [c for c in ["Name","Position","Cluster_Name","Performance",
                                  "Goals","Assists","Pace"] if c in df_show.columns]
        st.dataframe(df_show[show_cols].sort_values("Performance",ascending=False)
                     .reset_index(drop=True),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 5 — RECOMMENDER
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Player Recommender":
    st.markdown("## 🤖 Player Recommender")
    st.markdown("Find players with similar playing styles using **Cosine Similarity**.")

    rd     = models["rec_data"]
    sim    = models["rec_sim"]
    feats  = models["rec_feats"]

    name_col = "Name" if "Name" in rd.columns else None
    if name_col:
        all_names = rd[name_col].tolist()
        chosen    = st.selectbox("🔍 Search for a player", all_names)
        top_n     = st.slider("Number of similar players to show", 3, 15, 5)

        idx  = rd[rd[name_col]==chosen].index[0]
        sims = sorted(enumerate(sim[idx]), key=lambda x:x[1], reverse=True)
        sims = [(i,s) for i,s in sims if i!=idx][:top_n]

        # Radar chart comparison
        st.markdown("### 📡 Radar Comparison")
        categories = feats[:6]
        N = len(categories)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        scr  = models["rec_scaler"]
        scr2 = MinMaxScaler().fit(rd[categories])

        fig,ax = plt.subplots(figsize=(7,5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
        ax.spines["polar"].set_color("#222222")
        ax.tick_params(colors="#F0F0F0", labelsize=8)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, color="#F0F0F0", size=9)
        ax.yaxis.set_tick_params(labelcolor="#555555")
        ax.grid(color="#222222", linewidth=0.8)

        palette = ["#FF6B00","#FF8C00","#FFA040","#FFC080","#f59e0b"]
        t_val = scr2.transform([rd.iloc[idx][categories].values])[0].tolist() + [0]
        t_val[-1] = t_val[0]
        ax.plot(angles, t_val, color=palette[0], lw=2, label=str(chosen)[:16])
        ax.fill(angles, t_val, color=palette[0], alpha=0.2)

        for ri,(i,s) in enumerate(sims[:4]):
            rv = scr2.transform([rd.iloc[i][categories].values])[0].tolist()
            rv += rv[:1]
            nm = str(rd.iloc[i][name_col])[:16] if name_col else f"P{i}"
            ax.plot(angles, rv, color=palette[ri+1], lw=1.5, label=nm)
            ax.fill(angles, rv, color=palette[ri+1], alpha=0.08)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35,1.1),
                  facecolor="#111111", labelcolor="#F0F0F0", fontsize=8, edgecolor="#222222")
        show_fig(fig)

        # Similar players cards
        st.markdown("### 🃏 Most Similar Players")
        for rank,(i,s) in enumerate(sims,1):
            p = rd.iloc[i]
            pname = str(p[name_col]) if name_col else f"Player {i}"
            pos   = players_df.loc[players_df.get("Name","_")==pname,"Position"].values
            pos_t = pos[0] if len(pos)>0 else "—"
            perf  = players_df.loc[players_df.get("Name","_")==pname,"Performance"].values
            perf_v= f"{perf[0]:.0f}" if len(perf)>0 else "—"
            st.markdown(f"""
            <div class="player-card">
              <b>#{rank} {pname}</b>
              &nbsp; {role_tag(pos_t)}
              &nbsp;&nbsp;
              <span style="color:#4fffc8;font-weight:600">{s*100:.1f}% similar</span>
              &nbsp;&nbsp;
              <span style="color:#6b7a8d">Score: {perf_v}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 6 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "🌍 EDA Explorer":
    st.markdown("## 🌍 Data Explorer")

    tab1,tab2 = st.tabs(["👤 Player Data","⚽ Match Data"])

    with tab1:
        st.markdown("### Filter Players")
        col1,col2,col3 = st.columns(3)
        positions = ["All"] + (players_df["Position"].dropna().unique().tolist()
                               if "Position" in players_df.columns else [])
        sel_pos  = col1.selectbox("Position", positions)
        min_ovr  = col2.slider("Min Overall", 0, 99, 60)
        max_age  = col3.slider("Max Age", 16, 50, 35)

        df_f = players_df.copy()
        if sel_pos != "All" and "Position" in df_f.columns:
            df_f = df_f[df_f["Position"]==sel_pos]
        if "Overall" in df_f.columns:
            df_f = df_f[pd.to_numeric(df_f["Overall"],errors="coerce").fillna(0)>=min_ovr]
        if "Age" in df_f.columns:
            df_f = df_f[pd.to_numeric(df_f["Age"],errors="coerce").fillna(99)<=max_age]

        st.write(f"**{len(df_f):,} players** match your filters")

        num_cols = [c for c in ["Overall","Performance","Goals","Assists",
                                 "Pace","Shooting","Passing","Dribbling"]
                    if c in df_f.columns]
        if len(num_cols) >= 2:
            col1,col2 = st.columns(2)
            xax = col1.selectbox("X axis", num_cols, index=0)
            yax = col2.selectbox("Y axis", num_cols, index=1)

            fig,ax = plt.subplots(figsize=(9,5))
            fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
            sc = ax.scatter(df_f[xax], df_f[yax],
                            c=df_f["Performance"] if "Performance" in df_f.columns else "#FF6B00",
                            cmap="plasma", s=50, alpha=0.7, edgecolors="none")
            cbar = plt.colorbar(sc, ax=ax, label="Performance")
            cbar.ax.yaxis.label.set_color("#F0F0F0")
            cbar.ax.tick_params(colors="#888888")
            ax.tick_params(colors="#F0F0F0"); ax.spines[:].set_visible(False)
            ax.set_xlabel(xax, color="#888888"); ax.set_ylabel(yax, color="#888888")
            show_fig(fig)

        show_c = [c for c in ["Name","Position","Age","Overall","Performance"]
                  if c in df_f.columns]
        st.dataframe(df_f[show_c].sort_values("Performance",ascending=False)
                     .head(50).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Match Statistics")
        if "HomeTeam" in matches_df.columns:
            all_teams = sorted(set(matches_df["HomeTeam"].tolist()+matches_df["AwayTeam"].tolist()))
            sel_team  = st.selectbox("Filter by team", ["All"]+all_teams)
            mdf = matches_df if sel_team=="All" \
                  else matches_df[(matches_df["HomeTeam"]==sel_team)|(matches_df["AwayTeam"]==sel_team)]
        else:
            mdf = matches_df

        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(5,3.5))
            fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
            rc = mdf["FTR"].map({"H":"Home Win","D":"Draw","A":"Away Win"}).value_counts()
            pie_colors = ["#FF6B00","#f59e0b","#ef4444"]
            wedges, texts, autotexts = ax.pie(
                rc, labels=rc.index, autopct="%1.1f%%",
                colors=pie_colors[:len(rc)], startangle=90)
            for t in texts: t.set_color("#F0F0F0"); t.set_fontsize(9)
            for at in autotexts: at.set_color("#0a0a0a"); at.set_fontsize(8); at.set_fontweight("bold")
            ax.set_title("Results", color="#F0F0F0")
            show_fig(fig)
        with col2:
            if "HS" in mdf.columns and "AS" in mdf.columns:
                fig,ax = plt.subplots(figsize=(5,3.5))
                fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
                ax.hist(mdf["HS"].dropna(), bins=15, alpha=0.75, color="#FF6B00", label="Home")
                ax.hist(mdf["AS"].dropna(), bins=15, alpha=0.75, color="#FFA040", label="Away")
                ax.legend(facecolor="#111111", labelcolor="#F0F0F0", edgecolor="#222222")
                ax.tick_params(colors="#F0F0F0"); ax.spines[:].set_visible(False)
                ax.set_title("Shots Distribution", color="#F0F0F0")
                show_fig(fig)

# ══════════════════════════════════════════════════════════════
#  PAGE 7 — LEAGUE TABLE
# ══════════════════════════════════════════════════════════════
elif page == "📊 League Table":
    st.markdown("## 📊 League Table & Team Analytics")

    if "HomeTeam" in matches_df.columns:
        teams = list(set(matches_df["HomeTeam"].tolist()+matches_df["AwayTeam"].tolist()))
        rows  = []
        for t in teams:
            h = matches_df[matches_df["HomeTeam"]==t]
            a = matches_df[matches_df["AwayTeam"]==t]
            gf   = h["FTHG"].sum() + a["FTAG"].sum()
            ga   = h["FTAG"].sum() + a["FTHG"].sum()
            wins  = (h["FTR"]=="H").sum() + (a["FTR"]=="A").sum()
            draws = (h["FTR"]=="D").sum() + (a["FTR"]=="D").sum()
            games = len(h)+len(a)
            pts   = wins*3+draws
            rows.append({"Team":t,"P":games,"W":int(wins),"D":int(draws),
                         "L":int(games-wins-draws),"GF":int(gf),"GA":int(ga),
                         "GD":int(gf-ga),"Pts":int(pts)})
        table = pd.DataFrame(rows).sort_values("Pts",ascending=False).reset_index(drop=True)
        table.index += 1

        st.markdown("### 🏆 Full Standings")
        st.dataframe(table, use_container_width=True)

        st.markdown("### 📈 Goals Scored vs Conceded")
        top8 = table.head(8)
        fig,ax = plt.subplots(figsize=(10,4))
        fig.patch.set_facecolor("#0a0a0a"); ax.set_facecolor("#111111")
        x = np.arange(len(top8))
        ax.bar(x-0.2, top8["GF"], 0.4, label="Goals For",     color="#FF6B00", edgecolor="none")
        ax.bar(x+0.2, top8["GA"], 0.4, label="Goals Against",  color="#FF8C00", edgecolor="none", alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(top8["Team"], rotation=30, ha="right", color="#F0F0F0")
        ax.tick_params(colors="#F0F0F0"); ax.spines[:].set_visible(False)
        ax.legend(facecolor="#111111", labelcolor="#F0F0F0", edgecolor="#222222")
        show_fig(fig)

        st.markdown("### 🎯 Head-to-Head Lookup")
        all_teams_sorted = sorted(teams)
        c1,c2 = st.columns(2)
        t1 = c1.selectbox("Home Team", all_teams_sorted, index=0)
        t2 = c2.selectbox("Away Team", all_teams_sorted, index=1 if len(all_teams_sorted)>1 else 0)
        h2h = matches_df[
            ((matches_df["HomeTeam"]==t1)&(matches_df["AwayTeam"]==t2)) |
            ((matches_df["HomeTeam"]==t2)&(matches_df["AwayTeam"]==t1))
        ]
        if len(h2h) > 0:
            st.write(f"**{len(h2h)} meetings found**")
            t1_wins = ((h2h["HomeTeam"]==t1)&(h2h["FTR"]=="H")).sum() + \
                      ((h2h["AwayTeam"]==t1)&(h2h["FTR"]=="A")).sum()
            t2_wins = ((h2h["HomeTeam"]==t2)&(h2h["FTR"]=="H")).sum() + \
                      ((h2h["AwayTeam"]==t2)&(h2h["FTR"]=="A")).sum()
            draws   = (h2h["FTR"]=="D").sum()
            cc1,cc2,cc3 = st.columns(3)
            cc1.metric(f"{t1} Wins", int(t1_wins))
            cc2.metric("Draws",      int(draws))
            cc3.metric(f"{t2} Wins", int(t2_wins))
        else:
            st.info("No head-to-head matches found between these teams.")
    else:
        st.warning("Match team data not available.")

# ══════════════════════════════════════════════════════════════
#  PAGE 8 — ML CHART GALLERY  (ported from sports_ml_real.py)
# ══════════════════════════════════════════════════════════════
elif page == "🖼️ ML Chart Gallery":
    st.markdown("## 🖼️ ML Chart Gallery")
    st.markdown(
        "All charts are generated live from loaded data .  "
        "Hit **🔄 Regenerate** to get a fresh random palette & seed."
    )

    # ── Dynamic seed button ────────────────────────────────────
    if "gallery_seed" not in st.session_state:
        st.session_state.gallery_seed = int(time.time())

    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.gallery_seed = int(time.time())
            st.rerun()
    with col_info:
        st.caption(f"Current seed: `{st.session_state.gallery_seed}` — changes palette, splits & chart styles")

    GSEED = st.session_state.gallery_seed
    np.random.seed(GSEED)
    random.seed(GSEED)

    # ── Dynamic palettes (plasma-orange based) ────────────────
    PALETTES = [
        ["#FF6B00","#FF8C00","#FFA040","#FFC080","#FFD9B3","#FF4500"],
        ["#FF6B00","#FF9F1C","#FFBF69","#FF6B35","#F7C59F","#FF4500"],
        ["#FF6B00","#E85D04","#DC2F02","#9D0208","#F48C06","#FAA307"],
        ["#FF6B00","#FF8500","#FFA500","#FFBE00","#FF5500","#FF3300"],
        ["#FF6B00","#FF7C00","#FF9000","#FFAA00","#FFC000","#FFD700"],
    ]
    CMAPS = ["plasma","inferno","magma","hot","YlOrRd","Oranges","autumn"]
    PAL  = random.choice(PALETTES)
    CMAP = random.choice(CMAPS)
    BG   = random.choice(["#0a0a0a","#0d0a00","#100800","#080808","#0a0800"])

    # ── Helper to render a fig into the app ───────────────────
    def gallery_fig(fig):
        show_fig(fig)

    # ─────────────────────────────────────────────────────────
    #  CHART 1 — EDA Dashboard
    # ─────────────────────────────────────────────────────────
    with st.expander("📊 Chart 1 — EDA Dashboard", expanded=True):
        st.caption("Position distribution · Top nationalities · Performance histogram · Match results · Correlation heatmap · Attribute scatter")

        NUM_COLS = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
                    "Dribbling","Defending","Physical","Performance"]
                    if c in players_df.columns]
        HEAT_COLS = random.sample(NUM_COLS, min(6, len(NUM_COLS)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"EDA Dashboard  (Seed: {GSEED})", fontsize=16, fontweight='bold',
                     color='#F0F0F0')
        fig.patch.set_facecolor(BG)
        for ax in axes.flat:
            ax.set_facecolor("#111111")
            ax.tick_params(colors='#F0F0F0', labelsize=8)
            ax.spines[:].set_color('#222222')

        if "Position" in players_df.columns:
            pos_c = players_df["Position"].value_counts().head(6)
            wedges, texts, autotexts = axes[0,0].pie(
                pos_c, labels=pos_c.index, autopct="%1.1f%%",
                colors=PAL[:len(pos_c)], startangle=random.randint(0,360),
                explode=[0.05]*len(pos_c))
            for t in texts: t.set_color('#F0F0F0'); t.set_fontsize(8)
            for at in autotexts: at.set_color('#0a0a0a'); at.set_fontsize(7); at.set_fontweight('bold')
            axes[0,0].set_title("Position Distribution", fontweight='bold', color='#F0F0F0')

        if "Nationality" in players_df.columns:
            nat_c = players_df["Nationality"].value_counts().head(10)
            bar_colors = plt.colormaps[CMAP](np.linspace(0.2,0.9,len(nat_c)))
            if random.choice([True, False]):
                axes[0,1].barh(nat_c.index, nat_c.values, color=bar_colors)
                axes[0,1].invert_yaxis()
                axes[0,1].tick_params(axis='y', colors='#F0F0F0', labelsize=7)
            else:
                axes[0,1].bar(nat_c.index, nat_c.values, color=bar_colors)
                axes[0,1].tick_params(axis='x', rotation=40, labelsize=7)
                for t in axes[0,1].get_xticklabels(): t.set_color('#F0F0F0')
            axes[0,1].tick_params(axis='y', colors='#F0F0F0')
            axes[0,1].set_title("Top 10 Nationalities", fontweight='bold', color='#F0F0F0')

        if "Performance" in players_df.columns:
            perf = players_df["Performance"].dropna()
            if random.choice([True, False]):
                axes[0,2].hist(perf, bins=20, color=PAL[1], edgecolor='none', alpha=0.85)
            else:
                perf.plot.kde(ax=axes[0,2], color=PAL[1], lw=2)
                axes[0,2].fill_between(np.linspace(perf.min(),perf.max(),200),
                                       0, color=PAL[1], alpha=0.2)
            axes[0,2].axvline(perf.mean(), color='#FF6B00', linestyle='--', lw=1.5, label=f'Mean={perf.mean():.0f}')
            axes[0,2].set_title("Performance Score Distribution", fontweight='bold', color='#F0F0F0')
            axes[0,2].legend(facecolor='#111111', labelcolor='#F0F0F0', fontsize=8, edgecolor='#222222')

        res_map   = {"H":"Home Win","D":"Draw","A":"Away Win"}
        res_label = matches_df["FTR"].map(res_map)
        res_c     = res_label.value_counts()
        axes[1,0].bar(res_c.index, res_c.values,
                      color=[PAL[2],PAL[3],PAL[0]][:len(res_c)], edgecolor='none')
        axes[1,0].set_title("Match Result Distribution", fontweight='bold', color='#F0F0F0')
        for i,(idx,val) in enumerate(res_c.items()):
            axes[1,0].text(i, val+1, str(val), ha='center', fontweight='bold', color='#F0F0F0', fontsize=8)
        for t in axes[1,0].get_xticklabels(): t.set_color('#F0F0F0')

        corr = players_df[HEAT_COLS].dropna().corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=CMAP,
                    ax=axes[1,1], annot_kws={"size":8}, linewidths=0.5,
                    linecolor='#0a0a0a')
        axes[1,1].set_title("Correlation Heatmap", fontweight='bold', color='#F0F0F0')
        axes[1,1].tick_params(colors='#F0F0F0', labelsize=7)
        for t in axes[1,1].get_xticklabels(): t.set_color('#F0F0F0'); t.set_rotation(40)
        for t in axes[1,1].get_yticklabels(): t.set_color('#F0F0F0'); t.set_rotation(0)

        SCATTER_OPTS = [(a,b) for a,b in [("Goals","Assists"),("Shooting","Dribbling"),
                        ("Pace","Physical"),("Passing","Defending")]
                        if a in players_df.columns and b in players_df.columns]
        sx, sy = random.choice(SCATTER_OPTS) if SCATTER_OPTS else ("Goals","Assists")
        sc_df  = players_df[[sx,sy,"Performance"]].dropna()
        sp = axes[1,2].scatter(sc_df[sx], sc_df[sy], c=sc_df["Performance"],
                               cmap=CMAP, s=60, alpha=0.7, edgecolors='none')
        cbar = plt.colorbar(sp, ax=axes[1,2], label="Performance")
        cbar.ax.yaxis.label.set_color('#F0F0F0')
        cbar.ax.tick_params(colors='#888888')
        axes[1,2].set_title(f"{sx} vs {sy}", fontweight='bold', color='#F0F0F0')
        axes[1,2].set_xlabel(sx, color='#888888'); axes[1,2].set_ylabel(sy, color='#888888')
        axes[1,2].tick_params(colors='#F0F0F0')

        plt.tight_layout()
        gallery_fig(fig)

    # ─────────────────────────────────────────────────────────
    #  CHART 2 — Match Outcome Predictor
    # ─────────────────────────────────────────────────────────
    with st.expander("🏆 Chart 2 — Match Outcome Predictor"):
        st.caption("Confusion matrix · Feature importances — RandomForest or GradientBoosting chosen randomly")

        feat_cols = [c for c in ["HS","AS","HST","AST","HC","AC","HF","AF","HY","AY"]
                     if c in matches_df.columns]
        X_m = matches_df[feat_cols].dropna()
        le2 = LabelEncoder()
        y_m = le2.fit_transform(matches_df.loc[X_m.index,"FTR"])

        test_size = round(random.uniform(0.20, 0.30), 2)
        X_tr,X_te,y_tr,y_te = train_test_split(X_m, y_m, test_size=test_size,
                                                random_state=GSEED, stratify=y_m)
        use_gb = random.choice([True, False])
        if use_gb:
            from sklearn.ensemble import GradientBoostingClassifier
            clf2 = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                              learning_rate=0.1, random_state=GSEED)
            model_name2 = "Gradient Boosting"
        else:
            clf2 = RandomForestClassifier(n_estimators=150, max_depth=6,
                                          random_state=GSEED, class_weight='balanced')
            model_name2 = "Random Forest"

        clf2.fit(X_tr, y_tr)
        y_pred2 = clf2.predict(X_te)
        acc2    = accuracy_score(y_te, y_pred2)
        cv_sc2  = cross_val_score(clf2, X_m, y_m, cv=5)

        st.info(f"Model: **{model_name2}** | Accuracy: **{acc2*100:.1f}%** | CV: {cv_sc2.mean()*100:.1f}%±{cv_sc2.std()*100:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"Match Outcome Predictor — {model_name2}  (Seed:{GSEED})",
                     fontsize=13, fontweight='bold', color='#F0F0F0')
        for ax in axes:
            ax.set_facecolor("#111111")
            ax.tick_params(colors='#F0F0F0', labelsize=8)
            ax.spines[:].set_color('#222222')

        cm2 = confusion_matrix(y_te, y_pred2)
        sns.heatmap(cm2, annot=True, fmt='d', cmap=CMAP,
                    xticklabels=le2.classes_, yticklabels=le2.classes_, ax=axes[0],
                    linecolor='#0a0a0a', linewidths=0.5,
                    annot_kws={"size": 11, "weight": "bold", "color": "#F0F0F0"})
        axes[0].set_title(f"Confusion Matrix — Accuracy: {acc2*100:.1f}%", fontweight='bold', color='#F0F0F0')
        axes[0].set_xlabel("Predicted", color='#888888')
        axes[0].set_ylabel("Actual", color='#888888')
        for t in axes[0].get_xticklabels(): t.set_color('#F0F0F0')
        for t in axes[0].get_yticklabels(): t.set_color('#F0F0F0'); t.set_rotation(0)

        imp2 = pd.Series(clf2.feature_importances_, index=feat_cols).sort_values()
        imp2.plot(kind='barh', ax=axes[1],
                  color=plt.colormaps[CMAP](np.linspace(0.2, 0.9, len(imp2))))
        axes[1].set_title("Feature Importances", fontweight='bold', color='#F0F0F0')
        axes[1].set_xlabel("Importance Score", color='#888888')
        for t in axes[1].get_yticklabels(): t.set_color('#F0F0F0')
        plt.tight_layout()
        gallery_fig(fig)

    # ─────────────────────────────────────────────────────────
    #  CHART 3 — Player Performance Scorer
    # ─────────────────────────────────────────────────────────
    with st.expander("📈 Chart 3 — Player Performance Scorer"):
        st.caption("Actual vs Predicted scatter · Feature coefficients — Ridge or Linear Regression chosen randomly")

        PERF_FEATS = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
                      "Dribbling","Defending","Physical"] if c in players_df.columns]
        p_data = players_df[PERF_FEATS+["Performance"]].dropna()
        Xp = p_data[PERF_FEATS]; yp = p_data["Performance"]
        scaler_p = StandardScaler()
        Xp_s = scaler_p.fit_transform(Xp)
        Xp_tr,Xp_te,yp_tr,yp_te = train_test_split(Xp_s, yp,
                                    test_size=round(random.uniform(0.2,0.3),2),
                                    random_state=GSEED)
        use_ridge = random.choice([True,False])
        if use_ridge:
            alpha = round(random.uniform(0.5,5.0),2)
            from sklearn.linear_model import Ridge as _Ridge
            reg2 = _Ridge(alpha=alpha); reg_name2 = f"Ridge (α={alpha})"
        else:
            from sklearn.linear_model import LinearRegression as _LR
            reg2 = _LR(); reg_name2 = "Linear Regression"

        reg2.fit(Xp_tr, yp_tr)
        yp_pred2 = reg2.predict(Xp_te)
        r2_2  = r2_score(yp_te, yp_pred2)
        rmse2 = np.sqrt(mean_squared_error(yp_te, yp_pred2))

        st.info(f"Model: **{reg_name2}** | R²: **{r2_2:.3f}** | RMSE: {rmse2:.2f}")

        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"Performance Scorer — {reg_name2}  (Seed:{GSEED})",
                     fontsize=13, fontweight='bold', color='#F0F0F0')
        for ax in axes:
            ax.set_facecolor("#111111")
            ax.tick_params(colors='#F0F0F0', labelsize=8)
            ax.spines[:].set_color('#222222')

        axes[0].scatter(yp_te, yp_pred2, alpha=0.65, color=PAL[1],
                        edgecolors='none', s=60)
        mn,mx = min(yp_te.min(),yp_pred2.min()), max(yp_te.max(),yp_pred2.max())
        axes[0].plot([mn,mx],[mn,mx],'--',lw=2, color='#FF6B00', label='Perfect fit')
        axes[0].set_title(f"Actual vs Predicted  R²={r2_2:.3f}", fontweight='bold', color='#F0F0F0')
        axes[0].set_xlabel("Actual", color='#888888')
        axes[0].set_ylabel("Predicted", color='#888888')
        axes[0].legend(facecolor='#111111', labelcolor='#F0F0F0', edgecolor='#222222', fontsize=8)

        coef2 = pd.Series(reg2.coef_, index=PERF_FEATS).sort_values()
        bar_colors_c3 = ['#ef4444' if v < 0 else '#FF6B00' for v in coef2]
        coef2.plot(kind='barh', ax=axes[1], color=bar_colors_c3)
        axes[1].axvline(0, color='#F0F0F0', lw=0.8, alpha=0.4)
        axes[1].set_title("Feature Coefficients", fontweight='bold', color='#F0F0F0')
        axes[1].set_xlabel("Coefficient", color='#888888')
        for t in axes[1].get_yticklabels(): t.set_color('#F0F0F0')
        plt.tight_layout()
        gallery_fig(fig)

    # ─────────────────────────────────────────────────────────
    #  CHART 4 — Player Clustering
    # ─────────────────────────────────────────────────────────
    with st.expander("🗂️ Chart 4 — Player Clustering"):
        st.caption("Elbow method · Cluster scatter — k chosen randomly from {3,4,5}")

        CLUST_FEATS = [c for c in ["Goals","Assists","Passing","Dribbling",
                       "Defending","Physical","Pace"] if c in players_df.columns]
        c_data2   = players_df[CLUST_FEATS].dropna()
        scaler_c2 = StandardScaler()
        c_scaled2 = scaler_c2.fit_transform(c_data2)

        inertias2 = []
        for k in range(2,9):
            km_tmp = KMeans(n_clusters=k, random_state=GSEED, n_init=10)
            km_tmp.fit(c_scaled2)
            inertias2.append(km_tmp.inertia_)

        optimal_k2 = random.choice([3,4,5])
        kmeans2    = KMeans(n_clusters=optimal_k2, random_state=GSEED, n_init=10)
        c_labels2  = kmeans2.fit_predict(c_scaled2)

        ALL_NAMES2 = {
            3: ["Defenders","Attackers","Playmakers"],
            4: ["Defensive Wall","Speed Merchant","Goal Machine","Playmaker"],
            5: ["Rock Solid","Speedster","Striker","Creator","Box-to-Box"],
        }
        cnames2     = ALL_NAMES2[optimal_k2]
        c_data_cp   = c_data2.copy()
        c_data_cp["Cluster"] = c_labels2
        c_data_cp["Type"]    = c_data_cp["Cluster"].map(lambda x: cnames2[x])

        st.info(f"k={optimal_k2} clusters: {' | '.join(cnames2)}")

        fig, axes = plt.subplots(1,2,figsize=(14,5))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"Player Clustering — k={optimal_k2}  (Seed:{GSEED})",
                     fontsize=13, fontweight='bold', color='#F0F0F0')
        for ax in axes:
            ax.set_facecolor("#111111")
            ax.tick_params(colors='#F0F0F0', labelsize=8)
            ax.spines[:].set_color('#222222')

        axes[0].plot(range(2,9), inertias2, 'o-', color=PAL[0], lw=2, ms=8)
        axes[0].axvline(optimal_k2, color='#FF6B00', linestyle='--', lw=1.5, label=f'k={optimal_k2}')
        axes[0].set_title("Elbow Method", fontweight='bold', color='#F0F0F0')
        axes[0].set_xlabel("k", color='#888888')
        axes[0].set_ylabel("Inertia", color='#888888')
        axes[0].legend(facecolor='#111111', labelcolor='#F0F0F0', edgecolor='#222222', fontsize=8)

        cx2,cy2 = random.choice([(CLUST_FEATS[0],CLUST_FEATS[1]),
                                  (CLUST_FEATS[-1],CLUST_FEATS[0])])
        for ci in range(optimal_k2):
            mk = c_data_cp["Cluster"]==ci
            axes[1].scatter(c_data_cp.loc[mk,cx2], c_data_cp.loc[mk,cy2],
                            c=PAL[ci], label=cnames2[ci], s=70, alpha=0.85, edgecolors='none')
        axes[1].set_title(f"Clusters: {cx2} vs {cy2}", fontweight='bold', color='#F0F0F0')
        axes[1].set_xlabel(cx2, color='#888888')
        axes[1].set_ylabel(cy2, color='#888888')
        axes[1].legend(fontsize=8, facecolor='#111111', labelcolor='#F0F0F0', edgecolor='#222222')
        plt.tight_layout()
        gallery_fig(fig)

    # ─────────────────────────────────────────────────────────
    #  CHART 5 — Player Recommender
    # ─────────────────────────────────────────────────────────
    with st.expander("🤖 Chart 5 — Player Recommender"):
        st.caption("Cosine similarity heatmap · Radar comparison for a random target player")

        REC_FEATS2 = [c for c in ["Goals","Assists","Pace","Shooting","Passing",
                      "Dribbling","Defending","Physical"] if c in players_df.columns]
        name_col2 = "Name" if "Name" in players_df.columns else None
        r_cols2   = REC_FEATS2 + ([name_col2] if name_col2 else [])
        r_data2   = players_df[r_cols2].dropna().reset_index(drop=True)

        scaler_r2 = MinMaxScaler()
        r_scaled2 = scaler_r2.fit_transform(r_data2[REC_FEATS2])
        sim_mat2  = cosine_similarity(r_scaled2)

        target_idx2  = random.randint(0, len(r_data2)-1)
        target_name2 = (r_data2.iloc[target_idx2][name_col2]
                        if name_col2 else f"Player_{target_idx2}")

        sims2 = sorted(enumerate(sim_mat2[target_idx2]), key=lambda x:x[1], reverse=True)
        sims2 = [(i,s) for i,s in sims2 if i!=target_idx2][:5]

        st.info(f"Target player: **{target_name2}** | Top match similarity: {sims2[0][1]*100:.1f}%")

        sample_idx2   = random.sample(range(len(r_data2)), min(12,len(r_data2)))
        sim_sub2      = sim_mat2[np.ix_(sample_idx2,sample_idx2)]
        sample_names2 = ([str(r_data2.iloc[i][name_col2])[:12] for i in sample_idx2]
                         if name_col2 else [f"P{i}" for i in sample_idx2])

        fig = plt.figure(figsize=(16,6))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"Player Recommender  (Seed:{GSEED})", fontsize=13, fontweight='bold',
                     color='#F0F0F0')

        ax_heat = fig.add_subplot(1, 2, 1)
        ax_heat.set_facecolor("#111111")
        sns.heatmap(sim_sub2, xticklabels=sample_names2, yticklabels=sample_names2,
                    annot=True, fmt=".2f", cmap=CMAP, ax=ax_heat,
                    annot_kws={"size":7}, linewidths=0.4, linecolor='#0a0a0a')
        ax_heat.set_title("Similarity Matrix (random 12-player sample)", fontweight='bold', color='#F0F0F0')
        ax_heat.tick_params(axis='x', rotation=45, labelsize=7, colors='#F0F0F0')
        ax_heat.tick_params(axis='y', rotation=0, labelsize=7, colors='#F0F0F0')

        categories2 = REC_FEATS2[:6]
        N2     = len(categories2)
        angles2 = np.linspace(0,2*np.pi,N2,endpoint=False).tolist() + [0]
        ax_r2   = fig.add_subplot(1, 2, 2, polar=True)
        ax_r2.set_facecolor("#111111")
        ax_r2.spines["polar"].set_color("#222222")
        ax_r2.grid(color='#222222', linewidth=0.8)
        ax_r2.tick_params(colors='#F0F0F0', labelsize=7)

        scaler_radar2 = MinMaxScaler()
        scaler_radar2.fit(r_data2[categories2])
        t_vals2 = scaler_radar2.transform([r_data2.iloc[target_idx2][categories2].values])[0].tolist()
        t_vals2 += t_vals2[:1]
        ax_r2.plot(angles2, t_vals2, color=PAL[0], lw=2, label=str(target_name2)[:14])
        ax_r2.fill(angles2, t_vals2, color=PAL[0], alpha=0.25)
        for ri,(i,s) in enumerate(sims2[:3]):
            rv2 = scaler_radar2.transform([r_data2.iloc[i][categories2].values])[0].tolist()
            rv2 += rv2[:1]
            lbl2 = (str(r_data2.iloc[i][name_col2])[:14]
                    if name_col2 else f"Sim{ri+1}")
            ax_r2.plot(angles2, rv2, color=PAL[ri+1], lw=2, label=lbl2)
            ax_r2.fill(angles2, rv2, color=PAL[ri+1], alpha=0.15)
        ax_r2.set_xticks(angles2[:-1])
        ax_r2.set_xticklabels(categories2, size=8, color='#F0F0F0')
        ax_r2.yaxis.set_tick_params(labelcolor='#555555')
        ax_r2.set_title(f"Radar: {str(target_name2)[:14]}", fontweight='bold', pad=20, color='#F0F0F0')
        ax_r2.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=7,
                     facecolor='#111111', labelcolor='#F0F0F0', edgecolor='#222222')
        plt.tight_layout()
        gallery_fig(fig)

    # ─────────────────────────────────────────────────────────
    #  CHART 6 — League Table & Team Form
    # ─────────────────────────────────────────────────────────
    with st.expander("📊 Chart 6 — League Table & Team Form"):
        st.caption("Top teams by points · Win/Draw/Loss stacked bars")

        if "HomeTeam" in matches_df.columns:
            def _team_stats(team, df):
                h = df[df["HomeTeam"]==team]; a = df[df["AwayTeam"]==team]
                gf   = h["FTHG"].sum() + a["FTAG"].sum()
                ga   = h["FTAG"].sum() + a["FTHG"].sum()
                wins  = (h["FTR"]=="H").sum() + (a["FTR"]=="A").sum()
                draws = (h["FTR"]=="D").sum() + (a["FTR"]=="D").sum()
                games = len(h)+len(a)
                return {"Team":team,"Games":games,"Wins":int(wins),"Draws":int(draws),
                        "Losses":int(games-wins-draws),"GF":int(gf),"GA":int(ga),
                        "GD":int(gf-ga),"Points":int(wins*3+draws)}

            all_teams2 = [t for t in set(matches_df["HomeTeam"].tolist()+
                          matches_df["AwayTeam"].tolist()) if t and str(t)!='nan']
            table2 = pd.DataFrame([_team_stats(t,matches_df) for t in all_teams2])
            table2.sort_values("Points", ascending=False, inplace=True)
            table2.reset_index(drop=True, inplace=True); table2.index += 1

            top10 = table2.head(10)
            bar_c2 = plt.colormaps[CMAP](np.linspace(0.2, 0.9, len(top10)))

            fig, axes = plt.subplots(1,2,figsize=(16,6))
            fig.patch.set_facecolor(BG)
            fig.suptitle(f"League Table & Team Stats  (Seed:{GSEED})",
                         fontsize=13, fontweight='bold', color='#F0F0F0')
            for ax in axes:
                ax.set_facecolor("#111111")
                ax.tick_params(colors='#F0F0F0', labelsize=8)
                ax.spines[:].set_color('#222222')

            axes[0].barh(top10["Team"][::-1], top10["Points"][::-1], color=bar_c2[::-1], edgecolor='none')
            axes[0].set_title("Top 10 Teams by Points", fontweight='bold', color='#F0F0F0')
            axes[0].set_xlabel("Points", color='#888888')
            for t in axes[0].get_yticklabels(): t.set_color('#F0F0F0')
            for i,(_,row) in enumerate(top10[::-1].iterrows()):
                axes[0].text(row["Points"]+0.3, i, str(row["Points"]),
                             va='center', fontsize=8, color='#F0F0F0')

            x2 = np.arange(len(top10)); w=0.6
            axes[1].bar(x2, top10["Wins"],   width=w, label="Wins",   color=PAL[0], edgecolor='none')
            axes[1].bar(x2, top10["Draws"],  width=w, label="Draws",  color=PAL[2],
                        bottom=top10["Wins"], edgecolor='none')
            axes[1].bar(x2, top10["Losses"], width=w, label="Losses", color=PAL[4] if len(PAL)>4 else PAL[1],
                        bottom=top10["Wins"]+top10["Draws"], edgecolor='none')
            axes[1].set_xticks(x2)
            axes[1].set_xticklabels(top10["Team"], rotation=40, ha='right', fontsize=8, color='#F0F0F0')
            axes[1].set_title("Win / Draw / Loss Breakdown", fontweight='bold', color='#F0F0F0')
            axes[1].legend(facecolor='#111111', labelcolor='#F0F0F0', edgecolor='#222222', fontsize=8)
            plt.tight_layout()
            gallery_fig(fig)
        else:
            st.warning("Match team data not available for this chart.")

    # ─────────────────────────────────────────────────────────
    #  CHART 7 — Master Dashboard
    # ─────────────────────────────────────────────────────────
    with st.expander("🎯 Chart 7 — Master Dashboard"):
        st.caption("Full overview: KPI strip · Performance histogram · Match results · Top teams")

        if "HomeTeam" in matches_df.columns:
            all_teams3 = [t for t in set(matches_df["HomeTeam"].tolist()+
                          matches_df["AwayTeam"].tolist()) if t and str(t)!='nan']
            table3 = pd.DataFrame([_team_stats(t,matches_df) for t in all_teams3])
            table3.sort_values("Points", ascending=False, inplace=True)
            table3.reset_index(drop=True, inplace=True); table3.index += 1
        else:
            table3 = pd.DataFrame(columns=["Team","Points"])

        fig = plt.figure(figsize=(22,13))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"SPORTS ANALYTICS ML DASHBOARD  |  Seed: {GSEED}",
                     fontsize=18, fontweight='bold', color='#F0F0F0', y=0.99)

        _acc  = models["match_acc"]
        _r2   = models["perf_r2"]
        _opt_k = 4
        _mn   = "Random Forest"

        kpis3 = [("Match Accuracy", f"{_acc*100:.1f}%", PAL[0]),
                 ("Player R²",      f"{_r2:.3f}",        PAL[1]),
                 ("Players",        f"{len(players_df):,}", PAL[2]),
                 ("Matches",        f"{len(matches_df):,}", PAL[3]),
                 ("Clusters",       str(_opt_k),            PAL[4]),
                 ("ML Model",       _mn.split()[0],         PAL[5] if len(PAL)>5 else PAL[0])]
        for i,(lbl,val,col) in enumerate(kpis3):
            ax_k = fig.add_axes([0.02+i*0.163, 0.87, 0.15, 0.09])
            ax_k.set_facecolor(col); ax_k.axis('off')
            ax_k.text(0.5,0.65,val,ha='center',va='center',fontsize=20,
                      fontweight='bold',color='#0a0a0a',transform=ax_k.transAxes)
            ax_k.text(0.5,0.18,lbl,ha='center',va='center',fontsize=8,
                      color='#0a0a0a',alpha=0.85,transform=ax_k.transAxes)

        PANEL_BG = '#111111'
        ax1d = fig.add_axes([0.02,0.48,0.28,0.33]); ax1d.set_facecolor(PANEL_BG)
        perf3 = players_df["Performance"].dropna()
        ax1d.hist(perf3, bins=20, color=PAL[1], edgecolor='none', alpha=0.9)
        ax1d.axvline(perf3.mean(),color='#FF6B00',lw=2,linestyle='--',
                     label=f'Mean={perf3.mean():.0f}')
        ax1d.set_title("Performance Distribution",color='#F0F0F0',fontweight='bold')
        ax1d.tick_params(colors='#F0F0F0'); ax1d.spines[:].set_color('#222222')
        ax1d.legend(fontsize=8,labelcolor='#F0F0F0',facecolor=PANEL_BG,edgecolor='#222222')

        ax2d = fig.add_axes([0.35,0.48,0.20,0.33]); ax2d.set_facecolor(PANEL_BG)
        res_c3 = matches_df["FTR"].map({"H":"Home Win","D":"Draw","A":"Away Win"}).value_counts()
        ax2d.bar(res_c3.index, res_c3.values,
                 color=[PAL[2],PAL[0],PAL[3]][:len(res_c3)], edgecolor='none')
        ax2d.set_title("Match Results",color='#F0F0F0',fontweight='bold')
        ax2d.tick_params(colors='#F0F0F0'); ax2d.spines[:].set_color('#222222')
        for t in ax2d.get_xticklabels(): t.set_color('#F0F0F0')

        ax3d = fig.add_axes([0.59,0.48,0.38,0.33]); ax3d.set_facecolor(PANEL_BG)
        if len(table3) > 0:
            top5d = table3.head(8)
            ax3d.barh(top5d["Team"][::-1], top5d["Points"][::-1],
                      color=plt.colormaps[CMAP](np.linspace(0.2,0.9,len(top5d))),
                      edgecolor='none')
        ax3d.set_title("Top Teams (Points)",color='#F0F0F0',fontweight='bold')
        ax3d.tick_params(colors='#F0F0F0'); ax3d.spines[:].set_color('#222222')
        for t in ax3d.get_yticklabels(): t.set_color('#F0F0F0')

        ax4d = fig.add_axes([0.02,0.03,0.95,0.38])
        ax4d.set_facecolor(PANEL_BG); ax4d.axis('off')
        summary3 = (
            f"  MODULE A — Match Outcome Predictor (Random Forest)\n"
            f"  Accuracy: {_acc*100:.1f}%  |  Features: {', '.join([c for c in ['HS','AS','HST','AST','HC','AC','HF','AF','HY','AY'] if c in matches_df.columns])}\n\n"
            f"  MODULE B — Player Performance Scorer (Ridge)\n"
            f"  R²: {_r2:.3f}  |  Features: Goals, Assists, Pace, Shooting, Passing, Dribbling, Defending, Physical\n\n"
            f"  MODULE C — Player Clustering (K-Means k=4)\n"
            f"  Groups: Defensive Wall | Speed Merchant | Goal Machine | Playmaker\n\n"
            f"  MODULE D — Player Recommender (Cosine Similarity)\n\n"
            f"  Seed: {GSEED}  — Hit 🔄 Regenerate above for fresh splits, colours and chart styles!"
        )
        ax4d.text(0.01,0.97,summary3,va='top',ha='left',fontsize=9,color='#FF8C00',
                  family='monospace',transform=ax4d.transAxes,linespacing=1.7)
        ax4d.set_title("Full Model Summary",color='#F0F0F0',fontweight='bold',loc='left',pad=8)
        ax4d.spines[:].set_color('#222222')

        import tempfile, os
        _dash_path = os.path.join(tempfile.gettempdir(), "_gallery_dash.png")
        plt.savefig(_dash_path, dpi=120, bbox_inches='tight', facecolor=BG)
        st.image(_dash_path, use_container_width=True)
        plt.close(fig)