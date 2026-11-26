import numpy as np
import pandas as pd
import streamlit as st

from sportpulse import config
from webapp import state, ui

mm, pm = state.match_metrics(), state.player_metrics()


@st.cache_data(show_spinner=False)
def column_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        r = {"column": c, "dtype": str(s.dtype), "null %": float(s.isna().mean() * 100),
             "distinct": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            v = s.dropna().to_numpy(float)
            r["min"], r["max"], r["mean"] = float(v.min()), float(v.max()), float(v.mean())
            r["distribution"] = np.histogram(v, bins=24)[0].tolist()
        elif pd.api.types.is_datetime64_any_dtype(s):
            r["min"], r["max"] = s.min().strftime("%Y-%m-%d"), s.max().strftime("%Y-%m-%d")
            r["distribution"] = np.histogram(s.dropna().astype("int64"), bins=24)[0].tolist()
        else:
            top = s.value_counts().head(3)
            r["top values"] = ", ".join(f"{k} ({v})" for k, v in top.items())
            r["distribution"] = s.value_counts().head(24).tolist()
        rows.append(r)
    out = pd.DataFrame(rows)
    for c in ["min", "max"]:
        if c in out:
            out[c] = out[c].astype(str).replace("nan", "")
    return out


ds = st.radio("Dataset", ["club-matches-2000-2026", "ea-fifa-players-15-22"], horizontal=True,
              label_visibility="collapsed")

if ds.startswith("club"):
    d = mm["data"]
    ui.repo_header("sportpulse", ds, "🗂️",
                   tag_items=[("task", "Tabular Classification"), ("task", "Time-series Forecasting"),
                              ("", "csv"), ("", f"{d['matches_total']:,} rows"), ("", f"{d['divisions']} leagues"),
                              ("lic", "football-data.co.uk terms")])
    df = state.recent_matches().sort_values("date", ascending=False)
    split = f"recent ({df.date.min():%Y-%m} → {df.date.max():%Y-%m}, {len(df):,} rows)"
    card = f"""
**Source.** [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025),
compiled from football-data.co.uk (results, match stats, odds) and clubelo.com, covering
{d['first_date']} → **{d['last_date']}**.

**Cleaning applied by the pipeline.** Odds outside (1, 100] or with an implausible overround → NaN;
negative or absurd stat counts → NaN; shots-on-target > shots → NaN; duplicate fixtures dropped; result
recomputed from the score. Missing statistics are **never imputed with random values**: LightGBM handles NaN
natively.

**Splits.** train {d['train'][0]} → {d['train'][1]} ({d['train'][2]:,}) · valid → {d['valid'][1]}
({d['valid'][2]:,}) · test → {d['test'][1]} ({d['test'][2]:,}).

**Leagues.** """ + ", ".join(config.division_label(k) for k in config.DIVISIONS)
else:
    d = pm["data"]
    ui.repo_header("sportpulse", ds, "🗂️",
                   tag_items=[("task", "Tabular Regression"), ("", "csv"),
                              ("", f"{d['player_seasons']:,} rows"), ("", f"{d['players']:,} players"),
                              ("", "FIFA 15 → 22")])
    df = state.players().drop(columns=["label", "player_face_url"])
    split = f"latest edition (FIFA {int(df.edition.max()) % 100}, {len(df):,} rows) + model outputs"
    card = """
**Source.** [305kishan/FIFA](https://github.com/305kishan/FIFA): sofifa.com scrape of EA Sports FIFA 15–22
career-mode databases. One row per player per edition, with 60+ attributes, positions, value, wage, contract,
club and league.

**Processing.** A stable player id is parsed from the sofifa asset URL, so players are linked across editions
(used for development pairs and group-wise CV). Primary position is mapped to GK / DEF / MID / FWD and to
positional lines. Contract years left = contract end − edition.

**Model columns added to the viewer.** `fair_value_eur` (out-of-fold value model), `value_gap`,
`proj_overall` / `proj_low` / `proj_high` (development model), `archetype`, `style_x` / `style_y` (GMM style space).
"""

t_view, t_card = st.tabs(["🔍 Dataset viewer", "📄 Dataset card"])
with t_view:
    st.markdown(f'<p class="hf-mono">Split: {split}</p>', unsafe_allow_html=True)
    stats = column_stats(df)
    st.dataframe(stats, hide_index=True, use_container_width=True, height=330,
                 column_config={"distribution": st.column_config.BarChartColumn("Distribution", y_min=0),
                                "null %": st.column_config.ProgressColumn("Null %", min_value=0, max_value=100,
                                                                          format="%.1f"),
                                "mean": st.column_config.NumberColumn("Mean", format="%.3g")})
    q = st.text_input("Filter rows (pandas query, e.g. `division == 'E0' and hg >= 4`)", "")
    view = df
    if q:
        try:
            view = df.query(q)
        except Exception as e:  # noqa: BLE001 - show the user their query error
            st.error(f"Query error: {e}")
    st.dataframe(view.head(500), use_container_width=True, height=520, hide_index=True)
    st.caption(f"Showing {min(len(view), 500):,} of {len(view):,} rows")
with t_card:
    st.markdown(card)
