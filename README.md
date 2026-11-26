---
title: SportPulse ML
emoji: ⚽
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.46.0
app_file: app.py
pinned: false
license: mit
short_description: Real-data football forecasting that edges out the betting market
---

# ⚽ SportPulse ML

**Production-grade football forecasting, trained on real data through the 2026-27 season.**

SportPulse ML is an end-to-end machine-learning system:

- A leak-free feature engine and a hierarchical LightGBM forecaster that **matches or beats the
  bookmaker market** on 17,000+ out-of-sample matches.
- A Poisson / Dixon-Coles scoreline model.
- Player valuation, development-forecast and style-archetype models.
- A Hugging Face–style Streamlit UI in the original black and plasma-orange theme.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

## What changed from v1

The original app trained on partly synthetic data and had target leakage. v2 is a ground-up rebuild:

| v1 problem | v2 fix |
|---|---|
| Match "predictor" used **in-match** stats (shots, corners) to predict the result, so it only worked after the match ended | Only **pre-match** information: ratings, form, league context and pre-match odds, computed in one chronological pass |
| Player "Performance" target was `Goals*6 + Assists*4 + …` with **randomly generated** goals and assists | Real targets: EA market value, next-season rating |
| Missing columns filled with `np.random` | Missing stats stay NaN (handled natively by LightGBM); nothing is fabricated |
| 1 EPL season, 500 players, random train/test split | 238,854 matches (38 leagues, 2000 → Sept 2026), 142,079 player-seasons, chronological / group-wise splits |
| K-Means cluster names hard-coded to arbitrary cluster ids | GMM with BIC model selection; names derived from each centroid's distinguishing traits |
| 1,900-line single file | Tested package: data → features → models → evaluation → serving → UI |

## Architecture

```text
sportpulse/
├── config.py                 sources, leagues, chronological split dates
├── data/loaders.py           download + cache + clean (no imputation with fake values)
├── features/
│   ├── engine.py             sequential leak-free engine: Elo, pi-ratings, EWM form, league context
│   └── market.py             Shin de-vigging of bookmaker odds
├── models/
│   ├── match_outcome.py      fundamentals LGBM → market-residual LGBM (init_score = log p_mkt) → temperature
│   ├── goals.py              Poisson LGBM rates + Dixon-Coles ρ → scorelines, O/U, BTTS
│   └── players.py            value LGBM, quantile development LGBM, GMM archetypes, similarity
├── evaluation/metrics.py     RPS, log-loss, Brier, ECE, reliability curves, betting backtest
├── pipelines/                match_pipeline.py, player_pipeline.py (train → evaluate → refit → artifacts)
├── inference.py              MatchPredictor: live fixture prediction from saved team states
└── train.py                  CLI
webapp/                       Streamlit UI (HF-style design system + 7 views)
artifacts/                    trained models, metrics, serving state (committed so the app runs immediately)
tests/                        unit tests (incl. leakage tests) + artifact integration tests
```

### Match model

1. **Feature engine.** One chronological pass over every match. Each row is built from team state *before*
   kick-off, then the state is updated with the result. The same `pre_match_row` function serves live inference,
   so training and serving features match.
   * Margin-of-victory Elo (per-country pools, tier-aware initialisation)
   * **pi-ratings** (Constantinou & Fenton 2013): separate home/away ratings on a goal-difference scale
   * Missing-aware EWM form (short ≈ 8 matches, long ≈ 40) for goals, shots, shots on target, corners and points
   * League scoring environment; rest days; experience
   * Market: Shin de-vigged 1X2, best-price probabilities, overround, O/U 2.5, Asian-handicap line
2. **Stage 1, Fundamentals.** LightGBM multiclass with no betting information (Optuna TPE-tuned). Used when no
   odds exist.
3. **Stage 2, Market residual.** LightGBM *boosted from the market*: `init_score = log p_market`, so trees only
   learn systematic corrections to an already efficient forecast.
4. **Stage 3.** Temperature scaling on the validation season.
5. **Goals.** Two Poisson LightGBMs for λ_home and λ_away, plus Dixon-Coles low-score dependence ρ fitted by MLE.

**Protocol.** Train 2002-07 → 2023-06, validate 2023-24 (early stopping, tuning, calibration), **test 2024-07 →
2026-09**, never touched until evaluation. Then refit on all data for serving.

### Player models (EA FIFA 15–22)
* **Market value.** LightGBM on log €, 5-fold **GroupKFold by player** (a player's every season is held out
  together), plus a forward-edition stress test.
* **Development.** Next-season overall: L1 LightGBM point forecast plus q10/q90 quantile LightGBMs for
  80% intervals. Temporal test on the FIFA 21 → 22 transition.
* **Archetypes.** Quality-normalised style vectors → PCA(12) → full-covariance Gaussian mixture, k by BIC.
* **Similarity.** Cosine similarity in standardised 40-dim attribute space with position, age and budget filters.

## The app (Hugging Face Hub style)

| Page | What it is |
|---|---|
| 🏠 Hub | Org page: model and dataset cards, headline out-of-sample metrics |
| ⚡ Match Forecaster | Model card plus an **Inference API widget**: pick any current-season fixture in 38 leagues, optionally add odds, and get 1X2, xG, O/U 2.5, BTTS and a scoreline heatmap |
| 🏆 Leaderboard | Benchmark vs bookmaker and rating baselines, per-league, monthly, betting backtest |
| 📊 Leagues | Live standings (2026-27), power ratings from the model's own state, Elo history |
| 🔎 Player Scout | Fair value vs listed, rating forecast with interval, similar players, undervalued finder, rising stars |
| 🧬 Archetypes | Interactive style map and data-derived role profiles |
| 🗂️ Datasets | HF-style dataset viewer with per-column stats and histograms |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # uses the committed artifacts
```

Retrain from scratch (downloads ~120 MB of public data):

```bash
pip install -r requirements-dev.txt
python -m sportpulse.train all --trials 30           # ~25 min on 4 CPU cores
python -m sportpulse.train matches --refresh         # pull the latest results and retrain the match models
pytest -q
```

**Deploy to Hugging Face Spaces.** Create a Streamlit Space and push this repo. The YAML header above configures it.

## Data & honesty notes
* **Matches.** [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
  (football-data.co.uk + clubelo), updated through 2026-09-03.
* **Players.** [305kishan/FIFA](https://github.com/305kishan/FIFA): sofifa scrapes of EA FIFA 15–22. FIFA 22 is the
  newest player edition in the public sources this pipeline uses, so player outputs describe that snapshot plus one
  projected season. A newer sofifa export with the same schema can be dropped into `data/raw/`.
* Football is noisy. ~50% 1X2 accuracy is what well-calibrated public-information models achieve, and the betting
  market is the benchmark to beat. Any model claiming 70–90% pre-match accuracy is leaking the result.

## License
MIT. See [LICENSE](LICENSE). Author: **B Sujan Kumar** ([@BSujanK](https://github.com/BSujanK)).
