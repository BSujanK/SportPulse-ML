---
title: SportPulse ML
emoji: ⚽
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
license: mit
short_description: Real-data football forecasting that edges out the betting market
---

# ⚽ SportPulse ML

**Production-grade football forecasting, trained on real data through November 2026.**

SportPulse ML is an end-to-end machine-learning system:

- A leak-free feature engine and a hierarchical LightGBM forecaster that **edges past the bookmaker
  market** on 17,000+ out-of-sample club matches (38 leagues).
- A Poisson / Dixon-Coles scoreline model.
- A **Premier League player forecaster** (FPL points, P(60′), P(goal), P(assist)) that is live for the
  next gameweek of 2026-27.
- A **national-team forecaster** (tournament-weighted World Football Elo + LightGBM) with data through
  November 2026, including the 2026 World Cup.
- An **analyst layer on StatsBomb event data** (1,999 matches): an xG model on par with StatsBomb's commercial
  xG, expected threat, team style profiles (PPDA, field tilt, press height, build-up), opponent reports with
  scouting notes, match reports with xG race, shot maps and pass networks.
- Player valuation, development-forecast and style-archetype models (EA FIFA attribute data).
- A Hugging Face–style Streamlit UI in the original black and plasma-orange theme.

| Data | Coverage | Freshness |
|---|---|---|
| Club matches + odds (football-data.co.uk via xgabora) | 238,934 matches, 38 leagues | 2026-09-03 (+ Premier League live to GW10, 2026-11-08) |
| Fantasy Premier League (vaastav) | 260k player-fixtures, 2016-17 → 2026-27 | GW10, 2026-11-08 |
| International results (martj42) | 49k internationals since 1872 | 2026-11-17 |
| StatsBomb open data (events + freeze-frames) | 1,999 matches, 50k shots, 2M passes | Euro 2024 / Copa América 2024 / Bundesliga 2023-24 |
| EA FIFA 15–22 (sofifa) | 142k player-seasons | 2021-22 snapshot |

<!-- RESULTS:START -->
### Results on the untouched test window (2024-07-01 → 2026-11-08, 17,054 matches, 38 leagues)

| Model | RPS ↓ | Log-loss ↓ | Brier ↓ | Accuracy | ECE |
|---|---|---|---|---|---|
| **SportPulse Market-Residual (production)** | **0.2033** | 1.0005 | 0.5984 | 50.7% | 1.17% |
| Bookmaker market (Shin) | 0.2035 | 1.0008 | 0.5986 | 50.5% | 1.11% |
| Poisson-DC goals model | 0.2038 | 1.0024 | 0.5995 | 50.6% | 1.57% |
| SportPulse Fundamentals (no odds) | 0.2078 | 1.0148 | 0.6081 | 49.3% | 1.03% |
| Pi-rating + Elo logistic | 0.2097 | 1.0206 | 0.6119 | 48.9% | 1.03% |
| Elo logistic | 0.2103 | 1.0223 | 0.6132 | 48.8% | 1.06% |
| Base rates (train prior) | 0.2283 | 1.0748 | 0.6503 | 43.6% | 0.78% |

**Over/Under 2.5** log-loss: model 0.6790 · market 0.6781 · base rate 0.6941.

| Player model | Metric | SportPulse | Best baseline |
|---|---|---|---|
| Market value (unseen players, FIFA 22) | R² log € | **0.9993** | 0.9608 (linear) |
| Market value | median abs. % error | **1.6%** | 15.1% |
| Development (21→22) | MAE rating pts | **1.591** | 1.764 (potential-pull) · 1.805 (persistence) |
| Development, U23 | MAE rating pts | **1.998** | 2.482 (persistence) |
| Development | 80% interval coverage | 86.5% | — |
| Archetypes | k by BIC | 11 roles | — |

*Data through **2026-11-08** · production models refit on 226,617 matches.*

### Premier League player forecaster (test: 2025-26 + 2026-27 GW1–10, 36,319 player-fixtures)

| Model | Points MAE ↓ | Spearman ↑ | Best-XI pts/player ↑ |
|---|---|---|---|
| **SportPulse LightGBM (production)** | 0.999 | 0.719 | **4.65** |
| Research: SportPulse + archived FPL xP | 0.873 | 0.707 | 6.34 |
| FPL official xP | 1.092 | 0.327 | 3.41 |
| Form (last 6 mean) | 1.099 | 0.686 | 3.83 |
| Form (EWM) | 1.095 | 0.657 | 4.14 |
| Oracle (actual points) | 0.000 | 1.000 | 12.51 |

P(60+ min) AUC 0.947 · P(goal) AUC 0.887 · P(assist) AUC 0.847.

### National-team forecaster (test: 2024-07-01 → 2026-11-17)

| Model | Competitive RPS ↓ | Accuracy | World Cup 2026 RPS ↓ |
|---|---|---|---|
| SportPulse International (LightGBM) | 0.1680 | 61.3% | 0.1612 |
| World Football Elo (logistic) | 0.1695 | 61.3% | 0.1598 |
| Base rates | 0.2329 | 46.9% | 0.2295 |

### Expected goals, StatsBomb open data (1,999 matches, 49,257 non-penalty shots)

| Model | CV log-loss ↓ | CV AUC ↑ | 2022–24 tournaments log-loss ↓ | 2022–24 ECE ↓ |
|---|---|---|---|---|
| **SportPulse xG (LightGBM)** | 0.2507 | 0.8109 | 0.2548 | 0.0072 |
| StatsBomb xG (commercial) | 0.2503 | 0.8123 | 0.2563 | 0.0104 |
| Distance + angle logistic | 0.2797 | 0.7451 | — | — |
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
├── data/loaders.py           download + cache + clean, team-name canonicalisation (no fake values)
├── data/fpl.py               FPL player data + live Premier League results feed
├── data/statsbomb.py         StatsBomb event ingest → shots / passes / carries / defence / line-ups
├── analysis/xg.py            freeze-frame xG model + evaluation vs StatsBomb xG
├── analysis/analyst.py       expected threat, team styles, player output, pass networks, heat grids
├── international.py          national-team Elo engine, model, predictor
├── features/
│   ├── engine.py             sequential leak-free engine: Elo, pi-ratings, EWM form, league context
│   └── market.py             Shin de-vigging of bookmaker odds
├── models/
│   ├── match_outcome.py      fundamentals LGBM → market-residual LGBM (init_score = log p_mkt) → temperature
│   ├── goals.py              Poisson LGBM rates + Dixon-Coles ρ → scorelines, O/U, BTTS
│   ├── players.py            value LGBM, quantile development LGBM, GMM archetypes, similarity
│   └── fpl_model.py          per-fixture player forecaster (points + 3 probability heads)
├── evaluation/metrics.py     RPS, log-loss, Brier, ECE, reliability curves, betting backtest
├── pipelines/                match_pipeline.py, player_pipeline.py (train → evaluate → refit → artifacts)
├── inference.py              MatchPredictor: live fixture prediction from saved team states
└── train.py                  CLI
webapp/                       Streamlit UI (HF-style design system + 7 views)
artifacts/                    trained models, metrics, serving state (committed so the app runs immediately;
                              the 71 MB value booster is regenerated by training, not committed)
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

### Premier League player forecaster (FPL 2016-17 → 2026-27)
* One row per player per fixture. All features are lagged: rolling 3 / 6 / EWM form for minutes, points,
  goals, assists, xG, xA, xGI, xGC, ICT, BPS, bonus, clean sheets, saves; per-90 rates; season-to-date;
  price, ownership and transfer momentum from the previous gameweek.
* **Fixture context from the match engine:** team and opponent Elo, pi-rating goal difference, goals and shots
  on target for/against, from the team's perspective.
* Four LightGBM heads: points (L2), P(60+ min), P(goal), P(assist). Train 2016-17 → 2023-24, early-stop
  2024-25, **test 2025-26 + 2026-27**. The production refit projects the next gameweek, scaled by FPL's
  chance-of-playing flag.
* Benchmarked against FPL's own expected points (xP). A research variant that stacks archived xP scores
  higher in backtest, but is not deployed because the live `ep_next` feed is distributed differently.

### National-team forecaster
* World Football Elo (eloratings.net): K = 60 World Cup, 50 continental finals, 40 qualifiers / Nations League,
  30 other tournaments, 20 friendlies; goal-difference multiplier; +100 home advantage only at non-neutral venues
  (a "neutral" match in a team's own country counts as home).
* EWM form (all matches and competitive-only) → LightGBM multiclass + temperature calibration. Train 1994 →
  2022, early-stop 2023 → mid-2024, test **mid-2024 → Nov 2026**.

### Analyst layer (StatsBomb open data)
* **Ingest** streams each match's events and keeps compact tables: shots (with freeze-frame geometry), passes,
  carries, defensive actions and line-ups. Penalty shoot-outs are excluded, and full names are mapped to common
  names.
* **xG**: LightGBM with monotone constraints (distance ↓, angle ↑) on geometry, technique, phase, the assisting
  pass and freeze-frame features (defenders and teammates in the shooting cone, goalkeeper position and
  distance, nearest defender, pressure count). Validated out-of-fold by match and on a temporal hold-out
  (trained on ≤ 2021, tested on 2022–24 tournaments), benchmarked against StatsBomb's own xG.
* **Expected threat** (Singh 2018), a 16 × 12 Markov grid from 3.5M passes and carries, gives each move's
  xT added.
* **Team style profiles** per competition, with percentiles: npxG for/against, possession, field tilt,
  PPDA, pressure height, high regains, progressive passes/carries, xT, crosses, long-ball share, set-piece xG,
  defensive-action height.
* **Opponent report**: style radar, strengths and watch-outs, style reading, rule-based game-plan cues,
  attacking and conceded shot maps, pressing and ball-winning heat maps, key players (npxG, xA, xT per 90),
  match log.
* **Match report**: xG race, two-way shot map, starters' pass networks, team-stat comparison, xG model card.

### Live data
`load_matches()` appends finished current-season Premier League fixtures from the FPL feed when they are newer
than the club dataset, so Premier League ratings are current to the latest gameweek. It also canonicalises
team names that the source renamed mid-history ("Nottm Forest" → "Nott'm Forest", "MGladbach" →
"M'gladbach"), which previously split those clubs' rating histories.

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
| Hub | Org page: model and dataset cards, headline out-of-sample metrics |
| Match Forecaster | Model card plus an **Inference API widget**: pick any current-season fixture in 38 leagues, optionally add odds, and get 1X2, xG, O/U 2.5, BTTS and a scoreline heatmap |
| Leaderboard | Benchmark vs bookmaker and rating baselines, per-league, monthly, betting backtest |
| Leagues | Live standings (2026-27), power ratings from the model's own state, Elo history |
| Opponent Report | Pre-match scouting of any team in the StatsBomb competitions: style radar, cues, shot and pressing maps, key players |
| Match Analysis | xG race, shot map, pass networks and team stats for any of 1,999 matches; xG model card |
| PL Player Forecaster | Next-gameweek projections for every Premier League player, a budget-constrained best XI, evaluation vs FPL xP |
| National Teams | World Elo ranking (Nov 2026), neutral-venue fixture predictor, World Cup 2026 evaluation |
| Player Scout | Fair value vs listed, rating forecast with interval, similar players, undervalued finder, rising stars |
| Archetypes | Interactive style map and data-derived role profiles |
| Datasets | HF-style dataset viewer with per-column stats and histograms |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # uses the committed artifacts
```

Retrain from scratch (downloads ~120 MB of public data):

```bash
pip install -r requirements-dev.txt
python -m sportpulse.train all --trials 30           # ~40 min on 4 CPU cores
python -m sportpulse.train fpl --refresh             # pull the newest gameweek and re-project
python -m sportpulse.train international --refresh   # pull the newest internationals
python -m sportpulse.train analyst                   # StatsBomb ingest + xG + analyst aggregates (~3 min)
python -m sportpulse.train matches --refresh         # pull the latest results and retrain the match models
pytest -q
```

**Deploy.** Either option works:
* [Streamlit Community Cloud, one click](https://share.streamlit.io/deploy?repository=BSujanK/SportPulse-ML&branch=claude/tender-hypatia-qfphwp&mainModule=app.py)
  (choose Python 3.12)
* Hugging Face Spaces: create a Streamlit Space and push this repo. The YAML header above configures it.

## Data & honesty notes
* **Events.** [StatsBomb open data](https://github.com/statsbomb/open-data). *Data provided by StatsBomb.*
  CC BY-NC-SA 4.0: attribution required, non-commercial use. A commercial club deployment would need a licensed
  event feed (StatsBomb, Opta, Wyscout) through the same ingest interface.
* **Matches.** [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
  (football-data.co.uk + clubelo), updated through 2026-09-03.
* **Players.** [305kishan/FIFA](https://github.com/305kishan/FIFA): sofifa scrapes of EA FIFA 15–22. FIFA 22 is the
  newest player edition in the public sources this pipeline uses, so player outputs describe that snapshot plus one
  projected season. A newer sofifa export with the same schema can be dropped into `data/raw/`.
* Football is noisy. ~50% 1X2 accuracy is what well-calibrated public-information models achieve, and the betting
  market is the benchmark to beat. Any model claiming 70–90% pre-match accuracy is leaking the result.

## License
MIT. See [LICENSE](LICENSE). Author: **B Sujan Kumar** ([@BSujanK](https://github.com/BSujanK)).
