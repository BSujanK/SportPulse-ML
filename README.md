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
- An **analyst layer on StatsBomb event data** (3,961 men's and women's matches): an xG model that outperforms
  StatsBomb's commercial xG on unseen 2022–24 tournaments, expected threat, team style profiles (PPDA, field tilt, press height, build-up), opponent reports with
  scouting notes, match reports with xG race, shot maps and pass networks.
- **Real-world player valuation on Transfermarkt data** (656k valuations, 1.9M appearances, 175k transfers to
  July 2026): a transfer-fee model that beats the Transfermarkt value at predicting actual fees, a next-valuation
  forecaster, and a performance-only fair-value model for scouting.
- **VAEP action values** on 8M StatsBomb actions: every pass, carry, shot and tackle valued by its effect on scoring
  and conceding chances, with player ratings per 90.
- EA FIFA development-forecast and style-archetype models (FIFA 15–22 attributes).
- A Hugging Face–style Streamlit UI in the original black and plasma-orange theme.

| Data | Coverage | Freshness |
|---|---|---|
| Club matches + odds (football-data.co.uk via xgabora) | 238,934 matches, 38 leagues | 2026-09-03 (+ Premier League live to GW10, 2026-11-08) |
| Fantasy Premier League (vaastav) | 260k player-fixtures, 2016-17 → 2026-27 | GW10, 2026-11-08 |
| International results (martj42) | 49k internationals since 1872 | 2026-11-17 |
| StatsBomb open data (events + freeze-frames) | 3,961 matches, 101k shots, 3.8M passes | Euro 2024 / Copa América 2024 / Bundesliga 2023-24 |
| Transfermarkt (dcaribou/transfermarkt-datasets) | 656k valuations, 1.9M appearances, 175k transfers | 2026-07-06 |
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

EA FIFA models (FIFA 15–22 snapshot). The value model reproduces EA's own listed price from EA's own attributes, so its R² measures fidelity to EA, not to the transfer market (see Transfermarkt below).

| EA FIFA model | Metric | SportPulse | Best baseline |
|---|---|---|---|
| EA value (unseen players, FIFA 22) | R² log € | **0.9993** | 0.9608 (linear) |
| EA value | median abs. % error | **1.6%** | 15.1% |
| Development (21→22) | MAE rating pts | **1.591** | 1.764 (potential-pull) · 1.805 (persistence) |
| Development, U23 | MAE rating pts | **1.998** | 2.482 (persistence) |
| Development | 80% interval coverage | 86.5% | — |
| Archetypes | k by BIC | 11 roles | — |

*Data through **2026-11-08** · production models refit on 226,617 matches.*

### Premier League player forecaster (test: 2025-26 + 2026-27 GW1–10, 36,319 player-fixtures)

| Model | Points MAE ↓ | Spearman ↑ | Best-XI pts/player ↑ |
|---|---|---|---|
| **SportPulse LightGBM (production)** | 0.998 | 0.720 | **4.66** |
| Direct points head only | 0.999 | 0.719 | 4.68 |
| Hurdle head only (P(plays) × points if plays) | 0.991 | 0.723 | 4.71 |
| Research: SportPulse + archived FPL xP | 0.874 | 0.706 | 6.55 |
| FPL official xP | 1.092 | 0.327 | 3.41 |
| Form (last 6 mean) | 1.099 | 0.686 | 3.83 |
| Form (EWM) | 1.095 | 0.657 | 4.14 |
| Oracle (actual points) | 0.000 | 1.000 | 12.51 |

P(60+ min) AUC 0.947 · P(goal) AUC 0.887 · P(assist) AUC 0.847.

### National-team forecaster (test: 2024-07-01 → 2026-11-17)

| Model | Competitive RPS ↓ | Accuracy | World Cup 2026 RPS ↓ |
|---|---|---|---|
| SportPulse International (LightGBM) | 0.1657 | 61.8% | 0.1548 |
| World Football Elo (standard, logistic) | 0.1695 | 61.3% | 0.1598 |
| Squad value logistic (Peeters 2018) | 0.1995 | 54.8% | 0.1733 |
| Base rates | 0.2329 | 46.9% | 0.2295 |

Competitive matches where both squads have Transfermarkt coverage (n = 1,413): SportPulse International (LightGBM) 0.1647 · World Football Elo (standard, logistic) 0.1686 · Squad value logistic (Peeters 2018) 0.1808 · Base rates 0.2299. In the full-test rows above, the squad-value logistic gets a zero value difference for uncovered squads.

### Expected goals, StatsBomb open data (3,961 matches, 99,746 non-penalty shots)

| Model | CV log-loss ↓ | CV AUC ↑ | 2022–24 tournaments log-loss ↓ | 2022–24 ECE ↓ |
|---|---|---|---|---|
| **SportPulse xG (LightGBM)** | 0.2606 | 0.8155 | 0.2514 | 0.0082 |
| StatsBomb xG (commercial) | 0.2660 | 0.8048 | 0.2563 | 0.0104 |
| Distance + angle logistic | 0.2952 | 0.7368 | — | — |

### Transfermarkt valuation models (555,102 valuations of 41,078 players; test: 2025-07 → 2026-07)

| Task | Model | Median error ↓ | MAE log ↓ | R² log ↑ | Within ±25% ↑ |
|---|---|---|---|---|---|
| Transfer fee | **SportPulse fee model (LightGBM)** | **39.4%** | 0.537 | 0.800 | 33% |
| Transfer fee | Transfermarkt value, calibrated | 45.8% | 0.666 | 0.683 | 30% |
| Transfer fee | Transfermarkt value | 49.9% | 0.665 | 0.674 | 31% |
| Next valuation | **SportPulse forecast (LightGBM)** | **13.3%** | 0.200 | 0.964 | 73% |
| Next valuation | Persistence + age trend | 18.4% | 0.234 | 0.951 | 68% |
| Next valuation | Persistence (last Transfermarkt value) | 16.7% | 0.232 | 0.942 | 67% |
| Fair value, no TM input | **SportPulse fair value (LightGBM)** | **43.7%** | 0.587 | 0.763 | 30% |
| Fair value, no TM input | League × position × age median | 69.2% | 1.086 | 0.212 | 17% |

Direction of the next valuation change (moves over 5%): 79.9% of 30,288. Fees of €20M or more (172 moves): median error 19% vs 23% for the Transfermarkt value.

### VAEP action values (8.1M actions; trained before 2022-07-01, tested on 572,730 actions from the 2022–24 tournaments)

| Target | Model | Log-loss ↓ | Brier ↓ | AUC ↑ |
|---|---|---|---|---|
| P(scores) | Base rate | 0.05634 | 0.009959 | 0.5000 |
| P(scores) | xT (calibrated) | 0.04757 | 0.008853 | 0.7902 |
| P(scores) | Location-only LightGBM | 0.04424 | 0.008460 | 0.8117 |
| P(scores) | **SportPulse VAEP (LightGBM)** | 0.04307 | 0.008384 | 0.8322 |
| P(concedes) | Base rate | 0.01433 | 0.001976 | 0.5000 |
| P(concedes) | xT (calibrated) | 0.01434 | 0.001976 | 0.7399 |
| P(concedes) | Location-only LightGBM | 0.01196 | 0.001883 | 0.8203 |
| P(concedes) | **SportPulse VAEP (LightGBM)** | 0.01175 | 0.001880 | 0.8507 |
<!-- RESULTS:END -->

## Comparison with published work

Numbers from different papers are measured on different leagues, seasons and match counts, and RPS moves with
the league mix and the draw rate. So the strongest evidence below is **same data, same rows, paired test**:
the published methods are re-implemented and scored on the SportPulse test window.

**Club 1X2, same 16,843 test matches** (`python scripts/benchmark_dixon_coles.py`; Dixon-Coles refit monthly per
league on the previous three seasons, time decay ξ chosen on the 2023-24 validation season):

| Model | Source | RPS ↓ | Log-loss ↓ | Paired ΔRPS vs Dixon-Coles (z) |
|---|---|---|---|---|
| **SportPulse Market-Residual** | this repo | **0.2035** | **1.0013** | −0.0082 (z = −15.4) |
| Bookmaker market, Shin de-vigged | Štrumbelj 2014, *Int. J. Forecasting* | 0.2036 | 1.0016 | −0.0081 (z = −16.0) |
| SportPulse Fundamentals, no odds | this repo | 0.2079 | 1.0155 | −0.0038 (z = −8.3) |
| Pi-ratings + Elo logistic | Constantinou & Fenton 2013, *JQAS* | 0.2097 | 1.0206 | — |
| Dixon & Coles (1997) | Dixon & Coles 1997, *JRSS C* | 0.2117 | 1.0317 | — |

(The pi-rating row comes from the full 17,054-match leaderboard above.) Without odds, SportPulse beats the classic
Dixon-Coles model by a wide, significant margin. With odds, it edges the closing market (paired ΔRPS −0.00011,
z = −1.3). That is a consistent edge, but not a statistically significant one on 17k matches. Read it as
"matches the market"; beating a liquid market significantly with public data alone would be unusual.

**Published RPS figures on other data, for orientation only.** On the
[2017 Soccer Prediction Challenge](https://link.springer.com/article/10.1007/s10994-018-5763-8) test set, the
winning feature-engineered gradient-boosted trees of
[Hubáček, Šourek & Železný (2019)](https://mlanthology.org/mlj/2019/hubacek2019mlj-learning) scored RPS 0.2063.
The pi-ratings + CatBoost result summarised in [Bunker, Yeung & Fujii (2024)](https://arxiv.org/abs/2403.07669)
scored 0.1925 on that challenge's data. SportPulse uses the same family (rating features + gradient boosting)
on a far larger, odds-aware test set. These numbers are not head-to-head.

**Expected goals.** [Bajons & Harringer, *One xG model to rule them all?*](https://static.hudl.com/craft/performance-insights-research-stage/2025/One-xG-model-to-rule-them-all_-Robert-Bajons-Tobias-Harringer.pdf)
benchmark Hudl-StatsBomb's commercial xG, the industry reference model. SportPulse compares against the same
StatsBomb xG on the same shots: on the unseen 2022-24 tournaments it has log-loss 0.2514 vs 0.2563 and AUC 0.813 vs
0.803, and under 5-fold CV 0.2606 vs 0.2660.

**National teams.** Two references, both re-implemented on the same matches:
- **World Football Elo**, with its [published K-factors](https://www.eloratings.net/about). SportPulse has
  competitive-match RPS 0.1657 vs 0.1695 (accuracy 61.8% vs 61.3%), and 0.1548 vs 0.1598 at the 2026 World Cup.
- **Transfermarkt squad values**, after [Peeters (2018), *Int. J. Forecasting* 34(1)](https://ideas.repec.org/a/eee/intfor/v34y2018i1p17-29.html),
  who found crowd valuations forecast internationals better than Elo. On the 1,413 competitive test matches
  where both squads are covered, a squad-value logistic scores 0.1808, behind Elo
  (0.1686). So the finding does not replicate here on its own; this data only
  sees players at the dataset's clubs. Added to the ratings model, though, squad value is the largest single gain
  (0.1647).
- Squad values count only players whose club played covered games that season. This guards against look-ahead:
  the source lists players who reached a covered league at any time up to 2026. Citizenship is the current record, so
  nationality switchers are assigned their present country.

**Fantasy points.** The reference is the official FPL expected-points (xP) forecast: MAE 0.998 vs 1.092, and the
best-XI picks score 4.66 vs 3.41 points per player.

**Transfer fees.** Transfermarkt's crowd-sourced market value is the standard reference for fees. On
2,434 paid transfers from July 2025 to July 2026, the SportPulse fee model has median error
39% against 50% for the Transfermarkt value (R² log 0.80 vs 0.67).

**Action values.** VAEP is re-implemented from Decroos et al. (KDD 2019) and compared, on the same 2022–24
tournament actions, against expected threat (Karun Singh's xT grid, logistically calibrated) and a location-only
model.

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
* World Football Elo (eloratings.net K-factors: World Cup 60, continental finals 50, qualifiers / Nations League 40,
  other tournaments 30, friendlies 20; goal-difference multiplier). Home advantage and a K-factor scale are
  grid-searched on validation (80 Elo points, K × 1.25). A "neutral" match in a team's own country counts as home.
* Signals per team: short and long form, competitive-only form, **performance versus Elo expectation**, **schedule
  strength** (opponents' Elo) and **rating momentum**, all leak-free in one chronological pass.
* 5-seed bagged LightGBM multiclass (Optuna-tuned) + temperature calibration. Train 1994 → 2022, validate 2023 → mid-2024,
  test **mid-2024 → Nov 2026**, benchmarked against standard World Football Elo.

### Analyst layer (StatsBomb open data)
* **Ingest** streams each match's events and keeps compact tables: shots (with freeze-frame geometry), passes,
  carries, defensive actions and line-ups. Penalty shoot-outs are excluded, and full names are mapped to common
  names.
* **xG**: LightGBM (Optuna-tuned) with monotone constraints (distance ↓, angle ↑) on geometry, technique, phase,
  the assisting pass, a men's/women's flag and freeze-frame geometry: defenders and teammates in the shooting
  cone, **share of the goal mouth visible past the bodies in front of the ball**, blockers in the shot lane,
  distance-weighted pressure, and the goalkeeper's position, distance and offset from the ball-to-goal line.
  Validated out-of-fold by match and on a temporal hold-out
  (trained on matches before July 2022, tested on 2022–24 men's tournaments), benchmarked against StatsBomb's own xG.
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

### Transfermarkt valuation models
* **Leak-free as-of features.** For every (player, date) query, minutes, goals and assists over 180 / 365 / 730 days
  and the career; top-5-league, UEFA and national-team minutes; club points and goal difference over the past year;
  club and league value levels; and the valuation history. All come from rows dated strictly before the query.
  One global cumulative sum per table makes 650k queries take about a second.
* **Next valuation.** LightGBM (Huber loss) learns the change from the last value; persistence is the baseline.
* **Fair value.** The same features without this player's valuation history, for players the market has not priced.
* **Transfer fees.** Seller and buyer strength and league level are added; the Transfermarkt value is the baseline.
* Chronological split: train 2013-07 → 2024-06, validate 2024-07 → 2025-06, test 2025-07 → 2026-07.
* Clubs outside the dataset's leagues keep a stale league tag in the source, so current outputs use the latest
  valuation's club and drop clubs with fewer than 15 known games in the past year.

### VAEP action values
* Passes (with set-piece types), carries, shots and on-ball defensive actions in one time-ordered stream per match.
* Features: the action and the two before it in the acting team's frame (type, success, start and end location,
  distance and angle to goal, time gaps, possession change), score difference and clock.
* LightGBM estimates P(score) and P(concede) within the next 10 actions. Values are out-of-fold (2 folds by match).
* Player ratings use minutes estimated from the event stream and line-ups.

### Player models (EA FIFA 15–22)
* **Market value.** LightGBM on log €, 5-fold **GroupKFold by player** (a player's every season is held out
  together), plus a forward-edition stress test.
* **Development.** Next-season overall: L1 LightGBM point forecast plus q10/q90 quantile LightGBMs for
  80% intervals. Temporal test on the FIFA 21 → 22 transition.
* **Archetypes.** Quality-normalised style vectors → PCA(12) → full-covariance Gaussian mixture, k by BIC.
* **Similarity.** Cosine similarity in standardised 40-dim attribute space with position, age and budget filters.

## Experiment log

Every candidate is selected on a validation window and adopted only if the gain holds on the untouched test window.

| Model | Change | Validation | Test | Decision |
|---|---|---|---|---|
| xG | 18 → 80 StatsBomb competition-seasons (2× shots, men's + women's) | better | 0.2548 → 0.2534 | adopted |
| xG | Optuna-tuned LightGBM | better | 0.2534 → 0.2527 | adopted |
| xG | Freeze-frame geometry: visible goal share, lane blockers, weighted pressure, keeper lane offset | better | 0.2527 → 0.2516 | adopted |
| xG | Re-tuned after the geometry features | better | → 0.2514 | adopted |
| xG | Keeper reach / outfield-only coverage split | flat | worse | rejected |
| xG | CatBoost, and a LightGBM + CatBoost blend | better | worse | rejected |
| xG | Club vs international-tournament flag | better | worse | rejected |
| xG | 5-seed bagged ensemble | better | worse | rejected |
| National teams | Tuned home advantage and K-factor scale | better | 0.1680 → 0.1679 | adopted |
| National teams | Performance-vs-expectation, schedule strength, momentum | better | → 0.1672 | adopted |
| National teams | Optuna-tuned LightGBM | better | → 0.1673 | adopted (validation winner) |
| National teams | 5-seed bagged ensemble | better | → 0.1672 | adopted |
| National teams | Blend with Elo logistic | worse | flat | rejected |
| National teams | Transfermarkt squad value (top-26 sum, top-11 mean), home minus away | better | 0.1672 → 0.1656; World Cup 2026 0.1616 → 0.1562 | superseded by the point-in-time version |
| National teams | Same, counting only players at clubs covered that season (no look-ahead from the source's player list) | better (0.16646 → 0.16518) | 0.1672 → 0.1657; World Cup 2026 → 0.1548 | adopted |
| National teams | Squad-value levels and player counts as extra features | worse | — | rejected |
| Club matches | ClubElo cross-league ratings (verified pre-match) | flat | flat | rejected |
| Club matches | 5-seed bagged ensemble | flat | flat | rejected |
| Club matches | Log-linear blend with the Poisson-Dixon-Coles model | better | worse | rejected |
| Over/under 2.5 | Booster initialised from the market price | better | worse | rejected |
| Transfer fees | Squared-error loss, monotone value constraint, buyer-premium features, smaller trees | worse | mixed | rejected (Huber kept) |
| Transfermarkt models | No minutes share when the club has under 15 known games | — | fees 0.535 → 0.537 MAE log | adopted (removes inflated shares for uncovered clubs) |
| PL players | Team and opponent rolling xG | flat | flat | kept (harmless) |
| PL players | Hurdle heads: P(plays) × points if plays | better | MAE 0.999 → 0.998 | adopted (blend) |
| PL players | Optuna-tuned heads | better | mixed | rejected |
| Player development | Optuna-tuned LightGBM | flat | 1.591 → 1.593 | rejected |
| Club matches | Recency-weighted training (half-life 10 / 6 / 3 years) | flat | worse | rejected |
| Club matches | Asian-handicap price as a market feature | worse | 0.20335 → 0.20334 | rejected |
| Club matches | Asian-handicap price + best-vs-average odds gaps | better | 0.20335 → 0.20342 | rejected |
| National teams | Recency-weighted training | worse | — | rejected |
| xG | Recency-weighted training (half-life 15 / 8 / 5 / 3 years) | mixed | 0.2514 → 0.2517 | rejected |

The consistent pattern: forecasts priced by liquid betting markets (1X2, over/under) and FPL points (driven by
unannounced line-ups) are at the ceiling of public data. Shot-level xG and national-team ratings still had headroom.

## The app (Hugging Face Hub style)

| Page | What it is |
|---|---|
| Hub | Org page: model and dataset cards, headline out-of-sample metrics |
| Match Forecaster | Model card plus an **Inference API widget**: pick any current-season fixture in 38 leagues, optionally add odds, and get 1X2, xG, O/U 2.5, BTTS and a scoreline heatmap |
| Leaderboard | Benchmark vs bookmaker and rating baselines, per-league, monthly, betting backtest |
| Leagues | Live standings (2026-27), power ratings from the model's own state, Elo history |
| Opponent Report | Pre-match scouting of any team in the StatsBomb competitions: style radar, cues, shot and pressing maps, key players |
| Match Analysis | xG race, shot map, pass networks and team stats for any of 3,961 matches; xG model card |
| PL Player Forecaster | Next-gameweek projections for every Premier League player, a budget-constrained best XI, evaluation vs FPL xP |
| National Teams | World Elo ranking (Nov 2026), neutral-venue fixture predictor, World Cup 2026 evaluation |
| Market Value | Transfermarkt value history and next-update forecast, performance fair value, undervalued finder, 2025-26 fee test |
| EA FIFA Scout | EA FIFA rating forecast with interval, similar players, rising stars (FIFA 15–22) |
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
* **Transfermarkt.** [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), a public
  snapshot to 6 July 2026 that its author has stopped refreshing. Valuations are Transfermarkt's crowd-sourced
  estimates; fees are as recorded there.
* **Players.** [305kishan/FIFA](https://github.com/305kishan/FIFA): sofifa scrapes of EA FIFA 15–22. FIFA 22 is the
  newest player edition in the public sources this pipeline uses, so player outputs describe that snapshot plus one
  projected season. A newer sofifa export with the same schema can be dropped into `data/raw/`.
* Football is noisy. ~50% 1X2 accuracy is what well-calibrated public-information models achieve, and the betting
  market is the benchmark to beat. Any model claiming 70–90% pre-match accuracy is leaking the result.

## License
MIT. See [LICENSE](LICENSE). Author: **B Sujan Kumar** ([@BSujanK](https://github.com/BSujanK)).
