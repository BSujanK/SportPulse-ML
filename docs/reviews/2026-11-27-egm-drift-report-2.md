## README.md

- **Scope:** `README.md` as committed at `c432d0f`; README:N line numbers refer to that version. It was checked
  against `sportpulse/`, `webapp/`, `scripts/`, `tests/`, `app.py`, `artifacts/*/metrics.json`,
  `artifacts/analyst/vaep_metrics.json`, `artifacts/match/benchmark_dixon_coles.json` and `*_meta.json`. Only
  divergences are listed; all other README numbers match the artifacts.
- **Scope:** during the audit, something outside it changed the working tree: it rewrote `README.md` (moving the
  results block into new `docs/*.md` files) and edited `scripts/render_results.py`. None of the reviewers made
  these changes. (The deletion of `docs/2026-11-27_DRIFT_REPORT.md` was already in the working tree when the
  audit started.) The uncommitted versions were not audited.
- **Scope:** `pytest` was **not run**: this session's permissions allowed only `ls`/`wc`-style shell commands.
- **Bug:** **Dixon-Coles benchmark gives promoted teams the wrong starting rating.** README:151 says promoted teams "start
  at the bottom quarter's average rating". `scripts/benchmark_dixon_coles.py:104` selects
  `np.argsort(att - dfn)[: n // 4]`. In this parameterisation `dfn` is *subtracted from the opponent's* rate
  (`:44-45`, `eh = mu + h + att[hi] - dfn[ai]`), so higher `dfn` means a better defence and strength is `att + dfn`.
  The selection therefore picks low-attack, *strong-defence* sides, not the weakest quarter. This biases the
  Dixon-Coles baseline (RPS 0.2117) and every paired ΔRPS / z against it (README:155-159).
- **Bug:** **Train/serve skew in live match inference, against README:266-267** ("training and serving features match"):
  - With odds entered, `sportpulse/inference.py:93-97` sets `mkt_max_p_*` to the Shin probabilities of the entered
    odds. Training uses de-vigged *best* prices (`features/engine.py:220`). `mkt_p_o25` and `ah_line` are never set,
    so they are always NaN at serve time, though they are populated for most training rows (`engine.py:222-223`)
    and are inputs of the production market-residual booster (`outcome_meta.json` `all_features`).
  - The no-odds booster (the app default, README:277-278) is trained on the four manager-tenure features
    (`match_pipeline.py:81,98`), but `inference.py:98` supplies only `cv18_diff`/`cv11_diff`. `mgr_*` is therefore
    always NaN at serve time, while it was populated through the test window (2024-07 → 2026-06). The reported
    no-odds test score uses information the app never receives.
  - (Minor, documented.) Club values: training rows after the Transfermarkt snapshot get NaN
    (`features/club_value.py:43-47`), while serving uses the frozen last month (`club_value.py:60`). README:491-492
    documents this ("pre-summer-2026 squads"), so it is not a contradiction, but the model never saw that combination
    in training.
- **Bug:** **Own goals are missing from the StatsBomb event stream.** Ingest never reads "Own Goal For/Against" events
  (no match for `Own Goal` in `sportpulse/`). VAEP goals are only successful shots (`analysis/vaep.py:70`). So the
  P(score)/P(concede) labels miss own goals, the running `goal_diff` feature drifts from the real score
  (`vaep.py:72-79`), and the "values reset … after a goal" rule (README:367-368) does not fire after an own goal.
  Small but systematic.
- **Bug:** **The app's budget best XI can return fewer than 11 players.** README:300-301 calls it a "greedy £85m squad with
  formation limits". `webapp/views/fpl.py:90-92` reserves budget only for *unfilled formation minimums* (7 slots),
  not for every remaining slot. After the minimums are met, an expensive pick can leave too little for the last
  slots, and the loop ends with 10 or fewer players. The KPIs (`:101-103`) then show points and formation for the
  short squad without a warning. The duplicate check also compares `web_name` (`:87`), so two different players
  with the same short name block each other. In a double gameweek only one fixture row per player counts.
- **Bug:** **The fee model uses today's league for historical transfers.** README:350 says all features come "from rows dated
  strictly before the query". `pipelines/market_value_pipeline.py:71-72` maps `from_club_id`/`to_club_id` to
  `clubs.domestic_competition_id` from the 2026 snapshot, so promoted or relegated clubs carry their *current*
  league into past transfers (`league`, `to_league`, league value indexes; `to_league` is 7.3% of gain). The
  valuation models correctly use the league recorded with each valuation.
- **Bug:** **The next-valuation forecast is evaluated as a nowcast.** README:120,352-353 present it as a forecast of the
  next valuation. Training and test rows are built at the *date of the target valuation*
  (`market_value_pipeline.py:57`), so `days_since_value` (7th feature by gain) encodes the real gap to the next
  update and the as-of features run up to that day. Serving queries every player at
  `V.date.max() + 1` (`:169`), and the app plots the result 180 days later (`webapp/views/market_value.py:68`).
  The 12.9% median error assumes the revaluation date is known.
- **Bug:** **Opponent-report cues use the wrong unit.** Pitch coordinates are StatsBomb yards, labelled "(yd)" in
  `analysis/analyst.py:132,137`, but `webapp/views/opponent_report.py:91,108` print "pressures at … m" and
  "actions at … m avg.".
- **Stale:** **The results block and the comparison table are stale after the manager-tenure commit (`c432d0f`).**
  - README:55, *SportPulse Fundamentals (no odds)*: `0.2076 | 1.0142 | 0.6077 | 49.3%`.
    `artifacts/match/metrics.json` gives 0.20752 / 1.01394 / 0.60754 / 49.2%, which renders as
    **0.2075 | 1.0139 | 0.6075 | 49.2%**.
  - README:157, comparison table: `0.2077 | 1.0150 | −0.0040 (z = −8.7)`. `benchmark_dixon_coles.json` gives
    **0.2076 | 1.0147 | −0.0041 (z = −8.8)**. `scripts/render_results.py` does not generate this section, so
    re-rendering will not fix it.
  - README:427 (experiment log) reports the adopted no-odds test value as 0.20757. The current artifact is 0.20752.
- **Wrong:** **The second fundamentals booster is described incorrectly.** README:241 and :277-278 say it "adds the club squad
  values". It also uses `mgr_ten_h/a` and `mgr_new_h/a` (manager tenure from Transfermarkt games;
  `features/club_value.py:66-93`, `outcome_meta.json` `extra_fund_features`). The Architecture section and the
  experiment log do not mention them.
- **Wrong:** **The FPL ranking metrics are mislabelled.** README:90 (generated by `scripts/render_results.py`) says "among
  players FPL expects to play (xP > 0) … top-11 NDCG 0.506 vs 0.267". Those values are `ndcg11_all` (all
  players). The xP > 0 values are `ndcg11_xp_gt0` = **0.831 vs 0.610** (`artifacts/fpl/metrics.json:93-104`).
  README:195 and :421 repeat the same labelling.
- **Wrong:** **The "over all players" tie explanation contradicts the metrics.** README:90 and :197-198 say xP's Spearman "over
  all players" looks higher because of ties. Over all players, xP's Spearman is *lower* (0.327 vs 0.708). The
  0.720 vs 0.603 gap is the within-gameweek Spearman over players with recent minutes (`gw_spearman_active`,
  `fpl_pipeline.py:235`), as README:196 says correctly.
- **Unsupported:** **The Kendall figures have no source.** README:195 says "Kendall 0.613 vs 0.512". No code computes Kendall's τ
  and no artifact contains it (`fpl_pipeline.py:254-257` computes only Spearman and NDCG).
- **Wrong:** **The xG hold-out is not "men's tournaments".** README:329 says the model is "tested on 2022–24 men's tournaments".
  The hold-out set (README:111, `pipelines/analyst_pipeline.py:21-22`) includes Bundesliga 2023-24, Ligue 1 2022-23
  and MLS 2023. The same wording is in `analysis/xg.py:128`, `webapp/views/match_analysis.py:184` and
  `pipelines/vaep_pipeline.py:6`.
- **Wrong:** **The hold-outs are not untouched.** README:386 says candidates are "adopted only if the gain holds on the
  untouched test window". The experiment log shows the xG 2022–24 hold-out and the club/national test windows
  deciding many accept/reject choices (e.g. README:394-397, 408-409, 425-431). Several rows were adopted or
  rejected on test even when validation disagreed. The headline edges (xG 0.2525 vs 0.2563; market 0.2033 vs 0.2035)
  are therefore slightly optimistic.
- **Wrong:** **The Asian handicap is not de-vigged.** README:238 and :273-274 list "proportional de-vigging (O/U, Asian
  handicap)" and an "Asian-handicap line (proportionally de-vigged)". Only the raw `ah_line` is used
  (`engine.py:223`); handicap prices are not de-vigged or used. `ah_line` is also not range-cleaned
  (`data/loaders.py:127-131` cleans only `ah_h`/`ah_a`).
- **Wrong:** **The VAEP features are overstated.** README:369-370 says distance and angle to goal are features for "the
  action and the two before it". `analysis/vaep.py:89-113` computes them only for the current action. The
  previous-action block has type, success, team, locations and time gaps.
- **Wrong:** **The fair-value features are overstated.** README:355-356 and :414 list "national-team minutes **and goal
  involvements**". `HISTORY` (`market_value_pipeline.py:31`) has `ga_top5` and `ga_uefa` only. There is no
  national-team goal-involvement feature.
- **Wrong:** **The "80%" development interval is miscalibrated, and the coverage rule is not disclosed.**
  README:70 reports 86.5% coverage for the "80% interval". `pipelines/player_pipeline.py:91` counts a hit when
  `q10 − 0.5 ≤ y ≤ q90 + 0.5`. That is a reasonable continuity correction for integer ratings, but the README does
  not mention it. Under the rule actually used, the nominal 80% interval covers 86.5%, so the intervals are too
  wide; that is not the 80% calibration the README implies. The q10 booster early-stopped at 30 iterations, against
  462 and 584 for the others (`player_meta.json`), which points to a poorly fitted lower bound.
- **Wrong:** **The fee test set is narrower than stated.** README:201 says "2,434 paid transfers from July 2025 to July 2026".
  The test set also requires an earlier Transfermarkt valuation (`market_value_pipeline.py:141`,
  `mf = fte & fh`).
- **Unsupported:** **The "all 38 leagues" test coverage is unsupported.** README:48 says "17,054 matches … across all 38 leagues".
  `metrics.json` `data.divisions` = 38 counts the whole dataset, not the test window. Per-league results cover only
  the 24 leagues with ≥ 150 test matches (`match_pipeline.py:146`), which hold 15,772 of 17,054 matches. README:446
  ("Leagues … Elo history") also omits that Elo history covers only `FEATURED_DIVISIONS`, 8 of 38 leagues
  (`match_pipeline.py:242`).
- **Unsupported:** **"Edges the closing market" is unsupported.** README:162 says "closing". Nothing in the loaders
  establishes that the stored odds are closing prices.
- **App text:** **PL Player Forecaster model card** (`webapp/views/fpl.py:174-176`): says projections are "scaled by FPL's
  published chance-of-playing flag … injured players projected at zero". README:298-299 and
  `fpl_pipeline.py:298-299` do not apply the flag (`xpts = xpts_raw`); the best XI only filters on availability
  > 0.5 (`fpl.py:79`). The "What the model sees" list (`fpl.py:158-166`) omits FPL xP, the #2 feature
  (README:302).
- **App text:** **Match Forecaster** (`webapp/views/match_forecaster.py:103`): says "LightGBM, Optuna-tuned" and shows an Optuna
  tag. README:276-277 says the parameters are fixed defaults (`metrics.json` `"tuning": null`), and Optuna is not in
  `requirements.txt`. The page picks any two teams, not "any current-season fixture" (README:444).
- **App text:** **Match Analysis xG model card** (`match_analysis.py:172-173`): lists only cone counts, keeper position and
  nearest-defender pressure. It omits the visible goal-mouth share, lane blockers, distance-weighted pressure and
  keeper lane offset (README:325-327), which are among the strongest features.
- **App text:** **Hub "Updated" date** (`webapp/views/hub.py:23`): taken from the file mtime of `metrics.json`. On a fresh
  clone this is the clone time, not `production.trained_through` (2026-11-08).
- **App text:** **Leaderboard "Production configuration"** (`webapp/views/leaderboard.py:66-73`): shows the first fundamentals
  booster, not the club-value/manager booster that serves the no-odds forecasts.
- **App text:** **Docstrings** (`sportpulse/models/market_value.py:7`, `market_value_pipeline.py:117`): say fair value has "no
  Transfermarkt input"; README:356-358 correctly says it does.
- **Tests:** README:260 promises "leakage tests", and `tests/test_core.py:61` is headed "leakage & parity". The leakage tests
  only cover the club feature engine. There is **no train/serve parity test** between `pre_match_row` and
  `build_features` (README:266-267), which is how the inference skew above went unnoticed. The FPL lags,
  international Elo, Transfermarkt as-of features and point-in-time club values, all described as leak-free, have
  no leakage tests. `tests/test_artifacts.py` has no tests for the `market_value` or VAEP artifacts.
- **Tests:** `scripts/layout_audit.js:5-6` audits 11 pages; it skips Market Value, one of the 12 pages in README:439-454.
