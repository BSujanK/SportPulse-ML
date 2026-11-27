# Experiment log

Every candidate change, with its validation and test effect and the decision. A change is adopted only if it improves both the validation window and the test window. Because the test window is
used in this decision too, headline test numbers are slightly optimistic.

[← Back to the README](../README.md)

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
| Transfer fees | Optuna-tuned LightGBM (40 trials) | 0.5486 → 0.5465 MAE log | 0.5375 → 0.5414 | rejected |
| Next valuation | Optuna-tuned LightGBM (20 trials) | 0.1967 → 0.1948 MAE log | 0.1998 → 0.1971; median error 13.3% → 12.9% | adopted |
| Fair value | Optuna-tuned LightGBM (20 trials) | 0.5474 → 0.5517 | 0.5874 → 0.5954 | rejected |
| Fair value | Career and 730-day top-5-league, UEFA and national-team minutes, and top-5-league and UEFA goal involvements | 0.5474 → 0.5343 MAE log | 0.5874 → 0.5821; median error 43.7% → 42.5% | adopted |
| Transfermarkt models | No minutes share when the club has under 15 known games | — | fees 0.535 → 0.537 MAE log | adopted (removes inflated shares for uncovered clubs) |
| PL players | Team and opponent rolling xG | flat | flat | kept (harmless) |
| PL players | Hurdle heads: P(plays) × points if plays | better | MAE 0.999 → 0.998 | adopted (blend) |
| PL players | Optuna-tuned heads | better | mixed | rejected |
| PL players | FPL's pre-deadline xP as a feature (archive verified to equal the live feed; no outcome leak) | MAE 1.043 → 0.690, best XI 5.49 → 10.12 | MAE 0.998 → 0.874, best XI 4.66 → 6.56, within-GW Spearman 0.561 → 0.603 | adopted |
| PL players | Ordering players by a blend of the prediction and FPL xP (weights 0.25 / 0.5 / 0.75) | within-GW Spearman 0.756 → 0.754 / 0.740 / 0.720 | 0.603 → 0.604 / 0.601 / 0.594 | rejected: validation prefers no blend |
| PL players | Ranking metric audit: rank only players FPL expects to play, and NDCG@11 for top-11 picks | — | SportPulse 0.753 vs xP 0.645; NDCG@11 0.506 vs 0.267 | metric adopted: the all-player Spearman gap to xP was a tie artifact |
| PL players | LambdaRank head per gameweek (for ordering players), alone or rank-blended | within-GW Spearman 0.574 → 0.553 / 0.567 | 0.561 → 0.542 / 0.555 | rejected: the gap to FPL xP (0.720) is team news, not the objective |
| Player development | Optuna-tuned LightGBM | flat | 1.591 → 1.593 | rejected |
| Club matches | Recency-weighted training (half-life 10 / 6 / 3 years) | flat | worse | rejected |
| Club matches | Asian-handicap price as a market feature | worse | 0.20335 → 0.20334 | rejected |
| Club matches | Asian-handicap price + best-vs-average odds gaps | better | 0.20335 → 0.20342 | rejected |
| Club matches, no odds | Transfermarkt club squad value (point-in-time, home minus away) | 0.20788 → 0.20758 | 0.20776 → 0.20757 | adopted (second fundamentals booster) |
| Club matches, no odds | Manager tenure and a new-manager flag from Transfermarkt games | 0.20758 → 0.20750 | 0.20757 → 0.20752 | rejected after review: the app cannot supply it (Transfermarkt games stop at the July 2026 snapshot), so the gain is not one live users get |
| Club matches, with odds | Same feature in the market-residual model | 0.20420 → 0.20415 | 0.20335 → 0.20338 | rejected (production unchanged) |
| Club matches, no odds | CatBoost fundamentals; 50/50 log-linear blend with LightGBM | CatBoost worse; blend 0.20758 → 0.20757 | CatBoost worse; blend 0.20757 → 0.20753 | not adopted: gain within noise, and the blend would add CatBoost to the deployed app |
| National teams | Recency-weighted training | worse | — | rejected |
| xG | Recency-weighted training (half-life 15 / 8 / 5 / 3 years) | mixed | 0.2514 → 0.2517 | rejected |
| VAEP | Third previous action + possession context | P(scores) 0.046735 → 0.046561, P(concedes) 0.013719 → 0.013512 | AUC 0.8322 → 0.8347 and 0.8507 → 0.8607 | adopted |
| VAEP | Goal width in yards (8.0, was 7.32) for the angle feature | — | AUC 0.8347 → 0.8353, 0.8607 → 0.8606 | adopted (correctness fix) |
| xG | Hold-out round count chosen on pre-July-2022 matches only (was CV including the hold-out) | — | 0.2514 → 0.2525 | adopted (removes a small leak) |

The consistent pattern: forecasts priced by liquid betting markets (1X2, over/under) and FPL points (driven by
unannounced line-ups) are at the ceiling of public data. Shot-level xG and national-team ratings still had headroom.
