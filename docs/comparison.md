# Comparison with published work

How SportPulse compares with published football models, re-implemented on the same matches where possible.

[← Back to the README](../README.md)

Numbers from different papers are measured on different leagues, seasons and match counts, and RPS moves with
the league mix and the draw rate. So the strongest evidence below is **same data, same rows, paired test**:
the published methods are re-implemented and scored on the SportPulse test window.

**Club 1X2, same 16,843 test matches** (`python scripts/benchmark_dixon_coles.py`; Dixon-Coles refit monthly per
league on the previous 1,095 days, time decay ξ chosen on the 2023-24 validation season, ρ fitted in a second step
after the Poisson rates; league-months with under 200 history matches are skipped, which leaves 211 of the 17,054
matches unscored; promoted teams start at the bottom quarter's average rating):

| Model | Source | RPS ↓ | Log-loss ↓ | Paired ΔRPS vs Dixon-Coles (z) |
|---|---|---|---|---|
| **SportPulse Market-Residual** | this repo | **0.2035** | **1.0013** | −0.0082 (z = −15.4) |
| Bookmaker market, Shin de-vigged | Štrumbelj 2014, *Int. J. Forecasting* | 0.2036 | 1.0016 | −0.0081 (z = −16.0) |
| SportPulse Fundamentals, no odds (with club squad values) | this repo | 0.2077 | 1.0150 | −0.0040 (z = −8.7) |
| Logistic on pi-rating goal difference + Elo | pi-ratings from Constantinou & Fenton 2013, *JQAS* | 0.2097 | 1.0206 | — |
| Dixon & Coles (1997) | Dixon & Coles 1997, *JRSS C* | 0.2117 | 1.0317 | — |

(The pi-rating row comes from the full 17,054-match leaderboard in [results](results.md).) Without odds, SportPulse beats the classic
Dixon-Coles model by a wide, significant margin. With odds, it edges the market (paired ΔRPS −0.00011,
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
StatsBomb xG on the same shots: on the 2022-24 hold-out it has log-loss 0.2525 vs 0.2563 and AUC 0.811 vs 0.803, and
under 5-fold CV 0.2606 vs 0.2660.

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

**Fantasy points.** The reference is the official FPL expected-points (xP) forecast, which SportPulse now uses as an
input. Over the test window SportPulse has MAE 0.874 vs 1.092 and its best-XI picks score 6.56 vs 3.41 points per
player (7.92 vs 4.96 in 2026-27 so far). It also ranks better within each gameweek: among players FPL expects to
play, Spearman 0.753 vs 0.645 and top-11 NDCG 0.831 vs 0.610 (0.506 vs 0.267 over all players). An earlier
version of this README said xP ranked better (Spearman 0.720 vs 0.603 over players with recent minutes). That gap came
from ties: about 80% of xP values are exactly 0, mostly for players who then score 0, and Spearman over that group rewards those
tied zeros.

**Transfer fees.** Transfermarkt's crowd-sourced market value is the standard reference for fees. On
2,434 paid transfers from July 2025 to July 2026 (those with an earlier Transfermarkt valuation), the SportPulse fee model has median error
39% against 50% for the Transfermarkt value (R² log 0.80 vs 0.67).

**Action values.** VAEP is re-implemented from Decroos et al. (KDD 2019) and compared, on the same 2022–24
hold-out actions, against expected threat (Singh's method, with this repo's 16 × 12 grid, logistically calibrated on
start xT, end xT × success and shot flags) and a LightGBM on action type, success and locations only. The xT grid is
fitted on all matches, test matches included, which if anything favours the xT baseline.
