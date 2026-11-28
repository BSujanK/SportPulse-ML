"""Render the headline results table into README.md from the trained metrics.

    python scripts/render_results.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
mm = json.loads((ROOT / "artifacts/match/metrics.json").read_text())
pm = json.loads((ROOT / "artifacts/players/metrics.json").read_text())
d = mm["data"]

lines = [
    f"### Club match results on the test window ({d['test'][0][:7]} → {mm['monthly'][-1]['month']}: "
    f"{d['test_with_odds']:,} matches with pre-match odds across all {d['divisions']} leagues)",
    "",
    "| Model | RPS ↓ | Log-loss ↓ | Brier ↓ | Accuracy | ECE |",
    "|---|---|---|---|---|---|",
]
for r in mm["leaderboard"]:
    b = "**" if r["model"].startswith("SportPulse Market") else ""
    lines.append(f"| {b}{r['model']}{b} | {b}{r['rps']:.4f}{b} | {r['log_loss']:.4f} | {r['brier']:.4f} | "
                 f"{r['accuracy'] * 100:.1f}% | {r['ece'] * 100:.2f}% |")
ou = mm["over_under_25"]
v, dv = pm["value"], pm["development"]
lines += [
    "",
    f"**Over/Under 2.5** log-loss: model {ou['logloss_model']:.4f} · market {ou['logloss_market']:.4f} · "
    f"base rate {ou['logloss_base']:.4f}.",
    "",
    "EA FIFA models (FIFA 15–22 snapshot). The value model reproduces EA's own listed price from EA's "
    "own attributes, so its R² measures fidelity to EA, not to the transfer market (see Transfermarkt below).",
    "",
    "| EA FIFA model | Metric | SportPulse | Best baseline |",
    "|---|---|---|---|",
    f"| EA value (unseen players, FIFA 22) | R² log € | **{v['model']['r2']:.4f}** | "
    f"{v['baseline_linear_log']['r2']:.4f} (linear) |",
    f"| EA value | median abs. % error | **{v['model']['median_ape'] * 100:.1f}%** | "
    f"{v['baseline_linear_log']['median_ape'] * 100:.1f}% |",
    f"| Development (21→22) | MAE rating pts | **{dv['model']['mae']:.3f}** | "
    f"{dv['baseline_potential_pull']['mae']:.3f} (potential-pull) · {dv['baseline_persistence']['mae']:.3f} (persistence) |",
    f"| Development, U23 | MAE rating pts | **{dv['model_u23']['mae']:.3f}** | "
    f"{dv['baseline_persistence_u23']['mae']:.3f} (persistence) |",
    f"| Development | 80% interval coverage | {dv['model']['interval_80_coverage'] * 100:.1f}% | — |",
    f"| Archetypes | k by BIC | {pm['archetypes']['k']} roles | — |",
    "",
    f"*Data through **{d['last_date']}** · production models refit on {mm['production']['n_train']:,} matches.*",
]
fm = json.loads((ROOT / "artifacts/fpl/metrics.json").read_text())
im = json.loads((ROOT / "artifacts/international/metrics.json").read_text())
fd = fm["data"]
lines += ["", f"### Premier League player forecaster (test: 2025-26 + 2026-27 GW1–{fd['latest_gameweek']}, "
          f"{fd['test_rows']:,} player-fixtures)", "",
          "| Model | Points MAE ↓ | Spearman ↑ | Best-XI pts/player ↑ |", "|---|---|---|---|"]
for r in fm["leaderboard"]:
    b = "**" if "production" in r["model"] else ""
    lines.append(f"| {b}{r['model']}{b} | {r['mae']:.3f} | {r['spearman']:.3f} | {b}{r['top_xi']:.2f}{b} |")
c = fm["classifiers"]
lines += ["", f"P(60+ min) AUC {c['p_60']['auc']:.3f} · P(goal) AUC {c['p_goal']['auc']:.3f} · "
          f"P(assist) AUC {c['p_assist']['auc']:.3f}.", ""]
if fm.get("ranking"):
    rk = fm["ranking"]
    lines += [f"Ranking within each gameweek, among players FPL expects to play (xP > 0): Spearman "
              f"{rk['pred']['spearman_xp_gt0']:.3f} vs {rk['xP']['spearman_xp_gt0']:.3f} for FPL xP, top-11 NDCG "
              f"{rk['pred']['ndcg11_xp_gt0']:.3f} vs {rk['xP']['ndcg11_xp_gt0']:.3f}. Top-11 NDCG over all players: "
              f"{rk['pred']['ndcg11_all']:.3f} vs {rk['xP']['ndcg11_all']:.3f}. Among players with recent minutes "
              f"(including those FPL rules out), xP's Spearman is higher "
              f"({fm['gw_spearman_active']['xP']:.3f} vs {fm['gw_spearman_active']['pred']:.3f}) because about 80% "
              "of its values are tied at zero, mostly for players who then score zero.", ""]
lines += [
          f"### National-team forecaster (test: {im['data']['test_window'][0]} → {im['data']['test_window'][1]})", "",
          "| Model | Competitive RPS ↓ | Accuracy | World Cup 2026 RPS ↓ |", "|---|---|---|---|"]
wc = {r["model"]: r for r in im["world_cup_2026"]}
for r in im["competitive"]:
    lines.append(f"| {r['model']} | {r['rps']:.4f} | {r['accuracy'] * 100:.1f}% | "
                 f"{wc.get(r['model'], {}).get('rps', float('nan')):.4f} |")
if im.get("competitive_covered"):
    cc = im["competitive_covered"]
    lines += ["", f"Competitive matches where both squads have Transfermarkt coverage (n = {cc[0]['n']:,}): "
              + " · ".join(f"{r['model']} {r['rps']:.4f}" for r in cc)
              + ". In the full-test rows above, the squad-value logistic gets a zero value difference for uncovered squads."]
an = json.loads((ROOT / "artifacts/analyst/metrics.json").read_text())
HOLDOUT = ("2022–24 hold-out: World Cup 2022, Euro 2024, Copa América 2024, AFCON 2023, Bundesliga 2023-24, "
           "Ligue 1 2022-23 and MLS 2023; trained only on matches before July 2022.")
x = an["xg"]
lines += ["", f"### Expected goals, StatsBomb open data ({x['matches']:,} matches, {x['non_penalty_shots']:,} non-penalty shots)",
          "", "| Model | CV log-loss ↓ | CV AUC ↑ | 2022–24 hold-out log-loss ↓ | 2022–24 hold-out ECE ↓ |", "|---|---|---|---|---|"]
tt = x.get("temporal_recent_test", {})
for name, r in x["non_penalty_cv"].items():
    t = tt.get(name, {})
    b = "**" if name.startswith("SportPulse") else ""
    lines.append(f"| {b}{name}{b} | {r['log_loss']:.4f} | {r['auc']:.4f} | "
                 + (f"{t['log_loss']:.4f} | {t['ece']:.4f} |" if t else "— | — |"))
lines += ["", HOLDOUT + f" The hold-out model chooses its own round count on pre-July-2022 matches; penalties use "
          f"the empirical conversion rate ({x['penalty_rate'] * 100:.1f}%)."]
mvp = ROOT / "artifacts/market_value/metrics.json"
if mvp.exists():
    mv = json.loads(mvp.read_text())
    md = mv["data"]
    lines += ["", f"### Transfermarkt valuation models ({md['valuations']:,} valuations of {md['players']:,} players; "
              "test: valuations 2025-07 → 2026-06, fees 2025-07 → 2026-07)", "",
              "| Task | Model | Median error ↓ | MAE log ↓ | R² log ↑ | Within ±25% ↑ |", "|---|---|---|---|---|---|"]
    for task, lbl in (("fees", "Transfer fee"), ("forecast", "Next valuation"), ("fair_value", "Fair value, no valuation history")):
        for r in mv[task]["leaderboard"]:
            b = "**" if r["model"].startswith("SportPulse") else ""
            lines.append(f"| {lbl} | {b}{r['model']}{b} | {b}{r['median_ape'] * 100:.1f}%{b} | {r['mae_log']:.3f} | "
                         f"{r['r2_log']:.3f} | {r['within_25pct'] * 100:.0f}% |")
    bm = mv["fees"]["big_moves"]
    lines += ["", f"Direction of the next valuation change (moves over 5%): {mv['forecast']['direction_accuracy'] * 100:.1f}% "
              f"of {mv['forecast']['direction_n']:,}. Fees of €20M or more ({bm['n']} moves): median error "
              f"{bm['SportPulse']['median_ape'] * 100:.0f}% vs {bm['Transfermarkt value']['median_ape'] * 100:.0f}% "
              "for the Transfermarkt value."]
vp = ROOT / "artifacts/analyst/vaep_metrics.json"
if vp.exists():
    vm = json.loads(vp.read_text())
    lines += ["", f"### VAEP action values ({vm['actions'] / 1e6:.1f}M actions; trained before {vm['trained_before']}, "
              f"tested on {vm['test_actions']:,} actions from the 2022–24 hold-out competitions)", "",
              "| Target | Model | Log-loss ↓ | Brier ↓ | AUC ↑ |", "|---|---|---|---|---|"]
    for k, rows in vm["evaluation"].items():
        for mdl, r in rows.items():
            b = "**" if mdl.startswith("SportPulse") else ""
            lines.append(f"| P({k}) | {b}{mdl}{b} | {r['log_loss']:.5f} | {r['brier']:.6f} | {r['auc']:.4f} |")
block = "\n".join(lines)
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/results.md").write_text(
    "# Results\n\nEvery number below is measured on data the models never saw during training, tuning or "
    "calibration. Generated by `python scripts/render_results.py` from `artifacts/*/metrics.json`.\n\n"
    "[← Back to the README](../README.md)\n\n" + block.replace("### ", "## ") + "\n")

# ── README scorecard: one row per model, generated from the same metrics ─────────────────────────
lb = {r["model"]: r for r in mm["leaderboard"]}
ilb = {r["model"]: r for r in im["competitive"]}
iwc = {r["model"]: r for r in im["world_cup_2026"]}
flb = {r["model"]: r for r in fm["leaderboard"]}
xcv = x["temporal_recent_test"]
rows = [
    ("Club results, with odds", "RPS ↓", lb["SportPulse Market-Residual (production)"]["rps"], "Bookmaker (Shin)",
     lb["Bookmaker market (Shin)"]["rps"], "{:.4f}"),
    ("Club results, no odds", "RPS ↓", lb["SportPulse Fundamentals (no odds)"]["rps"], "Elo",
     lb["Elo logistic"]["rps"], "{:.4f}"),
    ("National teams", "RPS ↓", ilb["SportPulse International (LightGBM)"]["rps"], "World Football Elo",
     ilb["World Football Elo (standard, logistic)"]["rps"], "{:.4f}"),
    ("National teams, World Cup 2026", "RPS ↓", iwc["SportPulse International (LightGBM)"]["rps"],
     "World Football Elo", iwc["World Football Elo (standard, logistic)"]["rps"], "{:.4f}"),
    ("Expected goals (2022–24 hold-out)", "log-loss ↓", xcv["SportPulse xG (LightGBM)"]["log_loss"],
     "StatsBomb xG", xcv["StatsBomb xG (commercial)"]["log_loss"], "{:.4f}"),
]
if vp.exists():
    ev = vm["evaluation"]["scores"]
    rows.append(("Action values (VAEP)", "AUC ↑", ev["SportPulse VAEP (LightGBM)"]["auc"], "Expected threat (xT)",
                 ev["xT (calibrated)"]["auc"], "{:.3f}"))
if mvp.exists():
    fe = {r["model"]: r for r in mv["fees"]["leaderboard"]}
    fc = {r["model"]: r for r in mv["forecast"]["leaderboard"]}
    rows += [("Transfer fees", "median error ↓", fe["SportPulse fee model (LightGBM)"]["median_ape"],
              "Transfermarkt value", fe["Transfermarkt value"]["median_ape"], "{:.0%}"),
             ("Next Transfermarkt valuation", "median error ↓", fc["SportPulse forecast (LightGBM)"]["median_ape"],
              "Last value carried forward", fc["Persistence (last Transfermarkt value)"]["median_ape"], "{:.1%}")]
rows.append(("Fantasy Premier League points", "MAE ↓", flb["SportPulse LightGBM (production)"]["mae"],
             "FPL's own xP", flb["FPL official xP"]["mae"], "{:.3f}"))
card = ["| Task | Metric | SportPulse | Reference | Reference score |", "|---|---|---|---|---|"]
for task, metric, ours, ref, theirs, f in rows:
    card.append(f"| {task} | {metric} | **{f.format(ours)}** | {ref} | {f.format(theirs)} |")
scorecard = "\n".join(card)
readme = (ROOT / "README.md").read_text()
readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                f"<!-- RESULTS:START -->\n{scorecard}\n<!-- RESULTS:END -->", readme, flags=re.S)
(ROOT / "README.md").write_text(readme)
print(scorecard)
