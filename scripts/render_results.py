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
    f"### Results on the untouched test window ({d['test'][0]} → {d['test'][1]}, "
    f"{d['test_with_odds']:,} matches, {d['divisions']} leagues)",
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
    "| Player model | Metric | SportPulse | Best baseline |",
    "|---|---|---|---|",
    f"| Market value (unseen players, FIFA 22) | R² log € | **{v['model']['r2']:.4f}** | "
    f"{v['baseline_linear_log']['r2']:.4f} (linear) |",
    f"| Market value | median abs. % error | **{v['model']['median_ape'] * 100:.1f}%** | "
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
block = "\n".join(lines)
readme = (ROOT / "README.md").read_text()
readme = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", readme, flags=re.S)
(ROOT / "README.md").write_text(readme)
print(block)
