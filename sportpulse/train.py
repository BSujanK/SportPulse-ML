"""Training CLI.

    python -m sportpulse.train all              # everything (default)
    python -m sportpulse.train matches --trials 30
    python -m sportpulse.train players
"""
from __future__ import annotations

import argparse
import logging


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train SportPulse models on real data")
    ap.add_argument("target", nargs="?", default="all", choices=["all", "matches", "players", "fpl", "international"])
    ap.add_argument("--trials", type=int, default=0,
                    help="Optuna trials for the fundamentals booster (0 = use defaults)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download source data (pull the latest matches) before training")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.target in ("all", "matches"):
        from sportpulse.pipelines.match_pipeline import run as run_matches
        m = run_matches(trials=args.trials, refresh=args.refresh)
        for row in m["leaderboard"]:
            print(f"{row['model']:42s} RPS {row['rps']:.4f}  LL {row['log_loss']:.4f}  "
                  f"ACC {row['accuracy']:.3f}")
    if args.target in ("all", "fpl"):
        from sportpulse.pipelines.fpl_pipeline import run as run_fpl
        f = run_fpl(refresh=args.refresh)
        for row in f["leaderboard"]:
            print(f"{row['model']:28s} MAE {row['mae']:.3f}  rho {row['spearman']:.3f}  top-XI {row['top_xi']:.2f}")
    if args.target in ("all", "international"):
        from sportpulse.international import run as run_int
        i = run_int(refresh=args.refresh)
        for row in i["competitive"]:
            print(f"{row['model']:40s} RPS {row['rps']:.4f}  ACC {row['accuracy']:.3f}")
    if args.target in ("all", "players"):
        from sportpulse.pipelines.player_pipeline import run as run_players
        p = run_players(refresh=args.refresh)
        print({k: v for k, v in p.items() if k in ("value", "development", "archetypes")})


if __name__ == "__main__":
    main()
