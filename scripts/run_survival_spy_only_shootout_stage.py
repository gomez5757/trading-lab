from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from run_survival_spy_only_adaptive_stage import main as adaptive_main  # noqa: E402
from run_survival_spy_only_beam_stage import main as beam_main  # noqa: E402
from run_survival_spy_only_meta_stage import main as meta_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one equal-budget SPY-only shootout stage.")
    parser.add_argument("--method", choices=["adaptive", "beam", "bayesian", "bandit", "genetic"], required=True)
    parser.add_argument("--config", default="configs/survival_spy_only_github.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=12)
    parser.add_argument("--budget", type=int, default=720)
    args = parser.parse_args()

    if args.method == "adaptive":
        sys.argv = [
            "run_survival_spy_only_adaptive_stage.py",
            "--config",
            args.config,
            "--stage",
            str(args.stage),
            "--total-stages",
            str(args.total_stages),
            "--candidates-per-stage",
            str(args.budget),
        ]
        return adaptive_main()
    if args.method == "beam":
        if args.budget < 100:
            seed_pool = max(4, args.budget // 2)
            beam_width = max(2, args.budget // 4)
            generations = 1
            mutations_per_parent = max(1, (args.budget - seed_pool) // max(1, beam_width))
        else:
            seed_pool = 160
            beam_width = 20
            generations = 4
            mutations_per_parent = 7
        sys.argv = [
            "run_survival_spy_only_beam_stage.py",
            "--config",
            args.config,
            "--stage",
            str(args.stage),
            "--total-stages",
            str(args.total_stages),
            "--seed-pool",
            str(seed_pool),
            "--beam-width",
            str(beam_width),
            "--generations",
            str(generations),
            "--mutations-per-parent",
            str(mutations_per_parent),
        ]
        return beam_main()

    sys.argv = [
        "run_survival_spy_only_meta_stage.py",
        "--method",
        args.method,
        "--config",
        args.config,
        "--stage",
        str(args.stage),
        "--total-stages",
        str(args.total_stages),
        "--candidates-per-stage",
        str(args.budget),
    ]
    return meta_main()


if __name__ == "__main__":
    raise SystemExit(main())
