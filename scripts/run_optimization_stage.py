from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_optimization_config
from trading_lab.data_loader import load_market_data
from trading_lab.optimize import run_optimization_stage, save_stage_leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one optimization stage.")
    parser.add_argument("--config", default="configs/optimization.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    total_stages = args.total_stages or config.total_stages
    data = load_market_data(config.data_path)
    leaderboard = run_optimization_stage(
        data,
        parameter_space=config.parameter_space,
        stage=args.stage,
        total_stages=total_stages,
        initial_cash=config.execution.initial_cash,
        commission_bps=config.execution.commission_bps,
        slippage_bps=config.execution.slippage_bps,
    )
    output_path = Path(config.output_dir) / f"stage_{args.stage}.csv"
    save_stage_leaderboard(leaderboard, output_path)

    print(
        json.dumps(
            {
                "stage": args.stage,
                "total_stages": total_stages,
                "rows": len(leaderboard),
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
