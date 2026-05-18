from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_optimization_config
from trading_lab.data_loader import load_market_data
from trading_lab.survival import build_survival_grid, evaluate_survival_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one survival search stage.")
    parser.add_argument("--config", default="configs/survival_github.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    total_stages = args.total_stages or config.total_stages
    data = load_market_data(config.data_path)
    grid = build_survival_grid(config.parameter_space, stage=args.stage, total_stages=total_stages)
    rows = [
        evaluate_survival_candidate(
            data,
            params,
            initial_cash=config.execution.initial_cash,
            commission_bps=config.execution.commission_bps,
            slippage_bps=config.execution.slippage_bps,
        )
        for params in grid
    ]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"survival_stage_{args.stage}.csv"
    pd.DataFrame(rows).sort_values("survival_score", ascending=False).to_csv(output_path, index=False)
    print(json.dumps({"stage": args.stage, "rows": len(rows), "output_path": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
