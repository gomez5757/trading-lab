from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.backtest import run_backtest
from trading_lab.config import load_backtest_config
from trading_lab.data_loader import load_market_data
from trading_lab.results import save_backtest_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one reproducible backtest.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    data = load_market_data(config.data_path)
    result = run_backtest(
        data,
        params=config.strategy.model_dump(),
        initial_cash=config.execution.initial_cash,
        commission_bps=config.execution.commission_bps,
        slippage_bps=config.execution.slippage_bps,
    )
    output_dir = save_backtest_outputs(
        config.output_dir,
        metrics=result.metrics,
        trades=result.trades,
        equity_curve=result.equity_curve,
    )

    print(json.dumps({"output_dir": str(output_dir), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
