from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from trading_lab.backtest import run_backtest


def build_parameter_grid(parameter_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not parameter_space:
        raise ValueError("parameter_space cannot be empty")
    names = list(parameter_space)
    values = [parameter_space[name] for name in names]
    if any(not value for value in values):
        raise ValueError("parameter lists cannot be empty")
    return [dict(zip(names, combination, strict=True)) for combination in product(*values)]


def split_grid_for_stage(
    grid: list[dict[str, Any]],
    *,
    stage: int,
    total_stages: int,
) -> list[dict[str, Any]]:
    if total_stages <= 0:
        raise ValueError("total_stages must be greater than zero")
    if stage < 0 or stage >= total_stages:
        raise ValueError(f"stage must be between 0 and {total_stages - 1}")
    return [params for index, params in enumerate(grid) if index % total_stages == stage]


def run_optimization_stage(
    data: pd.DataFrame,
    *,
    parameter_space: dict[str, list[Any]],
    stage: int,
    total_stages: int,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    grid = build_parameter_grid(parameter_space)
    stage_grid = split_grid_for_stage(grid, stage=stage, total_stages=total_stages)
    rows = []

    for params in stage_grid:
        result = run_backtest(
            data,
            params=params,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
        rows.append({**params, **result.metrics})

    if not rows:
        return pd.DataFrame(columns=leaderboard_columns(parameter_space))

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def save_stage_leaderboard(leaderboard: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(path, index=False)
    return path


def leaderboard_columns(parameter_space: dict[str, list[Any]]) -> list[str]:
    return [
        *parameter_space.keys(),
        "initial_cash",
        "final_equity",
        "total_return_pct",
        "max_drawdown_pct",
        "trade_count",
        "win_rate_pct",
        "profit_factor",
        "sharpe",
        "score",
    ]
