from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ExecutionConfig(BaseModel):
    initial_cash: float = Field(default=10_000.0, gt=0)
    commission_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)


class StrategyConfig(BaseModel):
    fast_window: int = Field(default=5, gt=0)
    slow_window: int = Field(default=20, gt=0)


class BacktestConfig(BaseModel):
    data_path: str
    output_dir: str = "outputs/backtest"
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)


class OptimizationConfig(BaseModel):
    data_path: str
    output_dir: str = "outputs/optimization"
    total_stages: int = Field(default=16, gt=0)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    parameter_space: dict[str, list[Any]]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("config file must contain a YAML object")
    return data


def load_backtest_config(path: str | Path) -> BacktestConfig:
    return BacktestConfig.model_validate(load_yaml(path))


def load_optimization_config(path: str | Path) -> OptimizationConfig:
    return OptimizationConfig.model_validate(load_yaml(path))
