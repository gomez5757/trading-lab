import pytest
import pandas as pd

from trading_lab.optimize import build_parameter_grid, run_optimization_stage, split_grid_for_stage


def test_build_parameter_grid_creates_cartesian_product() -> None:
    grid = build_parameter_grid(
        {
            "fast_window": [2, 3],
            "slow_window": [5, 8],
        }
    )

    assert grid == [
        {"fast_window": 2, "slow_window": 5},
        {"fast_window": 2, "slow_window": 8},
        {"fast_window": 3, "slow_window": 5},
        {"fast_window": 3, "slow_window": 8},
    ]


def test_split_grid_for_stage_covers_without_duplicates() -> None:
    grid = [{"value": value} for value in range(10)]

    chunks = [split_grid_for_stage(grid, stage=stage, total_stages=3) for stage in range(3)]
    flattened = [item["value"] for chunk in chunks for item in chunk]

    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))


def test_split_grid_for_stage_rejects_invalid_stage() -> None:
    with pytest.raises(ValueError, match="stage must be between"):
        split_grid_for_stage([{"value": 1}], stage=3, total_stages=3)


def test_run_optimization_stage_returns_header_for_empty_stage() -> None:
    leaderboard = run_optimization_stage(
        data=pd.DataFrame(
            {
                "open": [1, 2, 3, 4, 5],
                "high": [1, 2, 3, 4, 5],
                "low": [1, 2, 3, 4, 5],
                "close": [1, 2, 3, 4, 5],
                "volume": [1, 1, 1, 1, 1],
            },
            index=pd.date_range("2026-01-01", periods=5, freq="min"),
        ),
        parameter_space={"fast_window": [2], "slow_window": [4]},
        stage=3,
        total_stages=4,
        initial_cash=1000,
        commission_bps=0,
        slippage_bps=0,
    )

    assert leaderboard.empty
    assert "score" in leaderboard.columns
