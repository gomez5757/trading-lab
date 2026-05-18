from pathlib import Path

import pandas as pd

from trading_lab.results import merge_leaderboards, save_backtest_outputs


def test_save_backtest_outputs_writes_metrics_and_tables(tmp_path: Path) -> None:
    metrics = {"total_return_pct": 12.5, "trade_count": 2}
    trades = pd.DataFrame([{"entry_time": "a", "exit_time": "b", "pnl": 10.0}])
    equity = pd.DataFrame([{"timestamp": "a", "equity": 10000.0}])

    save_backtest_outputs(tmp_path, metrics=metrics, trades=trades, equity_curve=equity)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()


def test_merge_leaderboards_sorts_by_score_descending(tmp_path: Path) -> None:
    first = tmp_path / "stage_0.csv"
    second = tmp_path / "stage_1.csv"
    pd.DataFrame([{"fast_window": 2, "score": 3.0}]).to_csv(first, index=False)
    pd.DataFrame([{"fast_window": 3, "score": 8.0}]).to_csv(second, index=False)

    merged = merge_leaderboards([first, second])

    assert merged["score"].tolist() == [8.0, 3.0]
    assert merged.iloc[0]["fast_window"] == 3
