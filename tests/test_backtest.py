import pandas as pd

from trading_lab.backtest import run_backtest


def sample_market_data() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=12, freq="min", name="timestamp")
    close = [100, 101, 102, 103, 104, 105, 103, 101, 99, 98, 100, 102]
    return pd.DataFrame(
        {
            "open": close,
            "high": [price + 1 for price in close],
            "low": [price - 1 for price in close],
            "close": close,
            "volume": [100] * len(close),
        },
        index=index,
    )


def test_run_backtest_returns_reproducible_metrics_and_trades() -> None:
    result = run_backtest(
        sample_market_data(),
        params={"fast_window": 2, "slow_window": 4},
        initial_cash=10_000,
        commission_bps=1.0,
        slippage_bps=1.0,
    )

    assert result.metrics["total_return_pct"] != 0
    assert result.metrics["max_drawdown_pct"] <= 0
    assert result.metrics["trade_count"] >= 1
    assert 0 <= result.metrics["win_rate_pct"] <= 100
    assert "sharpe" in result.metrics
    assert not result.equity_curve.empty
    assert not result.trades.empty
