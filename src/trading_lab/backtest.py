from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategy import moving_average_crossover_signals


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def run_backtest(
    data: pd.DataFrame,
    *,
    params: dict[str, Any],
    initial_cash: float = 10_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    if data.empty:
        raise ValueError("data cannot be empty")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be greater than zero")

    signals = moving_average_crossover_signals(
        data,
        fast_window=int(params["fast_window"]),
        slow_window=int(params["slow_window"]),
    )
    desired_position = signals.shift(1).fillna(0).astype(int)

    cash = float(initial_cash)
    units = 0.0
    entry_price = 0.0
    entry_time = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    cost_rate = (commission_bps + slippage_bps) / 10_000

    for timestamp, row in data.iterrows():
        close = float(row["close"])
        target = int(desired_position.loc[timestamp])

        if target == 1 and units == 0:
            entry_price = close * (1 + cost_rate)
            units = cash / entry_price
            cash = 0.0
            entry_time = timestamp
        elif target == 0 and units > 0:
            exit_price = close * (1 - cost_rate)
            cash = units * exit_price
            pnl = cash - initial_cash if not trades else cash - trades[-1]["exit_equity"]
            trade_return = (exit_price / entry_price - 1) * 100
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": trade_return,
                    "exit_equity": cash,
                }
            )
            units = 0.0
            entry_price = 0.0
            entry_time = None

        equity = cash + units * close
        equity_rows.append({"timestamp": timestamp, "equity": equity})

    if units > 0:
        timestamp = data.index[-1]
        close = float(data.iloc[-1]["close"])
        exit_price = close * (1 - cost_rate)
        cash = units * exit_price
        pnl = cash - initial_cash if not trades else cash - trades[-1]["exit_equity"]
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "return_pct": (exit_price / entry_price - 1) * 100,
                "exit_equity": cash,
            }
        )
        equity_rows[-1]["equity"] = cash

    equity_curve = pd.DataFrame(equity_rows)
    trades_frame = pd.DataFrame(
        trades,
        columns=[
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
            "exit_equity",
        ],
    )
    metrics = calculate_metrics(equity_curve, trades_frame, initial_cash)
    return BacktestResult(metrics=metrics, trades=trades_frame, equity_curve=equity_curve)


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, float]:
    final_equity = float(equity_curve.iloc[-1]["equity"])
    total_return_pct = (final_equity / initial_cash - 1) * 100

    equity = equity_curve["equity"].astype(float)
    drawdown = equity / equity.cummax() - 1
    max_drawdown_pct = float(drawdown.min() * 100)

    returns = equity.pct_change().dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * sqrt(252))

    if trades.empty:
        win_rate_pct = 0.0
        profit_factor = 0.0
        trade_count = 0.0
    else:
        pnl = trades["pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate_pct = float((len(wins) / len(pnl)) * 100)
        if losses.empty:
            profit_factor = float(wins.sum()) if not wins.empty else 0.0
        else:
            profit_factor = float(wins.sum() / abs(losses.sum()))
        trade_count = float(len(trades))

    return {
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": max_drawdown_pct,
        "trade_count": trade_count,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "score": float(total_return_pct + max_drawdown_pct + np.nan_to_num(sharpe)),
    }
