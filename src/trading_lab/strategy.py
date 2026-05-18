from __future__ import annotations

import pandas as pd

from trading_lab.indicators import simple_moving_average


def moving_average_crossover_signals(
    data: pd.DataFrame,
    *,
    fast_window: int,
    slow_window: int,
) -> pd.Series:
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("moving-average windows must be greater than zero")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    fast = simple_moving_average(data["close"], fast_window)
    slow = simple_moving_average(data["close"], slow_window)
    signal = (fast > slow).astype(int)
    signal[(fast.isna()) | (slow.isna())] = 0
    signal.name = "signal"
    return signal
