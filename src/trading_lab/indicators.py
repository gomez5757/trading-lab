from __future__ import annotations

import pandas as pd


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be greater than zero")
    return series.rolling(window=window, min_periods=window).mean()
