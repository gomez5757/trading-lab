from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


class DataValidationError(ValueError):
    """Raised when market data cannot be safely used for a backtest."""


def load_market_data(source: str | Path) -> pd.DataFrame:
    """Load and validate OHLCV market data from a local CSV path."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    raw = pd.read_csv(path)
    return normalize_market_data(raw)


def load_market_data_from_api(url: str, api_key: str | None = None) -> pd.DataFrame:
    """Load OHLCV CSV data from an HTTP endpoint."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        raw = pd.read_csv(response)
    return normalize_market_data(raw)


def normalize_market_data(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")

    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=False, errors="coerce")
    if data["timestamp"].isna().any():
        raise DataValidationError("Invalid timestamps found")

    if data["timestamp"].duplicated().any():
        raise DataValidationError("Duplicate timestamps found")

    for column in PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.loc[:, PRICE_COLUMNS].isna().any().any():
        raise DataValidationError("Non numeric OHLCV values found")

    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("OHLC prices must be greater than zero")

    if (data["volume"] < 0).any():
        raise DataValidationError("Volume cannot be negative")

    extra_columns = [column for column in data.columns if column not in REQUIRED_COLUMNS]
    for column in extra_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values("timestamp").set_index("timestamp")
    data.index.name = "timestamp"
    ordered_columns = [*PRICE_COLUMNS, *extra_columns]
    return data.loc[:, ordered_columns]
