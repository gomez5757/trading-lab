from pathlib import Path

import pandas as pd
import pytest

from trading_lab.data_loader import DataValidationError, load_market_data


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def valid_rows() -> list[dict[str, object]]:
    return [
        {
            "timestamp": "2026-01-01 00:00:00",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10,
        },
        {
            "timestamp": "2026-01-01 00:01:00",
            "open": 101.0,
            "high": 103.0,
            "low": 100.0,
            "close": 102.0,
            "volume": 12,
        },
    ]


def test_load_market_data_normalizes_valid_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "prices.csv"
    write_csv(csv_path, valid_rows())

    data = load_market_data(csv_path)

    assert list(data.columns) == ["open", "high", "low", "close", "volume"]
    assert data.index.name == "timestamp"
    assert data.index.is_monotonic_increasing
    assert data["close"].tolist() == [101.0, 102.0]


def test_load_market_data_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "broken.csv"
    rows = valid_rows()
    for row in rows:
        del row["volume"]
    write_csv(csv_path, rows)

    with pytest.raises(DataValidationError, match="Missing required columns: volume"):
        load_market_data(csv_path)


def test_load_market_data_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicates.csv"
    rows = valid_rows()
    rows[1]["timestamp"] = rows[0]["timestamp"]
    write_csv(csv_path, rows)

    with pytest.raises(DataValidationError, match="Duplicate timestamps"):
        load_market_data(csv_path)
