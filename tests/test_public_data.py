from pathlib import Path

import pandas as pd
import pytest

from trading_lab.public_data import PublicDataError, download_stooq_csv, normalize_stooq_csv, write_public_data


def test_normalize_stooq_csv_writes_expected_ohlcv_columns() -> None:
    raw = pd.DataFrame(
        [
            {
                "Date": "2020-01-02",
                "Open": 100.0,
                "High": 102.0,
                "Low": 99.0,
                "Close": 101.0,
                "Volume": 12345,
            }
        ]
    )

    data = normalize_stooq_csv(raw)

    assert list(data.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert data.loc[0, "timestamp"] == "2020-01-02"
    assert data.loc[0, "close"] == 101.0


def test_normalize_stooq_csv_rejects_empty_source() -> None:
    with pytest.raises(PublicDataError, match="empty"):
        normalize_stooq_csv(pd.DataFrame())


def test_download_stooq_csv_rejects_apikey_page(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b"Get your apikey"

    monkeypatch.setattr("trading_lab.public_data.urlopen", lambda *_args, **_kwargs: FakeResponse())

    with pytest.raises(PublicDataError, match="Stooq"):
        download_stooq_csv()


def test_write_public_data_creates_parent_and_csv(tmp_path: Path) -> None:
    path = tmp_path / "public" / "spy_daily.csv"
    data = pd.DataFrame(
        [
            {
                "timestamp": "2020-01-02",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 12345,
            }
        ]
    )

    write_public_data(data, path)

    assert path.exists()
    written = pd.read_csv(path)
    assert written["timestamp"].tolist() == ["2020-01-02"]
