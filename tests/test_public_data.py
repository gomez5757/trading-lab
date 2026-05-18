from pathlib import Path

import pandas as pd
import pytest

from trading_lab.public_data import (
    PublicDataError,
    download_public_feature_data,
    download_stooq_csv,
    download_yahoo_chart,
    normalize_stooq_csv,
    write_public_data,
)


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


def test_download_yahoo_chart_uses_adjusted_close(monkeypatch) -> None:
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [1577923200],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0],
                                "high": [110.0],
                                "low": [90.0],
                                "close": [100.0],
                                "volume": [12345],
                            }
                        ],
                        "adjclose": [{"adjclose": [50.0]}],
                    },
                }
            ],
        }
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            import json

            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr("trading_lab.public_data.urlopen", lambda *_args, **_kwargs: FakeResponse())

    data = download_yahoo_chart("SPY")

    assert data.loc[0, "close"] == 50.0
    assert data.loc[0, "open"] == 50.0
    assert data.loc[0, "high"] == 55.0
    assert data.loc[0, "low"] == 45.0


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


def test_download_public_feature_data_writes_cross_asset_features(monkeypatch, tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=320, freq="B")

    def fake_download(symbol: str = "SPY") -> pd.DataFrame:
        offset = 0 if symbol == "SPY" else 10
        close = [100 + offset + index for index in range(len(dates))]
        return pd.DataFrame(
            {
                "timestamp": dates.strftime("%Y-%m-%d"),
                "open": close,
                "high": [price + 1 for price in close],
                "low": [price - 1 for price in close],
                "close": close,
                "volume": [1000] * len(dates),
            }
        )

    monkeypatch.setattr("trading_lab.public_data.download_yahoo_chart", fake_download)
    monkeypatch.setattr("trading_lab.public_data.PUBLIC_FEATURE_SYMBOLS", ("QQQ",))

    path = download_public_feature_data(tmp_path / "spy_features.csv")

    written = pd.read_csv(path)
    assert "qqq_ret_20" in written.columns
    assert "spy_vs_qqq_ret_60" in written.columns
