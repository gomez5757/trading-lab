from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_annual_public_data import build_annual_public_data  # noqa: E402


def test_build_annual_public_data_adds_valuation_features(monkeypatch, tmp_path: Path) -> None:
    dates = pd.bdate_range("1980-01-01", periods=320, freq="B")

    def fake_yahoo(symbol: str) -> pd.DataFrame:
        offset = 0 if symbol == "^GSPC" else 10
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

    valuation = pd.DataFrame(
        {
            "cape": [10.0, 11.0],
            "earnings_yield": [0.08, 0.07],
            "dividend_yield": [0.04, 0.035],
        },
        index=pd.to_datetime(["1980-01-01", "1981-01-01"]),
    )

    monkeypatch.setattr("download_annual_public_data.download_yahoo_chart", fake_yahoo)
    monkeypatch.setattr("download_annual_public_data._download_fred_series", lambda _series_id: pd.Series(dtype=float))
    monkeypatch.setattr("download_annual_public_data._download_shiller_valuation_data", lambda: valuation)

    path = build_annual_public_data(tmp_path / "annual.csv")

    written = pd.read_csv(path)
    assert "cape" in written.columns
    assert "earnings_yield" in written.columns
    assert "dividend_yield" in written.columns
    assert written["cape"].notna().any()
