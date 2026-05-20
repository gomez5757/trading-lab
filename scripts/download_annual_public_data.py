from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.public_data import _download_fred_series, download_yahoo_chart, write_public_data  # noqa: E402


FRED_ANNUAL_SERIES = {
    "fed_funds": "DFF",
    "yield_2y": "DGS2",
    "yield_10y": "DGS10",
    "yield_30y": "DGS30",
    "curve_10y_2y": "T10Y2Y",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "industrial_production": "INDPRO",
    "financial_stress": "STLFSI4",
}

YAHOO_FEATURE_SYMBOLS = {
    "vix": "^VIX",
    "russell_2000": "^RUT",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public long-history daily data for annual SP500 search.")
    parser.add_argument("--output", default="data/public/sp500_annual_daily.csv")
    args = parser.parse_args()
    path = build_annual_public_data(args.output)
    print(path)
    return 0


def build_annual_public_data(output: str | Path) -> Path:
    base = download_yahoo_chart("^GSPC")
    base = base.rename(columns={"close": "sp500_close"})
    base["timestamp_dt"] = pd.to_datetime(base["timestamp"])
    base = base.set_index("timestamp_dt")
    close = base["sp500_close"].astype(float)
    panel = pd.DataFrame(
        {
            "timestamp": base["timestamp"],
            "open": base["open"],
            "high": base["high"],
            "low": base["low"],
            "close": close,
            "volume": base["volume"],
        },
        index=base.index,
    )
    for name, symbol in YAHOO_FEATURE_SYMBOLS.items():
        try:
            asset = download_yahoo_chart(symbol)
        except Exception:
            continue
        asset["timestamp_dt"] = pd.to_datetime(asset["timestamp"])
        aligned = asset.set_index("timestamp_dt").reindex(panel.index).ffill()
        series = aligned["close"].astype(float)
        panel[f"{name}_level"] = series
        panel[f"{name}_ret_12m"] = series.pct_change(252)
        panel[f"sp500_vs_{name}_ret_12m"] = close.pct_change(252) - series.pct_change(252)
    for name, series_id in FRED_ANNUAL_SERIES.items():
        try:
            series = _download_fred_series(series_id).shift(5, freq="B")
        except Exception:
            continue
        panel[name] = series.reindex(panel.index).ffill()
    if {"hy_oas", "ig_oas"}.issubset(panel.columns):
        panel["hy_minus_ig_oas"] = panel["hy_oas"] - panel["ig_oas"]
    if {"yield_10y", "yield_2y"}.issubset(panel.columns):
        panel["yield_10y_minus_2y"] = panel["yield_10y"] - panel["yield_2y"]
    panel = panel.reset_index(drop=True)
    return write_public_data(panel, output)


if __name__ == "__main__":
    raise SystemExit(main())
