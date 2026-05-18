from __future__ import annotations

from pathlib import Path
import json
import time
from urllib.parse import quote as url_quote
from urllib.request import Request
from urllib.request import urlopen

import pandas as pd

from trading_lab.data_loader import normalize_market_data

STOOQ_SPY_DAILY_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"
YAHOO_SPY_DAILY_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/SPY"
    "?period1=0&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
)
PUBLIC_DATA_SYMBOL = "SPY"
PUBLIC_FEATURE_SYMBOLS = (
    "QQQ",
    "IWM",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "UUP",
    "SHY",
    "IEI",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "XLU",
    "XLI",
    "XLB",
    "XLRE",
    "^VIX",
)
FEATURE_WINDOWS = (5, 20, 60)


class PublicDataError(ValueError):
    """Raised when public market data cannot be downloaded or normalized."""


def download_stooq_csv(url: str = STOOQ_SPY_DAILY_URL) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    if "apikey" in text.lower() or not text.lstrip().lower().startswith("date,"):
        raise PublicDataError("Stooq did not return public CSV data")
    return pd.read_csv(__import__("io").StringIO(text))


def download_yahoo_chart(symbol: str = "SPY") -> pd.DataFrame:
    url = YAHOO_SPY_DAILY_URL.format(period2=int(time.time()))
    if symbol != "SPY":
        url = url.replace("/SPY?", f"/{url_quote(symbol, safe='')}?")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    error = payload.get("chart", {}).get("error")
    if error:
        raise PublicDataError(f"Yahoo chart error: {error}")
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise PublicDataError("Yahoo chart returned no data")
    data = result[0]
    timestamps = data.get("timestamp") or []
    quote = (data.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (data.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s").strftime("%Y-%m-%d"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adjclose": adjclose,
            "volume": quote.get("volume"),
        }
    ).dropna()
    if raw.empty:
        raise PublicDataError("Yahoo chart returned an empty CSV")
    adjustment = raw["adjclose"] / raw["close"]
    raw["open"] = raw["open"] * adjustment
    raw["high"] = raw["high"] * adjustment
    raw["low"] = raw["low"] * adjustment
    raw["close"] = raw["adjclose"]
    raw = raw.drop(columns=["adjclose"])
    checked = normalize_market_data(raw)
    return checked.reset_index().assign(
        timestamp=lambda frame: frame["timestamp"].dt.strftime("%Y-%m-%d")
    )


def normalize_stooq_csv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise PublicDataError("public data source returned an empty CSV")

    normalized_names = {column.lower(): column for column in raw.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [name for name in required if name not in normalized_names]
    if missing:
        raise PublicDataError(f"public data missing columns: {', '.join(missing)}")

    data = pd.DataFrame(
        {
            "timestamp": raw[normalized_names["date"]].astype(str),
            "open": raw[normalized_names["open"]],
            "high": raw[normalized_names["high"]],
            "low": raw[normalized_names["low"]],
            "close": raw[normalized_names["close"]],
            "volume": raw[normalized_names["volume"]],
        }
    )
    checked = normalize_market_data(data)
    return checked.reset_index().assign(
        timestamp=lambda frame: frame["timestamp"].dt.strftime("%Y-%m-%d")
    )


def write_public_data(data: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def download_public_data(output_path: str | Path, url: str = STOOQ_SPY_DAILY_URL) -> Path:
    try:
        data = normalize_stooq_csv(download_stooq_csv(url))
    except Exception:
        data = download_yahoo_chart(PUBLIC_DATA_SYMBOL)
    return write_public_data(data, output_path)


def download_public_feature_data(output_path: str | Path) -> Path:
    base = download_yahoo_chart(PUBLIC_DATA_SYMBOL)
    base_frame = base.copy()
    base_frame["timestamp_dt"] = pd.to_datetime(base_frame["timestamp"])
    base_frame = base_frame.set_index("timestamp_dt")
    base_close = base_frame["close"].astype(float)
    base_return = base_close.pct_change()
    feature_columns: dict[str, pd.Series] = {}

    for symbol in PUBLIC_FEATURE_SYMBOLS:
        try:
            asset = download_yahoo_chart(symbol)
        except Exception:
            continue
        asset["timestamp_dt"] = pd.to_datetime(asset["timestamp"])
        asset = asset.set_index("timestamp_dt").reindex(base_frame.index).ffill()
        prefix = _feature_prefix(symbol)
        close = asset["close"].astype(float)
        returns = close.pct_change()
        feature_columns[f"{prefix}_close_ratio"] = close / base_close
        feature_columns[f"{prefix}_ret_1"] = returns
        for window in FEATURE_WINDOWS:
            feature_columns[f"{prefix}_ret_{window}"] = close.pct_change(window)
            feature_columns[f"{prefix}_vol_{window}"] = returns.rolling(window).std(ddof=0)
            feature_columns[f"spy_vs_{prefix}_ret_{window}"] = base_close.pct_change(window) - close.pct_change(window)

    for window in FEATURE_WINDOWS:
        feature_columns[f"spy_ret_{window}"] = base_close.pct_change(window)
        feature_columns[f"spy_vol_{window}"] = base_return.rolling(window).std(ddof=0)
        feature_columns[f"spy_drawdown_{window}"] = base_close / base_close.rolling(window).max() - 1.0

    features = pd.DataFrame(feature_columns, index=base_frame.index)
    panel = pd.concat([base_frame.drop(columns=["timestamp"]), features], axis=1).reset_index(drop=True)
    panel.insert(0, "timestamp", pd.to_datetime(base["timestamp"]).dt.strftime("%Y-%m-%d"))
    panel = panel.replace([float("inf"), float("-inf")], pd.NA).reset_index(drop=True)
    if panel.empty:
        raise PublicDataError("public feature panel is empty")
    return write_public_data(panel, output_path)


def _feature_prefix(symbol: str) -> str:
    return (
        symbol.lower()
        .replace("^", "")
        .replace("-", "_")
        .replace(".", "_")
    )
