from __future__ import annotations

from pathlib import Path
import json
import time
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
        url = url.replace("/SPY?", f"/{symbol}?")
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
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s").strftime("%Y-%m-%d"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    ).dropna()
    if raw.empty:
        raise PublicDataError("Yahoo chart returned an empty CSV")
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
