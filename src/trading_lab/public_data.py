from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import re
import time
from urllib.parse import quote as url_quote
from urllib.request import Request
from urllib.request import urlopen

import numpy as np
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
    "RSP",
    "DIA",
    "EFA",
    "EEM",
    "EWJ",
    "EWG",
    "EWU",
    "FXI",
    "TLT",
    "IEF",
    "HYG",
    "LQD",
    "GLD",
    "SLV",
    "USO",
    "CPER",
    "DBC",
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
    "XLC",
    "^VIX",
    "^VIX3M",
    "^VVIX",
    "^SKEW",
    "^TNX",
    "^IRX",
    "^FVX",
    "^TYX",
)
FEATURE_WINDOWS = (5, 10, 20, 21, 60, 63, 126, 252)
FRED_FEATURE_SERIES = {
    "nfci": "NFCI",
    "adjusted_nfci": "ANFCI",
    "stlouis_fsi": "STLFSI4",
    "m2": "M2SL",
    "fed_funds": "DFF",
    "sofr": "SOFR",
    "yield_2y": "DGS2",
    "yield_5y": "DGS5",
    "yield_10y": "DGS10",
    "yield_30y": "DGS30",
    "curve_10y_2y": "T10Y2Y",
    "curve_10y_3m": "T10Y3M",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "ccc_oas": "BAMLH0A3HYCEY",
    "breakeven_5y": "T5YIE",
    "breakeven_10y": "T10YIE",
    "unemployment": "UNRATE",
    "consumer_sentiment": "UMCSENT",
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "industrial_production": "INDPRO",
}
CBOE_PUT_CALL_URLS = {
    "total": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
    "index": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
    "equity": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
    "etp": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/etppc.csv",
    "vix": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/vixpc.csv",
    "spx": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/spxpc.csv",
}
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"


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
    close_by_symbol: dict[str, pd.Series] = {"SPY": base_close}

    _add_spy_structure_features(feature_columns, base_frame)
    _add_calendar_features(feature_columns, base_frame.index)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_yahoo_chart, symbol): symbol for symbol in PUBLIC_FEATURE_SYMBOLS}
        downloaded_assets = []
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                downloaded_assets.append((symbol, future.result()))
            except Exception:
                continue

    for symbol, asset in sorted(downloaded_assets, key=lambda item: item[0]):
        asset["timestamp_dt"] = pd.to_datetime(asset["timestamp"])
        asset = asset.set_index("timestamp_dt").reindex(base_frame.index).ffill()
        prefix = _feature_prefix(symbol)
        close = asset["close"].astype(float)
        returns = close.pct_change()
        close_by_symbol[symbol] = close
        feature_columns[f"{prefix}_close_ratio"] = close / base_close
        feature_columns[f"{prefix}_ret_1"] = returns
        for window in FEATURE_WINDOWS:
            feature_columns[f"{prefix}_ret_{window}"] = close.pct_change(window)
            feature_columns[f"{prefix}_vol_{window}"] = returns.rolling(window).std(ddof=0)
            feature_columns[f"spy_vs_{prefix}_ret_{window}"] = base_close.pct_change(window) - close.pct_change(window)

    _add_cross_asset_ratio_features(feature_columns, close_by_symbol)
    _add_fred_features(feature_columns, base_frame.index)
    _add_cboe_put_call_features(feature_columns, base_frame.index)
    _add_epu_features(feature_columns, base_frame.index)

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


def _add_spy_structure_features(out: dict[str, pd.Series], spy: pd.DataFrame) -> None:
    open_ = spy["open"].astype(float)
    high = spy["high"].astype(float)
    low = spy["low"].astype(float)
    close = spy["close"].astype(float)
    volume = spy["volume"].astype(float)
    prev_close = close.shift(1)
    out["range_high_low"] = high / low - 1.0
    out["range_intraday_norm_close"] = (high - low) / close
    out["gap_overnight"] = open_ / prev_close - 1.0
    out["return_intraday_open_close"] = close / open_ - 1.0
    out["return_overnight_close_open"] = open_ / prev_close - 1.0
    out["vol_parkinson_20"] = (np.log(high / low) ** 2 / (4.0 * math.log(2.0))).rolling(20).mean().pow(0.5)
    log_h_o = np.log(high / open_)
    log_l_o = np.log(low / open_)
    log_c_o = np.log(close / open_)
    out["vol_garman_klass_20"] = (
        0.5 * np.log(high / low) ** 2 - (2.0 * math.log(2.0) - 1.0) * log_c_o**2
    ).rolling(20).mean().pow(0.5)
    out["vol_rogers_satchell_20"] = (
        log_h_o * np.log(high / close) + log_l_o * np.log(low / close)
    ).rolling(20).mean().pow(0.5)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["range_expansion_20"] = out["range_high_low"] / out["range_high_low"].rolling(20).mean() - 1.0
    out["intraday_reversal"] = np.sign(out["gap_overnight"]) * -np.sign(out["return_intraday_open_close"])
    out["gap_cluster_5"] = out["gap_overnight"].abs().rolling(5).mean()
    out["opening_range_proxy"] = (high - open_) / close
    out["relative_volume_20"] = volume / volume.rolling(20).mean() - 1.0
    out["dollar_volume"] = close * volume
    out["close_location"] = (close - low) / (high - low).replace(0.0, np.nan)


def _add_calendar_features(out: dict[str, pd.Series], index: pd.DatetimeIndex) -> None:
    idx = pd.DatetimeIndex(index)
    out["calendar_month"] = pd.Series(idx.month, index=idx, dtype=float)
    out["calendar_week"] = pd.Series(idx.isocalendar().week.astype(float).to_numpy(), index=idx)
    out["calendar_day_of_week"] = pd.Series(idx.dayofweek, index=idx, dtype=float)
    out["calendar_turn_of_month"] = pd.Series(((idx.day <= 3) | (idx.day >= 28)).astype(float), index=idx)
    out["calendar_quarter_end"] = pd.Series(idx.is_quarter_end.astype(float), index=idx)
    out["calendar_year_end"] = pd.Series(((idx.month == 12) & idx.is_month_end).astype(float), index=idx)
    out["calendar_january"] = pd.Series((idx.month == 1).astype(float), index=idx)
    out["calendar_sell_in_may"] = pd.Series(idx.month.isin([5, 6, 7, 8, 9, 10]).astype(float), index=idx)
    out["calendar_opex_week_proxy"] = pd.Series(
        ((idx.day >= 15) & (idx.day <= 21) & (idx.dayofweek == 4)).astype(float),
        index=idx,
    )


def _add_cross_asset_ratio_features(out: dict[str, pd.Series], close_by_symbol: dict[str, pd.Series]) -> None:
    pairs = {
        "vix_term": ("^VIX", "^VIX3M"),
        "qqq_spy": ("QQQ", "SPY"),
        "iwm_spy": ("IWM", "SPY"),
        "rsp_spy": ("RSP", "SPY"),
        "efa_spy": ("EFA", "SPY"),
        "eem_spy": ("EEM", "SPY"),
        "japan_spy": ("EWJ", "SPY"),
        "china_spy": ("FXI", "SPY"),
        "copper_gold": ("CPER", "GLD"),
        "oil_gold": ("USO", "GLD"),
        "energy_spy": ("XLE", "SPY"),
        "hyg_lqd": ("HYG", "LQD"),
        "xly_xlp": ("XLY", "XLP"),
        "xlk_xlu": ("XLK", "XLU"),
        "xlf_spy": ("XLF", "SPY"),
    }
    for name, (left, right) in pairs.items():
        if left not in close_by_symbol or right not in close_by_symbol:
            continue
        ratio = close_by_symbol[left] / close_by_symbol[right]
        out[f"{name}_level"] = ratio
        for window in (21, 63, 126):
            out[f"{name}_ret_{window}"] = ratio.pct_change(window)
    if "^VIX" in close_by_symbol and "SPY" in close_by_symbol:
        spy = close_by_symbol["SPY"]
        realized = spy.pct_change().rolling(21).std(ddof=0) * math.sqrt(252.0) * 100.0
        out["vix_minus_realized_vol_21"] = close_by_symbol["^VIX"] - realized
    if "^SKEW" in close_by_symbol:
        out["skew_daily_change"] = close_by_symbol["^SKEW"].diff()


def _add_fred_features(out: dict[str, pd.Series], index: pd.DatetimeIndex) -> None:
    raw: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_download_fred_series, series_id): name
            for name, series_id in FRED_FEATURE_SERIES.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                raw[name] = future.result()
            except Exception:
                continue
    if not raw:
        return
    aligned = _align_frame(pd.DataFrame(raw).shift(5, freq="B"), index)
    for column in aligned.columns:
        _add_level_change_z(out, f"free_fred_{column}", aligned[column])
    _add_spread(out, aligned, "free_fred_hy_minus_ig_oas", "hy_oas", "ig_oas")
    _add_spread(out, aligned, "free_fred_sofr_minus_fed_funds", "sofr", "fed_funds")
    _add_spread(out, aligned, "free_fred_10y_minus_2y", "yield_10y", "yield_2y")
    _add_spread(out, aligned, "free_fred_30y_minus_5y", "yield_30y", "yield_5y")


def _download_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    request = Request(url, headers={"User-Agent": "trading-lab-public-features"})
    with urlopen(request, timeout=20) as response:
        raw = response.read()
    df = pd.read_csv(io.BytesIO(raw))
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise PublicDataError(f"FRED returned unexpected shape for {series_id}")
    index = pd.to_datetime(df["observation_date"], errors="coerce")
    values = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
    return pd.Series(values.to_numpy(dtype=float), index=index, name=series_id).dropna()


def _add_cboe_put_call_features(out: dict[str, pd.Series], index: pd.DatetimeIndex) -> None:
    pieces = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_download_bytes, url, timeout=15): name
            for name, url in CBOE_PUT_CALL_URLS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                pieces.append(_parse_cboe_put_call_csv(future.result(), name))
            except Exception:
                continue
    if not pieces:
        return
    aligned = _align_frame(pd.concat(pieces, axis=1).sort_index().shift(1, freq="B"), index)
    for column in aligned.columns:
        _add_level_change_z(out, f"free_cboe_{column}", aligned[column])


def _parse_cboe_put_call_csv(raw: bytes, name: str) -> pd.DataFrame:
    rows = raw.decode("utf-8-sig", errors="ignore").splitlines()
    header_index = next(
        idx for idx, row in enumerate(rows)
        if re.search(r"(DATE|Trade_date)", row, flags=re.I) and re.search(r"P[/_]?C", row, flags=re.I)
    )
    df = pd.read_csv(io.StringIO("\n".join(rows[header_index:])))
    df.columns = [str(column).strip().lower().replace(" ", "_").replace("/", "_") for column in df.columns]
    date_column = next(column for column in df.columns if "date" in column)
    pc_column = next(column for column in df.columns if "p_c" in column or "p/c" in column)
    output = pd.DataFrame(index=pd.to_datetime(df[date_column], errors="coerce"))
    output[f"{name}_pc_ratio"] = pd.to_numeric(df[pc_column], errors="coerce").to_numpy(dtype=float)
    for label in ("call", "put", "total"):
        column = next((candidate for candidate in df.columns if label in candidate), None)
        if column:
            output[f"{name}_{label}"] = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return output.dropna(how="all")


def _add_epu_features(out: dict[str, pd.Series], index: pd.DatetimeIndex) -> None:
    try:
        df = pd.read_csv(io.BytesIO(_download_bytes(EPU_URL)))
        df["date"] = pd.to_datetime(
            {"year": df["year"], "month": df["month"], "day": df["day"]},
            errors="coerce",
        )
        raw = df.set_index("date")[["daily_policy_index"]].rename(columns={"daily_policy_index": "epu_daily"})
        aligned = _align_frame(raw.shift(1, freq="B"), index)
        _add_level_change_z(out, "free_epu_daily", aligned["epu_daily"])
    except Exception:
        return


def _add_level_change_z(out: dict[str, pd.Series], prefix: str, series: pd.Series) -> None:
    out[prefix] = series
    for window in (5, 21, 63):
        out[f"{prefix}_chg_{window}"] = series.diff(window)
        out[f"{prefix}_z_{window}"] = _rolling_zscore(series, window)


def _add_spread(
    out: dict[str, pd.Series],
    frame: pd.DataFrame,
    name: str,
    left: str,
    right: str,
) -> None:
    if left in frame.columns and right in frame.columns:
        _add_level_change_z(out, name, frame[left] - frame[right])


def _align_frame(frame: pd.DataFrame | pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    aligned = frame.copy()
    aligned.index = pd.to_datetime(aligned.index).tz_localize(None)
    aligned = aligned.sort_index()
    if aligned.index.has_duplicates:
        aligned = aligned.groupby(level=0).mean(numeric_only=True)
    return aligned.reindex(index).ffill()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _download_bytes(url: str, *, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()
