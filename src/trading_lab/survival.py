from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.backtest import run_backtest

TRAIN_END = "2013-10-18"
VALIDATION_START = "2013-10-21"
VALIDATION_END = "2019-12-31"


@dataclass(frozen=True)
class SurvivalCriteria:
    min_train_calmar: float = 1.25
    min_validation_calmar: float = 1.25
    max_train_calmar: float = 8.0
    max_train_validation_ratio: float = 2.5
    min_train_cagr: float = 0.06
    min_validation_cagr: float = 0.06
    max_train_mdd: float = 0.30
    max_validation_mdd: float = 0.30
    min_trades_per_year: float = 12.0
    max_trades_per_year: float = 90.0
    min_long_fraction: float = 0.25
    max_long_fraction: float = 0.75
    max_validation_negative_years: int = 0

    def rejection_reason(self, row: dict[str, Any]) -> str | None:
        train_calmar = float(row["train_calmar"])
        validation_calmar = float(row["validation_calmar"])
        if train_calmar < self.min_train_calmar:
            return "train_calmar"
        if validation_calmar < self.min_validation_calmar:
            return "validation_calmar"
        if train_calmar > self.max_train_calmar:
            return "train_calmar_too_high"
        if validation_calmar == 0 or train_calmar / validation_calmar > self.max_train_validation_ratio:
            return "train_validation_gap"
        if float(row["train_cagr"]) < self.min_train_cagr:
            return "train_cagr"
        if float(row["validation_cagr"]) < self.min_validation_cagr:
            return "validation_cagr"
        if abs(float(row["train_mdd"])) > self.max_train_mdd:
            return "train_mdd"
        if abs(float(row["validation_mdd"])) > self.max_validation_mdd:
            return "validation_mdd"
        trades_per_year = float(row["trades_per_year"])
        if trades_per_year < self.min_trades_per_year:
            return "too_few_trades"
        if trades_per_year > self.max_trades_per_year:
            return "too_many_trades"
        long_fraction = float(row["long_fraction"])
        if long_fraction < self.min_long_fraction:
            return "too_little_long"
        if long_fraction > self.max_long_fraction:
            return "too_much_long"
        if int(row["validation_negative_years"]) > self.max_validation_negative_years:
            return "validation_negative_years"
        return None


def split_train_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data.index <= pd.Timestamp(TRAIN_END)]
    validation = data.loc[
        (data.index >= pd.Timestamp(VALIDATION_START))
        & (data.index <= pd.Timestamp(VALIDATION_END))
    ]
    return train.copy(), validation.copy()


def build_survival_grid(
    parameter_space: dict[str, list[Any]],
    *,
    stage: int,
    total_stages: int,
) -> list[dict[str, Any]]:
    names = list(parameter_space)
    values = [parameter_space[name] for name in names]
    base = [
        {"rule": "ma_crossover", **dict(zip(names, combination, strict=True))}
        for combination in product(*values)
    ]
    grid = [candidate for candidate in base if _valid_candidate(candidate)]
    return [params for index, params in enumerate(grid) if index % total_stages == stage]


def evaluate_survival_candidate(
    data: pd.DataFrame,
    params: dict[str, Any],
    *,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
    criteria: SurvivalCriteria | None = None,
) -> dict[str, Any]:
    criteria = criteria or SurvivalCriteria()
    train, validation = split_train_validation(data)
    train_result = _run_candidate(train, params, initial_cash, commission_bps, slippage_bps)
    validation_result = _run_candidate(validation, params, initial_cash, commission_bps, slippage_bps)
    train_metrics = _survival_metrics(train_result.equity_curve, train_result.metrics, train)
    validation_metrics = _survival_metrics(validation_result.equity_curve, validation_result.metrics, validation)
    row = {
        "candidate_id": _candidate_id(params),
        **params,
        "feature_count": _feature_count(params),
        "train_calmar": train_metrics["calmar"],
        "validation_calmar": validation_metrics["calmar"],
        "train_cagr": train_metrics["cagr"],
        "validation_cagr": validation_metrics["cagr"],
        "train_mdd": train_metrics["mdd"],
        "validation_mdd": validation_metrics["mdd"],
        "trades_per_year": train_metrics["trades_per_year"],
        "long_fraction": train_metrics["long_fraction"],
        "validation_negative_years": _negative_year_count(validation_result.equity_curve),
        "locked_opened": False,
    }
    row["survival_score"] = survival_score(row)
    row["rejection_reason"] = criteria.rejection_reason(row)
    row["accepted"] = row["rejection_reason"] is None
    return row


def survival_score(row: dict[str, Any]) -> float:
    validation_calmar = float(row.get("validation_calmar", 0.0) or 0.0)
    train_calmar = float(row.get("train_calmar", 0.0) or 0.0)
    gap = max(0.0, train_calmar - 2.0 * max(validation_calmar, 0.0))
    complexity = max(0, int(row.get("feature_count", 1)) - 2) * 0.15
    return float(validation_calmar - 0.35 * gap - complexity)


def _run_candidate(
    data: pd.DataFrame,
    params: dict[str, Any],
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
):
    if params.get("rule") == "ma_crossover":
        return run_backtest(
            data,
            params=params,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )
    signals = _momentum_signals(
        data,
        window=int(params["momentum_window"]),
        threshold=float(params["threshold"]),
    )
    return _run_signals_backtest(
        data,
        signals=signals,
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )


def _momentum_signals(data: pd.DataFrame, *, window: int, threshold: float) -> pd.Series:
    momentum = data["close"].pct_change(window)
    signal = (momentum > threshold).astype(int)
    signal[momentum.isna()] = 0
    return signal


def _run_signals_backtest(
    data: pd.DataFrame,
    *,
    signals: pd.Series,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
):
    from trading_lab.backtest import BacktestResult, calculate_metrics

    desired_position = signals.shift(1).fillna(0).astype(int)
    returns = data["close"].pct_change().fillna(0.0)
    position = desired_position.reindex(data.index).fillna(0).astype(float)
    cost_rate = (commission_bps + slippage_bps) / 10_000
    turnover = position.diff().abs().fillna(position.abs())
    strategy_returns = position * returns - turnover * cost_rate
    equity = initial_cash * (1.0 + strategy_returns).cumprod()
    equity_curve = pd.DataFrame({"timestamp": data.index, "equity": equity.to_numpy()})
    trades = _trades_from_position(data, position, initial_cash)
    metrics = calculate_metrics(equity_curve, trades, initial_cash)
    return BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve)


def _survival_metrics(
    equity_curve: pd.DataFrame,
    metrics: dict[str, float],
    data: pd.DataFrame,
) -> dict[str, float]:
    years = max(len(data) / 252.0, 1 / 252.0)
    final_equity = float(metrics["final_equity"])
    initial_cash = float(metrics["initial_cash"])
    cagr = (final_equity / initial_cash) ** (1.0 / years) - 1.0
    mdd = float(metrics["max_drawdown_pct"]) / 100.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    trades_per_year = float(metrics["trade_count"]) / years
    long_fraction = _long_fraction(equity_curve)
    return {
        "calmar": float(calmar),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "trades_per_year": float(trades_per_year),
        "long_fraction": float(long_fraction),
    }


def _long_fraction(equity_curve: pd.DataFrame) -> float:
    equity = equity_curve["equity"].astype(float)
    return float((equity.pct_change().fillna(0) != 0).mean())


def _negative_year_count(equity_curve: pd.DataFrame) -> int:
    frame = equity_curve.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["year"] = frame["timestamp"].dt.year
    negatives = 0
    for _, year_frame in frame.groupby("year"):
        first = float(year_frame.iloc[0]["equity"])
        last = float(year_frame.iloc[-1]["equity"])
        if last < first:
            negatives += 1
    return int(negatives)


def _trades_from_position(
    data: pd.DataFrame,
    position: pd.Series,
    initial_cash: float,
) -> pd.DataFrame:
    rows = []
    in_trade = False
    entry_time = None
    entry_price = 0.0
    current_equity = initial_cash
    for timestamp, target in position.items():
        close = float(data.loc[timestamp, "close"])
        if target == 1 and not in_trade:
            in_trade = True
            entry_time = timestamp
            entry_price = close
        elif target == 0 and in_trade:
            return_pct = close / entry_price - 1
            current_equity *= 1 + return_pct
            rows.append(
                {
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl": current_equity - initial_cash,
                    "return_pct": return_pct * 100,
                    "exit_equity": current_equity,
                }
            )
            in_trade = False
    return pd.DataFrame(
        rows,
        columns=["entry_time", "exit_time", "entry_price", "exit_price", "pnl", "return_pct", "exit_equity"],
    )


def _candidate_id(params: dict[str, Any]) -> str:
    clean = "_".join(f"{key}-{value}" for key, value in sorted(params.items()))
    return clean.replace(".", "p").replace("-", "_")


def _feature_count(params: dict[str, Any]) -> int:
    if params.get("rule") == "ma_crossover":
        return 2
    return 1


def _valid_candidate(params: dict[str, Any]) -> bool:
    if params.get("rule") == "ma_crossover":
        return int(params["fast_window"]) < int(params["slow_window"])
    return True
