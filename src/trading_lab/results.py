from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import EmptyDataError


def save_backtest_outputs(
    output_dir: str | Path,
    *,
    metrics: dict[str, float],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    with (path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True, default=str)

    trades.to_csv(path / "trades.csv", index=False)
    equity_curve.to_csv(path / "equity_curve.csv", index=False)
    return path


def merge_leaderboards(paths: Iterable[str | Path], *, score_column: str = "score") -> pd.DataFrame:
    frames = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        try:
            frame = pd.read_csv(file_path)
        except EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    if score_column not in merged.columns:
        raise ValueError(f"score column not found: {score_column}")

    return merged.sort_values(score_column, ascending=False).reset_index(drop=True)


def save_leaderboard(leaderboard: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(path, index=False)
    return path
