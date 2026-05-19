from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_optimization_config
from trading_lab.data_loader import load_market_data
from trading_lab.survival import (
    VALIDATION_END,
    VALIDATION_START,
    SurvivalCriteria,
    encode_feature_spec,
    evaluate_survival_candidate,
    public_feature_columns,
    split_train_validation,
    survival_score,
)
from trading_lab.survival import _run_candidate, _survival_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one SPY-only always-invested survival stage.")
    parser.add_argument("--config", default="configs/survival_spy_only_github.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int)
    parser.add_argument("--candidates-per-stage", type=int, default=2000)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    total_stages = args.total_stages or config.total_stages
    data = load_market_data(config.data_path)
    seeds = _build_feature_seeds(data)
    grid = list(
        _stage_candidates(
            seeds,
            stage=args.stage,
            total_stages=total_stages,
            candidates_per_stage=args.candidates_per_stage,
        )
    )
    criteria = SurvivalCriteria()
    rows = [
        _evaluate_walkforward_candidate(
            data,
            params,
            criteria=criteria,
            initial_cash=config.execution.initial_cash,
            commission_bps=config.execution.commission_bps,
            slippage_bps=config.execution.slippage_bps,
        )
        for params in grid
    ]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"survival_spy_only_stage_{args.stage}.csv"
    pd.DataFrame(rows).sort_values("survival_score", ascending=False).to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "rows": len(rows),
                "features": len(public_feature_columns(data)),
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


def _build_feature_seeds(data: pd.DataFrame) -> list[str]:
    specs: list[str] = []
    for name in public_feature_columns(data):
        series = pd.to_numeric(data[name], errors="coerce")
        if series.notna().sum() < 500 or series.nunique(dropna=True) < 20:
            continue
        specs.extend(
            [
                encode_feature_spec(name=name, kind="threshold", value=0.0, direction=1),
                encode_feature_spec(name=name, kind="threshold", value=0.0, direction=-1),
                encode_feature_spec(name=name, kind="zscore", value=0.0, window=60, direction=1),
                encode_feature_spec(name=name, kind="zscore", value=0.0, window=60, direction=-1),
                encode_feature_spec(name=name, kind="zscore", value=0.5, window=120, direction=1),
                encode_feature_spec(name=name, kind="zscore", value=0.5, window=120, direction=-1),
                encode_feature_spec(name=name, kind="zscore", value=-0.5, window=120, direction=1),
                encode_feature_spec(name=name, kind="zscore", value=-0.5, window=120, direction=-1),
            ]
        )
    if len(specs) < 4:
        raise ValueError("SPY-only search needs at least four usable feature specs")
    return specs


def _stage_candidates(
    specs: list[str],
    *,
    stage: int,
    total_stages: int,
    candidates_per_stage: int,
) -> Iterable[dict[str, object]]:
    rng = np.random.default_rng(10_000 + stage)
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(seen) < candidates_per_stage and attempts < candidates_per_stage * 20:
        attempts += 1
        combo_size = 4 if attempts % 3 else 5
        indexes = rng.choice(len(specs), size=combo_size, replace=False)
        chosen = [specs[int(index)] for index in indexes]
        long_count = combo_size // 2
        long_specs = tuple(sorted(chosen[:long_count]))
        short_specs = tuple(sorted(chosen[long_count:]))
        key = (*long_specs, "|SHORT|", *short_specs, str(attempts % 4))
        if key in seen:
            continue
        seen.add(key)
        yield {
            "rule": "spy_long_short_always",
            "long_specs": ";".join(long_specs),
            "short_specs": ";".join(short_specs),
            "long_min_votes": 1 if long_count == 2 else 2,
            "short_min_votes": 1 if combo_size - long_count == 2 else 2,
            "combo_size": combo_size,
        }


def _evaluate_walkforward_candidate(
    data: pd.DataFrame,
    params: dict[str, object],
    *,
    criteria: SurvivalCriteria,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
) -> dict[str, object]:
    row = evaluate_survival_candidate(
        data,
        params,
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        criteria=criteria,
    )
    train, validation = split_train_validation(data)
    train_blocks = _block_metrics(train, params, initial_cash, commission_bps, slippage_bps, blocks=6)
    validation_blocks = _block_metrics(validation, params, initial_cash, commission_bps, slippage_bps, blocks=6)
    row.update(
        {
            "train_block_min_calmar": _min_value(train_blocks, "calmar"),
            "validation_block_min_calmar": _min_value(validation_blocks, "calmar"),
            "train_blocks_positive": _count_at_least(train_blocks, "calmar", 0.0),
            "validation_blocks_positive": _count_at_least(validation_blocks, "calmar", 0.0),
            "walkforward_passes": _walkforward_passes(train_blocks, validation_blocks),
            "walkforward_total": 4,
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
        }
    )
    walkforward_rejection = _walkforward_rejection(row)
    if row["accepted"] and walkforward_rejection is not None:
        row["accepted"] = False
        row["rejection_reason"] = walkforward_rejection
    row["survival_score"] = survival_score(row) + float(row["walkforward_passes"]) * 3.0
    return row


def _block_metrics(
    data: pd.DataFrame,
    params: dict[str, object],
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
    *,
    blocks: int,
) -> list[dict[str, float]]:
    chunks = [data.iloc[index_chunk].copy() for index_chunk in np.array_split(np.arange(len(data)), blocks) if len(index_chunk) > 0]
    metrics = []
    for chunk in chunks:
        result = _run_candidate(chunk, params, initial_cash, commission_bps, slippage_bps)
        metrics.append(_survival_metrics(result.equity_curve, result.metrics, chunk))
    return metrics


def _min_value(metrics: list[dict[str, float]], key: str) -> float:
    return float(min((metric[key] for metric in metrics), default=0.0))


def _count_at_least(metrics: list[dict[str, float]], key: str, threshold: float) -> int:
    return int(sum(1 for metric in metrics if metric[key] >= threshold))


def _walkforward_passes(train_blocks: list[dict[str, float]], validation_blocks: list[dict[str, float]]) -> int:
    return sum(
        [
            _min_value(train_blocks, "calmar") >= 0.50,
            _min_value(validation_blocks, "calmar") >= 0.50,
            _count_at_least(train_blocks, "calmar", 0.0) == 6,
            _count_at_least(validation_blocks, "calmar", 0.0) == 6,
        ]
    )


def _walkforward_rejection(row: dict[str, object]) -> str | None:
    if float(row["train_block_min_calmar"]) < 0.50:
        return "train_block_min_calmar"
    if float(row["validation_block_min_calmar"]) < 0.50:
        return "validation_block_min_calmar"
    if int(row["train_blocks_positive"]) < 6:
        return "train_blocks_positive"
    if int(row["validation_blocks_positive"]) < 6:
        return "validation_blocks_positive"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
