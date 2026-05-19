from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
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
    evaluate_survival_candidate,
    split_train_validation,
    survival_score,
)
from trading_lab.survival import _run_candidate, _survival_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one walk-forward survival search stage.")
    parser.add_argument("--config", default="configs/survival_phase3_github.yaml")
    parser.add_argument("--seeds", default="configs/survival_phase2_seeds.json")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int)
    parser.add_argument("--max-candidates", type=int, default=360_000)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    total_stages = args.total_stages or config.total_stages
    data = load_market_data(config.data_path)
    seeds = _load_seeds(Path(args.seeds), data)
    grid = list(
        _stage_candidates(
            seeds,
            stage=args.stage,
            total_stages=total_stages,
            max_candidates=args.max_candidates,
        )
    )
    rows = [
        _evaluate_walkforward_candidate(
            data,
            params,
            initial_cash=config.execution.initial_cash,
            commission_bps=config.execution.commission_bps,
            slippage_bps=config.execution.slippage_bps,
        )
        for params in grid
    ]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"survival_walkforward_stage_{args.stage}.csv"
    pd.DataFrame(rows).sort_values("survival_score", ascending=False).to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "rows": len(rows),
                "seeds": len(seeds),
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


def _load_seeds(path: Path, data: pd.DataFrame) -> list[dict[str, object]]:
    seeds = json.loads(path.read_text(encoding="utf-8"))
    loaded = []
    seen_specs: set[str] = set()
    for seed in seeds:
        spec = str(seed["spec"])
        feature_name = str(seed["feature_name"])
        if feature_name not in data.columns or spec in seen_specs:
            continue
        seen_specs.add(spec)
        copied = dict(seed)
        copied["inverse_spec"] = _inverse_spec(spec)
        loaded.append(copied)
    if len(loaded) < 4:
        raise ValueError("walk-forward survival search needs at least four usable seeds")
    return loaded


def _inverse_spec(spec: str) -> str:
    parts = spec.split("|")
    if len(parts) != 5:
        raise ValueError(f"invalid feature spec: {spec}")
    parts[-1] = "-1"
    return "|".join(parts)


def _stage_candidates(
    seeds: list[dict[str, object]],
    *,
    stage: int,
    total_stages: int,
    max_candidates: int,
) -> Iterable[dict[str, object]]:
    produced = 0
    for params in _all_candidates(seeds):
        if produced >= max_candidates:
            return
        if produced % total_stages == stage:
            yield params
        produced += 1


def _all_candidates(seeds: list[dict[str, object]]) -> Iterable[dict[str, object]]:
    long_pool = seeds[:120]
    short_pool = seeds[:120]
    focused_long_pool = seeds[:70]
    focused_short_pool = seeds[:70]

    for long_combo in combinations(long_pool, 2):
        for short_combo in combinations(focused_short_pool, 2):
            if _combo_bucket(long_combo + short_combo, modulo=7) != 0:
                continue
            yield {
                "rule": "feature_vote_position",
                "long_specs": _join_specs(long_combo, inverse=False),
                "short_specs": _join_specs(short_combo, inverse=True),
                "long_min_votes": 1,
                "short_min_votes": 2,
                "combo_size": 4,
            }
            yield {
                "rule": "feature_vote_position",
                "long_specs": _join_specs(long_combo, inverse=False),
                "short_specs": _join_specs(short_combo, inverse=True),
                "long_min_votes": 2,
                "short_min_votes": 2,
                "combo_size": 4,
            }

    for long_combo in combinations(focused_long_pool, 3):
        for short_combo in combinations(focused_short_pool, 2):
            if _combo_bucket(long_combo + short_combo, modulo=18) != 0:
                continue
            yield {
                "rule": "feature_vote_position",
                "long_specs": _join_specs(long_combo, inverse=False),
                "short_specs": _join_specs(short_combo, inverse=True),
                "long_min_votes": 2,
                "short_min_votes": 2,
                "combo_size": 5,
            }
            yield {
                "rule": "feature_vote_position",
                "long_specs": _join_specs(long_combo, inverse=False),
                "short_specs": _join_specs(short_combo, inverse=True),
                "long_min_votes": 2,
                "short_min_votes": 1,
                "combo_size": 5,
            }


def _join_specs(combo: tuple[dict[str, object], ...], *, inverse: bool) -> str:
    key = "inverse_spec" if inverse else "spec"
    return ";".join(str(seed[key]) for seed in combo)


def _combo_bucket(combo: tuple[dict[str, object], ...], *, modulo: int) -> int:
    total = sum(int(seed["rank"]) for seed in combo)
    return total % modulo


def _evaluate_walkforward_candidate(
    data: pd.DataFrame,
    params: dict[str, object],
    *,
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
    if data.empty:
        return []
    index_chunks = [chunk for chunk in np.array_split(np.arange(len(data)), blocks) if len(chunk) > 0]
    chunks = [data.iloc[index_chunk].copy() for index_chunk in index_chunks]
    metrics = []
    for chunk in chunks:
        result = _run_candidate(chunk.copy(), params, initial_cash, commission_bps, slippage_bps)
        metrics.append(_survival_metrics(result.equity_curve, result.metrics, chunk))
    return metrics


def _min_value(metrics: list[dict[str, float]], key: str) -> float:
    if not metrics:
        return 0.0
    return float(min(metric[key] for metric in metrics))


def _count_at_least(metrics: list[dict[str, float]], key: str, threshold: float) -> int:
    return int(sum(1 for metric in metrics if metric[key] >= threshold))


def _walkforward_passes(train_blocks: list[dict[str, float]], validation_blocks: list[dict[str, float]]) -> int:
    return sum(
        [
            _min_value(train_blocks, "calmar") >= 0.0,
            _min_value(validation_blocks, "calmar") >= 0.0,
            _count_at_least(train_blocks, "calmar", 0.25) >= 4,
            _count_at_least(validation_blocks, "calmar", 0.25) >= 4,
        ]
    )


def _walkforward_rejection(row: dict[str, object]) -> str | None:
    if float(row["train_block_min_calmar"]) < 0.0:
        return "train_block_min_calmar"
    if float(row["validation_block_min_calmar"]) < 0.0:
        return "validation_block_min_calmar"
    if int(row["train_blocks_positive"]) < 4:
        return "train_blocks_positive"
    if int(row["validation_blocks_positive"]) < 4:
        return "validation_blocks_positive"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
