from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_optimization_config
from trading_lab.data_loader import load_market_data
from trading_lab.survival import evaluate_survival_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one phase-2 survival combination stage.")
    parser.add_argument("--config", default="configs/survival_phase2_github.yaml")
    parser.add_argument("--seeds", default="configs/survival_phase2_seeds.json")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int)
    parser.add_argument("--max-candidates", type=int, default=260_000)
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
        evaluate_survival_candidate(
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
    output_path = output_dir / f"survival_combo_stage_{args.stage}.csv"
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
        feature_name = str(seed["feature_name"])
        spec = str(seed["spec"])
        if feature_name not in data.columns or spec in seen_specs:
            continue
        seen_specs.add(spec)
        loaded.append(seed)
    if len(loaded) < 2:
        raise ValueError("phase-2 survival search needs at least two usable seeds")
    return loaded


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
    pair_seeds = seeds[:180]
    triple_seeds = seeds[:100]
    quad_seeds = seeds[:60]
    quint_seeds = seeds[:45]

    for combo in combinations(pair_seeds, 2):
        specs = _join_specs(combo)
        yield {"rule": "feature_vote", "feature_specs": specs, "min_votes": 1, "combo_size": 2}
        yield {"rule": "feature_vote", "feature_specs": specs, "min_votes": 2, "combo_size": 2}
        yield {"rule": "feature_score", "feature_specs": specs, "score_threshold": 0.0, "combo_size": 2}

    for combo in combinations(triple_seeds, 3):
        if _combo_bucket(combo, modulo=2) != 0:
            continue
        specs = _join_specs(combo)
        yield {"rule": "feature_vote", "feature_specs": specs, "min_votes": 2, "combo_size": 3}
        yield {"rule": "feature_vote", "feature_specs": specs, "min_votes": 3, "combo_size": 3}
        yield {"rule": "feature_score", "feature_specs": specs, "score_threshold": 0.0, "combo_size": 3}

    for combo in combinations(quad_seeds, 4):
        if _combo_bucket(combo, modulo=8) != 0:
            continue
        specs = _join_specs(combo)
        yield {"rule": "feature_vote", "feature_specs": specs, "min_votes": 3, "combo_size": 4}
        yield {"rule": "feature_score", "feature_specs": specs, "score_threshold": 0.0, "combo_size": 4}

    for combo in combinations(quint_seeds, 5):
        if _combo_bucket(combo, modulo=20) != 0:
            continue
        specs = _join_specs(combo)
        yield {"rule": "feature_vote", "feature_specs": specs, "min_votes": 3, "combo_size": 5}
        yield {"rule": "feature_score", "feature_specs": specs, "score_threshold": 0.0, "combo_size": 5}


def _join_specs(combo: tuple[dict[str, object], ...]) -> str:
    return ";".join(str(seed["spec"]) for seed in combo)


def _combo_bucket(combo: tuple[dict[str, object], ...], *, modulo: int) -> int:
    total = sum(int(seed["rank"]) for seed in combo)
    return total % modulo


if __name__ == "__main__":
    raise SystemExit(main())
